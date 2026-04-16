use crate::memory::working_memory::llm::{
    client::LlmClient, config::LLMConfig, prompt::PromptBuilder,
};
use anyhow::{Context, Error, Result};
use async_openai::{
    Client,
    config::Config,
    types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
    },
};
use dotenvy::{dotenv, var};
use secrecy::{ExposeSecret, SecretString};
use std::mem::take;
use std::sync::Arc;
use std::{collections::VecDeque, sync::atomic::AtomicUsize};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep};

//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow {
    window: Arc<RwLock<VecDeque<Information>>>,
    capacity: AtomicUsize,
    tag_count: AtomicUsize,
    summary: Arc<RwLock<MergedInformation>>,
}

impl SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        dotenv().ok();
        Self {
            window: Arc::new(RwLock::new(VecDeque::with_capacity(capacity + 1))),
            capacity: AtomicUsize::from(capacity),
            tag_count: AtomicUsize::from(capacity),
            summary: Arc::new(RwLock::new(MergedInformation::new())),
        }
    }
    //信息滑入
    pub async fn push(&self, value: &str, role: &str, client: &LlmClient) -> Result<()> {
        let mut text = Information::new(value, role);
        text = self.auto_tag(text);

        let mut window = self.window.write().await;
        window.push_back(text);

        //TODO: may have problem with the memory ordering, test it.
        let window_capacity = self.capacity.load(std::sync::atomic::Ordering::Relaxed);

        if window.len() == window_capacity + 1 {
            drop(window);
            self.pop(client).await?;
        }
        Ok(())
    }
    //信息滑出，若信息被标记则进行摘要
    pub async fn pop(&self, client: &LlmClient) -> Result<()> {
        let target = {
            let mut window = self.window.write().await;
            window.pop_front()
        };
        if let Some(value) = target {
            if value.is_tagged() {
                let _ = self.summarize(client).await?;
            }
        }
        Ok(())
    }
    pub async fn get_windows(&self) -> Arc<[Information]> {
        let window = self.window.read().await;
        Arc::from(window.iter().cloned().collect::<Vec<_>>())
    }
    //获取窗口大小
    pub async fn len(&self) -> usize {
        self.window.read().await.len()
    }
    //获取窗口容量
    pub fn get_capacity(&self) -> usize {
        //TODO: test the memory ordering
        self.capacity.load(std::sync::atomic::Ordering::Relaxed)
    }
    //获取窗口容量（可变）
    pub fn set_capacity(&self, val: usize) {
        self.capacity
            .store(val, std::sync::atomic::Ordering::Release);
    }
    //获取窗口中指定索引的信息
    pub async fn get(&self, index: usize) -> Option<Information> {
        self.window.read().await.get(index).cloned()
    }

    //判断窗口是否为空
    pub async fn is_empty(&self) -> bool {
        self.window.read().await.is_empty()
    }
    //清空窗口内容
    pub async fn clear(&self) {
        self.window.write().await.clear();
        self.tag_count
            .store(0, std::sync::atomic::Ordering::Release);
    }
    //标记用
    pub async fn tag_information(&self, index: usize) {
        if index < self.capacity.load(std::sync::atomic::Ordering::Relaxed) {
            let mut window = self.window.write().await;
            window[index].tag_information();
        }
    }
    //取消标记用
    pub async fn untag_information(&self, index: usize) {
        if index < self.capacity.load(std::sync::atomic::Ordering::Relaxed) {
            let mut window = self.window.write().await;
            window[index].untag_information();
        }
    }
    //每滑入capacity次信息时进行一次标记
    fn auto_tag(&self, mut value: Information) -> Information {
        self.tag_count
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if self.tag_count.load(std::sync::atomic::Ordering::Relaxed)
            >= self.capacity.load(std::sync::atomic::Ordering::Relaxed)
        {
            value.tag_information();
            self.tag_count
                .store(0, std::sync::atomic::Ordering::Release);
        }
        value
    }
    //整合摘要记忆和窗口信息
    async fn merge(&self) {
        let mut messages = self.summary.write().await;
        let mut previous =
            ChatCompletionRequestUserMessage::from(messages.previous_summary.as_str()).into();
        messages.content.clear();
        messages.content.push(ChatCompletionRequestSystemMessage::from(
            "Based on the summary of previous conversation and the information currently in the window, provide a new overall summary.").into());
        messages.content.push(previous);

        let window = self.window.read().await;
        for message in window.iter() {
            messages.content.push(message.to_message())
        }
    }

    //将摘要记忆和当前滑动窗口信息合并提供LLM
    async fn summarize(&self, client: &LlmClient) -> Result<String> {
        self.merge().await;
        let mut summary_arc = self.summary.write().await;
        let response = self.call_llm(client, &mut *summary_arc).await?;
        Ok(response)
    }

    // 获取当前摘要
    pub async fn get_summary_text(&self) -> String {
        let summary = self.summary.read().await;
        summary.get_previous_summary()
    }

    // 清空当前摘要
    pub async fn clear_summary(&self) {
        let mut summary = self.summary.write().await;
        summary.previous_summary.clear();
        summary.content.clear();
    }

    async fn call_llm(&self, client: &LlmClient, merged: &mut MergedInformation) -> Result<String> {
        let response = client.call_llm(merged).await?;
        let output = response.join(" ");
        merged.merge_summary(&output);
        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Information {
    User(UserInformation),
    Assistant(AssistantInformation),
}

impl From<UserInformation> for Information {
    fn from(info: UserInformation) -> Self {
        Information::User(info)
    }
}

impl From<AssistantInformation> for Information {
    fn from(info: AssistantInformation) -> Self {
        Information::Assistant(info)
    }
}

impl Information {
    pub fn new(value: &str, role: &str) -> Self {
        //TODO: careful with this string compare
        match role {
            "user" => Information::User(UserInformation::new(value)),
            "assistant" => Information::Assistant(AssistantInformation::new(value)),
            _ => Information::User(UserInformation::new(value)),
        }
    }
    pub fn is_tagged(&self) -> bool {
        match self {
            Information::User(info) => info.tag,
            Information::Assistant(info) => info.tag,
        }
    }
    pub fn tag_information(&mut self) {
        match self {
            Information::User(info) => info.tag = true,
            Information::Assistant(info) => info.tag = true,
        }
    }
    pub fn untag_information(&mut self) {
        match self {
            Information::User(info) => info.tag = false,
            Information::Assistant(info) => info.tag = false,
        }
    }
    pub fn get_str(&self) -> &str {
        match self {
            Information::User(info) => &info.text,
            Information::Assistant(info) => &info.text,
        }
    }
    pub fn to_message(&self) -> ChatCompletionRequestMessage {
        match self {
            Information::User(info) => ChatCompletionRequestMessage::from(
                ChatCompletionRequestUserMessage::from(info.get_str()),
            )
            .into(),
            Information::Assistant(info) => ChatCompletionRequestMessage::from(
                ChatCompletionRequestAssistantMessage::from(info.get_str()),
            )
            .into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserInformation {
    pub text: Arc<str>,
    pub tag: bool,
}

impl UserInformation {
    pub fn new(text: &str) -> Self {
        Self {
            text: Arc::from(text),
            tag: false,
        }
    }
    pub fn get_str(&self) -> &str {
        &self.text
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantInformation {
    pub text: Arc<str>,
    pub tag: bool,
}

impl AssistantInformation {
    pub fn new(text: &str) -> Self {
        Self {
            text: Arc::from(text),
            tag: false,
        }
    }
    pub fn get_str(&self) -> &str {
        &self.text
    }
}

struct MergedInformation {
    content: Vec<ChatCompletionRequestMessage>,
    previous_summary: String,
}
impl MergedInformation {
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
            previous_summary: String::new(),
        }
    }
    // //将整合后的自身交给call_llm处理，并根据结果自动更新自身
    // async fn call_llm(&mut self, config: MyConfig) -> Result<String> {
    //     let client:LlmClient<MyConfig> = LlmClient::new(config);
    //     let response = client.call_llm(self, 1).await?;
    //     let output = response.join(" ");
    //     self.merge_summary(&output);
    //     Ok(output)
    // }
    pub fn merge_summary(&mut self, content: &str) {
        self.previous_summary.push_str(content);
    }
    pub fn get_previous_summary(&self) -> String {
        self.previous_summary.clone()
    }
}
impl PromptBuilder for MergedInformation {
    fn build_prompt(&mut self) -> Vec<ChatCompletionRequestMessage> {
        take(&mut self.content)
    }
}

#[cfg(test)]
mod slidingwindow_test {
    use super::*;

    #[tokio::test]
    async fn sliding_window_test_push() {
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let mut window = SlidingWindow::new(10);
        let user_info = "user_info";
        window
            .push(user_info, "user", &client)
            .await
            .expect("Failed to push user_information");
        let assistant_info = "assistant_info";
        window
            .push(assistant_info, "assistant", &client)
            .await
            .expect("Failed to push assistant_information");
        assert_eq!(
            window
                .get(0)
                .await
                .expect("not found this information")
                .get_str(),
            "user_info"
        );
        assert_eq!(
            window
                .get(1)
                .await
                .expect("not found this information")
                .get_str(),
            "assistant_info"
        );
        assert_eq!(window.get_windows().await[0].get_str(), "user_info");
        assert_eq!(window.get_windows().await[1].get_str(), "assistant_info");
    }
    #[tokio::test]
    async fn sliding_window_test_pop() {
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let mut window = SlidingWindow::new(10);
        let user_info = "user_info";
        window
            .push(user_info, "user", &client)
            .await
            .expect("Failed to push user_information");
        let assistant_info = "assistant_info";
        window
            .push(assistant_info, "assistant", &client)
            .await
            .expect("Failed to push assistant_information");
        window
            .pop(&client)
            .await
            .expect("Failed to pop information");
        assert_eq!(
            window
                .get(0)
                .await
                .expect("not found this information")
                .get_str(),
            "assistant_info"
        );
    }
    #[tokio::test]
    async fn sliding_window_test_summary() {
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let mut window = SlidingWindow::new(2);
        let user_info = "user_info";
        window
            .push(user_info, "user", &client)
            .await
            .expect("Failed to push user_information");
        let assistant_info = "assistant_info";
        window
            .push(assistant_info, "assistant", &client)
            .await
            .expect("Failed to push assistant_information");
        let user_info2 = "user_info2";
        window
            .push(user_info2, "user", &client)
            .await
            .expect("Failed to push user_information");
        println!("{}", window.summary.read().await.previous_summary)
    }

    // #[tokio::test]
    // async fn sliding_window_test_summary2(){
    //     dotenvy::dotenv().ok();
    //     let client = LlmClient::new(LLMConfig::new(&var("API_KEY").unwrap_or_default(), &var("API_BASE").unwrap_or_default(),
    //         &var("MODEL").unwrap_or_default()));
    //     let mut window = SlidingWindow::new(3);
    //     {let mut summary = window.summary.write().await;
    //         summary.previous_summary = "".to_string();}
    //     let user_info = "user_info";
    //     window.push(user_info, "user", &client).await.expect("Failed to push user_information");
    //     let assistant_info = "assistant_info";
    //     window.push(assistant_info, "assistant", &client).await.expect("Failed to push assistant_information");
    //     let user_info2 = "user_info2";
    //     window.push(user_info2, "user", &client).await.expect("Failed to push user_information");
    //     let assistant_info2 = "assistant_info2";
    //     window.push(assistant_info2, "assistant", &client).await.expect("Failed to push assistant_information");
    //     println!("{}", window.summary.read().await.previous_summary)
    // }

    #[tokio::test]
    async fn test_concurrent_push() {
        dotenvy::dotenv().ok();
        let client1 = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let client2 = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let window = Arc::new(SlidingWindow::new(100));
        let window1 = window.clone();
        let window2 = window.clone();

        let handle = tokio::spawn(async move {
            for i in 0..50 {
                (&*window1)
                    .push(&format!("user_{}", i), "user", &client1)
                    .await
                    .expect("Failed to push user_information");
            }
        });

        for i in 50..100 {
            (&*window2)
                .push(&format!("user_{}", i), "user", &client2)
                .await
                .expect("Failed to push user_information");
        }

        handle.await.expect("Task join failed");
        assert_eq!(window.len().await, 100);
    }

    #[tokio::test]
    async fn test_concurrent_read_write() {
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let window = Arc::new(SlidingWindow::new(50));
        let window_write = window.clone();
        let window_read = window.clone();

        let write_handle = tokio::spawn(async move {
            for i in 0..25 {
                (&*window_write)
                    .push(&format!("msg_{}", i), "user", &client)
                    .await
                    .expect("Failed to push");
            }
        });

        let read_handle = tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            window_read.len().await
        });

        write_handle.await.expect("Write task join failed");
        let len = read_handle.await.expect("Read task join failed");
        assert!(len > 0);
        assert_eq!(window.len().await, 25);
    }

    #[tokio::test]
    async fn test_concurrent_pop_and_read() {
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let window = Arc::new(SlidingWindow::new(10));

        for i in 0..5 {
            window
                .push(&format!("initial_{}", i), "user", &client)
                .await
                .expect("Failed to push");
        }

        let window_clone = window.clone();
        let pop_handle = tokio::spawn(async move {
            for _ in 0..3 {
                (&*window_clone).pop(&client).await.expect("Failed to pop");
            }
        });

        pop_handle.await.expect("Pop task join failed");
        
        let len = window.len().await;
        assert_eq!(len, 2);
    }

    #[tokio::test]
    async fn test_clone_is_thread_safe() {
        dotenvy::dotenv().ok();
        let client1 = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let client2 = LlmClient::new(LLMConfig::new(
            &var("API_KEY").unwrap_or_default(),
            &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default(),
        ));
        let window = Arc::new(SlidingWindow::new(10));
        let window1 = window.clone();
        let window2 = window.clone();

        let handle1 = tokio::spawn(async move {
            for i in 0..5 {
                (&*window1).push(&format!("t1_{}", i), "user", &client1).await?;
            }
            Ok::<(), anyhow::Error>(())
        });

        let handle2 = tokio::spawn(async move {
            for i in 0..5 {
                (&*window2).push(&format!("t2_{}", i), "user", &client2).await?;
            }
            Ok::<(), anyhow::Error>(())
        });

        handle1
            .await
            .expect("Task 1 join failed")
            .expect("Task 1 failed");
        handle2
            .await
            .expect("Task 2 join failed")
            .expect("Task 2 failed");

        assert_eq!(window.len().await, 10);
    }

    #[tokio::test]
    async fn test_capacity_and_len_concurrent() {
        let window = Arc::new(SlidingWindow::new(20));

        assert_eq!(window.get_capacity(), 20);
        window.set_capacity(30);
        assert_eq!(window.get_capacity(), 30);

        let window_clone = window.clone();
        let window_for_check = window.clone();
        let handle = tokio::spawn(async move {
            window_clone.set_capacity(15);
            window_clone.get_capacity()
        });

        let capacity = handle.await.expect("Task join failed");
        assert_eq!(capacity, 15);
        assert_eq!(window_for_check.get_capacity(), 15);
    }
}
