use crate::memory::working_memory::llm::{
    client::LlmClient,
    config::LLMConfig,
    prompt::{PromptBuilder, PromptHistoryBuilder},
};
use anyhow::{Context, Error, Result};
use async_openai::{
    Client,
    config::Config,
    types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest, Role,
    },
};
use dotenvy::{dotenv, var};
use parking_lot::RwLock as ParkRwLock;
use secrecy::{ExposeSecret, SecretString};
use std::mem::take;
use std::sync::Arc;
use std::{collections::VecDeque, sync::atomic::AtomicUsize};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep};

//滑动窗口（容器、容量、标记计数、摘要用临时储存）
#[derive(Debug)]
pub struct SlidingWindow {
    window: Arc<ParkRwLock<VecDeque<Information>>>,
    capacity: AtomicUsize,
    tag_count: AtomicUsize,
    summary: Arc<ParkRwLock<Summary>>,
}
impl Default for SlidingWindow {
    fn default() -> Self {
        Self::new(20)
    }
}

impl SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        dotenv().ok();
        Self {
            window: Arc::new(ParkRwLock::new(VecDeque::with_capacity(capacity + 1))),
            capacity: AtomicUsize::from(capacity),
            tag_count: AtomicUsize::from(capacity),
            summary: Arc::new(ParkRwLock::new(Summary::new())),
        }
    }
    //信息滑入
    pub async fn push(&self, value: &str, role: &str, client: &LlmClient) -> Result<()> {
        let mut text = Information::new(value, role);
        text = self.auto_tag(text);

        {
            let mut window = self.window.write();
            window.push_back(text);
        }

        let window_len = {
            let window = self.window.read();
            window.len()
        };

        //TODO: may have problem with the memory ordering, test it.
        let window_capacity = self.capacity.load(std::sync::atomic::Ordering::Relaxed);

        if window_len == window_capacity + 1 {
            self.pop(client).await?;
        }
        Ok(())
    }
    //信息滑出，若信息被标记则进行摘要
    pub async fn pop(&self, client: &LlmClient) -> Result<()> {
        let target = {
            let mut window = self.window.write();
            window.pop_front()
        };
        if let Some(value) = target {
            if value.is_tagged() {
                let _ = self.summarize(client).await?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn window(&self) -> &Arc<ParkRwLock<VecDeque<Information>>> {
        &self.window
    }

    #[cfg(test)]
    pub fn summary(&self) -> &Arc<ParkRwLock<Summary>> {
        &self.summary
    }
    pub fn get_windows(&self) -> Arc<[Information]> {
        let window = self.window.read();
        Arc::from(window.iter().cloned().collect::<Vec<_>>())
    }
    pub fn get_summary(&self) -> Arc<str> {
        Arc::from(self.summary.read().get())
    }

    //获取窗口大小
    pub fn len(&self) -> usize {
        self.window.read().len()
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
    pub fn get(&self, index: usize) -> Option<Information> {
        self.window.read().get(index).cloned()
    }

    //判断窗口是否为空
    pub fn is_empty(&self) -> bool {
        self.window.read().is_empty()
    }
    //清空窗口内容
    pub fn clear(&self) {
        self.window.write().clear();
        self.tag_count
            .store(0, std::sync::atomic::Ordering::Release);
    }
    //标记用
    pub fn tag_information(&self, index: usize) {
        if index < self.capacity.load(std::sync::atomic::Ordering::Relaxed) {
            let mut window = self.window.write();
            window[index].tag_information();
        }
    }
    //取消标记用
    pub fn untag_information(&self, index: usize) {
        if index < self.capacity.load(std::sync::atomic::Ordering::Relaxed) {
            let mut window = self.window.write();
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
    fn prepare_prompt(&self) -> Vec<ChatCompletionRequestMessage> {
        let system_prompt = std::iter::once(
            ChatCompletionRequestSystemMessage::from(
                "You are a summary and compact agent. Based on the following conversation (which is happened before), provide a new summary.\n Only Output the summary content, no other text."
            ).into()
        );

        let snapshot = std::iter::once(self.summary.read().build_raw_prompt())
            .chain(self.window.read().iter().map(|msg| msg.build_raw_prompt()))
            .fold(String::new(), |acc, item| {
                acc + &format!("[{}]: {}\n", item.0, item.1)
            });

        system_prompt
            .chain(std::iter::once(
                ChatCompletionRequestUserMessage::from(snapshot).into(),
            ))
            .collect()
    }

    //将摘要记忆和当前滑动窗口信息合并提供LLM
    async fn summarize(&self, client: &LlmClient) -> Result<()> {
        let prompt_history = self.prepare_prompt();
        let mut response = client.call_llm(prompt_history).await?;

        if response.is_empty() {
            return Err(anyhow::anyhow!("Expected at least 1 response, got empty"));
        }

        self.summary
            .write()
            .update(std::mem::take(&mut response[0]));

        Ok(())
    }
}

impl PromptHistoryBuilder for SlidingWindow {
    fn build_history(&self) -> Vec<ChatCompletionRequestMessage> {
        std::iter::once(self.summary.read().build_prompt())
            .chain(self.window.read().iter().map(|msg| msg.build_prompt()))
            .collect()
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

impl PromptBuilder for Information {
    fn build_prompt(&self) -> ChatCompletionRequestMessage {
        self.to_message()
    }
    fn build_raw_prompt(&self) -> (&str, Role) {
        match self {
            Information::User(info) => (info.get_str(), Role::User),
            Information::Assistant(info) => (info.get_str(), Role::Assistant),
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

#[derive(Debug)]
pub struct Summary {
    summary: String,
}
impl Summary {
    pub fn new() -> Self {
        Self {
            summary: String::new(),
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
    pub fn update(&mut self, content: impl Into<String>) {
        self.summary = content.into();
    }
    pub fn get(&self) -> &str {
        self.summary.as_str()
    }
}
impl PromptBuilder for Summary {
    fn build_prompt(&self) -> ChatCompletionRequestMessage {
        //这是Agent的Summary，由LLM生成
        ChatCompletionRequestAssistantMessage::from(self.summary.as_str()).into()
    }
    fn build_raw_prompt(&self) -> (&str, Role) {
        (self.summary.as_str(), Role::Assistant)
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
            window.get(0).expect("not found this information").get_str(),
            "user_info"
        );
        assert_eq!(
            window.get(1).expect("not found this information").get_str(),
            "assistant_info"
        );
        assert_eq!(window.get_windows()[0].get_str(), "user_info");
        assert_eq!(window.get_windows()[1].get_str(), "assistant_info");
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
            window.get(0).expect("not found this information").get_str(),
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
        let user_info = "What is Rust?";
        window
            .push(user_info, "user", &client)
            .await
            .expect("Failed to push user_information");
        let assistant_info = "Rust is a systems programming language that runs blazingly fast, it emphasize on memory safety.";
        window
            .push(assistant_info, "assistant", &client)
            .await
            .expect("Failed to push assistant_information");
        let user_info2 = "Is Rust hard to learn?";
        window
            .push(user_info2, "user", &client)
            .await
            .expect("Failed to push user_information");
        println!("{}", window.summary.read().summary)
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
        assert_eq!(window.len(), 100);
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
            window_read.len()
        });

        write_handle.await.expect("Write task join failed");
        let len = read_handle.await.expect("Read task join failed");
        assert!(len > 0);
        assert_eq!(window.len(), 25);
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

        let len = window.len();
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
                (&*window1)
                    .push(&format!("t1_{}", i), "user", &client1)
                    .await?;
            }
            Ok::<(), anyhow::Error>(())
        });

        let handle2 = tokio::spawn(async move {
            for i in 0..5 {
                (&*window2)
                    .push(&format!("t2_{}", i), "user", &client2)
                    .await?;
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

        assert_eq!(window.len(), 10);
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
