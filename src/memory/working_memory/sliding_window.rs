use std::collections::VecDeque;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};
use tokio::runtime::Runtime;
use anyhow::{Error, Result, Context};
use crate::memory::working_memory::llm::{config::LLMConfig, client::LlmClient, prompt::PromptBuilder};
use async_openai::{
    types::chat::{CreateChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContentPart, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPartText, ChatCompletionRequestUserMessage, ChatCompletionRequestAssistantMessage},
    Client, config::Config,
};
use secrecy::{SecretString, ExposeSecret};
use std::mem::take;
use dotenvy::{dotenv, var};

//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow {
    window: VecDeque<Information>,
    capacity: usize,
    tag_count: usize,
    summary: Arc<RwLock<MergedInformation>>,
}

impl SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        dotenv().ok();
        Self {
            window: VecDeque::with_capacity(capacity + 1),
            capacity,
            tag_count: capacity,
            summary: Arc::new(RwLock::new(MergedInformation::new())),
        }
    }
    //信息滑入
    pub async fn push(&mut self, value: &str, role: &str, client: &LlmClient) -> Result<()> {
        let mut text = Information::new(value, role);
        text = self.auto_tag(text);
        self.window.push_back(text);
        if self.window.len() == (self.capacity+1) {
            self.pop(client).await?;
        }
        Ok(())
    }
    //信息滑出，若信息被标记则进行摘要
    pub async fn pop(&mut self, client: &LlmClient) -> Result<()> {
        let target = self.window.pop_front();
        if let Some(value) = target {
            if value.is_tagged() {
                let _ = self.summarize(client).await?;
            }
        }
        Ok(())
    }
    //获取窗口大小
    pub fn len(&self) -> usize {
        self.window.len()
    }
    //获取窗口容量
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }
    //获取窗口容量（可变）
    pub fn get_mut_capacity(&mut self) -> &mut usize {
        &mut self.capacity
    }
    //获取窗口中指定索引的信息
    pub fn get(&self, index: usize) -> Option<&Information> {
        self.window.get(index)
    }

    //判断窗口是否为空
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
    //清空窗口内容
    pub fn clear(&mut self) {
        self.window.clear();
        self.tag_count = 0;
    }
    //标记用
    pub fn tag_information(&mut self, index: usize) {
        if index < self.capacity {
            self.window[index].tag_information();
        }
    }
    //取消标记用
    pub fn untag_information(&mut self, index: usize) {
        if index < self.capacity {
            self.window[index].untag_information();
        }
    }
    //每滑入capacity次信息时进行一次标记
    fn auto_tag(&mut self, mut value: Information) -> Information {
        self.tag_count += 1;
        if self.tag_count >= self.capacity {
            value.tag_information();
            self.tag_count = 0;
        }
        value
    }
    //整合摘要记忆和窗口信息
    async fn merge(&self) {
        let mut messages = self.summary.write().await;
        let mut previous = ChatCompletionRequestUserMessage::from(messages.previous_summary.as_str()).into();
        messages.content.clear();
        messages.content.push(ChatCompletionRequestSystemMessage::from(
            "Based on the summary of previous conversation and the information currently in the window, provide a new overall summary.").into());
        messages.content.push(previous);
        for message in self.window.iter() {
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

    async fn call_llm(&self, client: &LlmClient, merged: &mut MergedInformation) -> Result<String> {
        let response = client.call_llm(merged).await?;
        let output = response.join(" ");
        merged.merge_summary(&output);
        Ok(output)
    }
}

pub enum Information {
    User(UserInformation),
    Assistant(AssistantInformation)
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
            Information::User(info) => ChatCompletionRequestMessage::from(ChatCompletionRequestUserMessage::from(info.get_str())).into(),
                Information::Assistant(info) => ChatCompletionRequestMessage::from(ChatCompletionRequestAssistantMessage::from(info.get_str())).into()
        }
    }
}

pub struct UserInformation {
    pub text: String,
    pub tag: bool,
}

impl UserInformation {
    pub fn new(text: &str) -> Self {
        Self { text: text.to_string(), tag: false }
    }
    pub fn get_str(&self) -> &str {
        &self.text
    }
    pub fn get_mut_str(&mut self) -> &mut String {
        &mut self.text
    }
}

pub struct AssistantInformation {
    pub text: String,
    pub tag: bool,
}

impl AssistantInformation {
    pub fn new(text: &str) -> Self {
        Self { text: text.to_string(), tag: false }
    }
    pub fn get_str(&self) -> &str {
        &self.text
    }
    pub fn get_mut_str(&mut self) -> &mut String {
        &mut self.text
    }
}

struct MergedInformation {
    content: Vec<ChatCompletionRequestMessage>,
    previous_summary: String
}
impl MergedInformation {
    pub fn new() -> Self {
        Self { content: Vec::new(), previous_summary: String::new() }
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
mod slidingwindow_test{
    use super::*;

    #[tokio::test]
    async fn sliding_window_test_push(){
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(&var("API_KEY").unwrap_or_default(), &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default()));
        let mut window = SlidingWindow::new(10);
        let user_info = "user_info";
        window.push(user_info, "user", &client).await.expect("Failed to push user_information");
        let assistant_info = "assistant_info";
        window.push(assistant_info, "assistant", &client).await.expect("Failed to push assistant_information");
        assert_eq!(window.get(0).expect("not found this information").get_str(), "user_info");
        assert_eq!(window.get(1).expect("not found this information").get_str(), "assistant_info");
    }
    #[tokio::test]
    async fn sliding_window_test_pop(){
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(&var("API_KEY").unwrap_or_default(), &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default()));
        let mut window = SlidingWindow::new(10);
        let user_info = "user_info";
        window.push(user_info, "user", &client).await.expect("Failed to push user_information");
        let assistant_info = "assistant_info";
        window.push(assistant_info, "assistant", &client).await.expect("Failed to push assistant_information");
        window.pop(&client).await.expect("Failed to pop information");
        assert_eq!(window.get(0).expect("not found this information").get_str(), "assistant_info");
    }
    #[tokio::test]
    async fn sliding_window_test_summary(){
        dotenvy::dotenv().ok();
        let client = LlmClient::new(LLMConfig::new(&var("API_KEY").unwrap_or_default(), &var("API_BASE").unwrap_or_default(),
            &var("MODEL").unwrap_or_default()));
        let mut window = SlidingWindow::new(2);
        let user_info = "user_info";
        window.push(user_info, "user", &client).await.expect("Failed to push user_information");
        let assistant_info = "assistant_info";
        window.push(assistant_info, "assistant", &client).await.expect("Failed to push assistant_information");
        let user_info2 = "user_info2";
        window.push(user_info2, "user", &client).await.expect("Failed to push user_information");
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
}
