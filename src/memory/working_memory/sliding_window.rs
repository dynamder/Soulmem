use std::collections::VecDeque;
use tokio::sync::mpsc;
use std::sync::{Arc, RwLock};
use tokio::time::{sleep, Duration};
use tokio::runtime::Runtime;
use anyhow::{Error, Result, Context};
use super::llm::{MyConfig, AIConfig, LlmClient, PromptBuilder};
use async_openai::{
    types::chat::{CreateChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPartText, ChatCompletionRequestUserMessage},
    Client, config::Config,
};


//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow {
    window: VecDeque<Information>,
    capacity: usize,
    tag_count: usize,
    summary: Arc<RwLock<MergedInformation>>,
    llm_config: MyConfig,
}

impl SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity + 1),
            capacity,
            tag_count: capacity,
            summary: Arc::new(RwLock::new(MergedInformation::new())),
            llm_config: MyConfig::new(),
        }
    }
    //信息滑入
    pub async fn push(&mut self, mut value: Information) -> Result<()> {
        value = self.auto_tag(value);
        self.window.push_back(value);
        if self.window.len() == (self.capacity+1) {
            self.pop().await?;
        }
        Ok(())
    }
    //信息滑出，若信息被标记则进行摘要
    pub async fn pop(&mut self) -> Result<()> {
        let target = self.window.pop_front();
        if let Some(value) = target {
            if value.is_tagged() {
                let _ = self.summarize().await?;
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
    fn merge(&self) {
        let mut summary_text = self.summary.write().unwrap_or_else(|e| e.into_inner());
        let mut merged_info = String::new();
        for (index, i) in self.window.iter().enumerate() {
            merged_info.push_str(&index.to_string());
            merged_info.push_str(&i.to_string());
        }
        summary_text.text = merged_info;

    }

    //将摘要记忆和当前滑动窗口信息合并提供LLM
    async fn summarize(&self) -> Result<String> {
        self.merge();
        let mut summary_arc = self.summary.write().unwrap_or_else(|e| e.into_inner());
        let response = summary_arc.call_llm(self.llm_config.clone()).await?;
        Ok(response)
    }

}

pub struct Information {
    pub text: String,
    pub tag: bool,
}

impl Information {
    pub fn new(text: String) -> Self {
        Self { text, tag: false }
    }
    pub fn tag_information(&mut self) {
        self.tag = true
    }
    pub fn untag_information(&mut self) {
        self.tag = false
    }
    pub fn is_tagged(&self) -> bool {
        self.tag
    }
    pub fn to_string(&self) -> String {
        self.text.clone()
    }
}

struct MergedInformation {
    text: String,
    Array: Vec<ChatCompletionRequestUserMessageContentPart>,
}
impl MergedInformation {
    pub fn new() -> Self {
        Self { text: String::new(), Array: Vec::new() }
    }
    pub fn push_content(&mut self, content: ChatCompletionRequestUserMessageContentPart) {
        self.Array.push(content);
    }
    //将整合后的自身交给call_llm处理，并根据结果自动更新自身
    async fn call_llm(&mut self, config: MyConfig) -> Result<String> {
        let client:LlmClient<MyConfig> = LlmClient::new(config);
        let response = client.call_llm(self).await?;
        let output = response.join(" ");
        self.merge_summary(&output);
        Ok(output)
    }
    pub fn merge_summary(&mut self, content: &str) {
        self.Array.push(ChatCompletionRequestUserMessageContentPart::Text(ChatCompletionRequestMessageContentPartText::from(content)));
    }
}
impl PromptBuilder for MergedInformation {
    fn build_prompt(&self) -> Vec<ChatCompletionRequestMessage> {
        let mut messages = Vec::new();
        messages.push(
            ChatCompletionRequestMessage::from(<ChatCompletionRequestUserMessage as From<&str>>::from(&self.text)));
        messages.push(
            ChatCompletionRequestMessage::from(<ChatCompletionRequestUserMessage as From<ChatCompletionRequestUserMessageContent>>::from(ChatCompletionRequestUserMessageContent::from(self.Array.clone()))));
        messages
    }
}





#[cfg(test)]
mod slidingwindow_test{
    use super::*;

    #[tokio::test]
    async fn sliding_window_test_push(){
        let mut window = SlidingWindow::new(10);
        let info = Information::new("test1".to_string());
        window.push(info).await;
        let info2 = Information::new("test2".to_string());
        window.push(info2).await;
        assert_eq!(window.get(0).expect("not found this information").text, "test1");
        assert_eq!(window.get(1).expect("not found this information").text, "test2");
    }
    #[tokio::test]
    async fn sliding_window_test_pop(){
        let mut window = SlidingWindow::new(10);
        let info = Information::new("test1".to_string());
        window.push(info).await;
        let info2 = Information::new("test2".to_string());
        window.push(info2).await;
        window.pop().await;
        assert_eq!(window.get(0).expect("not found this information").text, "test2");
    }
    #[tokio::test]
    async fn sliding_window_test_summary_and_tag(){
        let mut window = SlidingWindow::new(2);
        let info = Information::new("test1".to_string());
        window.push(info).await;
        let info2 = Information::new("test2".to_string());
        window.push(info2).await;
        let info3 = Information::new("test3".to_string());
        window.push(info3).await;
        assert_eq!(window.summary.read().unwrap().as_str(), "0test21test3");
        let test = window.get(1);
        if let Some(value) = test {
            assert!(value.is_tagged());
        }
    }
}
