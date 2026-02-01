use std::collections::VecDeque;
use tokio::sync::mpsc;
use async_openai::{
    types::{CreateChatCompletionRequest, ChatCompletionRequestMessage},
    Client,
};

//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow<Information> {
    window: VecDeque<Information>,
    capacity: usize,
    tag_count: usize,
    summary: String,
}

impl<Information> SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            tag_count: 0,
            summary: String::new(),
        }
    }
    //信息滑入
    pub fn push(&mut self, value: Information) {
        value = self.auto_tag(value);
        if self.window.len() == self.capacity {
            self.pop();
        }
        self.window.push_back(value);
    }
    //信息滑出，若信息被标记则进行摘要
    pub fn pop(&mut self) {
        let target = self.window.pop_front();
        if target.is_tagged() {
            let summary = self.summarize();
            tokio::spawn(async move {
                match call_llm(summary).await {
                    Ok(response) => {
                        self.set_summary(response);
                    }
                    Err(e) => eprintln!("LLM error for id {}: {}", id, e),
        }
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
    //判断窗口是否为空
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
    //清空窗口内容
    pub fn clear(&mut self) {
        self.window.clear();
        self.tag_count = 0;
        self.summary.clear();
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
    fn auto_tag(&mut self, value: Information) -> Information {
        self.tag_count += 1;
        if self.tag_count >= self.capacity {
            value.tag_information();
            self.tag_count = 0;
        }
        value
    }

    //将摘要记忆和当前滑动窗口信息合并
    pub fn summarize(&mut self) -> String {
        let mut summary = self.summary;
        for (index, i) in self.window.iter().enumerate() {
            summary.push_str(index.to_string());
            summary.push_str(&i.text);
        }
        summary
    }
    //将返回摘要记忆存入
    pub fn set_summary(&mut self, summary: String) {
        self.summary = summary;
    }
    //将摘要记忆清空
    pub fn clear_summary(&mut self) {
        self.summary.clear();
    }

}

pub struct Information {
    pub text: String,
    pub tag: bool,
}

impl Information {
    pub fn new(text: String) -> Self {
        Self { text, false }
    }
    pub fn tag_information(&mut self) {
        self.tag = true;
    }
    pub fn untag_information(&mut self) {
        self.tag = false;
    }
    pub fn is_tagged(&self) -> bool {
        self.tag
    }
    pub fn get_mut_capacity(&mut self) -> &mut usize {
        &mut self.capacity
    }
}

async fn call_llm(summary: String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();

    let request = CreateChatCompletionRequest {
        model: "unknown".to_string(),
        messages: vec![ChatCompletionRequestMessage {
            role: "user".to_string(),
            content: summary,
            ..Default::default()
        }],
        ..Default::default()
    };
    let response = client.chat().create(request).await?;
    let output = response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(output)
}


#[cfg(test)]
fn test_call(summary: String) -> Result<String, Err>{
    Ok("success".to_string())
}
