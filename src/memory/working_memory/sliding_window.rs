use std::collections::VecDeque;
use tokio::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::error::Error;
use tokio::time::{sleep, Duration};
use tokio::runtime::Runtime;

// use async_openai::{
//     types::{CreateChatCompletionRequest, ChatCompletionRequestMessage},
//     Client,
// };


//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow {
    window: VecDeque<Information>,
    capacity: usize,
    tag_count: usize,
    summary: Arc<RwLock<String>>,
}

impl SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity+1),
            capacity,
            tag_count: capacity,
            summary: Arc::new(RwLock::new(String::new())),
        }
    }
    //信息滑入
    pub async fn push(&mut self, mut value: Information) -> Result<(), Box<dyn Error>> {
        value = self.auto_tag(value);
        self.window.push_back(value);
        if self.window.len() == (self.capacity+1) {
            self.pop().await?;
        }
        Ok(())
    }
    //信息滑出，若信息被标记则进行摘要
    pub async fn pop(&mut self) -> Result<(), Box<dyn Error>> {
        let target = self.window.pop_front();
        if let Some(value) = target {
            if value.is_tagged() {
                self.summarize().await?;
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

    //将摘要记忆和当前滑动窗口信息合并提供LLM
    async fn summarize(&self) -> Result<Arc<RwLock<String>>, Box<dyn Error>> {
        let mut summary_text = match self.summary.write() {
            Ok(mut value) => value.clone(),
            Err(e) => {
                eprintln!("Summary Error: {}", e);
                String::new()
            }
        };

        for (index, i) in self.window.iter().enumerate() {
            summary_text.push_str(&index.to_string());
            summary_text.push_str(&i.to_string());
        }

        let summary_arc = self.summary.clone();

        match call_llm(&summary_text).await {
            Ok(response) => {
                match summary_arc.write() {
                    Ok(mut value) => {
                        println!("Summary updated in background.");
                        *value = response
                    },
                    Err(e) => eprintln!("LLM Error: {}", e),
                }
            }
            Err(e) => eprintln!("LLM Error: {}", e),
        }

        Ok(summary_arc)
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

fn test_summary(summary: String) -> String {
    println!("{}", summary.clone());
    summary
}


async fn call_llm(summary: &String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // let client = Client::new();

    // let request = CreateChatCompletionRequest {
    //     model: "unknown".to_string(),
    //     messages: vec![ChatCompletionRequestMessage {
    //         role: "user".to_string(),
    //         content: summary,
    //         ..Default::default()
    //     }],
    //     ..Default::default()
    // };
    // let response = client.chat().create(request).await?;
    // let output = response
    //     .choices
    //     .first()
    //     .and_then(|c| c.message.content.clone())
    //     .unwrap_or_default();
    sleep(Duration::from_millis(500)).await;
    let output = summary.clone();
    Ok(output)
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
