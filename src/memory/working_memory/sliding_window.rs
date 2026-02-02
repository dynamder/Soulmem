use std::collections::VecDeque;
use tokio::sync::mpsc;
// use async_openai::{
//     types::{CreateChatCompletionRequest, ChatCompletionRequestMessage},
//     Client,
// };


//滑动窗口（容器、容量、标记计数、摘要用临时储存）
pub struct SlidingWindow {
    window: VecDeque<Information>,
    capacity: usize,
    tag_count: usize,
    summary: String,
}

impl SlidingWindow {
    //新建
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity+1),
            capacity,
            tag_count: capacity,
            summary: String::new(),
        }
    }
    //信息滑入
    pub fn push(&mut self, mut value: Information) {
        value = self.auto_tag(value);
        self.window.push_back(value);
        if self.window.len() == (self.capacity+1) {
            self.pop();
        }
    }
    //信息滑出，若信息被标记则进行摘要
    pub fn pop(&mut self) {
        let target = self.window.pop_front();
        if let Some(value) = target {
            if value.is_tagged() {
                let summary = self.summarize();
                self.set_summary(test_summary(summary));
                // tokio::spawn(async move {
                //     match call_llm(summary).await {
                //         Ok(response) => {
                //             self.set_summary(response);
                //         }
                //         Err(e) => eprintln!("LLM Error"),
                //     }
                // });
            }
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
    //获取窗口中指定索引的信息
    pub fn get(&self, index: usize) -> Option<&Information> {
        self.window.get(index)
    }
    //获取摘要记忆
    pub fn get_summary(&self) -> &String {
        &self.summary
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
    fn auto_tag(&mut self, mut value: Information) -> Information {
        self.tag_count += 1;
        if self.tag_count >= self.capacity {
            value.tag_information();
            self.tag_count = 0;
        }
        value
    }

    //将摘要记忆和当前滑动窗口信息合并
    pub fn summarize(&mut self) -> String {
        let mut summary = self.summary.clone();
        for (index, i) in self.window.iter().enumerate() {
            summary.push_str(&index.to_string());
            summary.push_str(&i.to_string());
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

// async fn call_llm(summary: String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
//     let client = Client::new();

//     let request = CreateChatCompletionRequest {
//         model: "unknown".to_string(),
//         messages: vec![ChatCompletionRequestMessage {
//             role: "user".to_string(),
//             content: summary,
//             ..Default::default()
//         }],
//         ..Default::default()
//     };
//     let response = client.chat().create(request).await?;
//     let output = response
//         .choices
//         .first()
//         .and_then(|c| c.message.content.clone())
//         .unwrap_or_default();

//     Ok(output)
// }


#[cfg(test)]
mod slidingwindow_test{
    use super::*;

    // async fn SlidingWindowtest_call(summary: String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    //     Ok("success".to_string())
    }
    #[test]
    fn sliding_window_test_push(){
        let mut window = SlidingWindow::new(10);
        let info = Information::new("test1".to_string());
        window.push(info);
        let info2 = Information::new("test2".to_string());
        window.push(info2);
        assert_eq!(window.get(0).unwrap().text, "test1");
        assert_eq!(window.get(1).unwrap().text, "test2");
    }
    #[test]
    fn sliding_window_test_pop(){
        let mut window = SlidingWindow::new(10);
        let info = Information::new("test1".to_string());
        window.push(info);
        let info2 = Information::new("test2".to_string());
        window.push(info2);
        let _ = window.pop();
        assert_eq!(window.get(0).unwrap().text, "test2");
    }
    #[test]
    fn sliding_window_test_summary_and_tag(){
        let mut window = SlidingWindow::new(2);
        let info = Information::new("test1".to_string());
        window.push(info);
        let info2 = Information::new("test2".to_string());
        window.push(info2);
        let info3 = Information::new("test3".to_string());
        window.push(info3);
        assert_eq!(window.get_summary(), "0test21test3");
        let test = window.get(1);
        if let Some(value) = test {
            assert!(value.is_tagged());
        }
    }
}
