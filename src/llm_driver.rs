use std::sync::Arc;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tokio::task::{JoinError, JoinSet};

#[derive(Debug,Clone,Serialize,Deserialize,Default)]
pub struct LLMConfig {
    pub model: String,
    pub api_key: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub streaming: bool,
    pub timeout: std::time::Duration,
}
#[allow(dead_code)]
pub struct LLMConfigBuilder {
    model: String,
    api_key: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    streaming: Option<bool>,
    timeout: Option<std::time::Duration>,
}
#[allow(dead_code)]
impl LLMConfigBuilder {
    pub fn new(model: impl AsRef<str>, api_key: impl AsRef<str>) -> Self {
        LLMConfigBuilder {
            model: model.as_ref().to_string(),
            api_key: api_key.as_ref().to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            streaming: None,
            timeout: None,
        }
    }
    pub fn build(self) -> LLMConfig {
        LLMConfig {
            model: self.model,
            api_key: self.api_key,
            max_tokens: self.max_tokens.unwrap_or(1024),
            temperature: self.temperature.unwrap_or(0.7),
            top_p: self.top_p.unwrap_or(1.0),
            streaming: self.streaming.unwrap_or(false),
            timeout: self.timeout.unwrap_or(std::time::Duration::from_secs(60)),
        }
    }
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }
    pub fn streaming(mut self, streaming: bool) -> Self {
        self.streaming = Some(streaming);
        self
    }
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Function,
    Tool,
    Memory
}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}
pub type ChatContext = Vec<Message>;

#[allow(dead_code)]
pub trait Llm {
    async fn get_completion(&self, prompt: impl AsRef<str>, config: &LLMConfig) -> Result<serde_json::Value>; // response in JSON format
    async fn get_embedding(&self, content: impl AsRef<str>, config: &LLMConfig) -> Result<Vec<f32>>;
    async fn chat(&self, prompt: impl AsRef<&str>, context: &ChatContext, config: &LLMConfig) -> Result<String>;
    async fn batch_completion(self: Arc<Self>, prompts: &[impl AsRef<str>], config: &LLMConfig) -> Result<Vec<serde_json::Value>>;
}
//TODO: add batch operation
pub struct SiliconFlow {
    client: reqwest::Client,
}
#[allow(dead_code)]
impl SiliconFlow {
    pub fn new() -> Result<Self> {
        Ok(SiliconFlow {
            client: reqwest::ClientBuilder::new()
                .use_rustls_tls()
                .http2_prior_knowledge()
                .pool_max_idle_per_host(10)
                .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
                .pool_idle_timeout(std::time::Duration::from_secs(300))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()?,
        })
    }
    pub async fn warmup(&self) {
        // 发送HEAD请求预热连接
        let _ = self.client
            .head("https://api.siliconflow.cn")
            .send()
            .await;
    }
    pub fn from_client(client: reqwest::Client) -> Self {
        SiliconFlow {
            client,
        }
    }
    pub fn extract_completion_content(response: serde_json::Value) -> Option<serde_json::Value> {
        response.get("choices")?.get(0)?.get("message")?.get("content").cloned()
    }
}
impl Llm for SiliconFlow {
    async fn get_completion(&self, prompt: impl AsRef<str>, config: &LLMConfig) -> Result<serde_json::Value> {
        let body = json!({
            "model": config.model,
            "stream": config.streaming,
            "max_tokens": config.max_tokens as i32,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": [
                {"role": "user", "content": prompt.as_ref()}
            ]
        });
        let response = self.client
            .post("https://api.siliconflow.cn/v1/chat/completions")
            .bearer_auth(&config.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .timeout(config.timeout)
            .send()
            .await?;

        let status = response.status();
        //println!("version:{:?}",response.version());
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_else(|_| "Failed to read error response".to_string());
            return Err(anyhow::anyhow!("Error response from API: {}", error_body));
        }
        let response_json = response.json::<serde_json::Value>().await?;

        match SiliconFlow::extract_completion_content(response_json) {
            Some(content) => Ok(content),
            None => Err(anyhow::anyhow!("Failed to extract completion content"))
        }
    }
    async fn get_embedding(&self, content: impl AsRef<str>, config: &LLMConfig) -> Result<Vec<f32>> {
        todo!()
    }
    async fn chat(&self, prompt: impl AsRef<&str>, context: &ChatContext, config: &LLMConfig) -> Result<String> {
        todo!()
    }
    /// 批量获取 completions, 不保证与原顺序一致
    async fn batch_completion(self: Arc<Self>, prompts: &[impl AsRef<str>], config: &LLMConfig) -> Result<Vec<Value>> {
        let semaphore = Arc::new(Semaphore::new(3));
        let mut join_set = JoinSet::new();
        let config = Arc::new(config.clone());
        for prompt in prompts {
            let semaphore = semaphore.clone();
            let prompt = prompt.as_ref().to_string();
            let config = config.clone();
            let self_clone = self.clone();
            join_set.spawn(async move {
                let _permit = semaphore.acquire_owned().await?;
                self_clone.get_completion(&prompt, &config).await
            });
        }
        let (mut values, mut errors) = (Vec::new(), Vec::new());
        while let Some(join_result) = join_set.join_next_with_id().await {
            match join_result {
                Ok(result) => {
                    match result.1 {
                        Ok(value) => values.push(value),
                        Err(err) => errors.push(anyhow::anyhow!("completion error in task id: {}, {}", result.0, err)),
                    }
                },
                Err(err) => {
                    join_set.shutdown().await;
                    return Err(anyhow::anyhow!("join error in task id: {}, {}",JoinError::id(&err), err))
                },
            }
        }
        if !errors.is_empty() {
            Err(anyhow::anyhow!("{} tasks failed: \n {}", errors.len(), errors.iter().fold(String::new(), |acc, err| acc + &err.to_string() + "\n"))) // 聚合错误
        } else {
            Ok(values)
        }
    }
}

mod test {
    use std::env;
    #[allow(unused_imports)]
    use super::*;
    #[tokio::test]
    async fn test_siliconflow_function() {
        let env = dotenvy::dotenv().unwrap();
        let api_key = env::var("API_KEY").unwrap();
        let llm = SiliconFlow::new().unwrap();
        let config = LLMConfigBuilder::new("Qwen/Qwen3-8B", api_key).build();
        let response = llm.get_completion("好？", &config).await.unwrap();
        println!("{:?}", response);
    }
    #[tokio::test]
    async fn test_siliconflow_latency() {
        let env = dotenvy::dotenv().unwrap();
        let api_key = env::var("API_KEY").unwrap();
        let llm = SiliconFlow::new().unwrap();
        let config = LLMConfigBuilder::new("Qwen/Qwen3-8B", api_key).build();

        // 冷启动测试
        let start = std::time::Instant::now();
        llm.get_completion("好冷？", &config).await.unwrap();
        println!("冷启动延迟: {:?}", start.elapsed());

        // 预热后测试
        llm.warmup().await;
        let start = std::time::Instant::now();
        llm.get_completion("好热？", &config).await.unwrap();
        println!("预热后延迟: {:?}", start.elapsed()); //TODO: warm up mechanics failed

        // 连续请求测试
        for i in 0..5 {
            let start = std::time::Instant::now();
            llm.get_completion(&format!("请求{}", i), &config).await.unwrap();
            println!("请求{}延迟: {:?}", i, start.elapsed());
        }
    }
}