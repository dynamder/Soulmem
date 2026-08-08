use async_openai::config::{Config, OpenAIConfig};
use http::header::HeaderMap;

use secrecy::SecretString;

pub trait AIConfig: Config {
    fn get_config(&self) -> OpenAIConfig;
    fn get_model(&self) -> &str;
    fn get_temperature(&self) -> f32;
    fn get_n(&self) -> u8;
    fn get_max_tokens(&self) -> u32;
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    model: String,
    temprerature: f32,
    ai_config: OpenAIConfig,
    n: u8,
    max_tokens: u32,
}

impl LLMConfig {
    pub fn new(key: &str, base: &str, model: &str) -> Self {
        Self {
            model: model.to_string(),
            temprerature: 0.7,
            ai_config: OpenAIConfig::new().with_api_key(key).with_api_base(base),
            n: 1,
            max_tokens: 512,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temprerature = temperature;
        self
    }

    pub fn with_n(mut self, n: u8) -> Self {
        self.n = n;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

impl AIConfig for LLMConfig {
    fn get_config(&self) -> OpenAIConfig {
        self.ai_config.clone()
    }

    fn get_model(&self) -> &str {
        &self.model
    }

    fn get_temperature(&self) -> f32 {
        self.temprerature
    }

    fn get_n(&self) -> u8 {
        self.n
    }

    fn get_max_tokens(&self) -> u32 {
        self.max_tokens
    }
}
//Config
impl Config for LLMConfig {
    fn headers(&self) -> HeaderMap {
        self.ai_config.headers()
    }

    fn url(&self, path: &str) -> String {
        self.ai_config.url(path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        self.ai_config.query()
    }

    fn api_base(&self) -> &str {
        &self.ai_config.api_base()
    }

    fn api_key(&self) -> &SecretString {
        &self.ai_config.api_key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    #[test]
    fn test_llm_config_defaults() {
        let config = LLMConfig::new("key123", "https://api.example.com", "model-x");
        assert_eq!(config.get_model(), "model-x");
        assert_eq!(config.get_temperature(), 0.7);
        assert_eq!(config.get_n(), 1);
        assert_eq!(config.get_max_tokens(), 512);
        assert_eq!(config.api_base(), "https://api.example.com");
        assert_eq!(config.api_key().expose_secret(), "key123");
    }

    #[test]
    fn test_llm_config_builders() {
        let config = LLMConfig::new("k", "https://b.example.com", "m")
            .with_temperature(0.3)
            .with_n(3)
            .with_max_tokens(1024);
        assert_eq!(config.get_temperature(), 0.3);
        assert_eq!(config.get_n(), 3);
        assert_eq!(config.get_max_tokens(), 1024);
        assert_eq!(config.get_model(), "m");
    }

    #[test]
    fn test_llm_config_get_config_and_url() {
        let config = LLMConfig::new("k", "https://b.example.com/v1", "m");
        let ai_config = config.get_config();
        assert_eq!(ai_config.api_base(), "https://b.example.com/v1");
        let url = config.url("/chat/completions");
        assert!(url.contains("chat/completions"), "url was {url}");
    }

    #[test]
    fn test_llm_config_headers_with_api_key() {
        let config = LLMConfig::new("secret-key", "https://b.example.com/v1", "m");
        let headers = config.headers();
        // 带 API key 时 headers 应包含 Authorization，而非空
        assert!(
            headers.contains_key("authorization"),
            "headers should contain authorization, got: {headers:?}"
        );
    }

    #[test]
    fn test_llm_config_query_is_empty() {
        let config = LLMConfig::new("k", "https://b.example.com/v1", "m");
        assert!(config.query().is_empty());
    }
}
