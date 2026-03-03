use serde::{Deserialize, Serialize};
use anyhow::{Error, Result, Context};

use async_openai::config::{Config, OpenAIConfig};
use http::header::{HeaderMap, IntoHeaderName, HeaderValue, InvalidHeaderValue};
use std::collections::HashMap;
use secrecy::{SecretString, ExposeSecret};


pub trait AIConfig: Config{
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
            ai_config: OpenAIConfig::new()
                .with_api_key(key)
                .with_api_base(base),
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
