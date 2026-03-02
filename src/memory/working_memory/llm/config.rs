use serde::{Deserialize, Serialize};
use anyhow::{Error, Result, Context};
use dotenvy::{dotenv, var};
use async_openai::config::{Config, OpenAIConfig};
use http::header::{HeaderMap, IntoHeaderName, HeaderValue, InvalidHeaderValue};
use std::collections::HashMap;
use secrecy::{SecretString, ExposeSecret};


pub trait AIConfig: Config{
    fn get_config(&self) -> OpenAIConfig;
    fn get_model(&self) -> &str;
    fn get_temperature(&self) -> f32;
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    model: String,
    temprerature: f32,
    ai_config: OpenAIConfig,
}

impl LLMConfig {
    pub fn new(key: &str, base: &str) -> Self {
        Self {
            model: var("MODEL").unwrap_or_default(),
            temprerature: var("TEMPERATURE").unwrap_or_default().parse().unwrap_or_default(),
            ai_config: OpenAIConfig::new()
                .with_api_key(key)
                .with_api_base(base)
        }

    }

    pub fn with_header<K, V>(&self, key: K,  value: V) -> Result<OpenAIConfig>
    where
        K: IntoHeaderName,
        V: TryInto<HeaderValue>,
        V::Error: Into<InvalidHeaderValue>,
    {
        let new_ai_config = self.ai_config.clone().with_header(key, value)?;
        Ok(new_ai_config)
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
