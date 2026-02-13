use super::{config::{MyConfig, AIConfig}, prompt::PromptBuilder};
use async_openai::{
    types::chat::{ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse, Role},
    Client,
};
use serde::de::DeserializeOwned;
use anyhow::{Result, Error, Context};
use async_openai::config::{Config, OpenAIConfig};

pub struct LlmClient<C: Config + AIConfig + Clone> {
    client: Client<C>,
    config: C,
}

impl<C: Config + AIConfig + Clone> LlmClient<C> {
    pub fn new(config: C) -> Self {
        let client = Client::with_config(config.clone());
        Self { client, config }
    }
    pub async fn call_llm<T: PromptBuilder>(&self, content: &T) -> Result<Vec<String>> {
        let request = self.structed(content);
        let response = self.client.chat().create(request).await?;
        Ok(self.unstructed(response))
    }
    pub fn structed<T: PromptBuilder>(&self, content: &T) -> CreateChatCompletionRequest {
        let message = content.build_prompt();
        let request = CreateChatCompletionRequest {
            model: self.config.get_model().to_string(),
            messages: message,
            temperature: Some(self.config.get_temp()),
            ..Default::default()
        };
        request
    }
    pub fn unstructed(&self, response: CreateChatCompletionResponse) -> Vec<String> {
        response
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.content)
            .collect()
    }
}
