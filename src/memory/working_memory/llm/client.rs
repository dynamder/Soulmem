use super::{
    config::{AIConfig, LLMConfig},
    prompt::{PromptBuilder, PromptHistoryBuilder},
};
use anyhow::{Context, Error, Result};
use async_openai::config::{Config, OpenAIConfig};
use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        Role,
    },
};
use serde::de::DeserializeOwned;

pub struct LlmClient {
    client: Client<OpenAIConfig>,
    config: LLMConfig,
}

impl LlmClient {
    pub fn new(config: LLMConfig) -> Self {
        let client = Client::with_config(config.get_config());
        Self { client, config }
    }
    pub async fn call_llm(
        &self,
        content: Vec<ChatCompletionRequestMessage>,
    ) -> Result<Vec<String>> {
        let request = self.structured(content)?;
        let response = self.client.chat().create(request).await?;
        Ok(self.unstructured(response))
    }

    pub async fn simple_call(&self, message: ChatCompletionRequestMessage) -> Result<Vec<String>> {
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(self.config.get_max_tokens())
            .model(self.config.get_model().to_string())
            .messages(vec![message])
            .n(self.config.get_n())
            .build()?;
        let response = self.client.chat().create(request).await?;
        Ok(self.unstructured(response))
    }

    pub fn structured(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> Result<CreateChatCompletionRequest> {
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(self.config.get_max_tokens())
            .model(self.config.get_model().to_string())
            .messages(messages)
            .n(self.config.get_n())
            .build()?;
        Ok(request)
    }
    pub fn unstructured(&self, response: CreateChatCompletionResponse) -> Vec   <String> {
        response
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.content)
            .collect()
    }
}
