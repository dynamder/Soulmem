use super::{config::{LLMConfig, AIConfig}, prompt::PromptBuilder};
use async_openai::{
    types::chat::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        ChatCompletionRequestSystemMessage, Role, CreateChatCompletionRequest},
    Client,
};
use serde::de::DeserializeOwned;
use anyhow::{Result, Error, Context};
use async_openai::config::{Config, OpenAIConfig};


pub struct LlmClient {
    client: Client<OpenAIConfig>,
    config: LLMConfig,
}

impl LlmClient {
    pub fn new(config: LLMConfig) -> Self {
        let client = Client::with_config(config.get_config());
        Self { client, config }
    }
    pub async fn call_llm<T: PromptBuilder>(&self, content: &mut T) -> Result<Vec<String>> {
        let request = self.structured(content)?;
        let response = self.client.chat().create(request).await?;
        Ok(self.unstructured(response))
    }
    pub fn structured<T: PromptBuilder>(&self, content: &mut T) -> Result<CreateChatCompletionRequest> {
        let messages = content.build_prompt();
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(self.config.get_max_tokens())
            .model(self.config.get_model().to_string())
            .messages(messages)
            .n(self.config.get_n())
            .build()?;
        Ok(request)
    }
    pub fn unstructured(&self, response: CreateChatCompletionResponse) -> Vec<String> {
        response
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.content)
            .collect()
    }
}
