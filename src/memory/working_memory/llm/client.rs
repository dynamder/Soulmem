use super::{config::{MyConfig, AIConfig}, prompt::PromptBuilder};
use async_openai::{
    types::chat::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        ChatCompletionRequestSystemMessage, Role, CreateChatCompletionRequest},
    Client,
};
use serde::de::DeserializeOwned;
use anyhow::{Result, Error, Context};
use async_openai::config::{Config, OpenAIConfig};

pub struct LlmClient<C: AIConfig + Clone> {
    client: Client<C>,
    config: C,
}

impl<C: AIConfig + Clone> LlmClient<C> {
    pub fn new(config: C) -> Self {
        let client = Client::with_config(config.clone());
        Self { client, config }
    }
    pub async fn call_llm<T: PromptBuilder>(&self, content: &T, n: u8) -> Result<Vec<String>> {
        let request = self.structed(content, n)?;
        let response = self.client.chat().create(request).await?;
        Ok(self.unstructed(response))
    }
    pub fn structed<T: PromptBuilder>(&self, content: &T, n: u8) -> Result<CreateChatCompletionRequest> {
        let messages = content.build_prompt();
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(512u32)
            .model(self.config.get_model().to_string())
            .messages(messages)
            .n(n)
            .build()?;
        Ok(request)
    }
    pub fn unstructed(&self, response: CreateChatCompletionResponse) -> Vec<String> {
        response
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.content)
            .collect()
    }
}
