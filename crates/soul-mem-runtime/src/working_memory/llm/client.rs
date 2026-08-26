use super::config::{AIConfig, LLMConfig};
use anyhow::Result;
use async_openai::config::OpenAIConfig;
use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse, ResponseFormat,
    },
};
use soul_mem_query::consolidation::service::ConsolidationLlm;

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

    pub async fn call_structured_llm(
        &self,
        content: Vec<ChatCompletionRequestMessage>,
    ) -> Result<Vec<String>> {
        let request = self.structured_json(content)?;
        let response = self.client.chat().create(request).await?;
        Ok(self.unstructured(response))
    }

    pub fn structured(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> Result<CreateChatCompletionRequest> {
        let request = CreateChatCompletionRequestArgs::default()
            // 滑动窗口摘要不设置 max_tokens，使用模型服务端的默认输出上限。
            // .max_tokens(self.config.get_max_tokens())
            .model(self.config.get_model().to_string())
            .messages(messages)
            .n(self.config.get_n())
            .build()?;
        Ok(request)
    }

    fn structured_json(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> Result<CreateChatCompletionRequest> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.config.get_model().to_string())
            .messages(messages)
            .response_format(ResponseFormat::JsonObject)
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

#[async_trait::async_trait]
impl ConsolidationLlm for LlmClient {
    async fn call(&self, messages: Vec<ChatCompletionRequestMessage>) -> Result<Vec<String>> {
        self.call_llm(messages).await
    }

    async fn call_structured(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> Result<Vec<String>> {
        self.call_structured_llm(messages).await
    }
}
