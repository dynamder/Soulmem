use super::config::{AIConfig, LLMConfig};
use anyhow::Result;
use async_openai::config::OpenAIConfig;
use async_openai::{
    Client,
    types::chat::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse,
    },
};

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
    pub fn unstructured(&self, response: CreateChatCompletionResponse) -> Vec<String> {
        response
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.content)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{ChatChoice, ChatCompletionResponseMessage, Role};

    #[test]
    fn test_structured_builds_request() {
        let config = LLMConfig::new("key", "https://b.example.com", "model-x")
            .with_max_tokens(256)
            .with_n(2);
        let client = LlmClient::new(config);
        let message = ChatCompletionRequestMessage::from(
            async_openai::types::chat::ChatCompletionRequestUserMessage::from("hi"),
        );
        let request = client
            .structured(vec![message.clone()])
            .expect("request builds");
        assert_eq!(request.model, "model-x");
        assert_eq!(request.max_tokens, Some(256));
        assert_eq!(request.n, Some(2));
        assert_eq!(request.messages, vec![message]);
    }

    fn response_with_contents(contents: Vec<Option<String>>) -> CreateChatCompletionResponse {
        let choices = contents
            .into_iter()
            .enumerate()
            .map(|(i, content)| ChatChoice {
                index: i as u32,
                message: ChatCompletionResponseMessage {
                    content,
                    refusal: None,
                    tool_calls: None,
                    annotations: None,
                    role: Role::Assistant,
                    function_call: None,
                    audio: None,
                },
                finish_reason: None,
                logprobs: None,
            })
            .collect::<Vec<_>>();
        serde_json::from_value::<CreateChatCompletionResponse>(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "model-x",
            "choices": choices,
            "usage": null,
        }))
        .expect("response deserializes")
    }

    #[test]
    fn test_unstructured_extracts_content() {
        let config = LLMConfig::new("key", "https://b.example.com", "model-x");
        let client = LlmClient::new(config);
        let response = response_with_contents(vec![
            Some("first".to_string()),
            None,
            Some("third".to_string()),
        ]);
        let content = client.unstructured(response);
        assert_eq!(content, vec!["first".to_string(), "third".to_string()]);
    }

    #[test]
    fn test_unstructured_empty_choices() {
        let config = LLMConfig::new("key", "https://b.example.com", "model-x");
        let client = LlmClient::new(config);
        let response = response_with_contents(vec![]);
        assert!(client.unstructured(response).is_empty());
    }
}
