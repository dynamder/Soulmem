use async_openai::types::chat::ChatCompletionRequestMessage;
pub trait PromptBuilder {
    fn build_prompt(&self) -> Vec<ChatCompletionRequestMessage>;
}
