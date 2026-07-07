use async_openai::types::chat::{ChatCompletionRequestMessage, Role};

pub trait PromptBuilder {
    fn build_prompt(&self) -> ChatCompletionRequestMessage;
    fn build_raw_prompt(&self) -> (&str, Role);
}

pub trait PromptHistoryBuilder {
    fn build_history(&self) -> Vec<ChatCompletionRequestMessage>;
}
