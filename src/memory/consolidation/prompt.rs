use std::mem::take;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
};

use crate::memory::working_memory::llm::prompt::PromptBuilder;

pub struct ConsolidationPrompt {
    summary_text: String,
    hot_memories: Vec<String>,
    content: Vec<ChatCompletionRequestMessage>,
}

impl ConsolidationPrompt {
    pub fn new(summary_text: impl Into<String>, hot_memories: impl Into<Vec<String>>) -> Self {
        let mut prompt = Self {
            summary_text: summary_text.into(),
            hot_memories: hot_memories.into(),
            content: Vec::new(),
        };
        prompt.rebuild_messages();
        prompt
    }

    fn rebuild_messages(&mut self) {
        self.content.clear();

        let system_prompt = r#"You are a memory consolidation engine.
Task: split conversation summary into memory nodes and edges.
Output must be strict JSON only. Do not output markdown, explanation, or extra text.
All fields must conform to the provided JSON schema.
If information is uncertain, lower confidence instead of inventing details.
Every edge from/to must reference an existing node_id."#;

        let hot_memories_text = if self.hot_memories.is_empty() {
            "[]".to_string()
        } else {
            format!(
                "[\n{}\n]",
                self.hot_memories
                    .iter()
                    .map(|x| format!("  {:?}", x))
                    .collect::<Vec<_>>()
                    .join(",\n")
            )
        };

        let user_prompt = format!(
            "summary_text:\n{}\n\nhot_memories:\n{}\n\njson_schema:\n{}",
            self.summary_text,
            hot_memories_text,
            include_str!("consolidation.schema.json")
        );

        self.content
            .push(ChatCompletionRequestSystemMessage::from(system_prompt).into());
        self.content
            .push(ChatCompletionRequestUserMessage::from(user_prompt).into());
    }
}

impl PromptBuilder for ConsolidationPrompt {
    fn build_prompt(&mut self) -> Vec<ChatCompletionRequestMessage> {
        if self.content.is_empty() {
            self.rebuild_messages();
        }
        take(&mut self.content)
    }
}
