use anyhow::{Result, anyhow};

use crate::memory::working_memory::llm::client::LlmClient;

use super::{
    dto::ConsolidationOutput,
    mapper::{MappedConsolidation, map_output_to_notes},
    prompt::ConsolidationPrompt,
};

pub struct ConsolidationService;

impl ConsolidationService {
    pub fn new() -> Self {
        Self
    }

    pub async fn split_summary_to_output(
        &self,
        llm: &LlmClient,
        summary_text: &str,
        hot_memories: &[String],
    ) -> Result<ConsolidationOutput> {
        let prompt = ConsolidationPrompt::new(summary_text, hot_memories.to_vec());
        let raw = llm.call_llm(prompt.into_messages()).await?.join("\n");

        let output = parse_output(&raw)?;
        output.validate()?;
        Ok(output)
    }

    pub async fn split_summary_and_map(
        &self,
        llm: &LlmClient,
        summary_text: &str,
        hot_memories: &[String],
    ) -> Result<MappedConsolidation> {
        let output = self
            .split_summary_to_output(llm, summary_text, hot_memories)
            .await?;
        map_output_to_notes(output)
    }

    pub fn schema(&self) -> &'static str {
        include_str!("consolidation.schema.json")
    }
}

fn parse_output(raw: &str) -> Result<ConsolidationOutput> {
    let text = raw.trim();

    if let Ok(output) = serde_json::from_str::<ConsolidationOutput>(text) {
        return Ok(output);
    }

    let stripped = text
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if let Ok(output) = serde_json::from_str::<ConsolidationOutput>(stripped) {
        return Ok(output);
    }

    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) {
        if start < end {
            let candidate = &text[start..=end];
            if let Ok(output) = serde_json::from_str::<ConsolidationOutput>(candidate) {
                return Ok(output);
            }
        }
    }

    Err(anyhow!("LLM output is not valid consolidation JSON"))
}
