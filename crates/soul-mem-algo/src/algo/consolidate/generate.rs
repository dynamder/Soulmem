use std::future::Future;

use anyhow::Result;
use backon::{ExponentialBuilder, Retryable};
use chrono::Utc;
use serde::Deserialize;
use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory, SkillRecord};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::{Context, SituationType, SpecificSituation};
use soul_mem_core::memory_note::{MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_runtime::working_memory::sliding_window::Summary;

pub const DEFAULT_GENERATE_SYSTEM_PROMPT: &str = include_str!("generate_prompt.in");

/// LLM返回的单条记忆的JSON中间表示。
#[derive(Debug, Clone, Deserialize)]
pub struct MemorySpec {
    #[serde(rename = "type")]
    pub memory_type: String,
    pub content: String,
    pub description: Option<String>,
    pub concept_type: Option<String>,
    pub action_type: Option<String>,
    #[serde(default)]
    context: Option<Context>,
}

/// 构建记忆生成的 system + user prompt。
/// system_prompt控制LLM的角色设定与行为，传入None使用默认值。
pub fn build_generate_prompt(summary: &Summary, system_prompt: Option<&str>) -> (String, String) {
    let system = system_prompt
        .unwrap_or(DEFAULT_GENERATE_SYSTEM_PROMPT)
        .to_string();
    let user = format!("Summary text:\n{}", summary.get());
    (system, user)
}

/// 剥离LLM常带的markdown代码围栏（```json ... ```），只保留 JSON 本体。
fn strip_code_fence(response: &str) -> &str {
    let trimmed = response.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|s| s.strip_suffix("```").unwrap_or(s))
        .unwrap_or(trimmed);
    stripped.trim()
}

/// 将MemorySpec转换为MemoryNote。
impl From<MemorySpec> for MemoryNote {
    fn from(spec: MemorySpec) -> Self {
        let now = Utc::now();
        let mem_type = match spec.memory_type.as_str() {
            "semantic" => {
                let concept_type = match spec.concept_type.as_deref() {
                    Some(ct) if ct.eq_ignore_ascii_case("abstract") => ConceptType::Abstract,
                    _ => ConceptType::Entity,
                };
                MemoryType::Semantic(SemMemory::new(
                    spec.content,
                    concept_type,
                    spec.description.unwrap_or_default(),
                ))
            }
            "situation" => MemoryType::Situation(SituationType::SpecificSituation(
                SpecificSituation::new(spec.content, now, spec.context.unwrap_or_default()),
            )),
            _ => {
                let action_type = match spec.action_type.as_deref() {
                    Some(at) if at.eq_ignore_ascii_case("speak") => ActionType::new_speak(),
                    Some(at) if at.eq_ignore_ascii_case("skill") => {
                        ActionType::new_skill(SkillRecord {})
                    }
                    _ => ActionType::new_think(),
                };
                MemoryType::Procedure(ProcMemory::new(Action::new(spec.content, action_type)))
            }
        };
        MemoryNoteBuilder::new(mem_type)
            .create_time(now)
            .last_accessed_time(now)
            .build()
            .expect("memory note build cannot fail with equal times")
    }
}

/// 解析LLM返回的JSON数组并构建为Vec<MemoryNote>。
pub fn parse_memories_response(response: &str) -> Result<Vec<MemoryNote>> {
    let specs: Vec<MemorySpec> = serde_json::from_str(strip_code_fence(response))?;
    Ok(specs.into_iter().map(MemoryNote::from).collect())
}

/// 从Summary中拆分出多条记忆。
/// `llm_call` 需可重复调用（`Fn`），失败时按指数退避自动重试。
pub async fn generate_memories_from_summary<F, Fut>(
    summary: &Summary,
    system_prompt: Option<&str>,
    llm_call: F,
) -> Result<Vec<MemoryNote>>
where
    F: Fn(&str, &str) -> Fut,
    Fut: Future<Output = Result<String>>,
{
    let (system, user) = build_generate_prompt(summary, system_prompt);
    let response = (|| llm_call(&system, &user))
        .retry(ExponentialBuilder::default())
        .await?;
    parse_memories_response(&response)
}
