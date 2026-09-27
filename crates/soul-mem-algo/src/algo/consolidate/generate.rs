//! 巩固生成：把滑动窗口摘要拆解成多条记忆节点。
//!
//! 流程：提示词（`generate_prompt.in`）→ **统一的 [`soul_mem_llm::LlmEngine`]** → JSON 解析
//! （[`soul_mem_llm::json::parse_json_array`]）→ `Vec<MemoryNote>`。
//!
//! 这里**没有**自己的重试：传输层的 429/5xx/连接与引擎层的超时/流中断已经覆盖了
//! 重试场景，再套一层会让两层互相放大且不进 trace。完整链路见
//! `docs/architecture/llm-layer.md`。

use chrono::Utc;
use serde::Deserialize;
use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory, SkillRecord};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::{Context, SituationType, SpecificSituation};
use soul_mem_core::memory_note::{MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_llm::{LlmEngine, LlmError, Task};
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

/// 解析 LLM 返回的 JSON 数组并构建为 `Vec<MemoryNote>`。
///
/// 宽容抽取（剥 think 块、剥围栏、平衡括号扫描）统一由
/// [`soul_mem_llm::json::parse_json_array`] 负责：这里以前用 `rfind('}')` 截取，
/// 尾部解释里出现 `}` 就会解析失败。
pub fn parse_memories_response(response: &str) -> Result<Vec<MemoryNote>, LlmError> {
    let specs: Vec<MemorySpec> = soul_mem_llm::json::parse_json_array(response)?;
    Ok(specs.into_iter().map(MemoryNote::from).collect())
}

/// 从 Summary 中拆分出多条记忆。
///
/// 重试由 [`LlmEngine`] 统一负责（传输层管 429/5xx/连接，整调用管超时与流中断），
/// 这里不再自己套一层 `backon`——否则两层重试会互相放大，且重试次数不进 trace。
pub async fn generate_memories_from_summary(
    summary: &Summary,
    system_prompt: Option<&str>,
    engine: &LlmEngine,
) -> Result<Vec<MemoryNote>, LlmError> {
    let (system, user) = build_generate_prompt(summary, system_prompt);
    let completion = engine.complete(Task::system_user(system, user)).await?;
    parse_memories_response(&completion.text)
}
