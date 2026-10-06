//! 记忆文本渲染：把记忆节点投影为文本，并按模板组织成一段自然语言。
//!
//! 不使用 LLM：按类型取最核心的可读字段，再按固定顺序拼接。
//!
//! [`note_text`] **按值接收** `EmbeddedMemoryNote`，以便把节点里的字符串直接移出，
//! 不再为每个字段做一次 clone。

use soul_mem_core::memory_note::MemoryType;
use soul_mem_core::memory_note::situation_mem::{AbstractSituation, SituationType};
use soul_mem_llm::Role;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_runtime::working_memory::sliding_window::Information;

/// 把一条记忆节点渲染为文本；内容为空时返回 `None`。
pub fn note_text(embedded: EmbeddedMemoryNote) -> Option<String> {
    let (note, _embedding) = embedded.into_tuple();
    let text = match note.into_mem_type() {
        MemoryType::Semantic(sem) => {
            if sem.description.trim().is_empty() {
                sem.content
            } else {
                format!("{}（{}）", sem.content, sem.description)
            }
        }
        MemoryType::Situation(SituationType::SpecificSituation(situation)) => {
            situation.get_narrative().clone()
        }
        MemoryType::Situation(SituationType::AbstractSituation(abstract_situation)) => {
            abstract_text(abstract_situation)
        }
        MemoryType::Procedure(procedure) => procedure.get_action().get_content().to_string(),
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// 抽象情境的记忆没有叙事，用其代表字段代替。
fn abstract_text(situation: AbstractSituation) -> String {
    match situation {
        AbstractSituation::Location(location) => location.name,
        AbstractSituation::Participant(participant) => participant.name,
        AbstractSituation::Environment(environment) => {
            format!("{} {}", environment.atmosphere, environment.tone)
                .trim()
                .to_string()
        }
        AbstractSituation::Event(event) => event.action,
    }
}

/// 把检索产物组织成一段自然语言。
///
/// 顺序：摘要 → 相关记忆 → 最近对话。空的部分整段省略。
pub fn render_output(short_mem: &str, memories: &[String], history: &[Information]) -> String {
    let mut sections: Vec<String> = Vec::new();

    let summary = short_mem.trim();
    if !summary.is_empty() {
        sections.push(format!("【摘要】{summary}"));
    }
    if !memories.is_empty() {
        sections.push(format!("【相关记忆】{}", memories.join("；")));
    }
    if !history.is_empty() {
        let turns: Vec<String> = history
            .iter()
            .map(|info| format!("{}：{}", role_label(info.role()), info.get_str()))
            .collect();
        sections.push(format!("【最近对话】{}", turns.join(" / ")));
    }

    sections.join("\n")
}

/// 角色标签：assistant 记为"角色"，其余（含 user）记为"用户"。
fn role_label(role: Role) -> &'static str {
    match role {
        Role::Assistant => "角色",
        _ => "用户",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_sections_in_order_and_omits_empty() {
        let memories = vec!["记忆A".to_string()];
        let output = render_output("摘要", &memories, &[]);
        assert!(output.starts_with("【摘要】摘要"));
        assert!(output.contains("【相关记忆】记忆A"));
        assert!(!output.contains("【最近对话】"));
    }

    #[test]
    fn empty_everything_is_empty_string() {
        assert!(render_output("", &[], &[]).is_empty());
    }

    #[test]
    fn role_labels_map_assistant_and_others() {
        assert_eq!(role_label(Role::Assistant), "角色");
        assert_eq!(role_label(Role::User), "用户");
        assert_eq!(role_label(Role::System), "用户");
    }
}
