//! 记忆文本渲染：把检索产物按模板组织成一段自然语言。
//!
//! 单节点「MemoryNote → 内容文本」的投影统一走 `soul_mem_core::render::Render`
//! （产出 `RenderedMemoryNote { kind, score, content, context }`），本模块只负责
//! 把摘要、相关记忆、最近对话按固定顺序拼成一段话。不使用 LLM。

use soul_mem_llm::Role;
use soul_mem_runtime::working_memory::sliding_window::Information;

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
