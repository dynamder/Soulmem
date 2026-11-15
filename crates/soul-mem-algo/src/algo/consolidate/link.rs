//! 巩固连边：把本轮的活跃节点与新生成的记忆节点一起交给 LLM，拿回一组新边。
//!
//! 输入是 [`ActiveNode`]（只有 id，内容按 id 从工作记忆的记忆簇取回）与刚由
//! [`super::generate`] 产出的新节点。两者拼成一份节点清单，经统一的
//! [`soul_mem_llm::LlmEngine`] 发出，提示词见 `generate_link_prompt.in`。
//! 清单里新节点带 `[NEW]` 标记，提示词要求它们**必须**与旧节点连边。

use std::collections::HashSet;

use serde::Deserialize;
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkBuilder, MemoryLinkType};
use soul_mem_core::memory_note::proc_mem::ActionType;
use soul_mem_core::memory_note::situation_mem::{AbstractSituation, SituationType};
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryType};
use soul_mem_llm::{LlmEngine, LlmError, Task};
use soul_mem_runtime::working_memory::WorkingMemory;

use super::select::ActiveNode;

pub const DEFAULT_LINK_SYSTEM_PROMPT: &str = include_str!("generate_link_prompt.in");

/// 单轮最多接受多少条新边。
const MAX_LINKS: usize = 32;

/// LLM 返回的单条边。
#[derive(Debug, Deserialize)]
struct Edge {
    from: MemoryId,
    to: MemoryId,
    link_type: MemoryLinkType,
    /// 省略时按 [`MemoryLinkBuilder`] 的默认值（`1.0`）处理。
    #[serde(default)]
    intensity: Option<f64>,
}

/// 让 LLM 为这批节点生成连接边。
///
/// `active` 是 [`super::select`] 择出的本轮活跃节点，`new_nodes` 是
/// [`super::generate`] 从摘要新生成的节点。`active` 中无法在记忆簇里解析到内容的
/// id 会被跳过（陈旧或非法 id），不会打断整轮。
///
/// 引用不存在节点、以及自环的边会被丢弃；结果按 LLM 给出的顺序截断到
/// [`MAX_LINKS`] 条。
pub async fn generate_links(
    working_mem: &WorkingMemory,
    active: &[ActiveNode],
    new_nodes: &[MemoryNote],
    engine: &LlmEngine,
) -> Result<Vec<MemoryLink>, LlmError> {
    // 拼清单与收集合法 id 都在这一次读锁内完成——锁不跨 await 持有。
    let (user, known) = working_mem.memory_cluster().read_or_compute(|cluster| {
        let mut user = String::from("Memory nodes:\n");
        let mut known = HashSet::new();
        for node in active {
            if let Some(embedded) = cluster.get_node(node.id) {
                let note = embedded.note();
                user.push_str(&node_line(note, false));
                user.push('\n');
                known.insert(note.id());
            }
        }
        for note in new_nodes {
            user.push_str(&node_line(note, true));
            user.push('\n');
            known.insert(note.id());
        }
        (user, known)
    });

    let completion = engine
        .complete(Task::system_user(DEFAULT_LINK_SYSTEM_PROMPT, user))
        .await?;
    let edges: Vec<Edge> = soul_mem_llm::json::parse_json_array(&completion.text)?;

    let links = edges
        .into_iter()
        .filter(|edge| {
            edge.from != edge.to && known.contains(&edge.from) && known.contains(&edge.to)
        })
        .take(MAX_LINKS)
        .map(|edge| {
            let builder = MemoryLinkBuilder::new(edge.from, edge.to, edge.link_type);
            match edge.intensity {
                Some(intensity) => builder.intensity(intensity).build(),
                None => builder.build(),
            }
        })
        .collect();
    Ok(links)
}

/// 提示词里的一行节点：`- <id> [<kind>] [NEW] <摘要> [tags=...]`。
///
/// 非新节点不出现 `[NEW]`。
fn node_line(note: &MemoryNote, is_new: bool) -> String {
    let new_mark = if is_new { "[NEW] " } else { "" };
    let tags = if note.tags().is_empty() {
        String::new()
    } else {
        format!(" tags={}", note.tags().join("/"))
    };
    format!(
        "- {} [{}] {}{}{}",
        note.id(),
        kind_of(note),
        new_mark,
        describe(note),
        tags
    )
}

/// 节点类型标签。与 `generate_link_prompt.in` 的措辞一一对应，改动时同步。
fn kind_of(note: &MemoryNote) -> &'static str {
    match note.mem_type() {
        MemoryType::Semantic(_) => "semantic",
        MemoryType::Situation(SituationType::AbstractSituation(_)) => "abstract_situation",
        MemoryType::Situation(SituationType::SpecificSituation(_)) => "specific_situation",
        MemoryType::Procedure(_) => "procedure",
    }
}

/// 把节点压成喂给 LLM 的最小充分摘要。
fn describe(note: &MemoryNote) -> String {
    match note.mem_type() {
        MemoryType::Semantic(sem) => join(&[sem.content.clone(), sem.description.clone()]),
        MemoryType::Situation(SituationType::SpecificSituation(situation)) => {
            let context = situation.get_context();
            let mut parts = vec![situation.get_narrative().clone()];
            if let Some(location) = context.get_location()
                && !location.name.is_empty()
            {
                parts.push(format!("地点={}", location.name));
            }
            if !context.get_participants().is_empty() {
                parts.push(format!(
                    "参与者={}",
                    slash(context.get_participants().iter().map(|p| p.name.as_str()))
                ));
            }
            if !context.get_emotions().is_empty() {
                parts.push(format!(
                    "情绪={}",
                    slash(context.get_emotions().iter().map(|e| e.name.as_str()))
                ));
            }
            if !context.get_environment().atmosphere.is_empty() {
                parts.push(format!("氛围={}", context.get_environment().atmosphere));
            }
            if !context.get_event().is_empty() {
                parts.push(format!(
                    "事件={}",
                    slash(context.get_event().iter().map(|e| e.action.as_str()))
                ));
            }
            join(&parts)
        }
        MemoryType::Situation(SituationType::AbstractSituation(abstract_situation)) => {
            match abstract_situation {
                AbstractSituation::Location(location) => format!("地点={}", location.name),
                AbstractSituation::Participant(participant) => {
                    format!("参与者={}（{}）", participant.name, participant.role)
                }
                AbstractSituation::Environment(environment) => {
                    join(&[environment.atmosphere.clone(), environment.tone.clone()])
                }
                AbstractSituation::Event(event) => format!(
                    "事件={}（发起={}，目标={}）",
                    event.action, event.initiator, event.target
                ),
            }
        }
        MemoryType::Procedure(procedure) => {
            let action = procedure.get_action();
            let action_kind = match action.get_action_type() {
                ActionType::Speak => "speak",
                ActionType::Skill(_) => "skill",
                ActionType::Think => "think",
            };
            format!("{} [{action_kind}]", action.get_content())
        }
    }
}

/// 用 ` | ` 连接非空片段，丢掉纯空白项。
fn join(parts: &[String]) -> String {
    parts
        .iter()
        .filter(|part| !part.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" | ")
}

/// 把一组名字用 `/` 连起来（参与者 / 情绪 / 事件共用）。
fn slash<'a>(items: impl Iterator<Item = &'a str>) -> String {
    items.collect::<Vec<_>>().join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use soul_mem_core::memory_note::MemoryNoteBuilder;
    use soul_mem_core::memory_note::proc_mem::{Action, ProcMemory};
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{
        Context, Environment, Event, Location, Participant, SpecificSituation,
    };

    fn semantic() -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "小白怕辣".to_string(),
            ConceptType::Entity,
            "关于饮食的事实".to_string(),
        )))
        .tags(vec!["饮食".to_string()])
        .build()
        .expect("test note")
    }

    fn specific() -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(
            SpecificSituation::new(
                "在火锅店被辣到".to_string(),
                Utc::now(),
                Context::new(
                    Some(Location {
                        name: "火锅店".to_string(),
                        coordinates: String::new(),
                    }),
                    vec![Participant {
                        name: "小白".to_string(),
                        role: "朋友".to_string(),
                    }],
                    vec![],
                    vec![],
                    Environment {
                        atmosphere: "热闹".to_string(),
                        tone: String::new(),
                    },
                    vec![],
                ),
            ),
        )))
        .build()
        .expect("test note")
    }

    fn abstract_event() -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Situation(SituationType::AbstractSituation(
            AbstractSituation::Event(Event {
                action: "被他人关心".to_string(),
                action_intensity: 0.5,
                initiator: "他人".to_string(),
                target: "我".to_string(),
            }),
        )))
        .build()
        .expect("test note")
    }

    fn procedure() -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(Action::new(
            "当被关心时嘴上否认".to_string(),
            ActionType::new_speak(),
        ))))
        .build()
        .expect("test note")
    }

    #[test]
    fn node_line_carries_kind_and_summary() {
        let line = node_line(&semantic(), false);
        assert!(line.starts_with("- "), "{line}");
        assert!(line.contains("[semantic]"), "{line}");
        assert!(line.contains("小白怕辣"), "{line}");
        assert!(line.contains("tags=饮食"), "{line}");
        assert!(!line.contains("[NEW]"), "{line}");
    }

    #[test]
    fn node_line_marks_new_nodes() {
        let old = node_line(&semantic(), false);
        let new = node_line(&semantic(), true);
        assert!(new.contains("[NEW]"), "{new}");
        assert_ne!(old, new);
    }

    /// 四类节点的标签必须与提示词里的写法一致。
    #[test]
    fn kind_labels_match_prompt_wording() {
        assert_eq!(kind_of(&semantic()), "semantic");
        assert_eq!(kind_of(&specific()), "specific_situation");
        assert_eq!(kind_of(&abstract_event()), "abstract_situation");
        assert_eq!(kind_of(&procedure()), "procedure");
    }

    #[test]
    fn describe_exposes_fields_the_prompt_matches_on() {
        let specific_line = describe(&specific());
        assert!(specific_line.contains("在火锅店被辣到"), "{specific_line}");
        assert!(specific_line.contains("地点=火锅店"), "{specific_line}");
        assert!(specific_line.contains("参与者=小白"), "{specific_line}");
        assert!(specific_line.contains("氛围=热闹"), "{specific_line}");

        // 提示词按 Event{action} 做 Proc 语义匹配，action 必须出现
        let abstract_line = describe(&abstract_event());
        assert!(abstract_line.contains("事件=被他人关心"), "{abstract_line}");

        // Procedure 的触发条件写在 action content 里，必须原样带出
        let procedure_line = describe(&procedure());
        assert!(
            procedure_line.contains("当被关心时嘴上否认"),
            "{procedure_line}"
        );
        assert!(procedure_line.contains("[speak]"), "{procedure_line}");
    }
}
