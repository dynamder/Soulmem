//! [`Render`] 契约测试：类型映射、内容取值规则、`context` 语义、序列化形状。
//!
//! 单独成文件而非内联在 `render.rs` 里，是为了让契约断言（键集合、kind 拼写、
//! `context` 的 null 约定）与实现分开维护——改动实现时这些断言应当**仍然失败**，
//! 那正是它们的作用。

use chrono::{TimeZone, Utc};
use serde_json::json;

use super::{MemoryKind, Render, RenderedMemoryNote};
use crate::memory_note::proc_mem::{Action, ActionType, ProcMemory};
use crate::memory_note::sem_mem::{ConceptType, SemMemory};
use crate::memory_note::situation_mem::{
    AbstractSituation, Context, Emotion, Environment, Event, Location, Participant, SensoryData,
    SituationType, SpecificSituation,
};
use crate::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType};

/// 构造节点；仅供本模块测试使用，字段默认值足够。
fn note(mem_type: MemoryType) -> MemoryNote {
    MemoryNoteBuilder::new(mem_type)
        .build()
        .expect("test fixture: default timestamps satisfy builder invariants")
}

/// 六个字段全非空的上下文，用于断言 `context` 原样透传。
fn full_context() -> Context {
    Context::new(
        Some(Location {
            name: "红魔馆".to_string(),
            coordinates: "0,0".to_string(),
        }),
        vec![Participant {
            name: "蕾米莉亚".to_string(),
            role: "大小姐".to_string(),
        }],
        vec![Emotion {
            name: "安心".to_string(),
            intensity: 0.8,
        }],
        vec![SensoryData {
            name: "茶香".to_string(),
            intensity: 0.5,
        }],
        Environment {
            atmosphere: "安静".to_string(),
            tone: "温暖".to_string(),
        },
        vec![Event {
            action: "斟茶".to_string(),
            action_intensity: 0.4,
            initiator: "我".to_string(),
            target: "蕾米莉亚".to_string(),
        }],
    )
}

fn specific_situation(narrative: &str) -> SpecificSituation {
    SpecificSituation::new(
        narrative.to_string(),
        Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
            .single()
            .expect("test fixture: fixed timestamp is unambiguous"),
        full_context(),
    )
}

/// serde_json 保留大整数原样：`to_value(&struct)` 会把 `f32` 写成 `1e+2`，
/// `json!` 字面量却是 `100.0`，直接比较会假失败。先经由字符串往返统一数值表示。
fn lenient_json<T: serde::Serialize>(value: &T) -> serde_json::Value {
    let text = serde_json::to_string(value)
        .expect("test fixture: RenderedMemoryNote is JSON-serializable");
    serde_json::from_str(&text).expect("test fixture: serde_json output is valid JSON")
}

/// 语义记忆 → `Semantic`，内容取 `content`，`description` 非空时追加括号
#[test]
fn render_semantic_appends_description_when_present() {
    let with_desc = note(MemoryType::Semantic(SemMemory::new(
        "酒馆".to_string(),
        ConceptType::Entity,
        "人们喝酒聊天的地方".to_string(),
    )));
    let rendered = with_desc.render(0.75);
    assert_eq!(rendered.kind, MemoryKind::Semantic);
    assert_eq!(rendered.content, "酒馆（人们喝酒聊天的地方）");
    assert_eq!(rendered.score, 0.75);

    let no_desc = note(MemoryType::Semantic(SemMemory::new(
        "酒馆".to_string(),
        ConceptType::Entity,
        String::new(),
    )));
    assert_eq!(no_desc.render(0.75).content, "酒馆");
}

/// `description` 全为空白时视为空，不产出空括号
#[test]
fn render_semantic_treats_blank_description_as_empty() {
    let mem = note(MemoryType::Semantic(SemMemory::new(
        "酒馆".to_string(),
        ConceptType::Entity,
        "   ".to_string(),
    )));

    assert_eq!(mem.render(0.1).content, "酒馆");
}

/// 具体情景 → `SpecificSituation`，内容取 `narrative`，并带上完整 `context`
#[test]
fn render_specific_situation_carries_narrative_and_context() {
    let context = full_context();
    let mem = note(MemoryType::Situation(SituationType::SpecificSituation(
        SpecificSituation::new(
            "傍晚我在庭院里为大小姐斟茶".to_string(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
                .single()
                .expect("test fixture: fixed timestamp is unambiguous"),
            context.clone(),
        ),
    )));

    let rendered = mem.render(0.4);

    assert_eq!(rendered.kind, MemoryKind::SpecificSituation);
    assert_eq!(rendered.content, "傍晚我在庭院里为大小姐斟茶");
    assert_eq!(
        rendered.context,
        Some(context),
        "具体情景的 context 应原样透传（六个字段都不丢）"
    );
}

/// 抽象情景 → `AbstractSituation`，内容取代表字段
#[test]
fn render_abstract_situation_uses_representative_field() {
    let cases = [
        (
            AbstractSituation::Location(Location {
                name: "红魔馆".to_string(),
                coordinates: "0,0".to_string(),
            }),
            "红魔馆",
        ),
        (
            AbstractSituation::Participant(Participant {
                name: "蕾米莉亚".to_string(),
                role: "大小姐".to_string(),
            }),
            "蕾米莉亚",
        ),
        (
            AbstractSituation::Environment(Environment {
                atmosphere: "安静".to_string(),
                tone: "温暖".to_string(),
            }),
            "安静 温暖",
        ),
        (
            AbstractSituation::Event(Event {
                action: "斟茶".to_string(),
                action_intensity: 0.4,
                initiator: "我".to_string(),
                target: "蕾米莉亚".to_string(),
            }),
            "斟茶",
        ),
    ];

    for (variant, expected) in cases {
        let mem = note(MemoryType::Situation(SituationType::AbstractSituation(
            variant,
        )));
        let rendered = mem.render(0.1);

        assert_eq!(rendered.kind, MemoryKind::AbstractSituation);
        assert_eq!(rendered.content, expected);
        assert_eq!(
            rendered.context, None,
            "抽象情景的结构化内容刻意不进 context（见模块文档）"
        );
    }
}

/// 环境代表文本在字段为空时 trim 成空串，而不是留下一个空格
#[test]
fn render_abstract_environment_trims_blank_fields() {
    let mem = note(MemoryType::Situation(SituationType::AbstractSituation(
        AbstractSituation::Environment(Environment {
            atmosphere: String::new(),
            tone: String::new(),
        }),
    )));

    assert!(
        mem.render(0.1).content.is_empty(),
        "两个字段都为空时应产出空串"
    );
}

/// 程序性记忆 → `Procedure`，内容取 `Action::get_content`
#[test]
fn render_procedure_uses_action_content() {
    let mem = note(MemoryType::Procedure(ProcMemory::new(Action::new(
        "轻声细语地回应".to_string(),
        ActionType::new_speak(),
    ))));

    let rendered = mem.render(0.2);

    assert_eq!(rendered.kind, MemoryKind::Procedure);
    assert_eq!(rendered.content, "轻声细语地回应");
    assert_eq!(rendered.context, None);
}

/// 只有具体情景带 `context`，其余三种类型一律为 `None`
#[test]
fn render_context_present_only_for_specific_situation() {
    let cases = [
        MemoryType::Semantic(SemMemory::new(
            "c".to_string(),
            ConceptType::Entity,
            String::new(),
        )),
        MemoryType::Situation(SituationType::AbstractSituation(
            AbstractSituation::Participant(Participant {
                name: "p".to_string(),
                role: "r".to_string(),
            }),
        )),
        MemoryType::Procedure(ProcMemory::new(Action::new(
            "a".to_string(),
            ActionType::new_think(),
        ))),
    ];

    for mem_type in cases {
        let rendered = note(mem_type).render(0.5);
        assert_eq!(
            rendered.context, None,
            "{:?} 不应携带 context",
            rendered.kind
        );
    }
}

/// `score` 原样透传：含边界值，不做钳制也不做校验
#[test]
fn render_passes_score_through_verbatim() {
    let mem = note(MemoryType::Semantic(SemMemory::new(
        "概念".to_string(),
        ConceptType::Abstract,
        String::new(),
    )));

    for score in [0.0_f64, 1.0, f64::MIN_POSITIVE, -0.5, 2.0] {
        assert_eq!(
            mem.render(score).score,
            score,
            "render 不应改写 score（含越界值）"
        );
    }
}

/// 序列化形状即对外契约：键恰好是 kind / score / content / context 四项
#[test]
fn rendered_note_serializes_to_exactly_four_keys() {
    let mem = note(MemoryType::Semantic(SemMemory::new(
        "酒馆".to_string(),
        ConceptType::Entity,
        String::new(),
    )));

    let value = lenient_json(&mem.render(0.75));

    assert_eq!(
        value,
        json!({"kind": "semantic", "score": 0.75, "content": "酒馆", "context": null})
    );

    let object = value
        .as_object()
        .expect("test fixture: RenderedMemoryNote serializes to a JSON object");
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["content", "context", "kind", "score"],
        "对外契约的键集合被改动了"
    );
}

/// `context` 为 `None` 时保留键并输出 `null`（不用 skip_serializing_if）
#[test]
fn rendered_note_keeps_context_key_as_null() {
    for mem_type in [
        MemoryType::Semantic(SemMemory::new(
            "c".to_string(),
            ConceptType::Entity,
            String::new(),
        )),
        MemoryType::Procedure(ProcMemory::new(Action::new(
            "a".to_string(),
            ActionType::new_speak(),
        ))),
    ] {
        let value = lenient_json(&note(mem_type).render(0.5));
        let object = value
            .as_object()
            .expect("test fixture: RenderedMemoryNote serializes to a JSON object");

        assert!(
            object.contains_key("context"),
            "无 context 时键仍须存在，只是值为 null"
        );
        assert_eq!(object.get("context"), Some(&serde_json::Value::Null));
    }
}

/// `context` 的序列化形状对齐 proto 的 `Context` 消息，六个字段用 snake_case
#[test]
fn rendered_context_serializes_into_proto_shaped_object() {
    let mem = note(MemoryType::Situation(SituationType::SpecificSituation(
        specific_situation("傍晚我在庭院里为大小姐斟茶"),
    )));

    let value = lenient_json(&mem.render(0.9));
    let context = value
        .get("context")
        .and_then(serde_json::Value::as_object)
        .expect("test fixture: specific situation must carry a context object");

    let mut keys: Vec<&str> = context.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "emotions",
            "environment",
            "event",
            "location",
            "participants",
            "sensory_data"
        ],
        "context 的字段集合应与 proto 的 Context 消息一致"
    );

    assert_eq!(
        value,
        json!({
            "kind": "specific_situation",
            "score": 0.9,
            "content": "傍晚我在庭院里为大小姐斟茶",
            "context": {
                "location": {"name": "红魔馆", "coordinates": "0,0"},
                "participants": [{"name": "蕾米莉亚", "role": "大小姐"}],
                "emotions": [{"name": "安心", "intensity": 0.8}],
                "sensory_data": [{"name": "茶香", "intensity": 0.5}],
                "environment": {"atmosphere": "安静", "tone": "温暖"},
                "event": [{
                    "action": "斟茶",
                    "action_intensity": 0.4,
                    "initiator": "我",
                    "target": "蕾米莉亚"
                }]
            }
        })
    );
}

/// `kind` 的序列化拼写与 proto 枚举一一对应，改动即破坏对外兼容性
#[test]
fn rendered_note_serializes_kind_spelling() {
    let cases = [
        (
            MemoryType::Semantic(SemMemory::new(
                "c".to_string(),
                ConceptType::Entity,
                String::new(),
            )),
            "semantic",
        ),
        (
            MemoryType::Situation(SituationType::SpecificSituation(specific_situation("n"))),
            "specific_situation",
        ),
        (
            MemoryType::Situation(SituationType::AbstractSituation(
                AbstractSituation::Participant(Participant {
                    name: "p".to_string(),
                    role: "r".to_string(),
                }),
            )),
            "abstract_situation",
        ),
        (
            MemoryType::Procedure(ProcMemory::new(Action::new(
                "a".to_string(),
                ActionType::new_think(),
            ))),
            "procedure",
        ),
    ];

    for (mem_type, expected) in cases {
        let value = lenient_json(&note(mem_type).render(0.5));
        assert_eq!(
            value.get("kind").and_then(serde_json::Value::as_str),
            Some(expected),
            "kind 拼写是契约的一部分，不应改动"
        );
    }
}

/// 四种 `kind` 的中文名两两不同且非空（展示用，不参与协议）
#[test]
fn memory_kind_labels_are_distinct_and_non_empty() {
    let kinds = [
        MemoryKind::Semantic,
        MemoryKind::SpecificSituation,
        MemoryKind::AbstractSituation,
        MemoryKind::Procedure,
    ];

    let mut labels: Vec<&str> = Vec::new();
    for kind in kinds {
        let label = kind.label();
        assert!(!label.is_empty(), "{kind:?} 的中文名不能为空");
        assert!(!labels.contains(&label), "中文名重复: {label}");
        labels.push(label);
    }
    assert_eq!(labels.len(), kinds.len());
}

/// 渲染结果只依赖类型与内容，与 `MemoryId` 无关
#[test]
fn render_ignores_memory_id() {
    let make = |id: MemoryId| {
        MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "同样的内容".to_string(),
            ConceptType::Entity,
            String::new(),
        )))
        .id(id)
        .build()
        .expect("test fixture: default timestamps satisfy builder invariants")
    };

    let expected = RenderedMemoryNote {
        kind: MemoryKind::Semantic,
        score: 0.5,
        content: "同样的内容".to_string(),
        context: None,
    };

    assert_eq!(make(MemoryId::new()).render(0.5), expected);
    assert_eq!(make(MemoryId::new()).render(0.5), expected);
}
