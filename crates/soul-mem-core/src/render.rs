//! 对外渲染契约：把记忆节点转成可跨进程/跨语言传递的形态。
//!
//! `RenderedMemoryNote` 的字段集合与序列化名是协议的一部分：
//! - `MemoryKind` 的 `snake_case` 名与 proto 枚举一一对应，改动即破坏兼容；
//! - `context` 为 `None` 时保留键并输出 `null`，保证键集合恒定；
//! - `score` 由调用方传入并原样透传，本层不校验；
//! - `content` 为空串时由调用方决定是否丢弃，本层不丢。

use serde::{Deserialize, Serialize};

use crate::memory_note::situation_mem::{AbstractSituation, Context, SituationType};
use crate::memory_note::{MemoryNote, MemoryType};

#[cfg(test)]
mod tests;

/// 对外类型判别字段。serde 名与 proto 枚举一一对应，改动即破坏兼容。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Semantic,
    SpecificSituation,
    AbstractSituation,
    Procedure,
}

impl MemoryKind {
    /// 展示/日志用的中文名，不是协议字段。
    pub fn label(&self) -> &'static str {
        match self {
            Self::Semantic => "语义",
            Self::SpecificSituation => "情境",
            Self::AbstractSituation => "抽象情境",
            Self::Procedure => "流程",
        }
    }
}

/// 单个记忆节点的对外表示。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMemoryNote {
    pub kind: MemoryKind,
    pub score: f64,
    pub content: String,
    pub context: Option<Context>,
}

impl RenderedMemoryNote {
    pub fn label(&self) -> &'static str {
        self.kind.label()
    }
}

/// 把内部类型渲染为对外契约类型。
pub trait Render {
    type Output;

    fn render(&self, score: f64) -> Self::Output;
}

impl Render for MemoryNote {
    type Output = RenderedMemoryNote;

    fn render(&self, score: f64) -> Self::Output {
        let (kind, content, context) = match self.mem_type() {
            MemoryType::Semantic(sem) => {
                let content = if sem.description.trim().is_empty() {
                    sem.content.clone()
                } else {
                    format!("{}（{}）", sem.content, sem.description)
                };
                (MemoryKind::Semantic, content, None)
            }
            MemoryType::Situation(SituationType::SpecificSituation(specific)) => (
                MemoryKind::SpecificSituation,
                specific.get_narrative().clone(),
                Some(specific.get_context().clone()),
            ),
            // 抽象情景用代表字段；结构化内容不进 `context`，见模块文档
            MemoryType::Situation(SituationType::AbstractSituation(abstract_situation)) => (
                MemoryKind::AbstractSituation,
                abstract_text(abstract_situation),
                None,
            ),
            MemoryType::Procedure(proc) => (
                MemoryKind::Procedure,
                proc.get_action().get_content().to_string(),
                None,
            ),
        };

        RenderedMemoryNote {
            kind,
            score,
            content,
            context,
        }
    }
}

fn abstract_text(situation: &AbstractSituation) -> String {
    match situation {
        AbstractSituation::Location(location) => location.name.clone(),
        AbstractSituation::Participant(participant) => participant.name.clone(),
        AbstractSituation::Environment(environment) => {
            format!("{} {}", environment.atmosphere, environment.tone)
                .trim()
                .to_string()
        }
        AbstractSituation::Event(event) => event.action.clone(),
    }
}
