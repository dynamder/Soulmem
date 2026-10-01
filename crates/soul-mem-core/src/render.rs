//! 对外渲染契约：把记忆节点转成可跨进程/跨语言传递的形态。
//!
//! SoulMem 对外以服务形式提供（zenoh pub/sub、gRPC），输出是「检索到的记忆集合」。
//! 内部类型（[`MemoryNote`] / [`MemoryType`]）是为存储与算法优化的：变体带类型特定载荷、
//! 字段分散在 `memory_note/` 下的各个模块里，而且**不含分数**（分数由检索层算）。
//! 直接把内部类型序列化出去，等于让内部重构直接变成对外协议的破坏性变更。
//!
//! [`Render`] 就是这层稳定契约。它定义在本 crate（`soul-mem-core` 无任何内部依赖），
//! 因此服务层无需依赖算法 crate 就能拿到输出类型。
//!
//! # 输出形状
//!
//! [`RenderedMemoryNote`] 只有四个字段：类型、分数、内容、上下文。
//!
//! | 字段 | 序列化名 | 含义 |
//! |---|---|---|
//! | [`MemoryKind`] | `kind` | 节点类型判别字段，与 proto 枚举一一对应 |
//! | `score` | `score` | 检索分数，由调用方提供（见下） |
//! | `content` | `content` | 记忆正文，取值规则见下表 |
//! | `context` | `context` | 具体情景的结构化上下文；其余类型为 `null`（见下） |
//!
//! # 内容取值规则
//!
//! | [`MemoryType`] 变体 | [`MemoryKind`] | `content` 取值 | `context` |
//! |---|---|---|---|
//! | [`MemoryType::Semantic`] | [`MemoryKind::Semantic`] | [`SemMemory::content`](crate::memory_note::sem_mem::SemMemory::content)，`description` 非空时追加 `（description）` | `None` |
//! | [`MemoryType::Situation`] 的 `SpecificSituation` | [`MemoryKind::SpecificSituation`] | [`SpecificSituation::get_narrative`](crate::memory_note::situation_mem::SpecificSituation::get_narrative) | `Some`（[`Context`]） |
//! | [`MemoryType::Situation`] 的 `AbstractSituation` | [`MemoryKind::AbstractSituation`] | 代表字段，见下 | `None` |
//! | [`MemoryType::Procedure`] | [`MemoryKind::Procedure`] | [`Action::get_content`](crate::memory_note::proc_mem::Action::get_content) | `None` |
//!
//! 抽象情景没有叙事，用其**代表字段**代替：
//!
//! | `AbstractSituation` 变体 | `content` 取值 |
//! |---|---|
//! | `Location` | `name` |
//! | `Participant` | `name` |
//! | `Environment` | `"{atmosphere} {tone}"`（trim 后） |
//! | `Event` | `action` |
//!
//! 取代表字段而不是结构化全文是有意的：抽象情景是「地点/人物/氛围/事件」的**概念**，
//! 拼上 `coordinates`、`action_intensity` 这类字段会产出对外部消费者无意义的噪声。
//!
//! # `context`：只服务具体情景
//!
//! [`Context`] 的六个字段（location / participants / emotions / sensory_data /
//! environment / event）只有**具体情景**拿得出来，因此只有
//! [`MemoryKind::SpecificSituation`] 填 `Some`，其余三种类型一律 `None`。
//!
//! **抽象情景的结构化内容刻意不进本字段。** 它自己是 `Location` / `Participant` /
//! `Environment` / `Event` 四者之一，语义与「一条具体情景的上下文」不同，塞进
//! [`Context`] 的对应槽位会让那些字段变成「抽象时只有一格有值」的双重语义。
//! 现阶段需要它的调用方直接读 [`MemoryNote::mem_type`]；将来外部接口真要暴露抽象
//! 情景的结构化内容时，再按 proto 的 `oneof` 语义单独加字段。
//!
//! `context` 为 `None` 时序列化**保留键并输出 `null`**（不用
//! `skip_serializing_if`）：对外契约的键集合恒定，比"有时有、有时没有"更好断言、
//! 也更好演进。
//!
//! # 分数由调用方提供，本层不校验
//!
//! [`MemoryNote`] 不持有分数：检索管线的产物是 `(MemoryId, f64)` 配对，
//! 分数是检索过程的中间结果。因此 [`Render::render`] 把 `score` 作为入参。
//!
//! 本层**原样透传** `score`（含 `0.0` / `1.0` / 负数 / NaN），不做钳制也不做校验：
//! 「分数必须落在 0 到 1 之间」是检索管线的契约，在这里再钳一次只会把管线的 bug
//! 掩盖成一个看起来正常的值。
//!
//! # 空内容不在此处丢弃
//!
//! `content` 是 `String` 而不是 `Option<String>`：**推导内容**属于本层，
//! **判定可否丢弃**属于装配层（一条空记忆该不该出现在「相关记忆」段落里，
//! 是呈现策略）。本层如实返回空串，由调用方按 `content.is_empty()` 决定丢弃。

use serde::{Deserialize, Serialize};

use crate::memory_note::situation_mem::{AbstractSituation, Context, SituationType};
use crate::memory_note::{MemoryNote, MemoryType};

#[cfg(test)]
mod tests;

/// 记忆节点的对外类型判别字段。
///
/// 与 `MemoryType` 的分层不同：`MemoryType` 把「情景」的抽象/具体之分藏在
/// `SituationType` 内层，而对外协议需要**单层扁平**枚举，所以这里拆成两个变体。
///
/// 序列化名是本契约的一部分（proto 枚举名与之对应）：
/// `"semantic"` / `"specific_situation"` / `"abstract_situation"` / `"procedure"`。
/// 改动拼写等于破坏对外兼容性。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// 语义记忆（概念、实体）
    Semantic,
    /// 具体情景记忆（叙述 + 时间 + 上下文）
    SpecificSituation,
    /// 抽象情景记忆（地点 / 参与者 / 环境 / 事件）
    AbstractSituation,
    /// 程序性记忆（动作 / 行为倾向）
    Procedure,
}

impl MemoryKind {
    /// 人类可读的中文类型名（展示/日志用，**不要**当作协议字段）。
    pub fn label(&self) -> &'static str {
        match self {
            Self::Semantic => "语义",
            Self::SpecificSituation => "情境",
            Self::AbstractSituation => "抽象情境",
            Self::Procedure => "流程",
        }
    }
}

/// 单个记忆节点的对外表示：类型 + 分数 + 内容 + 上下文。
///
/// 由 [`Render::render`] 产出。字段刻意保持最小——每加一个字段都是对
/// 所有调用方的破坏性变更，需要时再加。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMemoryNote {
    /// 节点类型判别字段（序列化名 `kind`：proto 字段名，不是 Rust 保留字）
    pub kind: MemoryKind,
    /// 检索分数，由调用方传入，本层原样透传
    pub score: f64,
    /// 记忆正文，取值规则见[模块文档](self)；空串表示无正文，由调用方决定是否丢弃
    pub content: String,
    /// 具体情景的结构化上下文；其余三种类型为 `None`（序列化为 `null`），见[模块文档](self)
    pub context: Option<Context>,
}

impl RenderedMemoryNote {
    /// 节点类型的中文名，等价于 `self.kind.label()`
    pub fn label(&self) -> &'static str {
        self.kind.label()
    }
}

/// 把内部类型渲染为对外契约类型的 trait。
///
/// 当前只有 [`MemoryNote`] 一个实现。将来若 [`MemoryLink`](crate::memory_links::MemoryLink)
/// 也要对外暴露，**不要**把它塞进本 trait 的关联类型：边的输出是另一个结构体，
/// 用同一 trait 的另一个 `impl` 即可。
///
/// # 为什么 `score` 是入参而不是关联类型
///
/// 分数不属于节点本身（见[模块文档](self)）。若把它改成关联类型，就得为「带分数的节点」
/// 造一个包装类型，而那属于检索层的重构范围，不该由契约决定。
pub trait Render {
    /// 渲染产物类型
    type Output;

    /// 渲染为对外表示。
    ///
    /// `score` 由调用方（检索层）提供，本层不做范围校验。
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

/// 抽象情景的代表文本：取最能指代该节点的单个字段。
///
/// 没有叙事可用，因此不拼装多个字段——`Environment` 是唯一需要拼接的变体
/// （氛围与色调共同构成"环境"），拼接后 trim 掉两侧空白。
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
