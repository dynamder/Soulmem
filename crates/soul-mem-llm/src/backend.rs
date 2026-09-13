//! 顶层唯一的调用契约。
//!
//! [`ChatBackend`] 的签名里只有**语义**：要什么（[`Task`]）与得到了什么（[`Completion`] /
//! [`StreamEvent`]）。请求的 wire 格式与 chunk 的解析完全在实现内部，顶层不可见。
//!
//! 这里刻意只放当前调用方真正需要的东西：加字段前先问"谁会用"。
//! provider 专有扩展走 `OaiCompatConfig::extra_body`，不要往语义类型里搬。
//!
//! # 这一层在数据流中的位置
//!
//! ```text
//! 调用方 ── Task ──► [engine::LlmEngine] ──► ChatBackend::complete ──► 后端实现（如 oai_comp）
//!                       │                                                  │
//!                       └────────────── Completion ◄───────────────────────┘
//! ```
//!
//! - **本层不含任何逻辑**，只有类型定义与 trait：没有重试、没有超时、没有观测。
//!   这些都在 [`crate::engine`]（编排）与 [`crate::oai_comp`]（传输）里。
//! - **判断一个类型该不该放这里**：它描述的是"我们要什么 / 得到了什么"（放这里），
//!   还是"provider 管它叫什么"（放 `oai_comp/wire.rs`）？
//!
//! 新增后端时**不需要改动本文件**（除非某个语义概念对**所有**后端都成立）。实现清单见
//! `docs/architecture/llm-layer.md` 的"新增一个后端要做什么"。

use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// 消息角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// 一条纯语义消息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

impl Message {
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, text)
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, text)
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, text)
    }
}

/// 采样与长度约束。
///
/// `max_output_tokens` 是**语义**上限：映射到哪个 wire 字段（`max_tokens` 还是
/// `max_completion_tokens`）由后端按自己的 provider 决定。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

impl Sampling {
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_output_tokens(mut self, max: u32) -> Self {
        self.max_output_tokens = Some(max);
        self
    }
}

/// 对后端的**语义**要求。
///
/// 目前只有一项：抑制思考过程。这是本地 Qwen3 系的实际需求——不禁用 thinking 时
/// 正文可能只落在 `reasoning_content`；而它是 provider 专有开关，所以映射归后端。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hints {
    pub suppress_reasoning: bool,
}

impl Hints {
    pub fn with_suppress_reasoning(mut self, suppress: bool) -> Self {
        self.suppress_reasoning = suppress;
        self
    }
}

/// 一次调用请求：只有语义。
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub messages: Vec<Message>,
    pub sampling: Sampling,
    pub hints: Hints,
}

impl Task {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            sampling: Sampling::default(),
            hints: Hints::default(),
        }
    }

    /// 最常见的形状：一条 system + 一条 user。
    pub fn system_user(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self::new(vec![Message::system(system), Message::user(user)])
    }

    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_hints(mut self, hints: Hints) -> Self {
        self.hints = hints;
        self
    }
}

/// 停止原因（语义）。原始 `finish_reason` 字符串的映射由后端负责。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// 正常结束。
    Completed,
    /// 被输出上限截断。
    LengthCapped,
    /// 被内容策略拦截。
    Filtered,
    /// 转向工具调用（本项目目前不使用工具调用，出现即为异常信号）。
    ToolCall,
    /// 后端没有给出或给出了无法识别的停止原因。
    Unknown,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::LengthCapped => "length_capped",
            Self::Filtered => "filtered",
            Self::ToolCall => "tool_call",
            Self::Unknown => "unknown",
        }
    }
}

/// token 用量。后端拿不到就是 `None`——**不估算**，避免把猜测写进 trace 与统计。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// 一次性调用的结果。
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub text: String,
    /// 推理内容（如 `reasoning_content`）。后端没有这个概念时保持 `None`，不造假。
    pub reasoning: Option<String>,
    pub stop: StopReason,
    pub usage: Option<Usage>,
}

impl Completion {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            reasoning: None,
            stop: StopReason::Completed,
            usage: None,
        }
    }
}

/// 流式事件（语义）。每个后端负责把自己格式的 chunk 解析成这些事件。
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// 正文增量。
    Delta(String),
    /// 思考过程增量。
    ReasoningDelta(String),
    /// 流结束，携带聚合后的完整结果。
    Done(Box<Completion>),
}

/// 事件流。`Err` 之后流应终止。
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;

/// 后端身份。字段只服务于 trace 与"是否支持流式"这一项判断。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendInfo {
    pub name: String,
    /// 后端实际使用的模型标识（未知则 `None`）。进 trace，否则"这次跑的是哪个模型"
    /// 只能靠回忆配置。
    pub model: Option<String>,
    pub supports_streaming: bool,
}

impl BackendInfo {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model: None,
            supports_streaming: false,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_streaming(mut self, supported: bool) -> Self {
        self.supports_streaming = supported;
        self
    }
}

/// 统一的 LLM 对话后端。
///
/// 实现者负责：把 [`Task`] 编码成自己的请求格式、发送、解析自己的响应/chunk 格式、
/// 把失败映射成 [`LlmError`]（含 `Retry-After` 之类的建议等待时长）。
///
/// 实现者**不**负责重试与观测：那由 [`crate::LlmEngine`] 与传输层统一处理。
#[async_trait]
pub trait ChatBackend: Send + Sync {
    fn info(&self) -> BackendInfo;

    /// 一次性完成。
    async fn complete(&self, task: Task) -> Result<Completion, LlmError>;

    /// 流式完成。不支持流式的后端应返回 [`LlmErrorKind::Unsupported`]。
    ///
    /// [`LlmErrorKind::Unsupported`]: crate::LlmErrorKind::Unsupported
    async fn stream(&self, task: Task) -> Result<EventStream, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_builders_populate_expected_fields() {
        let task = Task::system_user("sys", "usr")
            .with_sampling(
                Sampling::default()
                    .with_temperature(0.3)
                    .with_max_output_tokens(64),
            )
            .with_hints(Hints::default().with_suppress_reasoning(true));

        assert_eq!(task.messages.len(), 2);
        assert_eq!(task.messages[0].role, Role::System);
        assert_eq!(task.messages[0].text, "sys");
        assert_eq!(task.messages[1].role, Role::User);
        assert_eq!(task.sampling.temperature, Some(0.3));
        assert_eq!(task.sampling.max_output_tokens, Some(64));
        assert!(task.hints.suppress_reasoning);
    }

    #[test]
    fn names_are_stable_for_trace_output() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(StopReason::Completed.as_str(), "completed");
        assert_eq!(StopReason::LengthCapped.as_str(), "length_capped");
        assert_eq!(StopReason::Filtered.as_str(), "filtered");
        assert_eq!(StopReason::ToolCall.as_str(), "tool_call");
        assert_eq!(StopReason::Unknown.as_str(), "unknown");
    }

    #[test]
    fn defaults_do_not_fake_values() {
        let usage = Usage::default();
        assert_eq!(usage.prompt_tokens, None);
        assert_eq!(usage.completion_tokens, None);
        assert_eq!(usage.total_tokens, None);

        let completion = Completion::new("x");
        assert_eq!(completion.reasoning, None, "没有思考内容时不得造假");
        assert_eq!(completion.usage, None);
        assert_eq!(completion.stop, StopReason::Completed);
    }

    #[test]
    fn backend_info_reports_streaming_capability() {
        let info = BackendInfo::new("mock")
            .with_model("m")
            .with_streaming(true);
        assert_eq!(info.name, "mock");
        assert_eq!(info.model.as_deref(), Some("m"));
        assert!(info.supports_streaming);
    }
}
