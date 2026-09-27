//! SoulMem 统一 LLM 调用层。
//!
//! # Quick Start
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use soul_mem_llm::{LlmEngine, OaiCompatBackend, OaiCompatConfig, Sampling, Task};
//! # #[tokio::main]
//! # async fn main() -> Result<(), soul_mem_llm::LlmError> {
//! // 1) 组装引擎：provider 配置 → 后端 → 引擎
//! let config = OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1", "qwen3-4b")
//!     .with_auth_header(None, None); // 本地服务不需要鉴权
//! let engine = LlmEngine::new(Arc::new(OaiCompatBackend::new(config)?));
//!
//! // 2) 描述"要什么"——只有语义，没有任何 wire 字段
//! let task = Task::system_user("你是摘要器", "把这段对话压缩成一句")
//!     .with_sampling(Sampling::default().with_max_output_tokens(512));
//!
//! // 3) 调用（引擎内部负责超时、重试、观测；后端内部负责编码、传输、解码）
//! let completion = engine.complete(task).await?;
//! println!("{}", completion.text);
//! # Ok(())
//! # }
//! ```
//!
//! **一次调用只经过两个类型**：[`Task`]（要什么）进去，[`Completion`]（得到了什么）出来。
//! 二者都只有语义，没有任何 provider 字段名。
//!
//! # 从哪里开始读
//!
//! | 你想知道 | 去看 |
//! |---|---|
//! | 我们能表达哪些语义要求 | [`Task`] / [`Sampling`] / [`Hints`]（[`backend`]） |
//! | 一次调用被怎么编排（观测 + 整调用重试） | [`LlmEngine::complete`] / [`LlmEngine::stream`]（[`engine`]） |
//! | 语义请求怎么变成 JSON、JSON 怎么变回来 | `oai_comp::OaiCompatBackend::{encode, decode}`（[`oai_comp`]） |
//! | HTTP 状态怎么变成语义错误 | `oai_comp::classify_error`（[`oai_comp`]） |
//! | 重试 / 退避 / `Retry-After` | `oai_comp::transport`（内层）+ [`engine`]（外层），见下方"两层重试" |
//! | **请求到底在哪发出** | 不在本 crate：`async_openai::middleware::ReqwestService::call`（我们装配的 tower 栈最内层） |
//! | trace 事件 | [`observer`] |
//! | LLM 输出里的 JSON 怎么宽容取出 | [`json`] |
//!
//! 完整数据流图（含 async-openai 侧的函数名）见
//! `docs/architecture/llm-layer.md`。
//!
//! # 分层原则
//!
//! - **顶层只定义契约**：[`ChatBackend`] trait 与语义类型（[`Task`] / [`Completion`] /
//!   [`StreamEvent`]）。这些类型里**没有任何 wire 字段**——不出现 `max_tokens` 与
//!   `max_completion_tokens` 之分、不出现 `chat_template_kwargs`、不出现
//!   `finish_reason` 的原始字符串、不出现 SSE chunk 形状。
//! - **编码与解码归各后端私有**：请求的 wire 格式、响应与 chunk 的解析、错误体的分类
//!   （含 `Retry-After` → [`LlmError::retry_after`]）全部由后端自己负责，见 [`oai_comp`]。
//! - **错误分类的"类别"在顶层，"映射"在后端**：只有后端知道自己的 HTTP status 与错误体形状。
//! - **判断规则**：类型描述"我们要什么 / 得到了什么"→ 属于 [`backend`]；
//!   描述"provider 管它叫什么"→ 属于 [`oai_comp`]。
//!
//! # 重试发生在两处，职责不重叠
//!
//! | 层 | 覆盖范围 | 位置 |
//! |---|---|---|
//! | 传输层（内层） | 429 / 5xx / 连接失败（发请求到拿到响应头之间） | `oai_comp::transport` 的 tower 重试策略 |
//! | 整调用（外层） | 超时、响应体读取失败、流未产出即中断 | [`LlmEngine`] |
//!
//! 分界来自 async-openai 的结构：响应体读取发生在 tower 边界**之上**
//! （`Client::execute_response` 拿到 `Response` 后由 `read_response` 读 body），
//! 因此 body 阶段与整体超时的失败**不可能**被传输层重试策略看到，必须由外层补。
//! 内层重试留在传输层还有一个不可替代的作用：它是**唯一能读到 `Retry-After` 响应头**
//! 的位置（`OpenAIError::ApiError` 只带 status 与错误体，没有头）。
//! 两层预算独立且有界，`(1 + 内层) × (1 + 外层)` 即单次语义调用的最大请求数。
//!
//! 内层重试发生在 tower 服务内部，调用方看不到；为了让它们进 trace，
//! 传输层通过 `ctx`（`src/ctx.rs`）的 task-local 上下文上报（这是"不污染
//! [`ChatBackend`] 签名"与"能观测到重试"之间唯一可用的机制）。

pub mod backend;
pub mod engine;
pub mod error;
pub mod json;
pub mod oai_comp;
pub mod observer;

pub(crate) mod ctx;

#[cfg(test)]
pub mod mock;

pub use backend::{
    BackendInfo, ChatBackend, Completion, EventStream, Hints, Message, Role, Sampling, StopReason,
    StreamEvent, Task, Usage,
};
pub use engine::LlmEngine;
pub use error::{LlmError, LlmErrorKind};
pub use oai_comp::{OaiCompatBackend, OaiCompatConfig, TokenFieldPolicy};
pub use observer::{
    CallEnd, CallStart, JsonlObserver, LlmObserver, NoopObserver, RetryEvent, RetryLayer,
};

#[cfg(test)]
pub use mock::{MockBackend, MockOutcome};
