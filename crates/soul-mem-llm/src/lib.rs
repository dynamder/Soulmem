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
//! **一次调用只经过两个类型**：[`Task`] 进去，[`Completion`] 出来；二者只有语义，没有 provider 字段名。
//!
//! # 两层重试
//!
//! | 层 | 覆盖范围 | 位置 |
//! |---|---|---|
//! | 传输层（内层） | 429 / 5xx / 连接失败（发请求到拿到响应头） | `oai_comp::transport` |
//! | 整调用（外层） | 超时、body 读取失败、流未产出即中断 | [`LlmEngine`] |
//!
//! 最大请求数 = `(1 + 内层) × (1 + 外层)`。内层是唯一能读到 `Retry-After` 的位置。

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
