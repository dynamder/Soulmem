//! OpenAI-compatible 后端子树（远程 API 与本地 llama-server 共用）。
//!
//! 顶层契约（[`crate::backend`]）之外的一切——wire 类型、SSE chunk 解析、
//! 鉴权头、错误体分类、tower 重试栈——全部关在这个目录里。
//!
//! # 子树内部的数据流
//!
//! ```text
//! ChatBackend::complete(&Task)            ← 来自 crate::engine
//!   ▼ backend.rs  OaiCompatBackend::complete
//!   │   encode(): Task ──────────────► wire.rs   OaiChatRequest（唯一的 provider 字段名）
//!   │   tokio::time::timeout(总时限)    ← 包住整次调用（含 body 读取）
//!   ▼ async_openai::Client::chat().create_byot::<OaiChatRequest, OaiResponse>
//!   │   URL / 鉴权头 ────────────────► provider.rs（impl async_openai::config::Config）
//!   ▼ ── 进入 async-openai：Client::post → executor.execute(factory) ──
//!   ▼ transport.rs  ★ 我们装配的 tower 栈 ★
//!   │   RetryLayer(JitterRetryPolicy) → 429/5xx/连接 → 重试（抖动 / Retry-After）
//!   │   上报重试 ────────────────────► crate::ctx（task-local）
//!   ▼ async_openai::middleware::ReqwestService::call   ← ★ HTTP 请求在此发出
//!   ▲ ── 回到 async-openai：读 body（在 tower 之上）→ 反序列化成 OaiResponse ──
//!   ▲ backend.rs  decode(): OaiResponse ─► crate::backend::Completion
//!   ▲ backend.rs  classify_error(): OpenAIError ─► crate::LlmError
//! ```
//!
//! 流式把最后两步换成 `create_stream_byot` + `backend.rs` 里的 `ChunkState` 状态机；
//! 传输层在流式下只跑到"拿到响应头"为止。
//!
//! | 文件 | 职责 |
//! |---|---|
//! | `backend.rs` | 编码 / 解码 / 错误映射 + 流式 chunk 状态机 |
//! | `wire.rs` | 私有 wire 类型与 `finish_reason` 映射 |
//! | `config.rs` | provider 配置（字段名策略、扩展体、超时、重试预算） |
//! | `provider.rs` | 鉴权头 / URL 拼接的 `Config` 实现 |
//! | `transport.rs` | tower 栈装配 + 重试策略 |

mod backend;
mod config;
mod provider;
mod transport;
mod wire;

#[cfg(test)]
mod e2e_tests;

pub use backend::OaiCompatBackend;
pub use config::{OaiCompatConfig, TokenFieldPolicy};
