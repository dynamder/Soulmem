//! 传输无关的服务核心。
//!
//! - [`types`]：领域 DTO。
//! - [`engine`]：[`MemoryService`] 与串行命令循环。
//! - [`merge`]：多 query 结果按优先级加权合并。
//! - [`render`]：记忆 -> 一段自然语言（无 LLM）。

pub mod engine;
pub mod merge;
pub mod render;
pub mod types;

pub use engine::{Command, MemoryService, ServiceHandle, spawn};
pub use types::{
    ControlAction, Delta, ServiceEvent, ServiceRequest, ServiceResponse, ServiceState,
};
