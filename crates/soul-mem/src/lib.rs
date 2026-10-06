//! SoulMem 运行时记忆服务。
//!
//! LLM 是**可选**依赖：未配置 `API_BASE` / `MODEL` 时服务照常启动，检索可用，
//! 只有摘要与巩固会失败（见 [`bootstrap`]）。
//!
//! # 分层
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`config`] | TOML 文件 + 环境变量（`db_path` / `key_prefix`） |
//! | [`error`] | 错误类型 |
//! | [`proto`] | protobuf 生成代码 |
//! | [`service`] | 传输无关的核心：串行命令循环与各用例 |
//! | [`transport`] | zenoh / gRPC 适配器与 proto 转换 |
//!
//! # 串行保证
//!
//! 核心持有唯一的 [`service::MemoryService`]，所有请求经内部命令通道**串行**处理，
//! 因此工作记忆永不被并发改写。适配器只负责编解码与路由。

pub mod bootstrap;
pub mod config;
pub mod error;
pub mod proto;
pub mod service;
pub mod transport;

pub use config::Config;
pub use error::{ServiceError, ServiceResult};
