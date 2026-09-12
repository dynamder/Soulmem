//! SoulMem 顶层服务封装：串联 core/query/runtime/algo。
//!
//! 对外**仅**通过 zenoh 的订阅/发布（pub/sub）与其他设备通信：
//! 请求-应答以「请求主题 + 按请求 id 的应答主题」模拟；服务发现/心跳用 zenoh liveliness。
//!
//! 模块组织（Rust 2024：同名 `.rs` 为模块根，子模块在同名目录下，无 `mod.rs`）：
//! - `config` / `error`：配置与统一错误
//! - `wire`：传输无关对外 DTO（含 pub/sub 请求-应答信封）
//! - `service`：Service 编排层（唯一业务入口）
//! - `store`：持久化抽象（文件快照）
//! - `background`：后台任务（调度骨架 + 占位）
//! - `zenoh`：唯一的对外通道（pub/sub + liveliness）
//! - `server`：运行时装配与优雅退出

pub mod background;
pub mod config;
pub mod error;
pub mod server;
pub mod service;
pub mod store;
pub mod wire;
#[cfg(feature = "zenoh")]
pub mod zenoh;
