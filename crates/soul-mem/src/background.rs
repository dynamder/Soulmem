//! 后台定时任务（对应 orchestration 的定时/被动流程，🔲 项在本 crate 只做调度骨架）。
//!
//! - `persist`：可用的真实实现（落快照）；
//! - `consolidate` / `forget`：调度位占位，调用后返回下层算法未实现的明确错误；
//! - `scheduler`：统一调度循环（间隔可配、可被控制信号强制触发、任务失败隔离）。

mod consolidate;
mod forget;
mod persist;
mod scheduler;

pub use scheduler::{BackgroundRuntime, run_background};
