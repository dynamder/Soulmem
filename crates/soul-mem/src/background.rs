//! 后台定时任务（对应 orchestration 的定时/被动流程，🔲 项在本 crate 只做调度骨架）。
//!
//! 任务实现集中在 `scheduler`：persist 真实可用；consolidate / forget 为占位
//! （Idle 门控后返回下层算法未实现的明确错误）。

mod scheduler;

pub use scheduler::{BackgroundRuntime, run_background};
