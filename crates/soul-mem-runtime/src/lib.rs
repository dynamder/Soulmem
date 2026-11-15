//! 运行时状态与持久化：工作记忆 + SurrealDB 仓储。
//!
//! 子域：`working_memory`（滑动窗口/摘要/`Record`）、`cluster`（petgraph 记忆图 + 并发句柄）、
//! `storage`（`MemoryRepository` + SurrealDB 实现）。

pub mod cluster;
pub mod storage;
pub mod working_memory;
