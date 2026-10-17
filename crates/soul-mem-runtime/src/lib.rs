//! 运行时状态与持久化：工作记忆 + SurrealDB 仓储。
//!
//! 子域：`working_memory`（滑动窗口/摘要/`Record`）、`cluster`（petgraph 记忆图 + 并发句柄）、
//! `storage`（`MemoryRepository` + SurrealDB 实现）。
//!
//! # 必须知道的约定
//!
//! - **滑动窗口淘汰是两步**：`push` 只入队并报告超容，`pop` 只移除队首，`summarize` 由调用方单独调；
//!   摘要失败不丢消息。
//! - **锁不得跨 await 持有**；摘要更新用的异步锁是唯一例外。
//! - **节点的 `links` 是出边完整真相源**：`upsert_notes` 会重建全部出边，传入部分边会静默丢弃其余边。
//! - **schema 用 `IF NOT EXISTS` 且没有迁移**：改动 `schema.surql` 时必须考虑已存在的库。
//! - **刻意不用 `MERGE`**：深合并会残留外部标签 enum 的旧变体键。

pub mod cluster;
pub mod storage;
pub mod working_memory;
