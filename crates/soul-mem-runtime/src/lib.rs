//! 运行时状态与持久化：工作记忆 + SurrealDB 仓储。
//!
//! # 三个子域
//!
//! | 子域 | 位置 | 内容 |
//! |---|---|---|
//! | 工作记忆 | [`working_memory`] | [`working_memory::sliding_window`] 滑动窗口与累加摘要、`Record` 活跃记录 |
//! | 记忆簇 | [`cluster`] | [`cluster::memory_cluster`] 基于 petgraph `StableDiGraph` 的记忆图、`MemoryClusterHandle` 并发封装 |
//! | 存储 | [`storage`] | `MemoryRepository` trait 与 [`storage::surreal`] 实现（行类型映射、schema、多列 HNSW） |
//!
//! 摘要需要 LLM，但本 crate 只通过 [`soul_mem_llm::LlmEngine`] 调用，
//! 不直接依赖任何 provider SDK。
//!
//! # 必须知道的约定
//!
//! - **摘要失败时窗口是"软上限"**：只有摘要成功才移除消息，失败时窗口会短暂超出容量，
//!   下一次成功后回落。这是刻意的（旧实现先出队再总结，失败即静默丢历史），
//!   见 [`working_memory::sliding_window`] 的模块注释。
//! - **锁不得跨 await 持有**：库内互斥一律用 `parking_lot`，唯一的例外是摘要更新用的
//!   异步锁——它必须串行化"读旧摘要 → await LLM → 覆写摘要"这个复合操作。
//! - **一个节点的 `links` 是它出边的完整真相源**：`upsert_notes` 会在同一事务内
//!   **重建**该节点的全部出边（先删后写）。传入只填了部分边的节点会静默丢弃其余出边。
//! - **schema 用 `IF NOT EXISTS` 且没有迁移**：所有数据库测试都跑在内存库上，
//!   因此"加一列"在 CI 全绿的同时，磁盘上已有的库不会被升级。
//!   改动 `schema.surql` 时必须考虑已存在的库。
//! - **刻意不用 SurrealDB 的 `MERGE`**：深合并会残留外部标签 enum 的旧变体键，
//!   原因与观测到的报错原文记在 `storage.rs` 与 `storage/surreal/repository.rs`。

pub mod cluster;
pub mod storage;
pub mod working_memory;
