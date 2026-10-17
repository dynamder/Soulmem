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
//! - **滑动窗口的淘汰是两步**：`push` 只入队并返回是否需要淘汰，`pop` 只移除并返回队首，
//!   摘要由调用方单独调 `summarize`。窗口自身不调用 LLM，被弹出的消息归调用方持有，
//!   所以摘要失败不丢数据，见 [`working_memory::sliding_window`] 的模块注释。
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
