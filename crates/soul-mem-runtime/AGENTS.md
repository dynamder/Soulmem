# AGENTS.md — soul-mem-runtime

运行时状态与持久化：工作记忆 + SurrealDB 仓储。

## 导航

- 先读 crate 根模块文档：「必须知道的约定」。
- `working_memory`：滑动窗口、记忆簇、活跃记录。
- `storage`：`MemoryRepository` trait 与 SurrealDB 实现；schema 定义在 `storage::surreal`。

## 风格

- 不跨 `await` 持有 `parking_lot` 锁。
- `WorkingMemory` 刻意**没有** `&mut self` 接口（`Arc::get_mut` 在共享计数大于 1 时静默返回 `None`）。

## 坑

- `upsert_notes` **全量重建出边**：传入只填了部分出边的节点，其余出边会被静默删除。read-modify-write 必须先 `fetch_notes`。
- schema 定义全 `IF NOT EXISTS` 且**无迁移**；所有 DB 测试跑内存库 → 磁盘上的旧库永不升级。改 schema 前必须考虑已存在的库。
- `cluster::memory_cluster` 里有一整块被注释掉的代码引用 `qdrant_client`——该依赖**不在** `Cargo.toml`，取消注释必然编译失败。
- 本 crate 唯一一处 `//SAFEUNWRAP:` 在 `cluster::memory_cluster`。

## 流程

- 刻意不用 SurrealDB 的 `MERGE`（`storage` 记录了实测报错与后果）。
- 多列 HNSW 的槽位列名是单一事实来源：新增槽位必须同时改 `EmbeddingSlot::ALL` 与 schema 定义。
- `similarity_fetch` 的 `candidate_k` 是**每槽位**预算，需按槽位数放大；精确重排由调用方做。
