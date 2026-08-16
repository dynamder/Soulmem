# soul-mem-runtime

> 依据 `feature/test_framework` 分支工作区代码整理。

## 1. 职责定位与依赖

`soul-mem-runtime` 是**运行时层纯库**，聚合 `WorkingMemory`（滑动窗口 + 记忆簇 + 活跃记录 +
LLM 摘要）。**无网络服务、无持久化、无任务调度**（这些属于规划中组件）。

```toml
[dependencies]
soul-mem-core = { path = "../soul-mem-core" }
soul-mem-query = { path = "../soul-mem-query" }
petgraph = { workspace = true }        # 记忆图（StableDiGraph）
async-openai = { version = "0.32.4", features = ["chat-completion"] }  # LLM 调用
http = "1.4.0"                          # 仅用于 HeaderMap
secrecy = "0.10.3"                      # 保护 API key
dotenvy = "0.15.7"                      # 加载 .env
parking_lot = "0.12.4"                  # 并发锁

[dev-dependencies]
soul-mem-algo = { path = "../soul-mem-algo" }   # 仅测试用
```

**关键事实**：
- LLM 调用走 OpenAI Chat Completions API（async-openai）；`.env` 实际指向 SiliconFlow
  兼容端点，model 为 `deepseek-ai/DeepSeek-V3.2`（见 `llm/config.rs` 与 `.env`）。
- **代码完全未使用 zenoh**（全工作区 grep 仅存在于 orchestration.md 文本中）。
- **无 SurrealDB / Qdrant 依赖**。

## 2. 模块结构

```text
src/
├── lib.rs                  # pub mod cluster; pub mod working_memory;
├── cluster.rs              # pub mod cluster_handle; pub mod memory_cluster;
│   ├── memory_cluster.rs   # MemoryCluster：进程内 StableDiGraph 记忆图
│   └── cluster_handle.rs   # MemoryClusterHandle：Arc<RwLock> + read_or_compute/write
└── working_memory.rs       # WorkingMemory：状态机 + 滑动窗口 + 记忆簇 + 记录 + LLM
    ├── record.rs           # Record：±1 反馈计分
    ├── sliding_window.rs   # SlidingWindow：Arc+原子窗口、摘要标记
    └── llm/                # client.rs / config.rs / prompt.rs
```

## 3. 工作记忆（WorkingMemory）

```rust,ignore
pub struct WorkingMemory { /* 状态机 + 滑动窗口 + 记忆簇 + 记录 */ }
```

公共 API：

```rust,ignore
pub fn new(window_capacity: usize) -> Self;
pub fn state(&self) -> &WorkingState;              // Working / Idle 状态机
pub fn transition_to_working(&mut self);
pub fn transition_to_idle(&mut self);
pub fn is_working(&self) -> bool;
pub fn sliding_window(&self) -> &SlidingWindow;
pub fn add_node(&mut self, node: EmbeddedMemoryNote);   // 新记忆入簇
pub fn remove_node(&mut self, node_id: MemoryId) -> Option<EmbeddedMemoryNote>;
pub fn memory_cluster(&self) -> MemoryClusterHandle;
pub fn record_retrieval(&mut self, node_id: MemoryId);  // 记录激活
pub fn add_feedback(&mut self, node_id: MemoryId, feedback: UserFeedback);
pub fn records(&self) -> &HashMap<MemoryId, Record>;
```

### record.rs — 活跃记录

`Record` 追踪检索中被激活的 MemoryNote，通过 `UserFeedback`（±1）计分，作为巩固时新
节点拓扑链接的候选（巩固为规划项，见 [编排](../architecture/orchestration.md)）。

### sliding_window.rs — 滑动窗口

- `SlidingWindow`：`VecDeque<Information>` + `capacity` + `tag_count` + 摘要
  （`Arc<RwLock<MergedInformation>>`）。
- 机制：**每 capacity 条消息标记一条**（摘要标记），被标记的消息滑出窗口时触发一次
  **累加性精简摘要**；窗口内信息无条件加入最终检索上下文。
- `Information` 枚举：`User(UserInformation)` / `Assistant(AssistantInformation)`。
- 并发：原子计数 + Arc 共享，支持 push/pop 与检索并发（详见
  [WorkingMemory 并发安全方案](../research/working_memory_fix.md)）。

### llm/ — LLM 摘要客户端

- `config.rs`：`LLMConfig`（OpenAIConfig + model + temperature 0.7 + max_tokens 512），
  builder 风格；`AIConfig` trait 提供 `get_config/get_model/get_temperature/get_n/get_max_tokens`。
- `client.rs`：Chat Completions 调用封装。
- `prompt.rs`：`PromptBuilder`（构建单条消息）/ `PromptHistoryBuilder`（构建历史）trait，
  具体摘要 prompt 模板在调用方实现。

## 4. 集群子系统（cluster/）

> ⚠️ **命名撞车**：代码中的 "cluster" 指**记忆簇**（进程内图），与
> [cluster.md](../architecture/cluster.md) 描述的 BEAM/Elixir **分布式集群**是两回事。
> 后者在本 crate **无任何实现**。

### MemoryCluster（memory_cluster.rs）

```rust,ignore
pub struct MemoryCluster {
    graph: StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink>,
    mem_id_to_index: HashMap<MemoryId, NodeIndex>,
    link_id_to_index: HashMap<LinkId, EdgeIndex>,
    incompletely_linked_note: HashMap<MemoryId, Vec<(MemoryId, MemoryLink)>>, // 待链接边缓冲
}
```

- 使用 `StableDiGraph`（删除节点/边不回收索引）。
- `incompletely_linked_note`：目标节点 uuid → 待链接源边，存 uuid 而非 NodeIndex 以
  避免 petgraph 索引复用导致错连。
- `GraphMemoryLink` 由 `MemoryLink` 转换而来（保留 id/intensity/link_type），
  `impl From<MemoryLink>`。

### MemoryClusterHandle（cluster_handle.rs）

```rust,ignore
pub struct MemoryClusterHandle { cluster: Arc<RwLock<MemoryCluster>> }
pub fn read_or_compute<R>(&self, closure: impl FnOnce(&MemoryCluster) -> R) -> R;
pub fn write<R>(&self, closure: impl FnOnce(&mut MemoryCluster) -> R) -> R;
```

仅 `Arc<RwLock>` + 读写闭包，**不存在任何任务管理/监督机制**。并发读安全经单测验证。

## 5. 与旧文档的出入

| 文档声称 | 实际代码 |
|---------|---------|
| orchestration.md：zenoh pub/sub 服务接口 | **未实现**，代码无 zenoh |
| orchestration.md：SurrealDB 持久化（🔲） | 未实现，无 DB 依赖 |
| orchestration.md：✅ 组件（滑动窗口/摘要/记忆簇/记录/状态机/LLM） | 属实，均已实现 |
| cluster.md：BEAM/Elixir 分布式集群 | 未实现；代码中 cluster = 记忆簇 |
| beta_ver.md：async-openai LLM 调用 | 保留（现指向本地/SiliconFlow 兼容端点） |
| 巩固/持久化/遗忘遮罩（🔲） | 未实现（soul-tune 有评测框架） |
