# WorkingMemory 并发安全与 API 重构方案

## 一、问题描述

### 1.1 并发场景

系统存在以下并发场景：
- **巩固算法与检索算法并发执行**：用户输入或 LLM 返回信息时，`push`/`pop` 操作可能与检索同时发生
- **检索算法内部并发**：query 被拆分为多个原子 query，各自独立执行以提升性能
- **`push`/`push` 并发**：用户连续发送信息或用户与 LLM 同时发送信息时，两个 `push` 可能并发执行
- **`tokio::spawn` 的 `Send` 约束**：闭包必须满足 `Send + Sync` 约束才能跨线程调度

### 1.2 当前架构

```rust,ignore
// src/memory/working_memory/sliding_window.rs

pub struct SlidingWindow {
    window: VecDeque<Information>,                    // 非线程安全
    capacity: usize,
    tag_count: usize,
    summary: Arc<RwLock<MergedInformation>>,         // 已有并发保护
}

pub enum Information {
    User(UserInformation),
    Assistant(AssistantInformation),
}

pub struct UserInformation {
    pub text: String,      // heap 分配，clone 代价高
    pub tag: bool,
}

pub struct AssistantInformation {
    pub text: String,      // heap 分配，clone 代价高
    pub tag: bool,
}

struct MergedInformation {
    content: Vec<ChatCompletionRequestMessage>,
    previous_summary: String,  // heap 分配，clone 代价高
}
```

```rust,ignore
// src/memory/working_memory.rs

pub struct WorkingMemory {
    state: WorkingState,
    sliding_window: SlidingWindow,       // 非 Arc，无法跨线程共享
    memory_cluster: MemoryCluster,      // HashMap，非线程安全
    records: HashMap<MemoryId, Record>,
}
```

### 1.3 当前问题

| 问题 | 说明 |
|------|------|
| **线程安全问题** | `VecDeque::push/pop` 不是原子操作，并发访问有数据竞争 |
| **字符串 clone 代价高** | `Information.text` 是 `String`，每次 clone 都要复制堆数据 |
| **API 设计缺陷** | `retrieve` 接受 `Request` 的所有权但返回引用，语义冲突 |
| **无法跨线程共享** | `SlidingWindow` 不是 `Arc`，无法通过 `tokio::spawn` 传递 |
| **MemoryCluster 非线程安全** | 内部 `HashMap` 并发读写有问题（计划改 DashMap） |

---

## 二、原因分析

### 2.1 `Information` 字符串 clone 问题

当前 `Information.text` 是 `String` 类型。当 `get_windows()` 返回 `Arc<VecDeque<Information>>` 时，如果每次 clone 要复制整个字符串，对于几十个中文字符的 message，代价不可接受。

### 2.2 `SlidingWindow` 线程安全问题

`VecDeque` 的 `push/pop` 操作不是线程安全的。即使包装在 `Arc` 中，多个线程同时修改 `VecDeque` 内部状态会导致数据竞争。

### 2.3 `RetrStrategy` API 设计缺陷

```rust,ignore
trait RetrStrategy {
    type Request: RetrRequest;
    type Return<'a>
    where
        Self: 'a;
    
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_>;
}
```

- `request` 按值传递，被消耗
- `Return<'a>` 是生命周期引用，语义上要求 `request` 仍然存活
- 两者冲突：无法同时满足"消耗 request"和"返回 request 内部数据的引用"

### 2.4 `tokio::spawn` 的 Send 约束

`tokio::spawn` 要求闭包满足 `Send + Sync`。如果 `SlidingWindow` 的方法需要 `&mut self`，则无法通过 `Arc<SlidingWindow>` 跨线程传递（`Arc` 只提供共享所有权，不提供独占访问）。

---

## 三、解决方案

### 3.1 核心理念

1. **`Information` 不可变设计**：`text` 使用 `Arc<str>`，创建后不变，返回新实例表示状态变更
2. **线程安全的数据结构**：`window` 使用 `Arc<RwLock<VecDeque>>`，读写分离
3. **`SlidingWindow` 完全可共享**：所有方法使用 `&self`，通过 `Arc<SlidingWindow>` 跨线程传递
4. **零 clone 字符串**：`Arc` 的 clone 只是引用计数操作，O(1)

### 3.2 结构变更

#### 3.2.1 `Information` 及相关结构

```rust,ignore
// src/memory/working_memory/sliding_window.rs

pub enum Information {
    User(UserInformation),
    Assistant(AssistantInformation),
}

impl Information {
    pub fn new(value: &str, role: &str) -> Self {
        match role {
            "user" => Information::User(UserInformation::new(value)),
            "assistant" => Information::Assistant(AssistantInformation::new(value)),
            _ => Information::User(UserInformation::new(value)),
        }
    }
    
    pub fn with_tag(self) -> Self {
        match self {
            Information::User(info) => Information::User(info.with_tag()),
            Information::Assistant(info) => Information::Assistant(info.with_tag()),
        }
    }
    
    pub fn without_tag(self) -> Self {
        match self {
            Information::User(info) => Information::User(info.without_tag()),
            Information::Assistant(info) => Information::Assistant(info.without_tag()),
        }
    }
    
    pub fn is_tagged(&self) -> bool {
        match self {
            Information::User(info) => info.tag,
            Information::Assistant(info) => info.tag,
        }
    }
    
    pub fn get_str(&self) -> &str {
        match self {
            Information::User(info) => &info.text,
            Information::Assistant(info) => &info.text,
        }
    }
    
    pub fn to_message(&self) -> ChatCompletionRequestMessage {
        match self {
            Information::User(info) => {
                ChatCompletionRequestMessage::from(ChatCompletionRequestUserMessage::from(info.get_str())).into()
            }
            Information::Assistant(info) => {
                ChatCompletionRequestMessage::from(ChatCompletionRequestAssistantMessage::from(info.get_str())).into()
            }
        }
    }
}

pub struct UserInformation {
    pub text: Arc<str>,  // 改为 Arc<str>
    pub tag: bool,
}

impl UserInformation {
    pub fn new(text: &str) -> Self {
        Self { 
            text: Arc::from(text),  // Arc::from 是 O(1)
            tag: false 
        }
    }
    
    pub fn with_tag(self) -> Self {
        Self { text: self.text, tag: true }  // Arc move，O(1)
    }
    
    pub fn without_tag(self) -> Self {
        Self { text: self.text, tag: false }
    }
    
    pub fn get_str(&self) -> &str {
        &self.text
    }
}

pub struct AssistantInformation {
    pub text: Arc<str>,  // 改为 Arc<str>
    pub tag: bool,
}

impl AssistantInformation {
    pub fn new(text: &str) -> Self {
        Self { 
            text: Arc::from(text),
            tag: false 
        }
    }
    
    pub fn with_tag(self) -> Self {
        Self { text: self.text, tag: true }
    }
    
    pub fn without_tag(self) -> Self {
        Self { text: self.text, tag: false }
    }
    
    pub fn get_str(&self) -> &str {
        &self.text
    }
}
```

#### 3.2.2 `MergedInformation`

```rust,ignore
struct MergedInformation {
    content: Vec<ChatCompletionRequestMessage>,
    previous_summary: Arc<str>,  // 改为 Arc<str>
}

impl MergedInformation {
    pub fn new() -> Self {
        Self { 
            content: Vec::new(), 
            previous_summary: Arc::from("") 
        }
    }
    
    pub fn merge_summary(&mut self, content: &str) {
        let new_summary = format!("{}{}", self.previous_summary, content);
        self.previous_summary = Arc::from(new_summary);
    }
    
    pub fn get_previous_summary(&self) -> String {
        self.previous_summary.to_string()
    }
}
```

#### 3.2.3 `SlidingWindow`

```rust,ignore
pub struct SlidingWindow {
    window: Arc<RwLock<VecDeque<Information>>>,  // 读写分离
    capacity: usize,
    tag_count: usize,
    summary: Arc<RwLock<MergedInformation>>,
}

impl SlidingWindow {
    pub fn new(capacity: usize) -> Self {
        Self {
            window: Arc::new(RwLock::new(VecDeque::with_capacity(capacity + 1))),
            capacity,
            tag_count: capacity,
            summary: Arc::new(RwLock::new(MergedInformation::new())),
        }
    }
    
    pub async fn push(&self, value: &str, role: &str, client: &LlmClient) -> Result<()> {
        let text = Information::new(value, role);
        let text = self.auto_tag(text);
        
        {
            let mut guard = self.window.write().unwrap();
            guard.push_back(text);
            if guard.len() == self.capacity + 1 {
                drop(guard);
                self.pop(client).await?;
            }
        }
        Ok(())
    }
    
    async fn pop(&self, client: &LlmClient) -> Result<()> {
        let target = {
            let mut guard = self.window.write().unwrap();
            guard.pop_front()
        };
        
        if let Some(value) = target {
            if value.is_tagged() {
                self.summarize(client).await?;
            }
        }
        Ok(())
    }
    
    pub fn get_windows(&self) -> Arc<VecDeque<Information>> {
        Arc::clone(&self.window)  // O(1)，只增加引用计数
    }
    
    pub fn len(&self) -> usize {
        self.window.read().unwrap().len()
    }
    
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }
    
    pub fn get(&self, index: usize) -> Option<&Information> {
        self.window.read().unwrap().get(index)
    }
    
    pub fn is_empty(&self) -> bool {
        self.window.read().unwrap().is_empty()
    }
    
    pub fn clear(&self) {
        self.window.write().unwrap().clear();
        self.tag_count = 0;
    }
    
    fn auto_tag(&self, value: Information) -> Information {
        self.tag_count += 1;
        if self.tag_count >= self.capacity {
            let tagged = value.with_tag();
            self.tag_count = 0;
            tagged
        } else {
            value
        }
    }
    
    async fn merge(&self) {
        let mut messages = self.summary.write().await;
        let previous = ChatCompletionRequestUserMessage::from((*messages.previous_summary).into()).into();
        messages.content.clear();
        messages.content.push(ChatCompletionRequestSystemMessage::from(
            "Based on the summary of previous conversation and the information currently in the window, provide a new overall summary.").into());
        messages.content.push(previous);
        
        let windows = self.window.read().unwrap();
        for message in windows.iter() {
            messages.content.push(message.to_message())
        }
    }
    
    async fn summarize(&self, client: &LlmClient) -> Result<String> {
        self.merge().await;
        let mut summary_arc = self.summary.write().await;
        let response = self.call_llm(client, &mut *summary_arc).await?;
        Ok(response)
    }
    
    async fn call_llm(&self, client: &LlmClient, merged: &mut MergedInformation) -> Result<String> {
        let response = client.call_llm(merged).await?;
        let output = response.join(" ");
        merged.merge_summary(&output);
        Ok(output)
    }
}
```

#### 3.2.4 `WorkingMemory`

```rust,ignore
// src/memory/working_memory.rs

pub struct WorkingMemory {
    state: WorkingState,
    sliding_window: Arc<SlidingWindow>,  // 改为 Arc 包装
    memory_cluster: MemoryCluster,
    records: HashMap<MemoryId, Record>,
}

impl WorkingMemory {
    pub fn new(window_capacity: usize) -> Self {
        Self {
            state: WorkingState::Idle,
            sliding_window: Arc::new(SlidingWindow::new(window_capacity)),
            memory_cluster: MemoryCluster::new(),
            records: HashMap::new(),
        }
    }
    
    pub fn sliding_window(&self) -> Arc<SlidingWindow> {
        Arc::clone(&self.sliding_window)
    }
    
    // ... 其他方法保持不变
}
```

---

## 四、Trait API 变更

### 4.1 `RetrStrategy` trait

```rust,ignore
pub trait RetrStrategy {
    type Request: RetrRequest;
    type Return;  // 不再带生命周期参数
    
    fn retrieve(&self, request: &Self::Request) -> Self::Return;
}
```

### 4.2 调用方使用方式

```rust,ignore
let working_mem: Arc<WorkingMemory> = Arc::new(WorkingMemory::new(10));

// tokio::spawn 中使用
tokio::spawn(async move {
    let request = ShortOnlyRequest {
        working_mem: working_mem.sliding_window(),  // Arc::clone，O(1)
    };
    
    let result = strategy.retrieve(&request);  // &request 是引用
    
    // request 可立即 drop，但 result 独立存在
    // result 是 Arc<VecDeque<Information>>，可在 task 间传递
});
```

---

## 五、MemoryCluster DashMap 改造（本期范围）

### 5.1 现状

`MemoryCluster` 内部使用 `HashMap`：

```rust,ignore
pub struct MemoryCluster {
    graph: StableDiGraph<MemoryNote, GraphMemoryLink>,
    mem_id_to_index: HashMap<MemoryId, NodeIndex>,
    link_id_to_index: HashMap<LinkId, EdgeIndex>,
    incompletely_linked_note: HashMap<MemoryId, Vec<(NodeIndex, MemoryLink)>>,
    embedding_store: HashMap<MemoryId, MemoryEmbedding>,
}
```

### 5.2 目标

改为 `DashMap` 以支持并发读写：

```rust,ignore
use dashmap::DashMap;

pub struct MemoryCluster {
    graph: StableDiGraph<MemoryNote, GraphMemoryLink>,
    mem_id_to_index: DashMap<MemoryId, NodeIndex>,
    link_id_to_index: DashMap<LinkId, EdgeIndex>,
    incompletely_linked_note: DashMap<MemoryId, Vec<(NodeIndex, MemoryLink)>>,
    embedding_store: DashMap<MemoryId, MemoryEmbedding>,
}
```

### 5.3 注意事项

- `DashMap` 的 `get`/`insert` 等操作返回 `Option<Ref<...>>` 或 `RefMut<...>`，需要相应调整方法实现
- `graph: StableDiGraph` 本身不是线程安全的，如果需要并发访问图结构，可能也需要加锁或改用其他方案
- 视情况决定是否将 `MemoryCluster` 也包装成 `Arc<MemoryCluster>`

---

## 六、实施计划

### Phase 1：`Information` 结构改造

- [ ] `UserInformation.text: String` → `text: Arc<str>`
- [ ] `AssistantInformation.text: String` → `text: Arc<str>`
- [ ] 添加 `with_tag(self) -> Self` 和 `without_tag(self) -> Self` 方法
- [ ] 移除 `tag_information(&mut self)` 和 `untag_information(&mut self)`
- [ ] 更新 `MergedInformation.previous_summary: String` → `Arc<str>`
- [ ] 更新 `merge_summary` 实现
- [ ] 更新 `get_previous_summary` 返回 `String`
- [ ] 更新测试用例

### Phase 2：`SlidingWindow` 结构改造

- [ ] `window: VecDeque<Information>` → `window: Arc<RwLock<VecDeque<Information>>>`
- [ ] `new()`: 初始化 `Arc::new(RwLock::new(...))`
- [ ] `push()`: 改为 `&self` + write lock
- [ ] `pop()`: 改为 `&self` + write lock
- [ ] `get_windows()`: 返回 `Arc<VecDeque<Information>>`
- [ ] `len()`: 通过 read lock
- [ ] `get()`: 通过 read lock
- [ ] `is_empty()`: 通过 read lock
- [ ] `clear()`: 通过 write lock
- [ ] `auto_tag()`: 改为 `&self`，返回新实例
- [ ] 移除 `get_mut_capacity()`（capacity 不可变）
- [ ] 更新测试用例

### Phase 3：`WorkingMemory` 结构改造

- [ ] `sliding_window: SlidingWindow` → `sliding_window: Arc<SlidingWindow>`
- [ ] `sliding_window()`: 返回 `Arc<SlidingWindow>`
- [ ] 移除 `sliding_window_mut()`
- [ ] 更新测试用例

### Phase 4：`RetrStrategy` trait 修改

- [ ] trait 定义：`retrieve` 接受 `&Self::Request`，返回 `Arc<VecDeque<Information>>`
- [ ] 所有实现同步修改
- [ ] 更新调用方代码

### Phase 5：`MemoryCluster` DashMap 改造

- [ ] 引入 `dashmap` crate
- [ ] `HashMap` → `DashMap`（`mem_id_to_index`、`link_id_to_index`、`incompletely_linked_note`、`embedding_store`）
- [ ] 调整相关方法实现以适配 `DashMap` 的 API
- [ ] 视情况决定 `MemoryCluster` 是否需要 `Arc` 包装
- [ ] 更新测试用例

---

## 七、技术债务清单

| 项目 | 说明 |
|------|------|
| **零 clone 字符串** | `Arc<str>` 使 `Information` clone 变成 O(1) 操作 |
| **读写锁分离** | `RwLock` 允许并发读、互斥写 |
| **API 设计合理化** | `retrieve` 接受引用而非消耗所有权 |
| **MemoryCluster 并发化** | `DashMap` 支持并发读写 |
| **capacity 不可变** | 移除 `get_mut_capacity()`，简化设计 |

---

## 八、风险与注意事项

1. **`Arc<RwLock<VecDeque>>` 的读写粒度**：每次 `push`/`pop` 需要获取写锁，锁竞争可能影响性能。但 `push` 频率不高，应可接受。

2. **`MergedInformation.previous_summary` 的 `Arc` 改造**：`merge_summary` 现在需要创建新的 `Arc<str>`，每次摘要都会产生新的 heap 分配。这是不可避免的，因为 `Arc` 不可变。

3. **`DashMap` 的 shard 数量**：`DashMap` 内部有多个 shard，过多或过少都会影响性能。默认通常足够，但如遇性能问题可调整。

4. **`StableDiGraph` 的线程安全**：`petgraph` 的 `StableDiGraph` 不是线程安全的。如果 `graph` 字段也需要并发访问，需要额外处理。

5. **测试覆盖**：改造后需要确保所有测试用例通过，特别是并发场景的测试。

---

## 九、相关文件清单

| 文件 | 改动 |
|------|------|
| `src/memory/working_memory/sliding_window.rs` | `Information`、`UserInformation`、`AssistantInformation`、`MergedInformation`、`SlidingWindow` |
| `src/memory/working_memory.rs` | `WorkingMemory` |
| `src/memory/memory_cluster.rs` | `MemoryCluster` 改 DashMap |
| `src/memory.rs` 或相关模块 | `RetrStrategy` trait 及实现 |
| `Cargo.toml` | 添加 `dashmap` 依赖 |

---

## 十、附录：关键类型变更对照表

| 原类型 | 新类型 | 说明 |
|--------|--------|------|
| `String` | `Arc<str>` | 字符串，clone O(1) |
| `&mut self` | `&self` | 所有 SlidingWindow 方法 |
| `VecDeque<Information>` | `Arc<RwLock<VecDeque<Information>>>` | 线程安全 |
| `SlidingWindow` | `Arc<SlidingWindow>` | 可跨线程共享 |
| `HashMap` | `DashMap` | 并发安全 HashMap |
| `fn retrieve(request: Self::Request) -> Self::Return<'_>` | `fn retrieve(&self, request: &Self::Request) -> Self::Return` | API 合理化 |

---

## 十一、方案二：Actor 模式异步化方案（备选）

### 11.1 背景与约束

- **`WorkingMemory`** 将被包装在 `Arc<WorkingMemory>` 中，并克隆到多个 tokio task
- **tokio 是多线程异步环境**，需要线程安全
- **`MemoryCluster`** 内部使用 `StableDiGraph`（petgraph），**不是 `Send + Sync`**
- **`SlidingWindow`** 有异步 LLM 调用，但不需要访问 `MemoryCluster`
- **consolidation 任务**需要同时访问 `SlidingWindow` 和 `MemoryCluster`

### 11.2 架构设计

```text
┌─────────────────────────────────────────────────────────────────┐
│  GraphActor  (运行在独立 tokio task 中)                           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  GraphState (原 MemoryCluster 内部结构)                    │ │
│  │  - 通过 mpsc channel 接收命令                             │ │
│  │  - 使用 spawn_blocking 处理 CPU-bound 操作                │ │
│  │  - 通过 oneshot 返回结果                                   │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ mpsc::Sender<GraphCommand>
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  WorkingMemoryHandle  (可 Clone，跨 task 共享)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  graph_handle: GraphHandle                               │  │
│  │  sliding_window: Arc<tokio::sync::RwLock<SlidingWindow>> │  │
│  │  records: Arc<tokio::sync::RwLock<HashMap<...>>>         │  │
│  │  state: Arc<parking_lot::RwLock<WorkingState>>           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  // 协调操作（consolidation）                                     │
│  pub async fn consolidate(&self, ...) -> Result<...> {        │
│      let sw = self.sliding_window.read().await;                │
│      let nodes = self.graph_handle.get_nodes(...).await?;       │
│      // 协调 sliding_window 和 graph                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 11.3 核心类型定义

#### 11.3.1 `GraphState`（内部结构，原 `MemoryCluster`）

```rust,ignore
// memory_cluster.rs

struct GraphState {
    graph: StableDiGraph<MemoryNote, GraphMemoryLink>,
    mem_id_to_index: HashMap<MemoryId, NodeIndex>,
    link_id_to_index: HashMap<LinkId, EdgeIndex>,
    incompletely_linked_note: HashMap<MemoryId, Vec<(NodeIndex, MemoryLink)>>,
    embedding_store: HashMap<MemoryId, MemoryEmbedding>,
}
```

#### 11.3.2 `GraphCommand` 枚举

```rust,ignore
enum GraphCommand {
    // 读操作
    GetNode(MemoryId, oneshot::Sender<Option<MemoryNote>>),
    GetEmbedding(MemoryId, oneshot::Sender<Option<MemoryEmbedding>>),
    ContainsNode(MemoryId, oneshot::Sender<bool>),
    HasEdge(LinkId, oneshot::Sender<bool>),
    GetDirectedLinkedEdges(MemoryId, Direction, oneshot::Sender<Option<Vec<LinkId>>>),
    GetAllLinkedEdges(MemoryId, oneshot::Sender<Option<Vec<LinkId>>>),
    
    // 写操作
    AddSingleNode(EmbeddedMemoryNote, oneshot::Sender<Result<NodeIndex>>),
    Merge(Vec<EmbeddedMemoryNote>, oneshot::Sender<Result<()>>),
    RemoveSingleNode(MemoryId, oneshot::Sender<Option<MemoryNote>>),
    RefreshNode(MemoryId, oneshot::Sender<Result<()>>),
    
    // 生命周期
    Shutdown(oneshot::Sender<()>),
}
```

#### 11.3.3 `GraphActor`

```rust,ignore
pub struct GraphActor {
    receiver: mpsc::Receiver<GraphCommand>,
    state: GraphState,
}

impl GraphActor {
    pub async fn run(&mut self) {
        while let Some(cmd) = self.receiver.recv().await {
            self.handle_command(cmd).await;
        }
    }
    
    async fn handle_command(&mut self, cmd: GraphCommand) {
        match cmd {
            GraphCommand::AddSingleNode(node, tx) => {
                // spawn_blocking 包装 CPU-bound 操作
                let result = tokio::task::spawn_blocking(move || {
                    let mut state = GraphState::new();
                    state.add_single_node(node);
                    Ok(())
                }).await;
                tx.send(result.unwrap()).ok();
            }
            GraphCommand::GetNode(id, tx) => {
                let result = tokio::task::spawn_blocking(move || {
                    self.state.get_node(id).cloned()
                }).await;
                tx.send(result.ok()).ok();
            }
            // ... 其他命令类似
            _ => {}
        }
    }
}
```

#### 11.3.4 `GraphHandle`

```rust,ignore
#[derive(Clone)]
pub struct GraphHandle {
    sender: mpsc::Sender<GraphCommand>,
}

impl GraphHandle {
    pub async fn get_node(&self, id: MemoryId) -> Option<MemoryNote> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(GraphCommand::GetNode(id, tx)).await.ok()?;
        rx.await.await.ok()?
    }
    
    pub async fn contains_node(&self, id: MemoryId) -> bool {
        let (tx, rx) = oneshot::channel();
        self.sender.send(GraphCommand::ContainsNode(id, tx)).await.ok()?;
        rx.await.await.unwrap_or(false)
    }
    
    pub async fn get_embedding(&self, id: MemoryId) -> Option<MemoryEmbedding> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(GraphCommand::GetEmbedding(id, tx)).await.ok()?;
        rx.await.await.ok()?
    }
    
    pub async fn add_single_node(&self, node: EmbeddedMemoryNote) -> Result<NodeIndex> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(GraphCommand::AddSingleNode(node, tx)).await??;
        rx.await??.map_err(Into::into)
    }
    
    pub async fn merge(&self, nodes: Vec<EmbeddedMemoryNote>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(GraphCommand::Merge(nodes, tx)).await??;
        rx.await??
    }
    
    pub async fn remove_single_node(&self, id: MemoryId) -> Option<MemoryNote> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(GraphCommand::RemoveSingleNode(id, tx)).await.ok()?;
        rx.await.await.ok()?
    }
    
    pub fn new() -> (Self, GraphActor) {
        let (tx, rx) = mpsc::channel(1024);
        let handle = Self { sender: tx };
        let actor = GraphActor::new(rx);
        (handle, actor)
    }
}
```

#### 11.3.5 `WorkingMemoryHandle`

```rust,ignore
// working_memory.rs

use tokio::sync::RwLock;
use std::sync::Arc;

pub struct WorkingMemoryHandle {
    graph_handle: GraphHandle,
    sliding_window: Arc<RwLock<SlidingWindow>>,
    records: Arc<RwLock<HashMap<MemoryId, Record>>>,
    state: Arc<parking_lot::RwLock<WorkingState>>,
}

impl WorkingMemoryHandle {
    pub fn new(window_capacity: usize) -> Self {
        // 启动 GraphActor
        let (graph_handle, graph_actor) = GraphHandle::new();
        tokio::spawn(async move { graph_actor.run().await });
        
        Self {
            graph_handle,
            sliding_window: Arc::new(RwLock::new(SlidingWindow::new(window_capacity))),
            records: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(parking_lot::RwLock::new(WorkingState::Idle)),
        }
    }
    
    // 状态
    pub async fn transition_to_working(&self) {
        *self.state.write() = WorkingState::Working;
    }
    
    pub async fn transition_to_idle(&self) {
        *self.state.write() = WorkingState::Idle;
    }
    
    pub fn state(&self) -> WorkingState {
        *self.state.read()
    }
    
    // Graph 操作（委托给 GraphHandle）
    pub async fn add_node(&self, node: EmbeddedMemoryNote) -> Result<NodeIndex> {
        let node_id = node.note().id();
        let result = self.graph_handle.add_single_node(node).await?;
        self.records.write().await.insert(node_id, Record::new(node_id));
        Ok(result)
    }
    
    pub async fn remove_node(&self, node_id: MemoryId) -> Option<MemoryNote> {
        self.records.write().await.remove(&node_id);
        self.graph_handle.remove_single_node(node_id).await
    }
    
    pub async fn get_node(&self, id: MemoryId) -> Option<MemoryNote> {
        self.graph_handle.get_node(id).await
    }
    
    pub async fn contains_node(&self, id: MemoryId) -> bool {
        self.graph_handle.contains_node(id).await
    }
    
    pub async fn merge(&self, nodes: Vec<EmbeddedMemoryNote>) -> Result<()> {
        for node in &nodes {
            let node_id = node.note().id();
            self.records.write().await.insert(node_id, Record::new(node_id));
        }
        self.graph_handle.merge(nodes).await
    }
    
    // SlidingWindow 操作
    pub async fn sliding_window(&self) -> Arc<RwLock<SlidingWindow>> {
        self.sliding_window.clone()
    }
    
    // Record 操作
    pub async fn record_retrieval(&self, node_id: MemoryId) {
        let mut records = self.records.write().await;
        if let Some(record) = records.get_mut(&node_id) {
            record.record_retrieval();
        } else {
            let mut record = Record::new(node_id);
            record.record_retrieval();
            records.insert(node_id, record);
        }
    }
    
    // Consolidation - 协调两个子系统
    pub async fn consolidate(&self, client: &LlmClient) -> Result<ConsolidationResult> {
        // 示例协调模式
        let window_snapshot = self.sliding_window.read().await.clone();
        let relevant_nodes = self.graph_handle.get_all_linked_edges(/* ... */).await?;
        // 协调 sliding_window 和 graph
        todo!()
    }
}
```

### 11.4 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **所有操作都通过 Actor** | 是 | 保证一致性，避免读写冲突 |
| **CPU-bound 操作** | `spawn_blocking` | 避免阻塞 async runtime |
| **SlidingWindow 和 records** | `Arc<RwLock<>>` | 独立组件，LLM 调用是 I/O-bound |
| **GraphHandle** | 只通过 channel 通信 | `StableDiGraph` 不是 `Send + Sync` |

### 11.5 与方案一（DashMap）的对比

| 维度 | 方案一 (DashMap) | 方案二 (Actor) |
|------|------------------|----------------|
| **复杂度** | 较低，改动较小 | 较高，需要 actor 模式 |
| **一致性** | 需要额外同步 | Actor 天然序列化 |
| **并发读** | DashMap 支持 | 所有操作序列化 |
| **petgraph 线程安全** | 仍需处理 | 不需要，graph 在单 task 内 |
| **适用场景** | 读多写少 | 写不频繁 |
| **consolidation 协调** | 需要额外锁 | 通过 handle 协调 |

### 11.6 风险与注意事项

1. **Actor 单点瓶颈**：如果写操作非常频繁，actor 的序列化可能成为瓶颈。但根据用户描述，写操作（merge、add_single_node）不频繁。

2. **spawn_blocking 的使用**：CPU-bound 的图操作通过 `spawn_blocking` 托付给阻塞线程池，保持 async runtime 的响应性。

3. **SlidingWindow 独立锁**：SlidingWindow 的 LLM 调用不会阻塞 GraphActor，两者可并行。

4. **consolidation 的原子性**：如果 consolidation 需要原子地访问 graph 和 sliding_window，可能需要额外的协调机制。

### 11.7 实施计划

#### Phase 1：`memory_cluster.rs` 重构

- [ ] 将 `MemoryCluster` 重命名为 `GraphState`（内部结构）
- [ ] 定义 `GraphCommand` 枚举
- [ ] 实现 `GraphActor`
- [ ] 实现 `GraphHandle`
- [ ] 保留 `MemorySubCluster`（同步视图）
- [ ] 重写测试用例

#### Phase 2：`working_memory.rs` 重构

- [ ] 实现 `WorkingMemoryHandle`
- [ ] 将 `SlidingWindow` 和 `records` 包装为 `Arc<RwLock<>>`
- [ ] 所有方法改为 async
- [ ] 实现 `consolidate()` 方法
- [ ] 重写测试用例

#### Phase 3：调用方更新

- [ ] 更新所有使用 `WorkingMemory` 的地方
- [ ] 将 `.await` 添加到异步方法调用
- [ ] 移除 `memory_cluster_mut()` 调用

### 11.8 相关文件变更

| 文件 | 变更 |
|------|------|
| `src/memory/memory_cluster.rs` | `MemoryCluster` → `GraphState`，新增 `GraphCommand`、`GraphActor`、`GraphHandle` |
| `src/memory/working_memory.rs` | 新增 `WorkingMemoryHandle`，async 化所有方法 |
| 其他调用方文件 | 添加 `.await`，使用 `WorkingMemoryHandle` |
