# soul-mem-core

> 依据 `feature/test_framework` 分支工作区代码整理。

## 1. 职责定位与依赖

`soul-mem-core` 是记忆系统的**领域数据模型层**（纯数据结构 crate）。它只定义"记忆节点"与
"记忆链接"及其三类记忆（情境/语义/程序性）的 Rust 结构体、ID 类型、Builder 和序列化，
**不包含**任何 I/O、存储、图算法、向量计算或 LLM 调用。图的构建（petgraph）、PPR 检索、
embedding 等全部由下游 crate 完成。

```toml
[dependencies]
serde = { workspace = true }   # 序列化/反序列化
uuid = { workspace = true }    # MemoryId/LinkId（v4）
chrono = { workspace = true }  # DateTime<Utc> 时间字段
thiserror = { workspace = true }
```

**关键点**：本 crate **不依赖 petgraph**（petgraph 仅出现在 `soul-mem-algo`、`soul-mem-runtime`、
`benches` 中）。旧文档中"工作记忆使用 petgraph"的职责在下游 runtime crate。

下游消费方：`soul-mem-runtime`、`soul-mem-algo`、`soul-mem-query`、`soul-tune`、`benches`。

## 2. 模块结构

```text
src/
├── lib.rs                 # pub mod memory_links; pub mod memory_note;
├── memory_note.rs         # MemoryId、MemoryNote、MemoryType、MemoryNoteBuilder
│   ├── sem_mem.rs         # ConceptType、SemMemory
│   ├── situation_mem.rs   # SituationType、AbstractSituation、SpecificSituation、Context、叶子结构
│   └── proc_mem.rs        # ActionType、SkillRecord、Action、ProcMemory
└── memory_links.rs        # LinkId、MemoryLink、MemoryLinkType
    ├── sem_mem.rs         # SemMemLink{verb, confidence}
    ├── situation_mem.rs   # SituationMemLink{AbstractToSpecific, SpecificToAbstract}
    └── proc_mem.rs        # ProcMemLink::TrigToAction{prob}
```

## 3. 记忆节点

### MemoryId

`Uuid` newtype，`Copy + Hash + Eq + Ord`，可直接作 HashMap 键：
`MemoryId::new()`（v4 随机）、`From<Uuid>`、`Default`、`Display`。

### MemoryNote

所有字段私有，仅经 getter 访问：

```rust,ignore
pub struct MemoryNote {
    id: MemoryId,
    tags: Vec<String>,                  // 标签（注释：暂定参与 embedding）
    retrieval_count: usize,             // 记忆被提取的次数
    create_time: DateTime<Utc>,
    last_accessed_time: DateTime<Utc>,  // 最后访问时间
    mem_type: MemoryType,               // 三类记忆之一
    mem_links: Vec<MemoryLink>,         // 该节点出发的全部出边
}
```

关键方法：`retrieval_increment()`（`retrieval_count += 1` 并刷新 `last_accessed_time`）、
`links()`/`links_mut()`、`mem_type()`/`mem_type_mut()`。

### MemoryType

```rust,ignore
pub enum MemoryType {
    Semantic(SemMemory),
    Situation(SituationType),
    Procedure(ProcMemory),
}
```

### MemoryNoteBuilder

`mem_type` 必填，其余可选；`build()` 校验 `last_accessed_time < create_time` 时报
`MemoryNoteBuildError::TimeConflict`。

> ⚠️ 实现隐患：`build()` 比较的是两个 `Option<DateTime<Utc>>`，Rust 中 `None < Some(_)` 为真，
> 因此**只设置 create_time 而未设置 last_accessed_time 时也会返回 TimeConflict**。

## 4. 三类记忆的数据结构

### 语义记忆（SemMemory）

```rust,ignore
pub enum ConceptType { Entity, Abstract }
pub struct SemMemory {
    pub content: String,        // 概念/事实内容
    pub aliases: Vec<String>,   // 别名，去重消歧
    pub concept_type: ConceptType,
    pub description: String,
}
```

### 情境记忆（SituationType）

```rust,ignore
pub enum SituationType {
    AbstractSituation(AbstractSituation),   // 抽象：Location/Participant/Environment/Event
    SpecificSituation(SpecificSituation),   // 具体：narrative + time_span + context
}

pub struct SpecificSituation {
    narrative: String,        // 情境的自然语言描述
    time_span: DateTime<Utc>, // 单个时间点（注意：非时间段）
    context: Context,
}

pub struct Context {
    location: Option<Location>,     // Location{name, coordinates}
    participants: Vec<Participant>, // Participant{name, role}
    emotions: Vec<Emotion>,         // Emotion{name, intensity: f32}
    sensory_data: Vec<SensoryData>, // SensoryData{name, intensity: f32}
    environment: Environment,       // Environment{atmosphere, tone}（必填，非 Option）
    event: Vec<Event>,              // Event{action, action_intensity, initiator, target}
}
```

### 程序性记忆（ProcMemory）

```rust,ignore
pub enum ActionType { Speak, Skill(SkillRecord), Think }  // SkillRecord 为占位空结构

pub struct Action { content: String, action_type: ActionType }
pub struct ProcMemory { action: Action }
```

**注意**：程序性记忆节点只有 `Action`，**没有**独立 trigger 节点类型——trigger→action 的
转移关系建模为**边** `ProcMemLink::TrigToAction{prob}`（见下）。

## 5. 记忆链接

### MemoryLink

```rust,ignore
pub struct MemoryLink {
    id: LinkId,
    from: MemoryId,
    to: MemoryId,
    pub intensity: f64,        // 唯一的 pub 字段：公共连接强度（默认 1.0）
    link_type: MemoryLinkType,
}
```

`new(from, to, link_type)` 默认 intensity=1.0；`from_tuple/into_tuple` 与
`(MemoryId, MemoryId, MemoryLinkType, f64)` 元组互转。

### MemoryLinkType

```rust,ignore
pub enum MemoryLinkType {
    Proc(ProcMemLink),                       // TrigToAction{prob: f64} 转移概率
    Sem(SemMemLink),                         // SemMemLink{verb: String, confidence: f32}
    Situation(SituationMemLink),             // AbstractToSpecific{} / SpecificToAbstract{}
}
```

`SpecificToAbstract` 的 doc 注释明确其角色：**具体情境 → 抽象情境的反向边**，PPR 从具体
情境种子游走到抽象模式节点后，抽象节点作为 Bayes 动作提取的优先源。

## 6. 图结构关系

- core **本身不用 petgraph**，图以"边随节点存储"表达：`MemoryNote.mem_links` 内嵌出边，
  `MemoryLink` 以 ID 引用两端节点。
- 权重体系：`intensity`（公共，f64）+ 类型特定置信度/概率（`SemMemLink.confidence: f32`、
  `TrigToAction.prob: f64`）。
- 下游真实图：`soul-mem-runtime` 的 `MemoryCluster` 用
  `petgraph::stable_graph::StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink>` 构建，
  维护 `mem_id_to_index`/`link_id_to_index` 映射与 `incompletely_linked_note`（待链接边缓冲）。

## 7. 与旧文档（beta_ver.md）的出入

| 旧设计 | 当前实现 |
|--------|---------|
| `time_span` 为起止时间段 | 单个 `DateTime<Utc>`（无结束时间） |
| context 含 `situation` 自然语言背景 | 无该字段；新增必填 `environment{atmosphere, tone}` 与 `event[]` |
| 语义边属性 verb+intensity+confidence | `intensity` 上移为 `MemoryLink` 公共字段；`SemMemLink` 仅 verb+confidence |
| 程序性记忆 trigger/action 两类节点（待定） | 无 trigger 节点；`ProcMemory` 仅 Action，trigger→action 为 `TrigToAction` 边；`ActionType` 细分为 Speak/Skill/Think |
| 抽象情境节点为"二级索引"抽象实体 | `AbstractSituation` 枚举（Location/Participant/Environment/Event 四类抽象元素），抽象↔具体关系由链接方向表达 |
| （未提及） | UUID ID 体系、`tags`/`retrieval_count`/时间元字段、Builder 校验、双方向情境链接 |

## 8. 实现备注

- 每个源文件均带内联单测（共 23 个），覆盖 ID、Builder、getter/mutator 往返、链接构造与
  元组互转。
- 序列化：无 `#[serde(rename)]` 等属性，字段名即 snake_case；`MemoryId`/`LinkId` 在 JSON 中
  为 UUID 字符串，`DateTime<Utc>` 为 RFC3339 字符串。
- 派生 trait 差异：含浮点字段的类型只有 `PartialEq/PartialOrd`；`ActionType`/`Action` 等有
  `Eq/Ord`；`TrigToAction` 是 `Copy`。
