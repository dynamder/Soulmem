# 记忆模型

记忆以图的方式组织——这个概念在 [核心概念 · 记忆图](../concepts/memory-graph.md) 里已经讲过：节点是记忆，边是关联。这一章回答另一个问题：**记忆图在代码里长什么样？**

我们从 `soul-mem-core` 的当前代码（`feature/test_framework` 分支工作区）出发，描述记忆的**数据结构模型**：三类记忆的节点结构、通用的节点/链接结构、以及图的构建方式。更完整的模块级说明见 [soul-mem-core](../crates/soul-mem-core.md)。

> [!note]
> 本文档只讲数据结构：节点有哪些字段、边有哪些类型、图如何构建。图上的算法（PPR 联想、遗忘衰减、巩固）见 [深入实现](../algorithm/retrieve.md) 相关章节。

## 1. 三类记忆

总记忆图分为三类子图，相互关联：

```mermaid
graph LR
	情境记忆 <--> 语义记忆
	语义记忆 <--> 程序性记忆
	情境记忆 <--> 程序性记忆
```

### 1.1 情境记忆（Situation / Episodic）

记忆具体事件经历（如"昨天中午和同学出去吃了麻辣烫"）。节点分两类：

- **具体情境 `SpecificSituation`**：`narrative`（叙述）+ `time_span`（时间）+ `context`（上下文）。
- **抽象情境 `AbstractSituation`**：`Location` / `Participant` / `Environment` / `Event`
  四类抽象元素之一，作为具象情境的**二级索引**与子图间接口节点。

`Context` 的六个字段（结构定义于 `soul-mem-core/src/memory_note/situation_mem.rs`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `location` | `Option<Location>` | ❌ | 地点（name + coordinates） |
| `participants` | `Vec<Participant>` | ❌ | 参与者（name + role） |
| `emotions` | `Vec<Emotion>` | ❌ | 情感（name + intensity） |
| `sensory_data` | `Vec<SensoryData>` | ❌ | 感官数据（name + intensity，当前保留） |
| `environment` | `Environment` | ✅ | 环境（atmosphere + tone） |
| `event` | `Vec<Event>` | ❌ | 事件（action + action_intensity + initiator + target） |

### 1.2 语义记忆（Semantic）

记忆通用概念、事实与关系（如"北京是中国的首都"），是图的中枢核心（类海马体索引）。

```rust,ignore
pub struct SemMemory {
    pub content: String,        // 概念/事实内容
    pub aliases: Vec<String>,   // 别名，去重消歧
    pub concept_type: ConceptType, // Entity（具象）/ Abstract（抽象）
    pub description: String,
}
```

语义链接负载：`SemMemLink { verb: String, confidence: f32 }`（谓词 + 置信度）。

### 1.3 程序性记忆（Procedural）

存储"肌肉记忆"、条件反射、行为习惯（如"沉思时会摸下巴"、"傲娇"）。

```rust,ignore
pub enum ActionType { Speak, Skill(SkillRecord), Think }
pub struct Action { content: String, action_type: ActionType }
pub struct ProcMemory { action: Action }
```

- 节点只有 `Action`（**无**独立 trigger 节点类型）。
- trigger→action 的转移关系建模为**边**：`ProcMemLink::TrigToAction { prob: f64 }`（转移概率）。
- 语义记忆可直接连到 action 作为"概念性补充/自我认知"，不触发动作；由情境（trigger）联想
  到的 action 才会进入动作提示词（旧设计约定，见 beta_ver.md）。

## 2. 通用节点与链接结构

### 2.1 记忆节点 MemoryNote

```rust,ignore
pub struct MemoryNote {
    id: MemoryId,                       // UUID newtype
    tags: Vec<String>,                  // 标签（暂定参与 embedding）
    retrieval_count: usize,             // 提取次数
    create_time: DateTime<Utc>,
    last_accessed_time: DateTime<Utc>,  // 最后访问时间（LRU/热度相关）
    mem_type: MemoryType,               // 三类记忆之一
    mem_links: Vec<MemoryLink>,         // 该节点出发的出边
}
```

### 2.2 记忆链接 MemoryLink

```rust,ignore
pub struct MemoryLink {
    id: LinkId,
    from: MemoryId,
    to: MemoryId,
    pub intensity: f64,       // 公共连接强度（默认 1.0），遗忘机制的关键载体
    link_type: MemoryLinkType,
}

pub enum MemoryLinkType {
    Proc(ProcMemLink),        // TrigToAction{prob}
    Sem(SemMemLink),          // SemMemLink{verb, confidence}
    Situation(SituationMemLink), // AbstractToSpecific{} / SpecificToAbstract{}
}
```

- `intensity`（f64，默认 1.0）为**公共**边权；`SemMemLink.confidence`（f32）、
  `TrigToAction.prob`（f64）为类型特定权重。
- `SituationMemLink::SpecificToAbstract`：具体情境 → 抽象情境的反向边，PPR 从具体情境种子
  游走到抽象模式节点后，抽象节点作为 Bayes 动作提取的优先源。

### 2.3 图结构

- core 层"边随节点存储"（`mem_links` 内嵌出边，边以 ID 引用两端节点）。
- 运行时由 `soul-mem-runtime` 的 `MemoryCluster` 构建真实图：
  `petgraph::StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink>`（`EmbeddedMemoryNote` =
  记忆 + 嵌入向量），并维护 `mem_id_to_index` / `link_id_to_index` 映射与
  `incompletely_linked_note` 待链接缓冲；通过 `MemoryClusterHandle` 并发访问。

## 3. 长期记忆与工作记忆

- **长期记忆**：持久化记忆；测试框架中以 `fixtures/graphs/<name>.json` 图文件加载
  （格式见 [测试数据规范](../testing/测试数据规范.md)）。
- **工作记忆**：运行时激活的子图（petgraph），修改频繁；新记忆先进入工作记忆。
- **滑动窗口**：最近几轮对话（短期记忆），窗口内记忆无条件加入最终检索上下文；滑出前
  未巩固的记忆丢失。详见 [编排与数据流](orchestration.md)。

## 4. 与旧设计（beta_ver.md）的差异速览

| 项 | 旧设计 | 当前实现 |
|----|--------|---------|
| time_span | 起止时间段 | 单个时间点 |
| Context | 含 situation 背景字段 | 无；新增必填 environment 与 event 列表 |
| 语义边 | verb + intensity + confidence | intensity 上移为公共字段 |
| 程序性记忆 | trigger/action 两类节点（待定） | 仅 Action 节点 + TrigToAction 边 |
| 抽象情境 | 独立抽象节点（二级索引） | AbstractSituation 枚举（四类抽象元素） |

---

下一章：[编排与数据流](orchestration.md)——这些数据结构如何被串联成一个可以运行的系统。
