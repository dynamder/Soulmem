# soul-mem-algo

> 依据 `feature/test_framework` 分支工作区代码整理（含未提交的遗忘模块新增）。

## 1. 职责定位与依赖

`soul-mem-algo` 是 SoulMem 的**记忆算法层**，负责两大类算法：

- **遗忘（forget）**：基于艾宾浩斯遗忘曲线的惰性遗忘（遮罩 + LLM 修订）与语义字段对齐。
- **检索/联想（retrieve）**：向量相似度检索、PPR 联想扩散、贝叶斯动作推理及组合管线
  `DefaultPipeline`。

该 crate **不直接接触数据库、不直接调用 LLM API**：数据均在内存工作记忆
（`WorkingMemory` 及其 `MemoryCluster`，petgraph `StableDiGraph`）上操作；LLM 调用通过
**调用方注入的闭包**（forget 的 `llm_call` 参数）间接完成。

```toml
[dependencies]
soul-mem-core    = { path = "../soul-mem-core" }
soul-mem-query   = { path = "../soul-mem-query" }
soul-mem-runtime = { path = "../soul-mem-runtime" }
petgraph = { workspace = true }      # 图结构
ordered-float = "5.0.0"              # OrdFloat 底层
serde = { workspace = true }
rayon = "1.12.0"                     # similarity 并行打分
chrono = { workspace = true }        # 遗忘时间计算（本分支新增）
jieba-rs = { workspace = true }      # 遗忘遮罩分词（本分支新增）
rand = { workspace = true }          # 遮罩随机选择（本分支新增）

[dev-dependencies]
insta = { workspace = true }         # 快照测试
tokio = { workspace = true }         # 遗忘模块异步测试（本分支新增）
```

要点：无数据库 crate、无 async-openai、无嵌入式向量库；`chrono`/`jieba-rs`/`rand`/`tokio`
是当前分支**未提交**的新增依赖，全部服务于新的 `algo/forget/` 模块。

## 2. 模块结构

```text
src/
├── lib.rs                 # pub mod algo; pub mod common;
├── algo.rs                # pub mod forget; pub mod retrieve;
├── common.rs              # pub mod ord_float; pub mod ppr;
├── common/
│   ├── ord_float.rs       # OrdFloat<F>：可全序比较/可运算的浮点包装（实现 UnitMeasure）
│   └── ppr.rs             # naive_ppr（幂迭代）/ weighted_ppr_fp（Forward-Push）
└── algo/
    ├── forget/            # 遗忘算法（本分支新增）
    │   ├── decay_calculator.rs  # 艾宾浩斯衰减三函数
    │   ├── decay_revise.rs      # 惰性遗忘主流程 lazy_forget / align_sem_fields
    │   └── mask.rs              # 文本遮罩 mask_text（jieba 分词 + 确定性随机遮罩）
    └── retrieve/
        ├── retrieve.rs          # RetrStrategy / RetrRequest / RetrRequestConfig
        ├── association.rs       # RetrAssociation：多源 PPR 联想 + DynWeightFuncBuilder
        ├── bayes_action.rs      # RetrBayesAction：经 Proc(TrigToAction) 边聚合动作概率
        ├── short_only.rs        # RetrShortOnly：滑动窗口 + 摘要
        ├── similarity.rs        # RetrSimilarity：向量相似度 top-k（rayon 并行）
        └── complex/
            ├── default_pipeline.rs  # RetrDefaultPipeline：三段式管线
            └── assoc_with_action.rs # RetrAssociateWithAction：PPR + 双源 Bayes
```

## 3. 检索抽象

```rust,ignore
pub trait RetrStrategy: 'static {
    type Request: RetrRequest;
    type Return<'a> where Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_>;
}
pub trait RetrRequest {}

#[derive(serde::Deserialize)]
#[serde(tag = "type")]
pub enum RetrRequestConfig {
    Association(AssociationConfig),
    BayesAction(BayesActionConfig),
    AssociateWithAction(AssociateWithActionConfig),
    ShortOnly(ShortOnlyConfig),
    Similarity(SimilarityConfig),
}
```

每个策略 = unit struct（`RetrAssociation` 等）+ Config（serde 反序列化，带默认值）+
Request（由 `Config::into_request(...)` 构造）。

## 4. 检索策略

### RetrAssociation（PPR 联想）

```rust,ignore
pub struct AssociationConfig {
    pub intensity_factor: Option<f64>,    // None → 默认 1.0
    pub confidence_factor: Option<f64>,   // None → 默认 0.8
    pub damping_factor: f64,              // 默认 0.65
    pub residue_threshold: f64,           // 默认 1e-5
    pub preference: TypePreference,       // 默认 Situation
    pub top_k: usize,                     // 默认 8
}
impl RetrStrategy for RetrAssociation { type Return<'a> = Vec<(MemoryId, f64)>; }
```

流程：source（相似度种子）→ 动态边权函数 → `weighted_ppr_fp`（damping 0.65、残差阈值 1e-5）
→ 分数降序 top_k。防御：source 非空但全部解析失败时直接返回空，避免 PPR 内部
"源权重和必须为正"的 assert panic。

**动态边权**（`DynWeightFuncBuilder`，`preference` 决定类型偏好数组）：
- 目标为 `Procedure` 的边权重 0.0（Proc 节点不参与 PPR 联想，交给 Bayes 动作推理）。
- `Situation` 边 → `(confidence_boost, type_boost) = (0.8, preference[1])`；
  `Sem` 边 → `(mem.confidence, preference[0])`；`Proc` 边 → (0.0, 0.0)。
- 归一化公式：`(intensity×i + confidence_boost×c + type_boost) / (i + c + type_boost)`，
  分母为 0 返回 0.0（防 NaN）。

### RetrBayesAction（动作推理）

```rust,ignore
pub struct BayesActionConfig { pub top_k: usize }   // 默认 5
impl RetrStrategy for RetrBayesAction { type Return<'a> = Vec<(MemoryId, f64)>; }
```

流程：对每个源节点，只保留**边类型为 `Proc` 且目标为 `Procedure`** 的出边邻居作为候选
动作；`possible_actions[id] += prob × weight`（多源累加）；降序取 top_k。

### RetrSimilarity（向量相似度）

```rust,ignore
pub struct SimilarityConfig {
    pub similarity_threshold: f32,   // 默认 0.35 —— 语义为"最低兜底分"
    pub max_results: usize,          // 默认 4
}
impl RetrStrategy for RetrSimilarity { type Return<'a> = Vec<(MemoryId, f32)>; }
```

rayon 并行 `compute_fused`（余弦 + 字符串通道，字符串只加分）；过滤非有限分与低于兜底分
的节点；降序取 top-k。

### RetrShortOnly（短期记忆）

```rust,ignore
pub struct ShortOnlyConfig {
    pub clipping_length: Option<usize>,  // 倒序计数裁剪
    pub include_summary: bool,           // 默认 false
}
impl RetrStrategy for RetrShortOnly { type Return<'a> = (Arc<[Information]>, Arc<str>); }
// (窗口消息, 摘要)
```

### RetrAssociateWithAction（PPR + 双源 Bayes）

```rust,ignore
pub struct AssociateWithActionConfig {
    pub association: AssociationConfig,
    pub action_top_k: usize,                  // 默认 3
    pub abstract_source_priority: f64,        // 默认 2.0 —— 抽象源加权
}
pub struct AssociateWithActionResult { pub memory: Vec<(MemoryId, f64)>, pub action: Vec<(MemoryId, f64)> }
```

流程：`RetrAssociation` 得 PPR 结果 → `merge_situation_sources`（相似度种子 ∪ PPR 结果，
同 id 取 max，**只保留 Situation 节点**，过滤 ≤0 分）→ 抽象源 ×2 / 具体源 ×1 →
`softmax` 归一化 → `RetrBayesAction` 得 action。

### RetrDefaultPipeline（三段式默认管线）

```rust,ignore
pub struct DefaultPipelineConfig {
    pub short_mem_with_history: ShortOnlyConfig,
    pub similarity: SimilarityConfig,
    pub assoc_with_action: AssociateWithActionConfig,
}
pub struct DefaultPipelineResult {
    pub association: Vec<(MemoryId, f64)>,  // 合并去重后 top-10
    pub action: Vec<(MemoryId, f64)>,       // 动作 top-k
    pub short_history: Arc<[Information]>,
    pub short_mem: Arc<str>,
    pub priority: u32,
}
```

严格按序：① ShortOnly → ② Similarity（PPR 源种子）→ ③ AssociateWithAction →
`merge_note_scores`（同 id 取高、降序、截断 `MAX_PIPELINE_NOTES = 10`）。

## 5. PPR 核心（common/ppr.rs）

- 目标方程：`ppr_s = damping × P × ppr_s + (1 - damping) × personalized_vec`。
- **多源**：`personalized_vec` 可含多个源节点，权重和 > 0 并归一化为概率分布。
- 无出度节点：与源节点建立"虚拟连接"，残差按源数量均分回源。
- `naive_ppr`：幂迭代 `nb_iter` 次，每次归一化；针对 StableDiGraph 处理空洞索引。
- `weighted_ppr_fp`（生产路径，EdgePush/Forward-Push 风格）：
  - `assert!(0 ≤ damping < 1)`——`damping == 1.0` 必须拒绝（残差无法转化为 reserve 会
    无限循环，注释明确写"防止 DoS"）。
  - 残差/保留模型：`reserve += (1-damping) × residue`；每次 push 残差最大的节点，残差
    ≤ 阈值停止。
  - 边权动态计算并缓存（`ppr_edge_weight_cache`），按总和归一化（和为 0 保持原值防 NaN）。
  - 迭代上限安全网：`node_bound × 1024`。
  - 最终输出 `(node_id, reserve/sum)` 概率分布。

## 6. 遗忘算法（forget/，本分支新增）

### 艾宾浩斯衰减（decay_calculator.rs）

```rust,ignore
pub const DEFAULT_MAX_ACTIVATION_CAP: usize = 50;
// R(t) = e^(-t/τ)；τ = adjusted_half_life / ln2；
// adjusted_half_life = base_half_life_hours × (1 + active_factor × min(retrieval_count, cap))
pub fn ebbinghaus_decay(...) -> f32;
pub fn compute_missing_degree(...) -> f32;        // 1.0 - decay，0~1
pub fn edge_decay_intensity(original_intensity: f64, ...) -> f64;  // 边强度 × 节点衰减
```

激活次数越多（封顶 50）半衰期越长、衰减越慢；`elapsed_hours <= 0` 返回 1.0。

### 惰性遗忘主流程（decay_revise.rs）

```rust,ignore
pub const DEFAULT_BASE_HALF_LIFE_HOURS: f32 = 24.0;
pub const DEFAULT_ACTIVE_FACTOR: f32 = 0.1;
pub const MASK_THRESHOLD: f32 = 0.05;      // 缺失度低于此 → 不遗忘
pub const REVISE_THRESHOLD: f32 = 0.15;    // 缺失度高于此 → 调 LLM 修订
pub const ALIGN_LENGTH_CAP_THRESHOLD: f32 = 0.6;

pub enum ForgetAction { NoAction, MaskOnly { .. }, Revised { .. } }
pub async fn lazy_forget<F, Fut>(node: &mut MemoryNote, current_time, jieba, llm_call) -> ForgetAction;
pub async fn align_sem_fields<F, Fut>(node: &mut MemoryNote, llm_call) -> Result<(), ...>;
pub fn get_summary(node: &MemoryNote) -> Option<String>;   // narrative / content
```

`lazy_forget` 流程：
1. 类型过滤：仅 `SpecificSituation` 与 `Semantic` 可遗忘，其余返回 `NoAction`。
2. `compute_missing_degree`（默认 24.0/0.1/50）算缺失度 `md`。
3. `md < 0.05` → NoAction；`md < 0.15` → 只遮罩（MaskOnly）；否则调 LLM 猜补（Revised），
   失败降级 MaskOnly。
4. LLM prompt：system 为固定英文"reconstructs partially masked memories"，user 为遮罩文本。

### 文本遮罩（mask.rs）

```rust,ignore
pub const MASK_WORD: &str = " [masked] ";
pub fn mask_text(text: &str, missing_degree: f32, jieba: &Jieba) -> MaskResult;
```

jieba 分词（HMM）→ 遮罩词数 `n = round(md × total)` → **确定性随机**（种子
`hash(text) ^ (degree.to_bits() × 114514)`，StdRng 洗牌）选词替换。同文本同缺失度输出恒等。

### 语义字段对齐（align_sem_fields）

仅 Semantic：LLM 逐行输出 `Aliases:` / `Description:` / `ConceptType:`；文意一致保留原值；
缺失度 < 0.6 时禁止 aliases 增长（防 LLM 幻觉膨胀）。

## 7. 与旧文档的出入

| 旧文档 | 实际代码 |
|--------|---------|
| SurrealDB/Qdrant 向量、图、时间序列存储 | **无任何数据库**；全部在内存工作记忆上操作 |
| async-openai 在 algo 中 | 无；LLM 通过注入闭包调用 |
| 遮罩法"记忆微元 + 透明度分层"精细模型 | 整段文本 jieba 分词 + 确定性随机遮罩 |
| 连接遗忘"低于阈值删边、删孤立节点" | 仅 `edge_decay_intensity` 计算函数，无删边实现 |
| 遗忘范围 | 仅 SpecificSituation 与 Semantic；Procedure/Abstract 不遗忘 |
| orchestration 中 DefaultPipeline 三步描述 | 与 `default_pipeline.rs` 一致 |
| 检索轨迹报告中 `SITUATION_SIMILARITY_THRESHOLD=0.5` | 已不存在；被"兜底分 floor + 必取 top-k"语义取代 |
| 抽象 PPR 检出报告 | 与 `assoc_with_action.rs` 当前实现完全一致 |

## 8. 工作区未提交修改说明

- 新增（untracked）：`src/algo/forget/` 四个文件（遗忘算法是本分支新增能力）。
- 修改：`algo.rs`（+ `pub mod forget;`）、`Cargo.toml`（+ chrono/jieba-rs/rand，dev + tokio）。
- 其余被 git 标记 M 的文件（association.rs、ppr.rs、snap 等）经校验**仅为 CRLF→LF 行尾
  转换，无语义变化**。
