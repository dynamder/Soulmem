# soul-mem-query

> 依据 `feature/test_framework` 分支工作区代码整理。

## 1. 职责定位与依赖

`soul-mem-query` 是 SoulMem 的**向量嵌入 + 检索评分层**（`lib.rs` 仅导出 `embedding` 与
`query` 两个模块）。它负责：嵌入模型接入、记忆节点/查询的向量化、查询构建、分层余弦
相似度计算与字符串通道融合。**不含**图检索、存储、阈值过滤（阈值与 top-k 在
`soul-mem-algo` 的 `similarity.rs` 中）。

```toml
[dependencies]
candle-core = { workspace = true }   # 仅用于 EmbeddingGenError::EmbeddingFailed 错误类型
embed_anything = "0.7.1"             # 本地模型加载（内部基于 candle）
rayon = "1.10.0"                     # 并行池化
text-splitter = "0.29.3"             # 长文本分块（按字符）
strsim = "0.11.1"                    # Jaro-Winkler / 归一化 Levenshtein
soul-mem-core = { path = "../soul-mem-core" }
```

## 2. 模块结构

```text
src/
├── lib.rs                # pub mod embedding; pub mod query;
├── embedding.rs          # Embeddable/EmbeddingModel trait、错误类型
│   ├── blend_weights.rs  # BlendWeights：16 个可调权重集中定义
│   ├── embedding_model/  # bge.rs（BGE 模型）、qwen3.rs（Qwen3 模型）
│   ├── note.rs           # EmbeddedMemoryNote（记忆节点 + 向量）、MemoryEmbedding
│   ├── query.rs          # 查询侧嵌入
│   ├── sem.rs            # 语义记忆嵌入
│   ├── situation.rs      # 情境记忆嵌入（context/emotion/environment/event/location/
│   │                     #   participant/sensory_data 子模块）
│   └── vec.rs            # EmbeddingVec、mean_pooling、raw_linear_blend
└── query.rs              # 查询结构定义
    ├── compute.rs        # 分层余弦相似度计算
    ├── retrieve.rs       # 检索查询结构体（PrioritizedMemoryRetrieveQuery 等）
    └── string_distance.rs# 字符串距离（Jaro-Winkler / 归一化 Levenshtein）
```

## 3. 嵌入模型层

### EmbeddingModel trait

```rust,ignore
#[async_trait]
pub trait EmbeddingModel {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>>;
    fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec>;
    fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec>;
    // 查询侧默认同侧；检索类模型（如 BGE v1.5）覆写为前置查询指令
    fn infer_query_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> { ... }
    fn infer_query_and_fuse(...) { ... }
    fn infer_query_with_chunk(...) { ... }
    fn max_input_token(&self) -> usize;
    fn dim(&self) -> usize;   // 模型输出维度，用于无有效输入时构造零向量
}
```

### 内置模型（embedding_model/）

| 模型 | 池化 | 维度 | 分块 | 查询指令 | 加载方式 |
|------|------|------|------|---------|---------|
| BGE `bge-small-zh-v1.5` | CLS | 512 | 200 字符 | `QUERY_INSTRUCTION` 前置到每个分块 | embed_anything + candle，进程级 `OnceLock` 单例 |
| Qwen3 `Qwen3-Embedding-0.6B` | — | 1024（F32） | 6000 字符 | 无覆写（默认同 passage 侧） | candle，按需加载 |

### Embeddable trait

```rust,ignore
pub trait Embeddable {
    type EmbeddingFused;
    type EmbeddingGen;
    fn embed_and_fuse(self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingFused>;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen>;
}
```

### BlendWeights（blend_weights.rs）

集中定义全部 16 个可调权重，经 `set_blend_weights` 递归传播到查询嵌入。关键默认值：

- `tag` = 0.3、`variant` = 0.7（tag/variant 通道权重）
- `string_blend_alpha` = 0.6（字符串通道混合系数）
- 以及各结构化字段（location/participant/emotion/environment/event 等）子权重

### 错误类型

`EmbeddingGenError`（InvalidInput / EmbeddingFailed / PostCalcFailed / Anyhow）与
`EmbeddingCalcError`（InvalidVec / ShapeMismatch / IncompatibleEmbeddingTypes /
InvalidNumValue）。

## 4. 查询构建层（embedding/query/ 与 situation/）

- **语义查询**（`sem.rs`）：基于 SemMemory 的 content/aliases 构建。
- **情境查询**（`situation.rs` 及各子模块）：environment / event / location / participant
  等各类查询的向量构建。
- **note 查询**（`note.rs`）：`EmbeddedMemoryNote { note: MemoryNote, embedding: MemoryEmbedding }`，
  记忆节点 + 嵌入向量的组合，作为记忆图节点。
- 结构化字段各自生成独立嵌入，评分时按 `max` 融合（见下）。

## 5. 检索计算层（query/）

### compute.rs — 分层余弦计算

- `AnonymousQueryCompute` / `QueryCompute` 两层：匿名查询（纯文本）与结构化查询。
- **Semantic**：content/aliases 的 max_pooling。
- **情境**：多信号 max 融合（各结构化字段）。
- **抽象情境**：`fused_self()` 叙事回退（无结构化输入时用叙事文本嵌入）。
- 顶层 `compute_fused`：`max(emb, 0.6·emb + 0.4·str)` —— 字符串通道**只加分不拉低**。

### string_distance.rs

- Jaro-Winkler 与归一化 Levenshtein 距离，用于标签/字符串匹配加分。

### retrieve.rs — 查询结构体

定义 `MemoryRetrieveQueryVariant`、`PrioritizedMemoryRetrieveQuery` 等查询类型（供
`soul-mem-algo` 的 `association.rs` 消费）。本 crate 不做阈值过滤与排序。

> 阈值过滤（默认 0.35）与 top-k（默认 4）在 `soul-mem-algo/src/algo/retrieve/similarity.rs`。

## 6. 与旧文档（beta_ver.md）的出入

| 旧设计 | 当前实现 |
|--------|---------|
| 向量相似度搜索定位 | 一致（本 crate 是向量层） |
| （未提及） | 新增**字符串通道**（只加分）、**tag 通道**、抽象情境叙事回退 |
| context 含 `situation` 背景字段 | 无该字段；`environment`/`event` 等结构化字段参与评分 |
| 程序性记忆 | 嵌入侧仍为空占位 `Procedure()` |
| time_span | 已定义但未参与评分 |
