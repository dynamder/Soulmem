//! 嵌入与评分层：把记忆与查询变成向量，并计算相似度分数。
//!
//! - [`embedding`]：`Embeddable` / `EmbeddingModel`、[`EmbeddingVec`](embedding::EmbeddingVec)、融合实现、`blend_weights`。
//! - [`query`]：查询类型、`compute` 评分、`string_distance`、`retrieve`。
//!
//! # 两个容易踩的点
//!
//! 1. **零向量是跨 crate 契约**：空标签写 `EmbeddingVec::zero(dim)`，runtime 仓储会跳过零向量、
//!    不写向量列。改这里会静默改变检索结果，且没有编译错误。
//! 2. **`EmbeddedMemoryNote::compute_fused` 是唯一评分入口**；`MemoryEmbedding` 不提供距离方法。
//!    不要造同名双份 API。
//!
//! 大量测试用 `BgeSmallZh::default_cpu()` 下载模型；离线失败属于环境问题。

pub mod embedding;
pub mod query;
