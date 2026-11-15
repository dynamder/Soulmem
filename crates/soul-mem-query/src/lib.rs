//! 嵌入与评分层：把记忆与查询变成向量，并计算相似度分数。
//!
//! - [`embedding`]：`Embeddable` / `EmbeddingModel`、[`EmbeddingVec`](embedding::EmbeddingVec)、融合实现、`blend_weights`。
//! - [`query`]：查询类型、`compute` 评分、`string_distance`、`retrieve`。

pub mod embedding;
pub mod query;
