//! 嵌入与评分层：把记忆与查询变成向量，并计算相似度分数。
//!
//! # 两部分
//!
//! - [`embedding`]：`Embeddable` / `EmbeddingModel` 两个 trait、向量容器
//!   [`EmbeddingVec`](embedding::EmbeddingVec)、各类"嵌入 + 融合"实现，
//!   以及可调权重的集中定义（`blend_weights`）。
//! - [`query`]：检索侧的查询类型、**评分**（`compute`）、字符串通道打分
//!   （`string_distance`），以及与数据库召回对接的入口（`retrieve`）。
//!
//! # 从哪里开始读
//!
//! | 你想知道 | 去看 |
//! |---|---|
//! | 一个记忆/查询怎么变成向量 | [`Embeddable`](embedding::Embeddable) 与 `embedding/{note,sem,situation}.rs` |
//! | 向量相似度到底怎么算 | `embedding::vec::fused_cosine_core`（单遍 lane SIMD，无 `unsafe`） |
//! | 最终分数怎么合成 | `query::compute::EmbeddedMemoryNote::compute_fused` |
//! | 各打分通道的权重从哪来 | [`BlendWeights`](embedding::blend_weights::BlendWeights) |
//! | 字符串相似度通道 | [`query::string_distance`] |
//! | 数据库召回入口 | [`query::retrieve`] |
//!
//! # 两个容易踩的点
//!
//! 1. **零向量是跨 crate 契约**：标签为空时会写入 `EmbeddingVec::zero(model.dim())`，
//!    而 `soul-mem-runtime` 的仓储层会**跳过零向量**、不写入 schema 的向量列
//!    （见 `storage/surreal/repository.rs` 的 `is_zero()` 分支）。改这里的零向量行为
//!    会静默改变检索结果，且不会有任何编译错误。
//! 2. **`EmbeddedMemoryNote::compute_fused` 是唯一的评分入口**。
//!    [`MemoryEmbedding`](embedding::note::MemoryEmbedding) **不提供**向量距离/相似度方法，
//!    只有 [`EmbeddingVec`](embedding::EmbeddingVec) 提供。
//!    **不要制造同名双份 API**——那会让调用方在一种类型上正常工作、在另一种上 abort，
//!    而编译器不会提示。
//!
//! # 测试的模型依赖
//!
//! 大量测试通过 `BgeSmallZh::default_cpu().unwrap()` 构造真实模型，会**下载**权重。
//! CI 靠 `actions/cache` 缓存 `~/.cache/huggingface`；离线环境下失败属于环境问题。
//! 纯数值测试（`embedding/vec.rs`、`query/string_distance.rs`）不依赖模型，可作为参考写法。

pub mod embedding;
pub mod query;
