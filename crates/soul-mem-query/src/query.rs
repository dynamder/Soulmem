//! 检索侧的查询类型与评分。
//!
//! # 三个模块
//!
//! - [`compute`]：**评分核心**。`AnonymousQueryCompute` / `QueryCompute` 两个 trait
//!   定义了"查询与某个嵌入类型怎么算分"，各嵌入类型分别实现；总入口是
//!   `EmbeddedMemoryNote::compute_fused`（融合 tag 通道与 variant 通道）。
//!   "Anonymous" 指的是**不携带身份**的打分形式（只返回 `f32`，而不是带来源的
//!   `QueryComputeResult`），便于在打分循环里复用。
//! - [`string_distance`]：字符串相似度通道。对结构化字段（概念、描述、动作文本等）
//!   做字符串比对并取 max pooling，权重同样来自
//!   [`BlendWeights`](crate::embedding::blend_weights::BlendWeights)。
//! - [`retrieve`]：查询侧的领域类型（情境查询单元、`MemoryRetrieveQuery` 等）
//!   与数据库召回入口。
//!
//! # 已知陷阱
//!
//! `SituationQueryUnit::with_time_span` 会**接受并保存** `time_span`，
//! 但当前评分逻辑**完全不使用它**（字段保留供后续实现，见 `retrieve.rs` 与
//! `compute.rs` 的 TODO）。也就是说带时间过滤的查询会得到"看起来合理"的结果，
//! 而过滤被静默丢弃。实现时间评分时需要一次性校正所有既有调用方的排序。

pub mod compute;
pub mod retrieve;
pub mod string_distance;
