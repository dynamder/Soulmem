//! 记忆算法层：检索、遗忘、巩固。
//!
//! 子域：`retrieve`（DefaultPipeline + 相似度 + 贝叶斯动作 + PPR）、`forget`（衰减/遮罩/LLM 补全）、
//! `consolidate`（摘要 → 记忆节点）。图算法在 `common`。

pub mod algo;
pub mod common;
