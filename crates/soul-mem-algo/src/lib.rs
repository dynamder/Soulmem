//! 记忆算法层：检索、遗忘、巩固。
//!
//! 子域：`retrieve`（DefaultPipeline + 相似度 + 贝叶斯动作 + PPR）、`forget`（衰减/遮罩/LLM 补全）、
//! `consolidate`（摘要 → 记忆节点）。图算法在 `common`。
//!
//! # 改这一层之前
//!
//! - `weighted_ppr_fp` 要求 `d < 1.0`，传 1.0 会 panic；`naive_ppr` 接受 `[0,1]`，仅参考。
//! - `retrieve::prefetch_db` 会做 DB IO 并写工作记忆；本 crate 因此依赖 `runtime`。
//! - 打分/合并分散在 `default_pipeline::merge_note_scores`、`assoc_with_action::merge_situation_sources`、
//!   `association` 的排序截断，NaN 策略不同；改合并必须同时看三处。
//! - 动作结果按类型分席（Speak/Think/Skill 各至多一个），同分按 `MemoryId` tiebreak；
//!   得分只由 `TrigToAction.prob × 源权重` 决定，Skill 席位恒空。

pub mod algo;
pub mod common;
