//! 记忆算法层：检索、遗忘、巩固。
//!
//! 这里是"记忆系统该做什么"的实现；IO 契约在 `soul-mem-runtime`，LLM 调用统一走
//! `soul-mem-llm`（需要 LLM 的算法入口一律接收 `&LlmEngine`，本层不碰传输细节）。
//!
//! # 三个子域
//!
//! | 子域 | 位置 | 内容 |
//! |---|---|---|
//! | 检索 | [`algo::retrieve`] | `DefaultPipeline`（ShortOnly → Similarity → AssociateWithAction）、纯相似度、贝叶斯动作推理、PPR 联想 |
//! | 遗忘 | [`algo::forget`] | 衰减计算（[`algo::forget::decay_calculator`]，**常数文档化的模板**）、随机遮罩、LLM 补全 |
//! | 巩固 | [`algo::consolidate`] | 把滑动窗口摘要拆解为记忆节点 |
//!
//! 图算法底座在 [`common`]：[`common::ppr`] 是 EdgePush PPR 变种，
//! [`common::ord_float`] 处理浮点在有序结构里的比较语义。
//!
//! # 改这一层之前要知道的
//!
//! - **`common::ppr` 的两个实现在参数域上不一致**：`naive_ppr` 接受 `d ∈ [0, 1]`，
//!   而生产路径实际调用的 `weighted_ppr_fp` 要求 `d < 1.0` 并在进入循环前 `assert!`。
//!   给 `damping_factor` 传 1.0 会 panic（不是返回 `Err`）。`naive_ppr` 只是参考实现，
//!   目前仅被测试调用。
//! - **`retrieve::prefetch_db` 会做数据库 IO**：它接收 `&dyn MemoryRepository` 并写入
//!   工作记忆，因此本 crate 依赖 `soul-mem-runtime`，"算法层不做 IO"这条线在检索路径上
//!   并不成立。`soul-mem-runtime` 反过来只在 `dev-dependencies` 依赖本 crate。
//!   它也返回 [`algo::retrieve::PrefetchOutcome`]（候选/邻居两组 id）：调用方观测召回
//!   应使用这份返回值，**不要为了观测重跑 `similarity_fetch`/`fetch_neighbors`**——
//!   重跑的副本会随实现演进与真实预取静默漂移。
//! - **打分/合并逻辑在多个文件里各有一份**（`default_pipeline::merge_note_scores`、
//!   `assoc_with_action::merge_situation_sources`、`association` 里的排序截断），
//!   且 NaN 策略不同（`total_cmp` vs `partial_cmp().unwrap_or(Equal)`）。
//!   改"分数怎么合并"必须同时看这三处。
//! - `AssociationConfig` 的调参常数目前**只有值没有文档**（单位/含义/取值范围），
//!   而本层质量几乎完全由这些数决定。新增或调整请照
//!   [`algo::forget::decay_calculator`] 的风格补上说明。

pub mod algo;
pub mod common;
