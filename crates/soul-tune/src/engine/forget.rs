//! 遗忘算法三阶段测试套件。
//!
//! soul-tune 本身是 SoulMem 的测试框架，这里不再复刻算法分支里的单元小测试，
//! 而是**直接驱动具体的遗忘算法管线**，并拆分为三个独立测试阶段
//! （对应 [`crate::base::ForgetMode`]）：
//!
//! 1. **Mask**（[`ForgetMaskSuite`]，`mask.rs`）：只验证遮罩模块 —— 纯算法、无 LLM、确定性。
//!    验证遮罩比例 ≈ 缺失度、确定性、`[masked]` 占位符计数、边界行为。
//! 2. **Revise**（[`ForgetReviseSuite`]，`revise.rs`）：只验证遮罩补全 —— 直接驱动 soul-tune
//!    的 `LlamaServer`（llama.cpp server），对**有上下文的长文本遮罩结果**做 LLM
//!    补全，贴出 LLM 原始回复并校验有效性（非空、不含占位符）。
//! 3. **Pipeline**（[`ForgetPipelineSuite`]，`pipeline/`）：全管线 —— fixture 角色图 + 模拟老化 →
//!    `compute_all_missing_degrees` → 逐节点 `lazy_forget`（衰减+遮罩+LLM 补全）→
//!    边衰减。区分**真实修订**与降级遮罩，LLM 可用但有效修订为 0 时用例失败。
//!
//! Pipeline 内嵌的**激发测试（excitation，黑盒效果）**：图克隆两份配对对照，
//! 按设计剂量梯度激发部分节点，验证"激发 → 遗忘被延缓"这一**可观察效果**
//! （断言 E1~E6 见 [`ForgetPipelineSuite::run_excitation_case`]）。soul-tune 是
//! 效果测试框架：不读取算法内部常量、不假设激发次数如何进入衰减公式，只通过
//! 公开接口驱动与观测。设计文档见 `docs/architecture/激发测试设计.md`。
//!
//! LLM 后端与 playtest 完全一致：统一来源解析（见 `engine::llm::resolver`）——
//! 先探测运行中的 llama-server，没有则自动拉起本地缓存模型，都没有则降级遮罩。
//!
//! # 文件布局
//!
//! 本模块按"共享 → 阶段 → 场景"分层组织：
//!
//! | 文件 | 内容 |
//! |---|---|
//! | `forget.rs`（本文件） | 共享常量、LLM 构造、通用判定/展示辅助，以及对外 `pub use` 汇总 |
//! | `mask.rs` | 阶段 1：遮罩（确定性，无 LLM） |
//! | `revise.rs` | 阶段 2：遮罩补全（需要 llama-server） |
//! | `pipeline.rs` | 阶段 3：场景规格、加载、老化、主流程、增量一致性、`TestSuite` 分发 |
//! | `pipeline/multistep.rs` | 多步遗忘场景（3 轮 × 24h 轨迹） |
//! | `pipeline/activation.rs` | 激活测试场景 |
//! | `pipeline/excitation.rs` | 激发测试场景（E1~E6，对照/实验双图配对） |
//! | `tests.rs` | 单元测试（原内嵌在同一文件底部） |
//!
//! 拆分边界全部是**条目边界**，方法体逐字未改。`ForgetPipelineSuite` 的固有 `impl`
//! 分散在 `pipeline/` 下多个文件是刻意且合法的（同一 crate 内可对同一类型写多个
//! 固有 `impl`）；跨文件调用的一律为 `pub(in crate::engine::forget) fn`。
//!
//! 目录组织遵循本仓库既有约定：**`xxx.rs` + `xxx/`，不使用 `mod.rs`**。
//!
//! 对外路径 `engine::forget::X` 保持不变——本文件用 `pub use` 重新导出全部公开项。

mod mask;
mod pipeline;
mod revise;

#[cfg(test)]
mod tests;

// 对外（soul-tune-api / main.rs）经全路径 `engine::forget::X` 使用这些项；
// 本模块内部未必直接引用，且 bin 目标里这些再导出不可达，故整体放行 unused_imports
// （与 `engine/llm.rs` 对 `resolver` 的处理同一理由）。
#[allow(unused_imports)]
pub use mask::{ForgetMaskSuite, MaskCaseData};
#[allow(unused_imports)]
pub use pipeline::{
    ForgetCaseData, ForgetCaseSpec, ForgetPipelineSuite, NodeForgetStat, NodeSeries, NodeStepStat,
    PIPELINE_CASES, PIPELINE_REVISE_MIN_WORDS, ideal_ebbinghaus_curve,
};
#[allow(unused_imports)]
pub use revise::{
    ForgetReviseSuite, REVISE_MASK_GRADIENTS, REVISE_MAX_SAMPLES, ReviseCaseData, ReviseMode,
    ReviseSample,
};

// 非公开项：仅供本模块内部（含 `tests`）通过 `use super::*` 引用，对外不可见。
// 单元测试是 `#[cfg(test)]`，非测试构建下这两个再导出没有使用者，故放行。
#[allow(unused_imports)]
pub(in crate::engine::forget) use mask::MASK_TEXTS;
#[allow(unused_imports)]
pub(in crate::engine::forget) use pipeline::ExcitationSchedule;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use jieba_rs::Jieba;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;

use soul_mem_algo::algo::forget::decay_calculator::{
    DEFAULT_MAX_ACTIVATION_CAP, compute_missing_degree, update_missing_degree_incremental,
};
use soul_mem_algo::algo::forget::decay_revise::{
    DEFAULT_ACTIVE_FACTOR, DEFAULT_BASE_HALF_LIFE_HOURS, ForgetAction, REVISE_THRESHOLD,
    compute_all_missing_degrees, current_missing_degree, decay_graph_edge, get_summary,
    lazy_forget, weight_placeholder,
};
use soul_mem_algo::algo::forget::llm_completion::build_reconstruct_prompt;
use soul_mem_algo::algo::forget::mask::{MASK_WORD, mask_text};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_runtime::cluster::memory_cluster::MemoryCluster;

// 同步外壳 trait：run_revise_case 直接调用 `chat()`，需要它在本作用域内
use crate::engine::llm::{LlamaServer, LlmBackend};
use crate::engine::loader::{build_reverse_id_map, load_graph_cluster};
use crate::engine::suite::{
    DetailRow, Series, SuiteReport, TestCaseOutcome, TestSuite, chart_metric, key_value_metric,
};

// ========================================================================
// 共享：LLM 闭包与通用工具
// ========================================================================

/// 记忆补全的最大生成 token 数（与 playtest 的生成调用量级一致）
const LLM_MAX_TOKENS: u32 = 1024;

/// 记忆重建 system prompt（记忆重建角色）。
/// 与算法层 `llm_completion::DEFAULT_RECONSTRUCT_SYSTEM_PROMPT` 保持一致：
/// 每个 [masked] 对应一个缺失 token，必须全部补全；全遮罩时输出固定遗忘句。
const FORGET_SYSTEM_PROMPT: &str = "You are a memory reconstruction assistant. \
    A segment of memory text has been partially masked with [masked] placeholders. \
    Each [masked] placeholder corresponds to exactly one missing word/token of the original text, \
    so the number of placeholders tells you how much information is missing. \
    Based on the remaining context, infer and fill in ALL placeholders naturally; \
    the output must contain NO [masked] placeholders — every one must be completed. \
    If the text is entirely masked with no remaining context to infer from, \
    output exactly: \"I totally forget it and cannot recall anything.\" \
    Output only the completed text, no explanation.";

/// 按统一来源解析创建 LLM 后端（见 [`crate::engine::llm::resolver`]）：
/// 复用运行中的 llama-server → 自动拉起本地缓存模型 → 降级为 None（遮罩路径）。
///
/// 返回 `None` 不再需要"伪造一个失败的闭包"：算法层的 `engine` 参数本身就是
/// `Option<&LlmEngine>`，"没有 LLM"是显式状态。
fn try_create_llm() -> Option<Arc<Mutex<LlamaServer>>> {
    let resolution = crate::engine::llm::resolve_llm();
    match resolution.server {
        Some(s) => Some(Arc::new(Mutex::new(s))),
        None => {
            let reason = resolution
                .status
                .reason
                .unwrap_or_else(|| "未知原因".to_string());
            eprintln!("llama-server 不可用（{reason}），降级为遮罩路径");
            None
        }
    }
}

/// 统计文本分词后的词数（与 mask 模块共用 jieba 分词）
fn mask_word_count(jieba: &Jieba, text: &str) -> usize {
    jieba.cut(text, true).len()
}

/// 统计遮罩文本中 `[masked]` 占位符的数量
fn count_masked(text: &str) -> usize {
    text.matches(MASK_WORD.trim()).count()
}

/// 是否可遮罩节点（SemMemory / SpecificSituation）
fn is_maskable(note: &MemoryNote) -> bool {
    matches!(
        note.mem_type(),
        MemoryType::Situation(SituationType::SpecificSituation(_)) | MemoryType::Semantic(_)
    )
}

fn forget_type_name(node: &MemoryNote) -> &'static str {
    match node.mem_type() {
        MemoryType::Semantic(_) => "SemMemory",
        MemoryType::Situation(SituationType::SpecificSituation(_)) => "SpecificSituation",
        MemoryType::Procedure(_) => "Procedure",
        _ => "Other",
    }
}

/// 语义 id 显示：优先 graph.json 可读 id（如 `sem_self`），缺失时回退 UUID。
/// 避免观测/明细里展示每次运行都不同的随机 MemoryId。
fn display_id(id_rev: &std::collections::HashMap<MemoryId, String>, id: MemoryId) -> String {
    id_rev.get(&id).cloned().unwrap_or_else(|| id.to_string())
}

/// 有效修订：LLM 回复非空且不含 `[masked]` 占位符（真正补全而非复述遮罩）
fn is_effective_revision(reply: &str) -> bool {
    let t = reply.trim();
    !t.is_empty() && !t.contains(MASK_WORD.trim())
}

// ========================================================================
// 阶段 1：遮罩验证（ForgetMaskSuite）—— 纯算法、无 LLM、确定性
// ========================================================================

fn is_maskable_type(type_name: &'static str) -> bool {
    matches!(type_name, "SemMemory" | "SpecificSituation")
}

/// 激活后的理论缺失度：半衰期随激活次数延长
/// `md = 1 - e^(-elapsed·ln2 / (base_hl × (1 + active_factor × min(count, cap))))`
fn activation_theory_md(retrieval_count: usize, elapsed_hours: f32) -> f32 {
    let capped = (retrieval_count as f32).min(DEFAULT_MAX_ACTIVATION_CAP as f32);
    let adjusted_hl = DEFAULT_BASE_HALF_LIFE_HOURS * (1.0 + DEFAULT_ACTIVE_FACTOR * capped);
    let tau = adjusted_hl / std::f32::consts::LN_2;
    1.0 - (-elapsed_hours / tau).exp()
}

/// 线性插值求缺失度到达阈值的时间（观测点含 t=0 处 md=0 的锚点，序列单调不减）。
/// 观测窗口内未达阈值返回 `None`（用于"实验组未达阈值"的报告与下限计算）。
fn crossing_time(mds: &[f32], hours: &[i64], threshold: f32) -> Option<f32> {
    let mut prev_h = 0i64;
    let mut prev_md = 0.0f32;
    for (h, &md) in hours.iter().zip(mds) {
        if md >= threshold {
            let span = (md - prev_md).max(1e-6);
            let frac = ((threshold - prev_md) / span).clamp(0.0, 1.0);
            return Some(prev_h as f32 + (*h as f32 - prev_h as f32) * frac);
        }
        prev_h = *h;
        prev_md = md;
    }
    None
}

fn compute_and_update(node: &mut MemoryNote, current_time: DateTime<Utc>) -> f32 {
    let md = update_missing_degree_incremental(
        node.missing_degree(),
        node.last_forget_time(),
        current_time,
        node.retrieval_count(),
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    );
    node.set_missing_degree(md);
    node.set_last_forget_time(current_time);
    md
}

// ========================================================================
// 测试（确定性，不依赖 LLM / 网络）
// ========================================================================
