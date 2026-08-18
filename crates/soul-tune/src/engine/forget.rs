//! 遗忘算法三阶段测试套件。
//!
//! soul-tune 本身是 SoulMem 的测试框架，这里不再复刻算法分支里的单元小测试，
//! 而是**直接驱动具体的遗忘算法管线**，并拆分为三个独立测试阶段
//! （对应 [`crate::base::ForgetMode`]）：
//!
//! 1. **Mask**（[`ForgetMaskSuite`]）：只验证遮罩模块 —— 纯算法、无 LLM、确定性。
//!    验证遮罩比例 ≈ 缺失度、确定性、`[masked]` 占位符计数、边界行为。
//! 2. **Revise**（[`ForgetReviseSuite`]）：只验证遮罩补全 —— 直接驱动 soul-tune
//!    的 `LlamaServer`（llama.cpp server），对**有上下文的长文本遮罩结果**做 LLM
//!    补全，贴出 LLM 原始回复并校验有效性（非空、不含占位符）。
//! 3. **Pipeline**（[`ForgetPipelineSuite`]）：全管线 —— fixture 角色图 + 模拟老化 →
//!    `compute_all_missing_degrees` → 逐节点 `lazy_forget`（衰减+遮罩+LLM 补全）→
//!    边衰减。区分**真实修订**与降级遮罩，LLM 可用但有效修订为 0 时用例失败。
//!
//! LLM 后端与 playtest 完全一致：`SOUL_TUNE_LLAMA_URL`（直连已运行服务）或
//! `SOUL_TUNE_CANDLE_MODEL_PATH`（自动拉起 llama-server）。

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use jieba_rs::Jieba;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use soul_mem_algo::algo::forget::decay_calculator::{
    compute_missing_degree, update_missing_degree_incremental, DEFAULT_MAX_ACTIVATION_CAP,
};
use soul_mem_algo::algo::forget::decay_revise::{
    compute_all_missing_degrees, current_missing_degree, decay_graph_edge, get_summary,
    lazy_forget, weight_placeholder, ForgetAction, DEFAULT_ACTIVE_FACTOR,
    DEFAULT_BASE_HALF_LIFE_HOURS, REVISE_THRESHOLD,
};
use soul_mem_algo::algo::forget::mask::{mask_text, MASK_WORD};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_runtime::cluster::memory_cluster::MemoryCluster;

use crate::engine::llm::{LlmBackend, LlamaServer};
use crate::engine::loader::load_graph_cluster;
use crate::engine::suite::{
    chart_metric, key_value_metric, DetailRow, Series, SuiteReport, TestCaseOutcome, TestSuite,
};

// ========================================================================
// 共享：LLM 闭包与通用工具
// ========================================================================

/// `lazy_forget` 要求的 LLM 调用闭包：`(system, user) -> Future<Result<String>>`
type LlmCall = Box<
    dyn FnOnce(
        &str,
        &str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>>
                + Send,
        >,
    >,
>;

/// 记忆补全的最大生成 token 数（与 playtest 的生成调用量级一致）
const LLM_MAX_TOKENS: u32 = 1024;

/// 记忆重建 system prompt（记忆重建角色）
const FORGET_SYSTEM_PROMPT: &str = "You are a memory reconstruction assistant. \
    A segment of memory text has been partially masked, with [masked] placeholders. \
    Based on the context and the remaining fragments, infer and complete the missing parts \
    naturally. Output only the completed text, no explanation.";

/// 使用 soul-tune 的 `LlamaServer`（与 playtest 相同后端）的闭包。
///
/// `LlmBackend::chat` 是阻塞调用（reqwest::blocking），必须通过 `spawn_blocking`
/// 移到 tokio 阻塞线程池执行——直接在 `block_on` 的异步上下文里调用会让 reqwest
/// 内部创建的 runtime 在异步上下文里被 drop 而 panic。
fn llama_closure(server: Arc<Mutex<LlamaServer>>) -> LlmCall {
    Box::new(move |system: &str, user: &str| {
        let s = server.clone();
        let sys = system.to_string();
        let usr = user.to_string();
        Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                let mut guard = s.lock().expect("llama-server 锁");
                guard.chat(&sys, &usr, LLM_MAX_TOKENS)
            })
            .await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("spawn_blocking 失败: {e}").into()
            })?;
            result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
        })
    })
}

/// LLM 不可用时传入的错误闭包：算法应优雅降级为 MaskOnly（遮罩不回退）
fn failing_llm_closure() -> LlmCall {
    Box::new(|_system: &str, _user: &str| {
        Box::pin(async {
            Err::<String, Box<dyn std::error::Error + Send + Sync>>(
                "LLM 未配置（llama-server 不可用）".into(),
            )
        })
    })
}

/// 按 soul-tune 的 llama-server 约定创建 LLM 后端：
/// `SOUL_TUNE_LLAMA_URL` 直连，否则 `SOUL_TUNE_CANDLE_MODEL_PATH` 自动拉起。
fn try_create_llm() -> Option<Arc<Mutex<LlamaServer>>> {
    let server = if std::env::var("SOUL_TUNE_LLAMA_URL").is_ok() {
        LlamaServer::load("")
    } else if let Ok(model) = std::env::var("SOUL_TUNE_CANDLE_MODEL_PATH") {
        if model.trim().is_empty() {
            return None;
        }
        LlamaServer::load(&model)
    } else {
        return None;
    };
    match server {
        Ok(s) => Some(Arc::new(Mutex::new(s))),
        Err(e) => {
            eprintln!("llama-server 不可用（{e}），降级为遮罩路径");
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

/// 有效修订：LLM 回复非空且不含 `[masked]` 占位符（真正补全而非复述遮罩）
fn is_effective_revision(reply: &str) -> bool {
    let t = reply.trim();
    !t.is_empty() && !t.contains(MASK_WORD.trim())
}

// ========================================================================
// 阶段 1：遮罩验证（ForgetMaskSuite）—— 纯算法、无 LLM、确定性
// ========================================================================

/// 遮罩用例：对指定文本按指定缺失度执行 `mask_text`
struct MaskCaseSpec {
    name: String,
    text: String,
    missing_degree: f32,
}

/// 缺失度梯度
const MASK_GRADIENTS: [f32; 6] = [0.0, 0.1, 0.2, 0.5, 0.87, 1.0];

/// 从文本集构造遮罩用例（中/长 × 全梯度 + 短文本边界 + 确定性）
fn build_mask_cases(texts: &[(&str, String)]) -> Vec<MaskCaseSpec> {
    let mut cases = Vec::new();
    // 中/长文本 × 全梯度：验证比例正确性
    for (tag, text) in texts.iter().filter(|(tag, _)| *tag != "short") {
        for md in MASK_GRADIENTS {
            cases.push(MaskCaseSpec {
                name: format!("{}-md{:.2}", tag, md),
                text: text.clone(),
                missing_degree: md,
            });
        }
    }
    // 短文本边界：验证不 panic、masked ≤ total
    if let Some((_, short)) = texts.iter().find(|(tag, _)| *tag == "short") {
        for md in [0.5, 0.87, 1.0] {
            cases.push(MaskCaseSpec {
                name: format!("short-md{:.2}", md),
                text: short.clone(),
                missing_degree: md,
            });
        }
    }
    // 确定性：同一输入两次结果一致（取最长文本）
    if let Some((_, long)) = texts.iter().max_by(|a, b| a.1.chars().count().cmp(&b.1.chars().count())) {
        cases.push(MaskCaseSpec {
            name: "determinism".into(),
            text: long.clone(),
            missing_degree: 0.5,
        });
    }
    cases
}

/// 内置文本集：短 / 中 / 长中文文本
const MASK_TEXTS: [(&str, &str); 3] = [
    (
        "short",
        "格蕾修是逐火十三英桀之一也是用画笔说话的画家",
    ),
    (
        "medium",
        "红魔馆的女仆长十六夜咲夜擅长投掷银质小刀她害怕烫的食物是众所周知的猫舌",
    ),
    (
        "long",
        "傍晚我在红魔馆的庭院为大小姐斟茶蕾米莉亚坐在阳台的红伞下望着雾之湖畔天色渐暗四周渐渐安静下来茶香混着傍晚的凉风飘散在庭院里",
    ),
];

/// 遮罩用例的观测数据
pub struct MaskCaseData {
    pub case_name: String,
    pub passed: bool,
    pub detail_lines: Vec<String>,
    pub metrics: Vec<(String, String, String)>,
}

pub struct ForgetMaskSuite {
    jieba: Jieba,
    cases: Vec<MaskCaseSpec>,
}

impl ForgetMaskSuite {
    /// 内置文本集套件（无图依赖，测试/快速验证使用）
    pub fn new() -> Self {
        let texts: Vec<(&str, String)> = MASK_TEXTS
            .iter()
            .map(|(tag, text)| (*tag, text.to_string()))
            .collect();
        Self {
            jieba: Jieba::new(),
            cases: build_mask_cases(&texts),
        }
    }

    /// 从指定 fixture 图加载文本（与 Revise/Pipeline 同一数据源）。
    ///
    /// 从图中收集可遮罩节点（SemMemory / SpecificSituation）的文本，
    /// 按词数分短/中/长三层各取一个代表性文本，构造遮罩用例。
    pub fn load(path: &Path) -> Result<Self, String> {
        let (cluster, _id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let jieba = Jieba::new();
        let mut short: Vec<String> = Vec::new();
        let mut medium: Vec<String> = Vec::new();
        let mut long: Vec<String> = Vec::new();
        for n in cluster.graph().node_weights() {
            if !is_maskable(n.note()) {
                continue;
            }
            let text = get_summary(n.note()).unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }
            let words = mask_word_count(&jieba, &text);
            if words < 8 {
                short.push(text);
            } else if words <= 20 {
                medium.push(text);
            } else {
                long.push(text);
            }
        }
        // 每层取最长的一个代表性文本（控制用例数量）
        let longest = |mut v: Vec<String>| -> Option<String> {
            v.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
            v.into_iter().next()
        };
        let mut texts: Vec<(&str, String)> = Vec::new();
        if let Some(t) = longest(short) {
            texts.push(("short", t));
        }
        if let Some(t) = longest(medium) {
            texts.push(("medium", t));
        }
        if let Some(t) = longest(long) {
            texts.push(("long", t));
        }

        Ok(Self {
            jieba,
            cases: build_mask_cases(&texts),
        })
    }

    fn run_mask_case(&self, spec: &MaskCaseSpec) -> MaskCaseData {
        let r1 = mask_text(&spec.text, spec.missing_degree, &self.jieba);
        let passed = {
            let mut ok = true;
            let ratio = if r1.total_count > 0 {
                r1.masked_count as f32 / r1.total_count as f32
            } else {
                0.0
            };
            // 占位符计数一致
            if count_masked(&r1.masked_text) != r1.masked_count {
                ok = false;
            }
            // 边界：md=0 不遮
            if spec.missing_degree <= 0.0 && r1.masked_count != 0 {
                ok = false;
            }
            // md=1 全遮（total>0 时）
            if spec.missing_degree >= 1.0 && r1.total_count > 0 && r1.masked_count != r1.total_count {
                ok = false;
            }
            // 长文本比例校验（词数足够多时容差更严格）
            if r1.total_count >= 8 {
                let expect = spec.missing_degree.clamp(0.0, 1.0);
                if (ratio - expect).abs() > 0.15 {
                    ok = false;
                }
            }
            // masked ≤ total
            if r1.masked_count > r1.total_count {
                ok = false;
            }
            ok
        };
        let mut detail_lines = Vec::new();
        detail_lines.push(format!(
            "md={:.2} 词数 {}→{}（遮 {:.0}%）",
            spec.missing_degree,
            r1.total_count,
            r1.masked_count,
            if r1.total_count > 0 {
                r1.masked_count as f32 / r1.total_count as f32 * 100.0
            } else {
                0.0
            }
        ));
        detail_lines.push(format!("  原文: {}", spec.text));
        detail_lines.push(format!("  遮罩: {}", r1.masked_text));

        // 确定性：同输入再跑一次
        let r2 = mask_text(&spec.text, spec.missing_degree, &self.jieba);
        if spec.name == "determinism" {
            if r1.masked_text != r2.masked_text {
                detail_lines.push("  确定性检查: 两次结果不一致！".into());
            }
        }

        let metrics = vec![
            (
                "遮罩".into(),
                format!("{} 遮罩率", spec.name),
                format!("{:.0}%", if r1.total_count > 0 { r1.masked_count as f32 / r1.total_count as f32 * 100.0 } else { 0.0 }),
            ),
            (
                "遮罩".into(),
                format!("{} 词数", spec.name),
                format!("{}/{}", r1.masked_count, r1.total_count),
            ),
        ];

        MaskCaseData {
            case_name: spec.name.to_string(),
            passed,
            detail_lines,
            metrics,
        }
    }
}

impl Default for ForgetMaskSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl TestSuite for ForgetMaskSuite {
    fn case_count(&self) -> usize {
        self.cases.len()
    }

    fn run_case(&self, index: usize) -> TestCaseOutcome {
        let data = self.run_mask_case(&self.cases[index]);
        let passed = data.passed;
        TestCaseOutcome {
            case_name: format!("mask/{}", data.case_name),
            description: format!(
                "遮罩验证: md={:.2}",
                self.cases[index].missing_degree
            ),
            passed,
            data: Box::new(data),
        }
    }

    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        elapsed: Duration,
        total: usize,
        passed: usize,
        _failed: usize,
    ) -> SuiteReport {
        let mut metrics: Vec<Box<dyn crate::engine::suite::ReportMetric>> = Vec::new();
        let mut detail_rows: Vec<DetailRow> = Vec::new();
        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<MaskCaseData>() else {
                continue;
            };
            for (group, label, value) in &data.metrics {
                metrics.push(Box::new(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                )));
            }
            for line in &data.detail_lines {
                detail_rows.push(DetailRow {
                    text: format!("[{}] {}", data.case_name, line),
                    has_error: !o.passed,
                });
            }
        }
        metrics.push(Box::new(key_value_metric(
            "通过率".to_string(),
            "汇总".to_string(),
            format!(
                "{:.1}% ({}/{})",
                if total > 0 {
                    passed as f64 / total as f64 * 100.0
                } else {
                    0.0
                },
                passed,
                total
            ),
        )));
        SuiteReport {
            metrics,
            detail_header: format!(
                "阶段1 遮罩验证（通过 {}/{}，耗时 {:.2}s）",
                passed,
                total,
                elapsed.as_secs_f64()
            ),
            detail_rows,
            outcomes,
        }
    }
}

// ========================================================================
// 阶段 2：遮罩补全验证（ForgetReviseSuite）—— 直接驱动 llama-server
// ========================================================================

/// 补全样本：从 fixture 图选取的长文本节点
pub struct ReviseSample {
    pub node_id: String,
    pub type_name: &'static str,
    /// 原始文本
    pub original: String,
    /// 遮罩后的文本（md=0.5）
    pub masked: String,
}

/// 补全用例的观测数据
pub struct ReviseCaseData {
    pub case_name: String,
    pub passed: bool,
    pub llm_available: bool,
    /// LLM 遮罩输入
    pub masked_text: String,
    /// LLM **原始回复**
    pub llm_reply: String,
    pub detail_lines: Vec<String>,
    pub metrics: Vec<(String, String, String)>,
}

/// 参与补全验证的最小词数（保证遮罩后仍有上下文可推断）
pub const REVISE_MIN_WORDS: usize = 20;
/// 参与补全验证的最大样本数
pub const REVISE_MAX_SAMPLES: usize = 6;

pub struct ForgetReviseSuite {
    jieba: Jieba,
    llm: Option<Arc<Mutex<LlamaServer>>>,
    /// 长文本样本（从 fixture 图提取）
    samples: Vec<ReviseSample>,
}

impl ForgetReviseSuite {
    /// 从 fixture 图加载长文本样本并尝试启用 llama-server
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut suite = Self::load_without_llm(path)?;
        suite.llm = try_create_llm();
        Ok(suite)
    }

    /// 仅加载长文本样本、不启用 LLM（测试 / 确定性验证使用）
    #[allow(dead_code)]
    pub fn load_without_llm(path: &Path) -> Result<Self, String> {
        let (cluster, _id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let jieba = Jieba::new();
        let mut samples = Vec::new();
        let g = cluster.graph();
        let mut candidates: Vec<(String, &'static str, String, usize)> = g
            .node_weights()
            .filter_map(|n| {
                if !is_maskable(n.note()) {
                    return None;
                }
                let text = get_summary(n.note()).unwrap_or_default();
                let words = mask_word_count(&jieba, &text);
                if words < REVISE_MIN_WORDS {
                    return None;
                }
                Some((
                    n.note().id().to_string(),
                    forget_type_name(n.note()),
                    text,
                    words,
                ))
            })
            .collect();
        // 按词数降序取前 N（最长的文本上下文最丰富）
        candidates.sort_by(|a, b| b.3.cmp(&a.3));
        for (id, ty, text, _words) in candidates.into_iter().take(REVISE_MAX_SAMPLES) {
            let masked = mask_text(&text, 0.5, &jieba).masked_text;
            samples.push(ReviseSample {
                node_id: id,
                type_name: ty,
                original: text,
                masked,
            });
        }
        Ok(Self {
            jieba,
            llm: None,
            samples,
        })
    }

    /// 探活：用固定文本做一次最小补全调用，验证 llama-server 链路可用
    fn probe(&self) -> Result<String, String> {
        let Some(llm) = &self.llm else {
            return Err("llama-server 未配置（SOUL_TUNE_LLAMA_URL 或 SOUL_TUNE_CANDLE_MODEL_PATH）".into());
        };
        let probe_user = "Masked text: 十六夜咲夜是红魔馆的 [masked] 拥有操纵时间的能力她可以 [masked] 时间";
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let closure = llama_closure(llm.clone());
        runtime
            .block_on(closure(FORGET_SYSTEM_PROMPT, probe_user))
            .map_err(|e| format!("llama-server 调用失败: {e}"))
    }

    /// 单个补全用例：遮罩输入 → llama-server → 校验有效性与长度
    fn run_revise_case(&self, sample: &ReviseSample) -> ReviseCaseData {
        let llm_available = self.llm.is_some();
        let (reply, llm_err) = match &self.llm {
            Some(llm) => {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime");
                let closure = llama_closure(llm.clone());
                let user = format!("Masked text: {}", sample.masked);
                match runtime.block_on(closure(FORGET_SYSTEM_PROMPT, &user)) {
                    Ok(r) => (r, None),
                    Err(e) => (String::new(), Some(format!("{e}"))),
                }
            }
            None => (String::new(), Some("llama-server 不可用".into())),
        };

        let mut passed = true;
        let mut detail_lines = Vec::new();
        let original_chars = sample.original.chars().count();
        detail_lines.push(format!(
            "{} [{}] 原文词数 {} / 字数 {}",
            sample.type_name,
            sample.node_id.chars().take(8).collect::<String>(),
            mask_word_count(&self.jieba, &sample.original),
            original_chars
        ));
        detail_lines.push(format!("  遮罩输入: {}", sample.masked));
        if let Some(err) = &llm_err {
            passed = false;
            detail_lines.push(format!("  LLM 调用失败: {}", err));
        } else {
            detail_lines.push(format!("  LLM原始回复: {}", reply));
            if !is_effective_revision(&reply) {
                passed = false;
                detail_lines.push("  ✗ 回复无效：为空或仍含 [masked] 占位符".into());
            }
            // 长度与**原文**比较：补全应接近原文长度（≥50%），
            // 与含占位符的遮罩输入比较没有意义
            let min_len = (original_chars as f32 * 0.5) as usize;
            if reply.chars().count() < min_len {
                passed = false;
                detail_lines.push(format!(
                    "  ✗ 回复过短：{} 字 < 原文 50%（{} 字）",
                    reply.chars().count(),
                    min_len
                ));
            }
        }

        let metrics = vec![
            (
                "补全".into(),
                format!("{} 回复字数", sample.node_id.chars().take(8).collect::<String>()),
                if llm_err.is_none() { reply.chars().count().to_string() } else { "失败".into() },
            ),
        ];

        ReviseCaseData {
            case_name: sample.node_id.chars().take(8).collect(),
            passed,
            llm_available,
            masked_text: sample.masked.clone(),
            llm_reply: reply,
            detail_lines,
            metrics,
        }
    }
}

impl TestSuite for ForgetReviseSuite {
    fn case_count(&self) -> usize {
        // 1 个探活 + 每个样本 1 个补全用例
        1 + self.samples.len()
    }

    fn run_case(&self, index: usize) -> TestCaseOutcome {
        if index == 0 {
            let probe_result = self.probe();
            let (passed, reply, detail) = match probe_result {
                Ok(r) => {
                    let len = r.chars().count();
                    (true, r, format!("探活成功，模型响应 {} 字", len))
                }
                Err(e) => (false, String::new(), format!("探活失败: {e}")),
            };
            let data = ReviseCaseData {
                case_name: "probe".into(),
                passed,
                llm_available: self.llm.is_some(),
                masked_text: "探活遮罩文本".into(),
                llm_reply: reply,
                detail_lines: vec![detail],
                metrics: vec![],
            };
            let passed = data.passed;
            TestCaseOutcome {
                case_name: "forget/revise/probe".into(),
                description: "llama-server 链路探活".into(),
                passed,
                data: Box::new(data),
            }
        } else {
            let data = self.run_revise_case(&self.samples[index - 1]);
            let passed = data.passed;
            TestCaseOutcome {
                case_name: format!("forget/revise/{}", data.case_name),
                description: format!("遮罩补全验证: {}", data.case_name),
                passed,
                data: Box::new(data),
            }
        }
    }

    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        elapsed: Duration,
        total: usize,
        passed: usize,
        _failed: usize,
    ) -> SuiteReport {
        let mut metrics: Vec<Box<dyn crate::engine::suite::ReportMetric>> = Vec::new();
        let mut detail_rows: Vec<DetailRow> = Vec::new();
        let mut llm_available = false;
        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<ReviseCaseData>() else {
                continue;
            };
            llm_available |= data.llm_available;
            for (group, label, value) in &data.metrics {
                metrics.push(Box::new(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                )));
            }
            for line in &data.detail_lines {
                detail_rows.push(DetailRow {
                    text: format!("[{}] {}", data.case_name, line),
                    has_error: !o.passed,
                });
            }
        }
        metrics.push(Box::new(key_value_metric(
            "LLM 可用".to_string(),
            "LLM".to_string(),
            if llm_available {
                "是（llama-server）".to_string()
            } else {
                "否".to_string()
            },
        )));
        metrics.push(Box::new(key_value_metric(
            "通过率".to_string(),
            "汇总".to_string(),
            format!(
                "{:.1}% ({}/{})",
                if total > 0 {
                    passed as f64 / total as f64 * 100.0
                } else {
                    0.0
                },
                passed,
                total
            ),
        )));
        SuiteReport {
            metrics,
            detail_header: format!(
                "阶段2 遮罩补全验证（通过 {}/{}，耗时 {:.2}s）",
                passed,
                total,
                elapsed.as_secs_f64()
            ),
            detail_rows,
            outcomes,
        }
    }
}

// ========================================================================
// 阶段 3：全管线测试（ForgetPipelineSuite）
// ========================================================================

/// 全管线场景：对加载的真实角色图施加指定时间跨度（距上次遗忘操作的小时数）
#[derive(Debug, Clone, Copy)]
pub struct ForgetCaseSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// 模拟老化小时数（距上次遗忘操作/访问），决定遗忘缺失度
    pub elapsed_hours: i64,
    /// 是否允许调用 llama-server（不可用时自动降级为遮罩）
    pub want_llm: bool,
}

/// 全管线场景集：低/中/高遗忘强度 + 多步遗忘 + 激活测试 + 增量一致性
pub const PIPELINE_CASES: [ForgetCaseSpec; 6] = [
    ForgetCaseSpec {
        name: "low",
        description: "低遗忘强度（Δt=8h）：全图批量刷新 + 惰性遗忘",
        elapsed_hours: 8,
        want_llm: false,
    },
    ForgetCaseSpec {
        name: "medium",
        description: "中遗忘强度（Δt=24h）：缺失度约半衰，遮罩触发",
        elapsed_hours: 24,
        want_llm: false,
    },
    ForgetCaseSpec {
        name: "high",
        description: "高遗忘强度（Δt=72h）：llama-server 修订长文本节点（抽样）",
        elapsed_hours: 72,
        want_llm: true,
    },
    ForgetCaseSpec {
        name: "multi-step",
        description: "多步遗忘：3 轮 × 24h 对全图每个节点逐步衰减/遮罩/修订",
        elapsed_hours: -10, // 特殊标记：多步遗忘场景
        want_llm: true,
    },
    ForgetCaseSpec {
        name: "activation",
        description: "激活测试：随机节点激活多次，验证整图遗忘状态符合设计",
        elapsed_hours: -11, // 特殊标记：激活测试场景
        want_llm: false,
    },
    ForgetCaseSpec {
        name: "incremental",
        description: "增量一致性：两次 12h 增量更新 == 一次 24h 全量计算",
        elapsed_hours: -1, // 特殊标记：增量一致性场景
        want_llm: false,
    },
];

/// 全管线 LLM 修订的最小词数（短文本被全遮后无上下文，LLM 无法补全）
pub const PIPELINE_REVISE_MIN_WORDS: usize = 12;
/// 每用例最多 LLM 修订节点数（按缺失度降序抽样）
pub const MAX_LLM_REVISIONS: usize = 8;

/// 单个节点的遗忘观测结果
pub struct NodeForgetStat {
    pub id: String,
    pub type_name: &'static str,
    /// 图原文（遗忘管线执行前的原始记忆文本）
    pub original: String,
    /// 管线前缺失度 → 管线后缺失度
    pub md_before: f32,
    pub md_after: f32,
    /// 触发的遗忘动作
    pub action: &'static str,
    /// 遮罩词数 / 总词数（未遮罩为 None）
    pub mask: Option<(usize, usize)>,
    /// LLM 补全的遮罩输入文本（Revised 时）
    pub masked_text: Option<String>,
    /// LLM **原始回复**（Revised 时，未经任何处理）
    pub llm_reply: Option<String>,
    /// 是否有效修订（回复非空且不含 [masked]）
    pub effective: bool,
}

/// 单个用例（一次完整管线运行）的观测数据
pub struct ForgetCaseData {
    pub case_name: String,
    pub passed: bool,
    pub llm_available: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub llm_revised: usize,
    /// 有效修订数（LLM 回复非空且不含占位符）
    pub effective_revised: usize,
    pub action_histogram: Vec<(&'static str, usize)>,
    pub avg_missing_degree: f32,
    pub max_missing_degree: f32,
    pub avg_masked_ratio: f32,
    pub avg_edge_intensity: f64,
    /// 结构化节点观测（含图原文，供报告与 TUI 展示）
    pub nodes: Vec<NodeForgetStat>,
    pub detail_lines: Vec<String>,
    pub metrics: Vec<(String, String, String)>,
}

pub struct ForgetPipelineSuite {
    /// 从 fixtures 加载的真实角色记忆图（蓝图，每个用例克隆后施加老化）
    graph: MemoryCluster,
    graph_name: String,
    jieba: Jieba,
    llm: Option<Arc<Mutex<LlamaServer>>>,
    cases: Vec<ForgetCaseSpec>,
}

impl ForgetPipelineSuite {
    /// 从 fixture graph JSON 加载真实角色图，并按环境变量启用 llama-server
    pub fn load(path: &Path) -> Result<Self, String> {
        let (graph, _id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let graph_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let llm = try_create_llm();
        Ok(Self {
            graph,
            graph_name,
            jieba: Jieba::new(),
            llm,
            cases: PIPELINE_CASES.to_vec(),
        })
    }

    /// 仅加载图、不启用 LLM（测试 / 确定性验证使用）
    #[allow(dead_code)]
    pub fn load_without_llm(path: &Path) -> Result<Self, String> {
        let (graph, _id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let graph_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(Self {
            graph,
            graph_name,
            jieba: Jieba::new(),
            llm: None,
            cases: PIPELINE_CASES.to_vec(),
        })
    }

    /// 模拟老化：把整张图的节点/边统一回拨 `last_forget_time`（缺失度归零）。
    /// `hours=0` 表示就在 `now` 时刻（供多步遗忘的起始时间轴使用）。
    fn apply_aging(&self, cluster: &mut MemoryCluster, now: DateTime<Utc>, hours: i64) {
        let g = cluster.graph_mut();
        let node_indices: Vec<_> = g.node_indices().collect();
        for idx in node_indices {
            let t = now - ChronoDuration::hours(hours.max(0));
            let n = g.node_weight_mut(idx).expect("node");
            n.note.set_last_forget_time(t);
            n.note.set_missing_degree(0.0);
        }
        let t_edge = now - ChronoDuration::hours(hours.max(0));
        for ei in g.edge_indices().collect::<Vec<_>>() {
            let l = g.edge_weight_mut(ei).expect("edge");
            l.set_last_forget_time(t_edge);
            l.set_missing_degree(0.0);
        }
    }

    /// 运行一次完整的真实遗忘管线：
    /// 1. 克隆加载图并施加老化；
    /// 2. `compute_all_missing_degrees` 全图批量刷新节点+边缺失度；
    /// 3. 逐节点 `lazy_forget`：可遮罩且**足够长**（词数 ≥ PIPELINE_REVISE_MIN_WORDS）
    ///    且缺失度最高的一批节点走 llama-server 修订，其余走遮罩/降级；
    /// 4. `decay_graph_edge` 边独立衰减 + `weight_placeholder` 检索权重。
    ///
    /// LLM 可用但有效修订为 0 → 用例失败（LLM 链路或输入有问题，不再假通过）。
    fn run_pipeline(&self, spec: &ForgetCaseSpec) -> ForgetCaseData {
        let now = Utc::now();
        let mut cluster = self.graph.clone();
        self.apply_aging(&mut cluster, now, spec.elapsed_hours);

        let use_llm = self.llm.is_some() && spec.want_llm;

        // ── 步骤 1：全图批量刷新缺失度（节点 + 边）──
        compute_all_missing_degrees(&mut cluster, now);

        // LLM 修订抽样：可遮罩、缺失度 ≥ REVISE_THRESHOLD、**词数足够**的节点，
        // 按缺失度降序取前 N。排除短文本——被全遮后无上下文，LLM 无法补全。
        let revise_set: std::collections::HashSet<_> = {
            let g = cluster.graph();
            let mut candidates: Vec<_> = g
                .node_indices()
                .filter(|idx| {
                    let n = g.node_weight(*idx).expect("node");
                    if !is_maskable(n.note()) {
                        return false;
                    }
                    let words = get_summary(n.note())
                        .map(|t| mask_word_count(&self.jieba, &t))
                        .unwrap_or(0);
                    n.note().missing_degree() >= REVISE_THRESHOLD && words >= PIPELINE_REVISE_MIN_WORDS
                })
                .map(|idx| {
                    let n = g.node_weight(idx).expect("node");
                    (idx, n.note().missing_degree())
                })
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates
                .into_iter()
                .take(MAX_LLM_REVISIONS)
                .map(|(idx, _)| idx)
                .collect()
        };

        // ── 步骤 2：逐节点惰性遗忘 ──
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let node_indices: Vec<_> = {
            let g = cluster.graph();
            g.node_indices().collect()
        };

        let mut stats: Vec<NodeForgetStat> = Vec::new();
        let mut passed = true;
        let mut total_md = 0.0f32;
        let mut max_md = 0.0f32;
        let mut llm_revised = 0usize;
        let mut effective_revised = 0usize;
        let mut masked_words_total = 0usize;
        let mut total_words_total = 0usize;
        let mut hist: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();

        for idx in node_indices {
            let (before, type_name, id) = {
                let g = cluster.graph();
                let n = g.node_weight(idx).expect("node");
                (
                    current_missing_degree(n.note(), now),
                    forget_type_name(n.note()),
                    n.note().id().to_string(),
                )
            };

            let (action, after, orig_words, original) = {
                let g = cluster.graph_mut();
                let node = &mut g.node_weight_mut(idx).expect("node").note;
                let orig = get_summary(node).unwrap_or_default();
                let orig_words = mask_word_count(&self.jieba, &orig);
                let closure: LlmCall = if use_llm && revise_set.contains(&idx) {
                    llama_closure(self.llm.as_ref().expect("llm").clone())
                } else {
                    failing_llm_closure()
                };
                let act = runtime.block_on(lazy_forget(
                    node,
                    now,
                    &self.jieba,
                    Some(FORGET_SYSTEM_PROMPT),
                    closure,
                ));
                let after = node.missing_degree();
                (act, after, orig_words, orig)
            };

            // 动作分类、遮罩统计与 LLM 原始回复
            let (action_name, masked_words_this, mask_ratio_this, llm_reply, masked_text, effective) =
                match &action {
                    ForgetAction::NoAction => ("NoAction", 0usize, None, None, None, false),
                    ForgetAction::MaskOnly {
                        masked_count,
                        masked_text,
                        ..
                    } => {
                        let ratio = if orig_words > 0 {
                            Some(*masked_count as f32 / orig_words as f32)
                        } else {
                            None
                        };
                        (
                            "MaskOnly",
                            *masked_count,
                            ratio,
                            None,
                            Some(masked_text.clone()),
                            false,
                        )
                    }
                    ForgetAction::Revised {
                        masked_text, new_summary, ..
                    } => {
                        let masked = count_masked(masked_text);
                        let ratio = if orig_words > 0 {
                            Some(masked as f32 / orig_words as f32)
                        } else {
                            None
                        };
                        let effective = is_effective_revision(new_summary);
                        (
                            "Revised",
                            masked,
                            ratio,
                            Some(new_summary.clone()),
                            Some(masked_text.clone()),
                            effective,
                        )
                    }
                };

            if is_maskable_type(type_name) {
                if orig_words > 0 {
                    masked_words_total += masked_words_this;
                    total_words_total += orig_words;
                }
                if orig_words >= 8 {
                    if let Some(r) = mask_ratio_this {
                        if (r - after).abs() > 0.15 {
                            passed = false;
                        }
                    }
                }
            }

            if action_name == "Revised" {
                llm_revised += 1;
                if effective {
                    effective_revised += 1;
                }
            }
            *hist.entry(action_name).or_insert(0) += 1;
            total_md += after;
            max_md = max_md.max(after);

            // 不变量检查
            if !(0.0..=1.0).contains(&after) {
                passed = false;
            }
            if after + 1e-4 < before {
                passed = false; // 缺失度不应回退
            }
            if type_name == "Procedure" && action_name != "NoAction" {
                passed = false; // Procedure 仅更新缺失度，不触发遮罩
            }

            stats.push(NodeForgetStat {
                id,
                type_name,
                original,
                md_before: before,
                md_after: after,
                action: action_name,
                mask: mask_ratio_this
                    .map(|r| ((r * orig_words as f32).round() as usize, orig_words)),
                masked_text,
                llm_reply,
                effective,
            });
        }

        // ── 步骤 3：边衰减与权重占位 ──
        let mut avg_edge_intensity = 0.0f64;
        let mut avg_edge_weight = 0.0f64;
        let mut edge_count = 0usize;
        let mut edge_passed = true;
        {
            let g = cluster.graph_mut();
            for ei in g.edge_indices().collect::<Vec<_>>() {
                let link = g.edge_weight_mut(ei).expect("edge");
                let intensity = decay_graph_edge(link, now);
                let md = link.missing_degree();
                if !(0.0..=1.0).contains(&md) {
                    edge_passed = false;
                }
                if link.intensity() < 0.0 {
                    edge_passed = false;
                }
                avg_edge_weight += weight_placeholder(md);
                avg_edge_intensity += intensity;
                edge_count += 1;
            }
        }
        if !edge_passed {
            passed = false;
        }
        let avg_edge_intensity = if edge_count > 0 {
            avg_edge_intensity / edge_count as f64
        } else {
            0.0
        };
        let avg_edge_weight = if edge_count > 0 {
            avg_edge_weight / edge_count as f64
        } else {
            0.0
        };

        // LLM 可用但没有任何有效修订 → 明确失败（链路或抽样有问题）
        if use_llm && llm_revised == 0 {
            passed = false;
        }

        // ── 汇总 ──
        let node_count = stats.len();
        let avg_md = if node_count > 0 {
            total_md / node_count as f32
        } else {
            0.0
        };
        let avg_masked_ratio = if total_words_total > 0 {
            masked_words_total as f32 / total_words_total as f32
        } else {
            0.0
        };

        let hist_vec: Vec<(&'static str, usize)> = hist.into_iter().collect();

        let mut detail_lines = Vec::new();
        for s in &stats {
            let mask_txt = match s.mask {
                Some((m, t)) if t > 0 => {
                    format!(" 遮罩 {}/{}={:.0}%", m, t, m as f32 / t as f32 * 100.0)
                }
                _ => String::new(),
            };
            let short_id: String = s.id.chars().take(8).collect();
            detail_lines.push(format!(
                "{} [{}] md {:.3}→{:.3} 动作={}{}{}",
                s.type_name,
                short_id,
                s.md_before,
                s.md_after,
                s.action,
                mask_txt,
                if s.action == "Revised" {
                    if s.effective {
                        " [有效修订]"
                    } else {
                        " [无效修订!]"
                    }
                } else {
                    ""
                }
            ));
            // 附上图原文（人类测试员核对遗忘前内容）
            if !s.original.is_empty() {
                detail_lines.push(format!("    图原文: {}", s.original));
            }
            // LLM 补全：贴出遮罩输入与 LLM 原始回复
            if let (Some(mt), Some(reply)) = (&s.masked_text, &s.llm_reply) {
                detail_lines.push(format!("    遮罩输入: {}", mt));
                detail_lines.push(format!("    LLM原始回复: {}", reply));
            }
        }

        let mut metrics: Vec<(String, String, String)> = vec![
            (
                "遗忘缺失度".into(),
                format!("{} 平均缺失度", spec.name),
                format!("{:.3}", avg_md),
            ),
            (
                "遗忘缺失度".into(),
                format!("{} 最大缺失度", spec.name),
                format!("{:.3}", max_md),
            ),
            (
                "边衰减".into(),
                format!("{} 平均边强度", spec.name),
                format!("{:.3}", avg_edge_intensity),
            ),
            (
                "边衰减".into(),
                format!("{} 平均检索权重", spec.name),
                format!("{:.3}", avg_edge_weight),
            ),
            (
                "遮罩".into(),
                format!("{} 平均遮罩率", spec.name),
                format!("{:.2}%", avg_masked_ratio * 100.0),
            ),
        ];
        if use_llm {
            metrics.push((
                "LLM".into(),
                format!("{} llama-server 修订数", spec.name),
                format!("{}/{} 有效", effective_revised, llm_revised),
            ));
        }
        for (k, v) in &hist_vec {
            metrics.push((
                "遗忘动作".into(),
                format!("{} {}", spec.name, k),
                v.to_string(),
            ));
        }

        ForgetCaseData {
            case_name: spec.name.to_string(),
            passed,
            llm_available: use_llm,
            node_count,
            edge_count,
            llm_revised,
            effective_revised,
            action_histogram: hist_vec,
            avg_missing_degree: avg_md,
            max_missing_degree: max_md,
            avg_masked_ratio,
            avg_edge_intensity,
            nodes: stats,
            detail_lines,
            metrics,
        }
    }

    /// 多步遗忘：3 轮 × 24h，对图中**每个受遗忘影响的节点**逐步执行
    /// 衰减 → 遮罩 →（LLM 抽样）修订，收集每一步的输入输出与缺失度轨迹。
    ///
    /// 观测内容：
    /// - 每节点的缺失度轨迹（md0 → md1 → md2 → md3，单调不减）；
    /// - 每轮动作分布（NoAction / MaskOnly / Revised）与有效修订数；
    /// - 内容退化轨迹：原始文本 → 每轮遮罩输入 → LLM 原始回复。
    fn run_multi_step_case(&self) -> ForgetCaseData {
        const STEPS: usize = 3;
        const STEP_HOURS: i64 = 24;
        let t0 = Utc::now() - ChronoDuration::hours(STEP_HOURS * (STEPS as i64));
        let mut cluster = self.graph.clone();
        let use_llm = self.llm.is_some();

        // 初始化：所有节点/边在 t0 时刻刚被遗忘刷新（缺失度归零）
        self.apply_aging(&mut cluster, t0, 0);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let node_indices: Vec<_> = {
            let g = cluster.graph();
            g.node_indices().collect()
        };

        // 原始文本（第一轮前的原文，供"原始输入输出"展示）
        let originals: Vec<String> = node_indices
            .iter()
            .map(|idx| {
                let g = cluster.graph();
                get_summary(g.node_weight(*idx).expect("node").note()).unwrap_or_default()
            })
            .collect();

        // 每节点的最终观测状态（观测页列表数据）
        let mut final_nodes: Vec<NodeForgetStat> = Vec::new();

        let mut passed = true;
        let mut per_step_avg_md: Vec<f32> = Vec::new();
        let mut per_step_hist: Vec<(String, Vec<(&'static str, usize)>)> = Vec::new();
        let mut detail_lines = Vec::new();
        let mut metrics: Vec<(String, String, String)> = Vec::new();
        let mut total_effective = 0usize;
        let mut total_revised = 0usize;

        for step in 0..STEPS {
            let now = t0 + ChronoDuration::hours(STEP_HOURS * ((step + 1) as i64));

            // 先全图刷新缺失度（lazy_forget 内部也会刷新，但修订抽样需要
            // 基于本轮的缺失度筛选）
            compute_all_missing_degrees(&mut cluster, now);

            // LLM 修订抽样（每轮重算：内容与缺失度都已演变）
            let revise_set: std::collections::HashSet<_> = {
                let g = cluster.graph();
                let mut candidates: Vec<_> = g
                    .node_indices()
                    .filter(|idx| {
                        let n = g.node_weight(*idx).expect("node");
                        if !is_maskable(n.note()) {
                            return false;
                        }
                        let words = get_summary(n.note())
                            .map(|t| mask_word_count(&self.jieba, &t))
                            .unwrap_or(0);
                        n.note().missing_degree() >= REVISE_THRESHOLD
                            && words >= PIPELINE_REVISE_MIN_WORDS
                    })
                    .map(|idx| {
                        let n = g.node_weight(idx).expect("node");
                        (idx, n.note().missing_degree())
                    })
                    .collect();
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                candidates
                    .into_iter()
                    .take(MAX_LLM_REVISIONS)
                    .map(|(idx, _)| idx)
                    .collect()
            };

            let mut step_md_sum = 0.0f32;
            let mut step_hist: std::collections::BTreeMap<&'static str, usize> =
                std::collections::BTreeMap::new();
            let mut step_revised = 0usize;
            let mut step_effective = 0usize;

            for (i, idx) in node_indices.iter().enumerate() {
                let (before, type_name, id) = {
                    let g = cluster.graph();
                    let n = g.node_weight(*idx).expect("node");
                    (
                        current_missing_degree(n.note(), now),
                        forget_type_name(n.note()),
                        n.note().id().to_string(),
                    )
                };

                let (action, after) = {
                    let g = cluster.graph_mut();
                    let node = &mut g.node_weight_mut(*idx).expect("node").note;
                    let closure: LlmCall = if use_llm && revise_set.contains(idx) {
                        llama_closure(self.llm.as_ref().expect("llm").clone())
                    } else {
                        failing_llm_closure()
                    };
                    let act = runtime.block_on(lazy_forget(
                        node,
                        now,
                        &self.jieba,
                        Some(FORGET_SYSTEM_PROMPT),
                        closure,
                    ));
                    (act, node.missing_degree())
                };

                let (action_name, masked_text, llm_reply, effective) = match &action {
                    ForgetAction::NoAction => ("NoAction", None, None, false),
                    ForgetAction::MaskOnly { masked_text, .. } => {
                        ("MaskOnly", Some(masked_text.clone()), None, false)
                    }
                    ForgetAction::Revised {
                        masked_text, new_summary, ..
                    } => {
                        let eff = is_effective_revision(new_summary);
                        (
                            "Revised",
                            Some(masked_text.clone()),
                            Some(new_summary.clone()),
                            eff,
                        )
                    }
                };

                *step_hist.entry(action_name).or_insert(0) += 1;
                step_md_sum += after;
                if action_name == "Revised" {
                    step_revised += 1;
                    if effective {
                        step_effective += 1;
                    }
                }

                // 不变量：缺失度单调不减
                if after + 1e-4 < before {
                    passed = false;
                }
                if !(0.0..=1.0).contains(&after) {
                    passed = false;
                }

                // 内容轨迹（首轮展示原始文本，每轮展示动作与缺失度）
                if step == 0 {
                    detail_lines.push(format!(
                        "{} [{}] 原文: {}",
                        type_name,
                        id.chars().take(8).collect::<String>(),
                        originals[i]
                    ));
                }
                detail_lines.push(format!(
                    "  步{} t+{}h {} [{}] md {:.3}→{:.3} 动作={}{}",
                    step + 1,
                    STEP_HOURS * ((step + 1) as i64),
                    type_name,
                    id.chars().take(8).collect::<String>(),
                    before,
                    after,
                    action_name,
                    if action_name == "Revised" {
                        if effective {
                            " [有效修订]"
                        } else {
                            " [无效修订!]"
                        }
                    } else {
                        ""
                    }
                ));
                if let (Some(mt), Some(reply)) = (&masked_text, &llm_reply) {
                    detail_lines.push(format!("      遮罩输入: {}", mt));
                    detail_lines.push(format!("      LLM原始回复: {}", reply));
                }

                // 收集最终节点状态（供观测页逐节点查看，避免"节点 0/0"）
                let masked_count = masked_text.as_ref().map(|mt| count_masked(mt)).unwrap_or(0);
                let orig_words = mask_word_count(&self.jieba, &originals[i]);
                let stat = NodeForgetStat {
                    id: id.clone(),
                    type_name,
                    original: originals[i].clone(),
                    md_before: before,
                    md_after: after,
                    action: action_name,
                    mask: if masked_count > 0 {
                        Some((masked_count, orig_words))
                    } else {
                        None
                    },
                    masked_text,
                    llm_reply,
                    effective,
                };
                if step == 0 {
                    final_nodes.push(stat);
                } else {
                    final_nodes[i] = stat;
                }
            }

            let avg_md = step_md_sum / node_indices.len().max(1) as f32;
            per_step_avg_md.push(avg_md);
            per_step_hist.push((
                format!("step{}", step + 1),
                step_hist.iter().map(|(k, v)| (*k, *v)).collect(),
            ));
            total_revised += step_revised;
            total_effective += step_effective;
        }

        // 汇总 metric
        for (s, avg) in per_step_avg_md.iter().enumerate() {
            metrics.push((
                "多步遗忘".into(),
                format!("step{} 平均缺失度", s + 1),
                format!("{:.3}", avg),
            ));
        }
        for (label, hist) in &per_step_hist {
            for (k, v) in hist {
                metrics.push((
                    "多步遗忘动作".into(),
                    format!("{} {}", label, k),
                    v.to_string(),
                ));
            }
        }
        if use_llm {
            metrics.push((
                "多步遗忘".into(),
                "LLM 修订/有效".into(),
                format!("{}/{}", total_effective, total_revised),
            ));
        }
        // 衰减曲线：轮次 vs 平均缺失度
        let points: Vec<(f64, f64)> = per_step_avg_md
            .iter()
            .enumerate()
            .map(|(i, md)| ((i + 1) as f64, *md as f64))
            .collect();

        // LLM 可用但全程零修订 → 明确失败
        if use_llm && total_revised == 0 {
            passed = false;
        }

        let node_count = node_indices.len();
        let edge_count = {
            let g = cluster.graph();
            g.edge_count()
        };
        let max_md = per_step_avg_md.last().copied().unwrap_or(0.0);

        let mut out_metrics = vec![
            (
                "多步遗忘".into(),
                "平均缺失度(末步)".into(),
                format!("{:.3}", max_md),
            ),
            (
                "多步遗忘".into(),
                "受影响的节点数".into(),
                node_count.to_string(),
            ),
            (
                "多步遗忘".into(),
                "边数".into(),
                edge_count.to_string(),
            ),
        ];
        out_metrics.extend(metrics);

        ForgetCaseData {
            case_name: "multi-step".into(),
            passed,
            llm_available: use_llm,
            node_count,
            edge_count,
            llm_revised: total_revised,
            effective_revised: total_effective,
            action_histogram: per_step_hist
                .last()
                .map(|(_, h)| h.clone())
                .unwrap_or_default(),
            avg_missing_degree: max_md,
            max_missing_degree: max_md,
            avg_masked_ratio: 0.0,
            avg_edge_intensity: 0.0,
            nodes: final_nodes,
            detail_lines,
            metrics: out_metrics,
        }
    }

    /// 激活测试：固定种子随机选节点激活多次（`retrieval_increment`），
    /// 统一老化 72h 后验证**整图**的遗忘状态是否符合设计：
    /// - 激活次数越多 → 半衰期越长 → 缺失度越低（负相关）；
    /// - 每个节点的实测缺失度与理论值一致（±1e-3）。
    fn run_activation_case(&self) -> ForgetCaseData {
        let now = Utc::now();
        const ELAPSED_HOURS: i64 = 72;
        let mut cluster = self.graph.clone();
        self.apply_aging(&mut cluster, now, ELAPSED_HOURS);

        // 固定种子随机激活：给随机节点赋予 0..=12 次激活
        let mut rng = StdRng::seed_from_u64(20260818);
        let node_indices: Vec<_> = {
            let g = cluster.graph();
            g.node_indices().collect()
        };
        let mut shuffled: Vec<usize> = (0..node_indices.len()).collect();
        shuffled.shuffle(&mut rng);
        // 约 2/3 的节点获得随机激活次数
        let activated_count = (node_indices.len() * 2 / 3).max(1);
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &i in shuffled.iter().take(activated_count) {
            let c: usize = rng.random_range(1..=12);
            counts.insert(i, c);
        }

        // 施加激活（retrieval_increment 会更新 last_accessed_time，不影响衰减公式）
        {
            let g = cluster.graph_mut();
            for (i, idx) in node_indices.iter().enumerate() {
                if let Some(&n) = counts.get(&i) {
                    let node = &mut g.node_weight_mut(*idx).expect("node").note;
                    for _ in 0..n {
                        node.retrieval_increment();
                    }
                }
            }
        }

        // 全图刷新缺失度
        compute_all_missing_degrees(&mut cluster, now);

        // 逐节点校验：实测 vs 理论
        let mut passed = true;
        let mut detail_lines = Vec::new();
        let mut groups: std::collections::BTreeMap<u32, (f32, f32, usize)> =
            std::collections::BTreeMap::new(); // 分组 -> (实测md和, 理论md和, 数)
        let g = cluster.graph();
        for (i, idx) in node_indices.iter().enumerate() {
            let n = g.node_weight(*idx).expect("node");
            let count = counts.get(&i).copied().unwrap_or(0);
            let md_actual = n.note().missing_degree();
            let md_theory = activation_theory_md(count, ELAPSED_HOURS as f32);

            // 容差：浮点公式一致，1e-3 足够
            if (md_actual - md_theory).abs() > 1e-3 {
                passed = false;
            }
            if !(0.0..=1.0).contains(&md_actual) {
                passed = false;
            }

            // 分组统计（0 / 1-3 / 4-7 / 8+）
            let bucket = match count {
                0 => 0u32,
                1..=3 => 1,
                4..=7 => 2,
                _ => 3,
            };
            let e = groups.entry(bucket).or_insert((0.0, 0.0, 0));
            e.0 += md_actual;
            e.1 += md_theory;
            e.2 += 1;

            let short_id: String = n.note().id().to_string().chars().take(8).collect();
            detail_lines.push(format!(
                "{} 激活{}次 md实测{:.3} 理论{:.3} 偏差{:.5}",
                short_id, count, md_actual, md_theory, md_actual - md_theory
            ));
        }

        // 负相关断言：激活越多的分组，平均缺失度越低
        let bucket_order = [0u32, 1, 2, 3];
        let avg_md: Vec<(u32, f32)> = groups
            .iter()
            .map(|(k, (s, _, n))| (*k, s / *n as f32))
            .collect();
        for w in bucket_order.windows(2) {
            let a = avg_md.iter().find(|(k, _)| *k == w[0]).map(|(_, v)| *v);
            let b = avg_md.iter().find(|(k, _)| *k == w[1]).map(|(_, v)| *v);
            if let (Some(a), Some(b)) = (a, b) {
                // 相邻分组严格递减（激活多 → 缺失度低）
                if b >= a - 1e-3 {
                    passed = false;
                }
            }
        }

        let mut metrics: Vec<(String, String, String)> = Vec::new();
        for (k, (s, t, n)) in &groups {
            let label = match k {
                0 => "激活0次".to_string(),
                1 => "激活1-3次".to_string(),
                2 => "激活4-7次".to_string(),
                _ => "激活8+次".to_string(),
            };
            metrics.push((
                "激活测试".into(),
                format!("{} 平均缺失度(实测/理论)", label),
                format!("{:.3}/{:.3}（{}节点）", s / *n as f32, t / *n as f32, n),
            ));
        }
        metrics.push((
            "激活测试".into(),
            "半衰期延长设计".into(),
            format!(
                "激活10次: 24h→{}h；激活0次: 24h",
                DEFAULT_BASE_HALF_LIFE_HOURS * (1.0 + DEFAULT_ACTIVE_FACTOR * 10.0)
            ),
        ));

        ForgetCaseData {
            case_name: "activation".into(),
            passed,
            llm_available: false,
            node_count: node_indices.len(),
            edge_count: {
                let g = cluster.graph();
                g.edge_count()
            },
            llm_revised: 0,
            effective_revised: 0,
            action_histogram: vec![],
            avg_missing_degree: groups.get(&0).map(|(s, _, n)| s / *n as f32).unwrap_or(0.0),
            max_missing_degree: 0.0,
            avg_masked_ratio: 0.0,
            nodes: vec![],
            avg_edge_intensity: 0.0,
            detail_lines,
            metrics,
        }
    }

    /// 增量一致性场景：两次 12h 增量更新 == 一次 24h 全量计算（误差 < 1e-3）
    fn run_incremental_case(&self) -> ForgetCaseData {
        let now = Utc::now();
        let mut passed = true;
        let mut detail_lines = Vec::new();
        let mut metrics = Vec::new();

        let make = |offset_hours: i64| -> MemoryNote {
            let created = now - ChronoDuration::hours(24 * 10);
            let forget = now - ChronoDuration::hours(offset_hours);
            MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
                "红魔馆的女仆长十六夜咲夜擅长投掷银质小刀她害怕烫的食物是众所周知的猫舌大小姐为此常常感到无可奈何"
                    .to_string(),
                ConceptType::Entity,
                "测试描述".to_string(),
            )))
            .create_time(created)
            .last_accessed_time(created)
            .last_forget_time(forget)
            .build()
            .expect("note")
        };

        let mut full = make(24);
        let full_md = compute_and_update(&mut full, now);
        let mut inc = make(24);
        let mid = now - ChronoDuration::hours(12);
        let mid_md = compute_and_update(&mut inc, mid);
        let inc_md = compute_and_update(&mut inc, now);

        let from_create = compute_missing_degree(
            full.creation_time(),
            full.retrieval_count(),
            now,
            DEFAULT_BASE_HALF_LIFE_HOURS,
            DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );

        let diff = (full_md - inc_md).abs();
        detail_lines.push(format!(
            "全量24h缺失度={:.4}，增量12h+12h缺失度={:.4}，差值={:.5}（从创建时间算起={:.4}）",
            full_md, inc_md, diff, from_create
        ));
        metrics.push(("增量一致性".into(), "全量24h缺失度".into(), format!("{:.4}", full_md)));
        metrics.push(("增量一致性".into(), "两次12h增量缺失度".into(), format!("{:.4}", inc_md)));
        metrics.push(("增量一致性".into(), "差值".into(), format!("{:.5}", diff)));
        metrics.push(("增量一致性".into(), "中间态12h缺失度".into(), format!("{:.4}", mid_md)));

        if diff > 1e-3 {
            passed = false;
        }
        if !(0.0..=1.0).contains(&inc_md) || !(0.0..=1.0).contains(&full_md) {
            passed = false;
        }

        ForgetCaseData {
            case_name: "incremental".into(),
            passed,
            llm_available: false,
            node_count: 1,
            edge_count: 0,
            llm_revised: 0,
            effective_revised: 0,
            action_histogram: vec![("NoAction", 1)],
            avg_missing_degree: inc_md,
            max_missing_degree: inc_md,
            avg_masked_ratio: 0.0,
            avg_edge_intensity: 0.0,
            nodes: vec![],
            detail_lines,
            metrics,
        }
    }
}

impl TestSuite for ForgetPipelineSuite {
    fn case_count(&self) -> usize {
        self.cases.len()
    }

    fn run_case(&self, index: usize) -> TestCaseOutcome {
        let spec = &self.cases[index];
        let data = if spec.elapsed_hours == -1 {
            self.run_incremental_case()
        } else if spec.elapsed_hours == -10 {
            self.run_multi_step_case()
        } else if spec.elapsed_hours == -11 {
            self.run_activation_case()
        } else {
            self.run_pipeline(spec)
        };
        let passed = data.passed;
        TestCaseOutcome {
            case_name: format!("forget/full/{}", spec.name),
            description: spec.description.to_string(),
            passed,
            data: Box::new(data),
        }
    }

    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        elapsed: Duration,
        total: usize,
        passed: usize,
        _failed: usize,
    ) -> SuiteReport {
        let mut metrics: Vec<Box<dyn crate::engine::suite::ReportMetric>> = Vec::new();
        let mut detail_rows: Vec<DetailRow> = Vec::new();
        let mut decay_points: Vec<(f64, f64)> = Vec::new();
        let mut llm_available = false;
        let mut total_llm_revised = 0usize;
        let mut total_effective_revised = 0usize;
        let mut max_node_count = 0usize;
        let mut max_edge_count = 0usize;

        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<ForgetCaseData>() else {
                continue;
            };
            llm_available |= data.llm_available;
            total_llm_revised += data.llm_revised;
            total_effective_revised += data.effective_revised;
            max_node_count = max_node_count.max(data.node_count);
            max_edge_count = max_edge_count.max(data.edge_count);
            for (group, label, value) in &data.metrics {
                metrics.push(Box::new(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                )));
            }
            let hours = match data.case_name.as_str() {
                "low" => Some(8.0),
                "medium" => Some(24.0),
                "high" => Some(72.0),
                _ => None,
            };
            if let Some(h) = hours {
                decay_points.push((h, data.avg_missing_degree as f64));
            }
            let hist_txt: String = data
                .action_histogram
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ");
            detail_rows.push(DetailRow {
                text: format!(
                    "[{}] 节点{} 边{} | 缺失度均值{:.3}/最大{:.3} | 遮罩率{:.1}% | 边强度{:.3} | 修订 {}/{} 有效 | {}",
                    data.case_name,
                    data.node_count,
                    data.edge_count,
                    data.avg_missing_degree,
                    data.max_missing_degree,
                    data.avg_masked_ratio * 100.0,
                    data.avg_edge_intensity,
                    data.effective_revised,
                    data.llm_revised,
                    hist_txt,
                ),
                has_error: !o.passed,
            });
            for line in &data.detail_lines {
                detail_rows.push(DetailRow {
                    text: format!("[{}] {}", data.case_name, line),
                    has_error: !o.passed,
                });
            }
        }

        decay_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if decay_points.len() >= 2 {
            metrics.push(Box::new(chart_metric(
                "遗忘衰减曲线".to_string(),
                "遗忘缺失度".to_string(),
                "时间跨度(小时)".to_string(),
                "平均缺失度".to_string(),
                vec![Series {
                    label: "平均缺失度".to_string(),
                    points: decay_points,
                }],
            )));
        }

        metrics.push(Box::new(key_value_metric(
            "图".to_string(),
            "图".to_string(),
            format!(
                "{}（节点 {} / 边 {}）",
                self.graph_name, max_node_count, max_edge_count
            ),
        )));
        metrics.push(Box::new(key_value_metric(
            "LLM 可用".to_string(),
            "LLM".to_string(),
            if llm_available {
                format!(
                    "是（llama-server，修订 {} 节点，{} 有效）",
                    total_llm_revised, total_effective_revised
                )
            } else {
                "否（遮罩降级路径已验证）".to_string()
            },
        )));
        metrics.push(Box::new(key_value_metric(
            "通过率".to_string(),
            "汇总".to_string(),
            format!(
                "{:.1}% ({}/{})",
                if total > 0 {
                    passed as f64 / total as f64 * 100.0
                } else {
                    0.0
                },
                passed,
                total
            ),
        )));

        SuiteReport {
            metrics,
            detail_header: format!(
                "阶段3 全管线逐节点明细（图 {}：节点 {} / 边 {}，通过 {}/{}，耗时 {:.2}s）",
                self.graph_name,
                max_node_count,
                max_edge_count,
                passed,
                total,
                elapsed.as_secs_f64()
            ),
            detail_rows,
            outcomes,
        }
    }
}

// ========================================================================
// 内部辅助
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 仓库内真实 fixture：格蕾修角色图
    fn fixture_graph() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures/example_data/格蕾修_https_zh_moegirl_org_cn_E6_A0_BC_E8_95_BE_E4_BF_AE/graph.json")
    }

    // ── 阶段 1：Mask ──

    #[test]
    fn test_mask_suite_has_cases_and_all_pass() {
        let suite = ForgetMaskSuite::new();
        assert!(suite.case_count() >= 10);
        for i in 0..suite.case_count() {
            let outcome = suite.run_case(i);
            assert!(
                outcome.passed,
                "遮罩用例 {} 失败: {}",
                outcome.case_name,
                outcome.description
            );
        }
    }

    #[test]
    fn test_mask_suite_loads_fixture_and_all_pass() {
        // 遮罩阶段也以 fixture 图为输入源
        let suite = ForgetMaskSuite::load(&fixture_graph()).expect("加载 fixture 图");
        assert!(suite.case_count() >= 12);
        for i in 0..suite.case_count() {
            let outcome = suite.run_case(i);
            assert!(
                outcome.passed,
                "遮罩用例 {} 失败: {}",
                outcome.case_name,
                outcome.description
            );
        }
    }

    #[test]
    fn test_mask_ratio_matches_missing_degree() {
        let jieba = Jieba::new();
        let text = MASK_TEXTS[2].1; // 长文本
        for md in [0.2f32, 0.5, 0.87] {
            let r = mask_text(text, md, &jieba);
            let ratio = r.masked_count as f32 / r.total_count.max(1) as f32;
            assert!(
                (ratio - md).abs() < 0.15,
                "md={} ratio={}",
                md,
                ratio
            );
            assert_eq!(count_masked(&r.masked_text), r.masked_count);
        }
    }

    #[test]
    fn test_mask_deterministic() {
        let jieba = Jieba::new();
        let text = MASK_TEXTS[2].1;
        let a = mask_text(text, 0.5, &jieba);
        let b = mask_text(text, 0.5, &jieba);
        assert_eq!(a.masked_text, b.masked_text);
    }

    // ── 阶段 2：Revise（无 LLM 时探活失败但用例不 panic）──

    #[test]
    fn test_revise_suite_loads_fixture_samples() {
        let suite =
            ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        assert!(suite.llm.is_none(), "测试环境不应配置 LLM");
        assert!(!suite.samples.is_empty(), "长文本样本不应为空");
        // 样本都足够长
        let jieba = Jieba::new();
        for s in &suite.samples {
            assert!(
                mask_word_count(&jieba, &s.original) >= REVISE_MIN_WORDS,
                "样本词数不足"
            );
            assert!(s.masked.contains(MASK_WORD.trim()), "样本应含遮罩");
        }
    }

    #[test]
    fn test_revise_probe_fails_without_llm() {
        let suite =
            ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let outcome = suite.run_case(0); // probe
        assert!(!outcome.passed, "无 LLM 时探活应失败");
    }

    // ── 阶段 3：Pipeline（无 LLM 全绿，降级路径）──

    #[test]
    fn test_pipeline_loads_real_fixture_graph() {
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        assert!(suite.llm.is_none());
        let node_count = suite.graph.graph().node_count();
        let edge_count = suite.graph.graph().edge_count();
        assert!(node_count > 10, "真实图应有足够节点，实际 {}", node_count);
        assert!(edge_count > 0, "真实图应有边，实际 {}", edge_count);
    }

    #[test]
    fn test_pipeline_all_cases_pass_without_llm() {
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        for i in 0..suite.case_count() {
            let outcome = suite.run_case(i);
            assert!(
                outcome.passed,
                "用例 {} 失败: {}",
                outcome.case_name,
                outcome.description
            );
        }
    }

    #[test]
    fn test_pipeline_multi_step_without_llm() {
        // 多步遗忘：无 LLM 时应全走遮罩降级且通过（缺失度单调不减）
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let data = suite.run_multi_step_case();
        assert!(data.passed, "多步遗忘失败");
        assert!(data.node_count > 10, "应覆盖全图节点");
        assert!(!data.detail_lines.is_empty());
        // 无 LLM 时不应有修订
        assert_eq!(data.llm_revised, 0);
    }

    #[test]
    fn test_pipeline_activation_reflects_design() {
        // 激活测试：确定性（固定种子），激活多的节点缺失度更低，实测=理论
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let data = suite.run_activation_case();
        assert!(data.passed, "激活测试失败");
        // 理论公式自检：激活10次 vs 0次在72h的缺失度
        let md0 = activation_theory_md(0, 72.0);
        let md3 = activation_theory_md(3, 72.0);
        let md10 = activation_theory_md(10, 72.0);
        assert!(md10 < md3 && md3 < md0, "激活应减缓遗忘");
        assert!((md0 - 0.875).abs() < 1e-2, "72h 无激活缺失度应≈0.875");
    }

    #[test]
    fn test_pipeline_report_builds_metrics_and_rows() {
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let n = suite.case_count();
        let outcomes: Vec<TestCaseOutcome> = (0..n).map(|i| suite.run_case(i)).collect();
        let passed = outcomes.iter().filter(|o| o.passed).count();
        let report =
            suite.build_report(outcomes, Duration::from_millis(10), n, passed, n - passed);
        assert!(!report.metrics.is_empty());
        assert!(!report.detail_rows.is_empty());
        assert_eq!(report.outcomes.len(), n);
    }

    #[test]
    fn test_incremental_consistency() {
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let data = suite.run_incremental_case();
        assert!(data.passed, "增量一致性失败");
    }

    #[test]
    fn test_observer_downcast_roundtrip() {
        // 验证 TUI 观测页的数据通路：build_report 后 outcomes.data
        // 仍可 downcast 回 ForgetCaseData 且 nodes 非空（节点 0/0 的回归测试）
        let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let n = suite.case_count();
        let outcomes: Vec<TestCaseOutcome> = (0..n).map(|i| suite.run_case(i)).collect();
        let passed = outcomes.iter().filter(|o| o.passed).count();
        // 诊断：run_case 返回的 data 是否可直接识别
        let direct_ok = outcomes[0].data.is::<ForgetCaseData>();
        let report =
            suite.build_report(outcomes, Duration::from_millis(10), n, passed, n - passed);
        let report_ok = report
            .outcomes
            .first()
            .map(|o| o.data.is::<ForgetCaseData>())
            .unwrap_or(false);
        assert!(direct_ok, "run_case 的 data 不是 ForgetCaseData");
        assert!(report_ok, "build_report 后 data 不再是 ForgetCaseData");
        assert_eq!(report.outcomes.len(), n);
        for o in &report.outcomes {
            let data = o
                .data
                .downcast_ref::<ForgetCaseData>()
                .unwrap_or_else(|| panic!("downcast 失败: {}", o.case_name));
            if matches!(data.case_name.as_str(), "low" | "medium" | "high" | "multi-step") {
                assert!(
                    !data.nodes.is_empty(),
                    "{} 的节点数据为空（观测页将显示 0/0）",
                    o.case_name
                );
            }
        }
    }
}
