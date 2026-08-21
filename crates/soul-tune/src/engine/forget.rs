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
//! Pipeline 内嵌的**激发测试（excitation，黑盒效果）**：图克隆两份配对对照，
//! 按设计剂量梯度激发部分节点，验证"激发 → 遗忘被延缓"这一**可观察效果**
//! （断言 E1~E6 见 [`ForgetPipelineSuite::run_excitation_case`]）。soul-tune 是
//! 效果测试框架：不读取算法内部常量、不假设激发次数如何进入衰减公式，只通过
//! 公开接口驱动与观测。设计文档见 `docs/architecture/激发测试设计.md`。
//!
//! LLM 后端与 playtest 完全一致：统一来源解析（见 `engine::llm::resolver`）——
//! 先探测运行中的 llama-server，没有则自动拉起本地缓存模型，都没有则降级遮罩。

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, TimeZone, Utc};
use jieba_rs::Jieba;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use soul_mem_algo::algo::forget::decay_calculator::{
    compute_missing_degree, update_missing_degree_incremental, DEFAULT_MAX_ACTIVATION_CAP,
};
use soul_mem_algo::algo::forget::decay_revise::{
    compute_all_missing_degrees, current_missing_degree, decay_graph_edge, get_summary,
    lazy_forget, weight_placeholder, ForgetAction, DEFAULT_ACTIVE_FACTOR,
    DEFAULT_BASE_HALF_LIFE_HOURS, REVISE_THRESHOLD,
};
use soul_mem_algo::algo::forget::llm_completion::build_reconstruct_prompt;
use soul_mem_algo::algo::forget::mask::{mask_text, MASK_WORD};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_runtime::cluster::memory_cluster::MemoryCluster;

use crate::engine::llm::{LlmBackend, LlamaServer};
use crate::engine::loader::{build_reverse_id_map, load_graph_cluster};
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

/// 按统一来源解析创建 LLM 后端（见 [`crate::engine::llm::resolver`]）：
/// 复用运行中的 llama-server → 自动拉起本地缓存模型 → 降级为 None（遮罩路径）。
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

/// 遮罩用例：对指定文本按指定缺失度执行 `mask_text`
struct MaskCaseSpec {
    /// 用例名：`{node_id}-md{md:.2}` 或 `{node_id}-determinism`
    name: String,
    /// 源记忆节点 id（内置文本集时用文本标签 short/medium/long）
    node_id: String,
    text: String,
    missing_degree: f32,
}

/// 缺失度梯度
const MASK_GRADIENTS: [f32; 6] = [0.0, 0.1, 0.2, 0.5, 0.87, 1.0];

/// 从节点列表构造遮罩用例：**每个节点 × 全梯度**（时间越长 → 缺失度越高 → 遮罩越多），
/// 外加确定性用例（取最长文本）。以记忆节点 id 为单位，供观测按节点聚合。
fn build_node_mask_cases(nodes: &[(String, String)]) -> Vec<MaskCaseSpec> {
    let mut cases = Vec::new();
    for (id, text) in nodes {
        for md in MASK_GRADIENTS {
            cases.push(MaskCaseSpec {
                name: format!("{id}-md{md:.2}"),
                node_id: id.clone(),
                text: text.clone(),
                missing_degree: md,
            });
        }
    }
    // 确定性：同一输入两次结果一致（取最长文本）
    if let Some((id, long)) = nodes
        .iter()
        .max_by(|a, b| a.1.chars().count().cmp(&b.1.chars().count()))
    {
        cases.push(MaskCaseSpec {
            name: format!("{id}-determinism"),
            node_id: id.clone(),
            text: long.clone(),
            missing_degree: 0.5,
        });
    }
    cases
}

/// 内置文本集：短 / 中 / 长中文文本（无图依赖，测试/快速验证使用）
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
#[derive(Serialize)]
pub struct MaskCaseData {
    pub case_name: String,
    /// 源记忆节点 id（观测按此聚合）
    pub node_id: String,
    pub passed: bool,
    /// 原文（遮罩前）
    pub original: String,
    /// 遮罩结果文本
    pub masked: String,
    /// 遮罩词数 / 总词数（供遮罩率）
    pub masked_count: usize,
    pub total_count: usize,
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
        let nodes: Vec<(String, String)> = MASK_TEXTS
            .iter()
            .map(|(tag, text)| (tag.to_string(), text.to_string()))
            .collect();
        Self {
            jieba: Jieba::new(),
            cases: build_node_mask_cases(&nodes),
        }
    }

    /// 从指定 fixture 图加载：**收集全部可遮罩节点**（SemMemory / SpecificSituation），
    /// 每个节点 × 全缺失度梯度构造用例 —— 观测以记忆节点 id 为单位展示遮罩演变。
    pub fn load(path: &Path) -> Result<Self, String> {
        let (cluster, id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let id_rev = build_reverse_id_map(&id_map);
        let jieba = Jieba::new();
        let mut nodes: Vec<(String, String)> = Vec::new();
        for n in cluster.graph().node_weights() {
            if !is_maskable(n.note()) {
                continue;
            }
            let text = get_summary(n.note()).unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }
            // 语义 id：graph.json 可读 id，避免每次运行不同的 UUID
            nodes.push((display_id(&id_rev, n.note().id()), text));
        }
        // 确定性排序，保证可复现
        nodes.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            jieba,
            cases: build_node_mask_cases(&nodes),
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
        if spec.name.ends_with("-determinism") {
            if r1.masked_text != r2.masked_text {
                detail_lines.push("  确定性检查: 两次结果不一致！".into());
            }
        }

        let metrics = vec![
            (
                "遮罩".into(),
                format!("{} 遮罩率", spec.node_id),
                format!("{:.0}%", if r1.total_count > 0 { r1.masked_count as f32 / r1.total_count as f32 * 100.0 } else { 0.0 }),
            ),
            (
                "遮罩".into(),
                format!("{} 词数", spec.node_id),
                format!("{}/{}", r1.masked_count, r1.total_count),
            ),
        ];

        MaskCaseData {
            case_name: spec.name.to_string(),
            node_id: spec.node_id.clone(),
            passed,
            original: spec.text.clone(),
            masked: r1.masked_text.clone(),
            masked_count: r1.masked_count,
            total_count: r1.total_count,
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
        let mut metrics: Vec<crate::engine::suite::MetricEntry> = Vec::new();
        let mut detail_rows: Vec<DetailRow> = Vec::new();
        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<MaskCaseData>() else {
                continue;
            };
            for (group, label, value) in &data.metrics {
                metrics.push(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                ));
            }
            for line in &data.detail_lines {
                detail_rows.push(DetailRow {
                    text: format!("[{}] {}", data.case_name, line),
                    has_error: !o.passed,
                });
            }
        }
        metrics.push(key_value_metric(
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
        ));
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

/// 参与修订验证的遮罩水平梯度（缺失度）：低 / 中 / 高
pub const REVISE_MASK_GRADIENTS: [f32; 3] = [0.2, 0.5, 0.87];

/// 补全样本：从 fixture 图选取的可遗忘节点 × 遮罩水平梯度
pub struct ReviseSample {
    pub node_id: String,
    pub type_name: &'static str,
    /// 原始文本
    pub original: String,
    /// 遮罩后的文本（按 `mask_md` 遮罩）
    pub masked: String,
    /// 遮罩水平（缺失度梯度，观测页 x 轴）
    pub mask_md: f32,
}

/// 补全用例的观测数据
#[derive(Serialize)]
pub struct ReviseCaseData {
    pub case_name: String,
    pub passed: bool,
    pub llm_available: bool,
    /// 源节点 id（供原文对照）
    pub node_id: String,
    /// 节点原文
    pub original: String,
    /// LLM 遮罩输入
    pub masked_text: String,
    /// LLM **原始回复**
    pub llm_reply: String,
    pub detail_lines: Vec<String>,
    pub metrics: Vec<(String, String, String)>,
}

/// 抽样模式的目标样本数（约 8 个：分层抽样）
pub const REVISE_MAX_SAMPLES: usize = 8;

/// 修订测试采样模式：全量（所有可遗忘节点）或分层抽样（固定种子，约 8 个）
#[derive(Debug, Clone, Copy)]
pub enum ReviseMode {
    /// 对所有可遗忘（SemMemory / SpecificSituation）节点执行遮罩修订
    Full,
    /// 按节点类型分层抽样，固定种子可复现，总数约 [`REVISE_MAX_SAMPLES`]
    Sampled(u64),
}

/// 构造修订样本：全量 = 全部可遗忘节点；抽样 = 按类型分层、固定种子层间轮转取约
/// [`REVISE_MAX_SAMPLES`] 个节点。每个节点按 [`REVISE_MASK_GRADIENTS`] 全梯度展开
/// （低/中/高遮罩水平各一个样本）。
fn build_revise_samples(
    jieba: &Jieba,
    candidates: Vec<(String, &'static str, String, usize)>,
    mode: ReviseMode,
) -> Vec<ReviseSample> {
    // 单个候选 × 全梯度展开
    let expand = |c: &(String, &'static str, String, usize)| -> Vec<ReviseSample> {
        REVISE_MASK_GRADIENTS
            .iter()
            .map(|&md| ReviseSample {
                node_id: c.0.clone(),
                type_name: c.1,
                original: c.2.clone(),
                masked: mask_text(&c.2, md, jieba).masked_text,
                mask_md: md,
            })
            .collect()
    };
    match mode {
        ReviseMode::Full => candidates.iter().flat_map(expand).collect(),
        ReviseMode::Sampled(seed) => {
            // 分层：按节点类型分组（保持候选顺序稳定）
            let mut layers: Vec<(&'static str, Vec<usize>)> = Vec::new();
            for (i, (_, ty, _, _)) in candidates.iter().enumerate() {
                match layers.iter_mut().find(|(t, _)| t == ty) {
                    Some((_, v)) => v.push(i),
                    None => layers.push((ty, vec![i])),
                }
            }
            let mut rng = StdRng::seed_from_u64(seed);
            let mut pools: Vec<Vec<usize>> = layers
                .iter()
                .map(|(_, v)| {
                    let mut p = v.clone();
                    p.shuffle(&mut rng);
                    p
                })
                .collect();
            // 层间轮转取目标节点数，保证每层都有代表
            let mut picked: Vec<usize> = Vec::new();
            while picked.len() < REVISE_MAX_SAMPLES {
                let mut progressed = false;
                for p in &mut pools {
                    if picked.len() >= REVISE_MAX_SAMPLES {
                        break;
                    }
                    if let Some(i) = p.pop() {
                        picked.push(i);
                        progressed = true;
                    }
                }
                if !progressed {
                    break;
                }
            }
            picked
                .into_iter()
                .flat_map(|i| expand(&candidates[i]))
                .collect()
        }
    }
}

pub struct ForgetReviseSuite {
    jieba: Jieba,
    llm: Option<Arc<Mutex<LlamaServer>>>,
    /// 长文本样本（从 fixture 图提取）
    samples: Vec<ReviseSample>,
}

impl ForgetReviseSuite {
    /// 从 fixture 图加载长文本样本并尝试启用 llama-server（默认全量模式）
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut suite = Self::load_with_mode(path, ReviseMode::Full)?;
        suite.llm = try_create_llm();
        Ok(suite)
    }

    /// 按采样模式加载长文本样本（不启用 LLM；GUI 全量/抽样由此入口驱动）
    pub fn load_with_mode(path: &Path, mode: ReviseMode) -> Result<Self, String> {
        let (cluster, id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let id_rev = build_reverse_id_map(&id_map);
        let jieba = Jieba::new();
        let mut candidates: Vec<(String, &'static str, String, usize)> = Vec::new();
        for n in cluster.graph().node_weights() {
            if !is_maskable(n.note()) {
                continue;
            }
            let text = get_summary(n.note()).unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }
            let words = mask_word_count(&jieba, &text);
            candidates.push((
                display_id(&id_rev, n.note().id()),
                forget_type_name(n.note()),
                text,
                words,
            ));
        }
        let samples = build_revise_samples(&jieba, candidates, mode);
        Ok(Self {
            jieba,
            llm: None,
            samples,
        })
    }

    /// 分层抽样模式（GUI 使用）：按种子分层抽约 [`REVISE_MAX_SAMPLES`] 个，并启用 llama-server。
    /// 供 soul-tune-api 调用（跨 crate pub API，lib 内无直接调用者）。
    #[allow(dead_code)]
    pub fn load_sampled(path: &Path, seed: u64) -> Result<Self, String> {
        let mut suite = Self::load_with_mode(path, ReviseMode::Sampled(seed))?;
        suite.llm = try_create_llm();
        Ok(suite)
    }

    /// 仅加载长文本样本、不启用 LLM（测试 / 确定性验证使用）
    #[allow(dead_code)]
    pub fn load_without_llm(path: &Path) -> Result<Self, String> {
        Self::load_with_mode(path, ReviseMode::Full)
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
                // 与算法层 `reconstruct_summary` 完全一致的提示词（user 带占位符计数说明），
                // 保证修订测试 = 真实管线行为
                let (system, user) =
                    build_reconstruct_prompt(&sample.masked, Some(FORGET_SYSTEM_PROMPT));
                match runtime.block_on(closure(&system, &user)) {
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
                format!(
                    "{} md{:.2} 回复字数",
                    sample.node_id.chars().take(8).collect::<String>(),
                    sample.mask_md
                ),
                if llm_err.is_none() { reply.chars().count().to_string() } else { "失败".into() },
            ),
        ];

        ReviseCaseData {
            // 用例名带遮罩水平（观测页按 节点 × 梯度 聚合，x 轴 = 缺失度）
            case_name: format!(
                "{}-md{:.2}",
                sample.node_id.chars().take(8).collect::<String>(),
                sample.mask_md
            ),
            passed,
            llm_available,
            node_id: sample.node_id.clone(),
            original: sample.original.clone(),
            masked_text: sample.masked.clone(),
            llm_reply: reply,
            detail_lines,
            metrics,
        }
    }
}

impl TestSuite for ForgetReviseSuite {
    fn case_count(&self) -> usize {
        self.samples.len()
    }

    fn run_case(&self, index: usize) -> TestCaseOutcome {
        let data = self.run_revise_case(&self.samples[index]);
        let passed = data.passed;
        TestCaseOutcome {
            case_name: format!("forget/revise/{}", data.case_name),
            description: format!("遮罩补全验证: {}", data.case_name),
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
        let mut metrics: Vec<crate::engine::suite::MetricEntry> = Vec::new();
        let mut detail_rows: Vec<DetailRow> = Vec::new();
        let mut llm_available = false;
        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<ReviseCaseData>() else {
                continue;
            };
            llm_available |= data.llm_available;
            for (group, label, value) in &data.metrics {
                metrics.push(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                ));
            }
            for line in &data.detail_lines {
                detail_rows.push(DetailRow {
                    text: format!("[{}] {}", data.case_name, line),
                    has_error: !o.passed,
                });
            }
        }
        metrics.push(key_value_metric(
            "LLM 可用".to_string(),
            "LLM".to_string(),
            if llm_available {
                "是（llama-server）".to_string()
            } else {
                "否".to_string()
            },
        ));
        metrics.push(key_value_metric(
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
        ));
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

/// 全管线场景集：低/中/高遗忘强度 + 多步遗忘 + 激活测试 + 激发测试 + 增量一致性
pub const PIPELINE_CASES: [ForgetCaseSpec; 9] = [
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
        description: "高遗忘强度（Δt=72h）：llama-server 修订全部满足条件的节点",
        elapsed_hours: 72,
        want_llm: true,
    },
    ForgetCaseSpec {
        name: "multi-step",
        description: "多步遗忘：3 轮 × 24h 对全图每个节点逐步衰减/遮罩/修订（全部满足条件者补全）",
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
        name: "excitation-early",
        description: "激发测试·前置：t=0 全部激发，配对对照验证遗忘被延缓（黑盒效果）",
        elapsed_hours: -12, // 特殊标记：激发测试场景（前置）
        want_llm: false,
    },
    ForgetCaseSpec {
        name: "excitation-spaced",
        description: "激发测试·均布：24/48/72h 分批激发，事件研究验证遗忘被延缓",
        elapsed_hours: -13, // 特殊标记：激发测试场景（均布）
        want_llm: false,
    },
    ForgetCaseSpec {
        name: "excitation-late",
        description: "激发测试·后置：t=48h 全部激发，激发前无差异、激发后出现延缓",
        elapsed_hours: -14, // 特殊标记：激发测试场景（后置）
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

/// 激发测试的三种时机子场景（黑盒效果：同一批总激发次数，激发时机不同）。
///
/// 测试**不假设**时机是否影响结果——三种场景各自独立跑一遍全部断言，
/// 无论算法如何演化（次数制 / 回鲜制），"激发了就被延缓"的效果断言都必须成立。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExcitationSchedule {
    /// 前置：全部激发发生在 t=0（首个检查点之前）
    Early,
    /// 均布：按 24h / 48h / 72h 分三批激发
    Spaced,
    /// 后置：全部激发发生在 t=48h（第 2 个检查点）
    Late,
}

impl ExcitationSchedule {
    fn tag(self) -> &'static str {
        match self {
            ExcitationSchedule::Early => "early",
            ExcitationSchedule::Spaced => "spaced",
            ExcitationSchedule::Late => "late",
        }
    }
}

/// 单个节点的遗忘观测结果
#[derive(Clone, Serialize)]
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

/// 单个节点在某时间步的遗忘观测（供"以节点为单位、时间步长为横轴"的曲线）。
#[derive(Clone, Serialize)]
pub struct NodeStepStat {
    /// x 轴：距遗忘刷新的累计小时数（多步用例为 24/48/72；单步用例为 elapsed_hours）
    pub hours: i64,
    /// 步序号（0 起始；单步用例恒为 0）
    pub step: usize,
    /// 该时间步后的缺失度（y 轴主指标；激发测试为**激发组** md）
    pub md: f32,
    /// 对照组（未激发）同刻缺失度：仅激发测试填充（配对对照），其余用例为 None。
    /// 观测界面据此画"对照组 vs 激发组"双曲线，直观展示遗忘被延缓。
    pub md_ctrl: Option<f32>,
    /// 该步触发的遗忘动作（NoAction / MaskOnly / Revised；激发测试为 Activated / Control）
    pub action: &'static str,
    /// LLM 补全的遮罩输入文本（Revised / MaskOnly 时）
    pub masked_text: Option<String>,
    /// LLM **原始回复**（Revised 时，未经任何处理）
    pub llm_reply: Option<String>,
    /// 是否有效修订（回复非空且不含 [masked]）
    pub effective: bool,
}

/// 单个节点的完整时间步长序列：遗忘以节点为单位，对节点内容按时间步变化。
#[derive(Clone, Serialize)]
pub struct NodeSeries {
    pub id: String,
    pub type_name: &'static str,
    /// 图节点原文（遗忘管线执行前的原始记忆文本）
    pub original: String,
    /// 时间步序列（按 hours 升序）
    pub steps: Vec<NodeStepStat>,
}

/// 理想艾宾浩斯遗忘曲线采样：`md(t) = 1 - exp(-t·ln2 / 基础半衰期)`，
/// 从 0h 到 max_hours 每 2h 一个点，供观测图叠加对比"实测 vs 理想"。
pub fn ideal_ebbinghaus_curve(max_hours: i64) -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    let mut h = 0i64;
    while h <= max_hours {
        let md = 1.0
            - (-(h as f64) * std::f64::consts::LN_2 / DEFAULT_BASE_HALF_LIFE_HOURS as f64).exp();
        pts.push((h as f64, md));
        h += 2;
    }
    pts
}

/// 单个用例（一次完整管线运行）的观测数据
#[derive(Serialize)]
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
    /// 逐节点时间步长序列（遗忘观测核心数据：以节点为单位随时间步变化）
    pub node_series: Vec<NodeSeries>,
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
    /// MemoryId → graph.json 语义 id 反向表（观测/明细展示可读 id，不用随机 UUID）
    id_rev: std::collections::HashMap<MemoryId, String>,
}

impl ForgetPipelineSuite {
    /// 从 fixture graph JSON 加载真实角色图，并按环境变量启用 llama-server
    pub fn load(path: &Path) -> Result<Self, String> {
        let (graph, id_map) = load_graph_cluster(path)
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
            id_rev: build_reverse_id_map(&id_map),
        })
    }

    /// 仅加载图、不启用 LLM（测试 / 确定性验证使用）
    #[allow(dead_code)]
    pub fn load_without_llm(path: &Path) -> Result<Self, String> {
        let (graph, id_map) = load_graph_cluster(path)
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
            id_rev: build_reverse_id_map(&id_map),
        })
    }

    /// 仅激发测试（GUI 独立入口）：只加载 `excitation-*` 三个时机子场景
    /// （前置/均布/后置），不启用 LLM——激发测试为纯效果验证（E1~E6），
    /// 无需 LLM，保持确定性与快速响应。
    pub fn load_excitation_only(path: &Path) -> Result<Self, String> {
        let mut suite = Self::load_without_llm(path)?;
        suite.cases = PIPELINE_CASES
            .iter()
            .filter(|c| c.name.starts_with("excitation-"))
            .copied()
            .collect();
        Ok(suite)
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
    ///    的节点**全部**走 llama-server 修订（不抽样），其余走遮罩/降级；
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

        // LLM 修订集：可遮罩、缺失度 ≥ REVISE_THRESHOLD、**词数足够**的节点——
        // **全部**进入 LLM 补全（不抽样；测试时间不是约束，保证每个满足条件的节点都验证补全）。
        // 排除短文本——被全遮后无上下文，LLM 无法补全。
        let revise_set: std::collections::HashSet<_> = {
            let g = cluster.graph();
            g.node_indices()
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
                    display_id(&self.id_rev, n.note().id()),
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

        // ── 逐节点时间步长序列（单步用例：每个节点一个时间点，x=elapsed_hours）──
        let node_series: Vec<NodeSeries> = stats
            .iter()
            .map(|s| NodeSeries {
                id: s.id.clone(),
                type_name: s.type_name,
                original: s.original.clone(),
                steps: vec![NodeStepStat {
                    hours: spec.elapsed_hours.max(0),
                    step: 0,
                    md: s.md_after,
                    md_ctrl: None,
                    action: s.action,
                    masked_text: s.masked_text.clone(),
                    llm_reply: s.llm_reply.clone(),
                    effective: s.effective,
                }],
            })
            .collect();

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
            node_series,
            detail_lines,
            metrics,
        }
    }

    /// 多步遗忘：3 轮 × 24h，对图中**每个受遗忘影响的节点**逐步执行
    /// 衰减 → 遮罩 →（LLM 修订，全部满足条件者）补全，收集每一步的输入输出与缺失度轨迹。
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
        // 每节点的逐时间步序列（步 → 缺失度/动作/输入输出，供"节点 × 时间步"曲线）
        let mut series_steps: Vec<Vec<NodeStepStat>> = vec![Vec::new(); node_indices.len()];

        let mut passed = true;
        let mut per_step_avg_md: Vec<f32> = Vec::new();
        let mut per_step_hist: Vec<(String, Vec<(&'static str, usize)>)> = Vec::new();
        let mut detail_lines = Vec::new();
        let mut metrics: Vec<(String, String, String)> = Vec::new();
        let mut total_effective = 0usize;
        let mut total_revised = 0usize;

        for step in 0..STEPS {
            let now = t0 + ChronoDuration::hours(STEP_HOURS * ((step + 1) as i64));

            // 先全图刷新缺失度（lazy_forget 内部也会刷新，但修订集需要
            // 基于本轮的缺失度筛选）
            compute_all_missing_degrees(&mut cluster, now);

            // LLM 修订集（每轮重算：内容与缺失度都已演变）——全部满足条件的节点
            // 都进入补全，不抽样（测试时间不是约束）。
            let revise_set: std::collections::HashSet<_> = {
                let g = cluster.graph();
                g.node_indices()
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
                        display_id(&self.id_rev, n.note().id()),
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

                // 收集该节点本时间步的观测（供逐节点时间步曲线与数据点展开）
                series_steps[i].push(NodeStepStat {
                    hours: STEP_HOURS * ((step + 1) as i64),
                    step,
                    md: after,
                    md_ctrl: None,
                    action: action_name,
                    masked_text: masked_text.clone(),
                    llm_reply: llm_reply.clone(),
                    effective,
                });

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

        // 逐节点时间步长序列（final_nodes[i] 携带节点元信息，series_steps[i] 为轨迹）
        let node_series: Vec<NodeSeries> = final_nodes
            .iter()
            .enumerate()
            .map(|(i, f)| NodeSeries {
                id: f.id.clone(),
                type_name: f.type_name,
                original: f.original.clone(),
                steps: series_steps[i].clone(),
            })
            .collect();

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
            node_series,
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

            let short_id: String = display_id(&self.id_rev, n.note().id()).chars().take(8).collect();
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

        // 逐节点观测：激活次数不同 → 半衰期不同 → 缺失度不同（曲线差异的真实来源）。
        // 观测以记忆节点为单位，每个节点一个时间点（x=72h），md 随激活次数变化。
        let nodes: Vec<NodeForgetStat> = node_indices
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let n = g.node_weight(*idx).expect("node");
                let count = counts.get(&i).copied().unwrap_or(0);
                NodeForgetStat {
                    id: display_id(&self.id_rev, n.note().id()),
                    type_name: forget_type_name(n.note()),
                    original: get_summary(n.note()).unwrap_or_default(),
                    md_before: 0.0,
                    md_after: n.note().missing_degree(),
                    action: if count > 0 { "Activated" } else { "NoAction" },
                    mask: None,
                    masked_text: None,
                    llm_reply: None,
                    effective: false,
                }
            })
            .collect();
        let node_series: Vec<NodeSeries> = nodes
            .iter()
            .map(|s| NodeSeries {
                id: s.id.clone(),
                type_name: s.type_name,
                original: s.original.clone(),
                steps: vec![NodeStepStat {
                    hours: ELAPSED_HOURS,
                    step: 0,
                    md: s.md_after,
                    md_ctrl: None,
                    action: s.action,
                    masked_text: None,
                    llm_reply: None,
                    effective: false,
                }],
            })
            .collect();

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
            nodes,
            node_series,
            avg_edge_intensity: 0.0,
            detail_lines,
            metrics,
        }
    }

    /// 激发测试（黑盒效果）：验证"记忆被激发/提取后，遗忘被延缓"这一**可观察效果**。
    ///
    /// 设计原则：soul-tune 是效果测试框架，不读取算法内部常量（如
    /// `DEFAULT_ACTIVE_FACTOR` / 激活封顶值），不假设激发次数如何进入衰减公式，
    /// 只通过公开接口驱动与观测：
    /// - 激发：`MemoryNote::retrieval_increment()`；
    /// - 老化：`apply_aging` 统一回拨 `last_forget_time`（测试框架侧，不触碰算法逻辑）；
    /// - 观测：只读 `current_missing_degree`（不写回，全程同一条模拟时间轴）。
    ///
    /// 结构（三种时机子场景 [`ExcitationSchedule`] 各自独立跑一遍全部断言）：
    /// 图克隆两份（对照/实验）→ 同一 72h 老化 → 实验组按设计剂量梯度
    /// `{0,1,3,10,30,50,100}` 激发（固定种子洗牌分配，每个节点以自身为对照）→
    /// 每 2h 一个检查点（36 点）逐节点配对观测。
    ///
    /// 统计口径：
    /// - 只统计**参与遗忘**的节点（SemMemory / SpecificSituation），
    ///   Procedure 不参与遗忘机制，从一开始就排除；
    /// - 对照组（未激发基线）= 全体参与节点的未激发 md 平均；
    /// - 激发组效果 = 仅被激发（dose>0）节点的平均 md / 延缓。
    ///
    /// 断言（全部基于可观察效果）：
    /// - E1：被激发节点在 72h 检查点 `Δmd = md对照 − md实验 > 1e-3`（激发延缓遗忘）；
    /// - E2：未激发（dose=0）节点 `|Δmd| < 1e-4`（激发无全局副作用）；
    /// - E3：剂量组平均 md 随剂量单调下降；饱和点（50 与 100 次）效果相同（可观察封顶）；
    /// - E4：`0≤md≤1`、激发后同刻 md 不上涨；
    /// - E5：事件研究——已激发节点在激发后的检查点出现延缓，未激发的不出现；
    /// - E6：确定性（单元测试中两次运行结果一致）。
    ///
    /// 质量门槛：dose ≥ 3 的组平均 Δmd(72h) ≥ 0.05（防止"理论上延缓、效果微不可察"）。
    ///
    /// 前瞻性接口建议（暂不改动算法，仅记录）：`retrieval_increment()` 把
    /// `last_accessed_time` 写为真实 `Utc::now()`。当前衰减公式不使用该字段，
    /// 因此本测试仍完全确定；若将来实现"激活回鲜"语义开始使用该字段，
    /// soul-tune 应要求 core 层提供 `retrieve_at(DateTime<Utc>)` 保持模拟时钟可控。
    fn run_excitation_case(&self, schedule: ExcitationSchedule) -> ForgetCaseData {
        const ELAPSED_HOURS: i64 = 72;
        const DOSES: [usize; 7] = [0, 1, 3, 10, 30, 50, 100];
        const THRESHOLD_MD: f32 = 0.5; // 延缓指标：到达 md=0.5（"遗忘到一半"）的时间
        // 观测检查点：每 2h 一个（36 点），曲线平滑（计算瞬时完成）。
        // 关键展示点（detail 文本 / 事件研究摘要）单独定义，避免明细过长。
        const KEY_HOURS: [i64; 3] = [24, 48, 72];
        let checkpoints: Vec<i64> = (2..=ELAPSED_HOURS).step_by(2).collect();
        let last_idx = checkpoints.len() - 1; // 72h 检查点索引

        // 固定模拟时钟锚点：整个场景与真实时间无关，保证确定性（E6）
        let sim_now = match Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0) {
            chrono::LocalResult::Single(t) => t,
            _ => panic!("固定模拟时钟解析失败"),
        };

        // 图克隆两份，统一老化 72h（last_forget_time = sim_now − 72h，md 归零）
        let mut ctrl = self.graph.clone();
        let mut trt = self.graph.clone();
        self.apply_aging(&mut ctrl, sim_now, ELAPSED_HOURS);
        self.apply_aging(&mut trt, sim_now, ELAPSED_HOURS);

        // 只保留**参与遗忘**的节点（SemMemory / SpecificSituation）。
        // Procedure 类型不参与遗忘机制，从一开始就不纳入激发测试的统计与观测。
        let node_indices: Vec<_> = {
            let g = trt.graph();
            g.node_indices()
                .filter(|idx| {
                    is_maskable_type(forget_type_name(&g.node_weight(*idx).expect("node").note))
                })
                .collect()
        };
        let n = node_indices.len();

        // 设计剂量梯度：固定种子洗牌节点顺序后按剂量表循环分配（确定性、类型混合）
        let mut rng = StdRng::seed_from_u64(0xE7C1_5EED);
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let mut dose: Vec<usize> = vec![0; n];
        for (k, &i) in order.iter().enumerate() {
            dose[i] = DOSES[k % DOSES.len()];
        }

        // 节点元信息（id / 类型 / 原文），先取好避免后续反复借用图
        let ids: Vec<String> = node_indices
            .iter()
            .map(|idx| {
                display_id(
                    &self.id_rev,
                    trt.graph().node_weight(*idx).expect("node").note.id(),
                )
            })
            .collect();
        let type_names: Vec<&'static str> = node_indices
            .iter()
            .map(|idx| forget_type_name(&trt.graph().node_weight(*idx).expect("node").note))
            .collect();
        let originals: Vec<String> = node_indices
            .iter()
            .map(|idx| {
                get_summary(&trt.graph().node_weight(*idx).expect("node").note)
                    .unwrap_or_default()
            })
            .collect();

        // 激发计划：batches = [(激活时刻, [(节点序号, 本次激发次数), ...])]。
        // 激活时刻与观测检查点解耦：Early 全部在 t=0（首个检查点前应用），
        // Spaced 在 24/48/72h 分三批，Late 全部在 t=48h。
        let batches: Vec<(i64, Vec<(usize, usize)>)> = match schedule {
            ExcitationSchedule::Early => vec![(
                0,
                (0..n).filter(|&i| dose[i] > 0).map(|i| (i, dose[i])).collect(),
            )],
            ExcitationSchedule::Spaced => {
                let mut b24 = Vec::new();
                let mut b48 = Vec::new();
                let mut b72 = Vec::new();
                for i in 0..n {
                    let d = dose[i];
                    if d == 0 {
                        continue;
                    }
                    let (base, rem) = (d / 3, d % 3);
                    let (b1, b2, b3) = (
                        base + usize::from(rem >= 1),
                        base + usize::from(rem >= 2),
                        base,
                    );
                    if b1 > 0 {
                        b24.push((i, b1));
                    }
                    if b2 > 0 {
                        b48.push((i, b2));
                    }
                    if b3 > 0 {
                        b72.push((i, b3));
                    }
                }
                vec![(24, b24), (48, b48), (72, b72)]
                    .into_iter()
                    .filter(|(_, v)| !v.is_empty())
                    .collect()
            }
            ExcitationSchedule::Late => vec![(
                48,
                (0..n).filter(|&i| dose[i] > 0).map(|i| (i, dose[i])).collect(),
            )],
        };

        // 每节点首次激发时刻（用于 E5 事件研究："激发前无差异、激发后出现延缓"）
        let mut activated_at: Vec<Option<i64>> = vec![None; n];
        for (at, items) in &batches {
            for &(i, _) in items {
                if activated_at[i].is_none() {
                    activated_at[i] = Some(*at);
                }
            }
        }

        let mut passed = true;
        let mut detail_lines = Vec::new();
        let mut metrics: Vec<(String, String, String)> = Vec::new();
        // 每节点每检查点的只读观测（对照/实验）
        let mut ctrl_series: Vec<Vec<f32>> = vec![Vec::with_capacity(checkpoints.len()); n];
        let mut trt_series: Vec<Vec<f32>> = vec![Vec::with_capacity(checkpoints.len()); n];

        // 循环前：应用激活时刻早于首个检查点的批次（Early 的 t=0）。
        // t0 时刻所有节点 md=0，激发后仍为 0，E4 不涨断言恒成立；观测自首个检查点开始。
        {
            let t0 = sim_now - ChronoDuration::hours(ELAPSED_HOURS);
            let early: Vec<(usize, usize)> = batches
                .iter()
                .filter(|(at, _)| *at < checkpoints[0])
                .flat_map(|(_, items)| items.iter().copied())
                .collect();
            let md_before: Vec<f32> = early
                .iter()
                .map(|&(i, _)| {
                    current_missing_degree(
                        &trt.graph().node_weight(node_indices[i]).expect("node").note,
                        t0,
                    )
                })
                .collect();
            {
                let g = trt.graph_mut();
                for &(i, cnt) in &early {
                    let node = &mut g.node_weight_mut(node_indices[i]).expect("node").note;
                    for _ in 0..cnt {
                        node.retrieval_increment();
                    }
                }
            }
            for (j, &(i, _)) in early.iter().enumerate() {
                let after = current_missing_degree(
                    &trt.graph().node_weight(node_indices[i]).expect("node").note,
                    t0,
                );
                if after > md_before[j] + 1e-4 {
                    passed = false;
                    detail_lines.push(format!(
                        "E4失败: [{}] 激发后 md 上涨（t0）",
                        ids[i].chars().take(8).collect::<String>()
                    ));
                }
            }
        }

        for &t_hours in checkpoints.iter() {
            let t = sim_now - ChronoDuration::hours(ELAPSED_HOURS - t_hours);

            // 本时刻的激活批次（Spaced 24/48/72h、Late 48h；Early 已在循环前应用）
            let batch: Vec<(usize, usize)> = batches
                .iter()
                .filter(|(at, _)| *at == t_hours)
                .flat_map(|(_, items)| items.iter().copied())
                .collect();

            // E4 前置读：本批激发前，批次内节点的 md（只读）
            let md_before_batch: Vec<f32> = batch
                .iter()
                .map(|&(i, _)| {
                    current_missing_degree(
                        &trt.graph().node_weight(node_indices[i]).expect("node").note,
                        t,
                    )
                })
                .collect();

            // 施加本时刻的激发（公开接口；retrieval_increment 会把 last_accessed_time
            // 写为真实 Utc::now()——当前衰减公式不使用该字段，故不影响确定性，见函数文档）
            {
                let g = trt.graph_mut();
                for &(i, cnt) in &batch {
                    let node = &mut g.node_weight_mut(node_indices[i]).expect("node").note;
                    for _ in 0..cnt {
                        node.retrieval_increment();
                    }
                }
            }

            // E4 后置读：激发后同刻 md 不得上涨
            for (j, &(i, _)) in batch.iter().enumerate() {
                let after = current_missing_degree(
                    &trt.graph().node_weight(node_indices[i]).expect("node").note,
                    t,
                );
                if after > md_before_batch[j] + 1e-4 {
                    passed = false;
                    detail_lines.push(format!(
                        "E4失败: [{}] 激发后 md 上涨 {:.4} → {:.4}",
                        ids[i].chars().take(8).collect::<String>(),
                        md_before_batch[j],
                        after
                    ));
                }
            }

            // 逐节点配对观测 + E5 事件研究 + 不变量
            for (i, idx) in node_indices.iter().enumerate() {
                let md_c =
                    current_missing_degree(&ctrl.graph().node_weight(*idx).expect("node").note, t);
                let md_t =
                    current_missing_degree(&trt.graph().node_weight(*idx).expect("node").note, t);
                ctrl_series[i].push(md_c);
                trt_series[i].push(md_t);

                // 不变量：md 必须在 [0,1]；对照（无激发）随时间单调不减
                if !(0.0..=1.0).contains(&md_c) || !(0.0..=1.0).contains(&md_t) {
                    passed = false;
                }
                if ctrl_series[i].len() >= 2
                    && md_c + 1e-4 < ctrl_series[i][ctrl_series[i].len() - 2]
                {
                    passed = false;
                    detail_lines.push(format!(
                        "E4失败: [{}] 对照 md 随时间回退 {:.4} → {:.4}",
                        ids[i].chars().take(8).collect::<String>(),
                        ctrl_series[i][ctrl_series[i].len() - 2],
                        md_c
                    ));
                }

                // E5 事件研究：已激发节点出现延缓；未激发节点无差异
                let delta = md_c - md_t;
                let activated_by_now = activated_at[i].map_or(false, |at| at <= t_hours);
                if activated_by_now {
                    if delta <= 1e-3 {
                        passed = false;
                        detail_lines.push(format!(
                            "E5失败: [{}] dose={} 检查点{}h 已激发但 Δmd={:.4} ≤ 1e-3",
                            ids[i].chars().take(8).collect::<String>(),
                            dose[i],
                            t_hours,
                            delta
                        ));
                    }
                } else if delta.abs() > 1e-4 {
                    passed = false;
                    detail_lines.push(format!(
                        "E5失败: [{}] 检查点{}h 未激发但 Δmd={:.4}",
                        ids[i].chars().take(8).collect::<String>(),
                        t_hours,
                        delta
                    ));
                }
            }
        }

        // 持久化最终缺失度（仅用于 missing_degree() 读取一致；断言全部基于只读观测）
        compute_all_missing_degrees(&mut ctrl, sim_now);
        compute_all_missing_degrees(&mut trt, sim_now);

        // ── 剂量组统计（72h 检查点）──
        let mut groups: std::collections::BTreeMap<usize, (f32, f32, usize)> =
            std::collections::BTreeMap::new();
        for i in 0..n {
            let e = groups.entry(dose[i]).or_insert((0.0, 0.0, 0));
            e.0 += ctrl_series[i][last_idx];
            e.1 += trt_series[i][last_idx];
            e.2 += 1;
        }
        let mut dose_means: Vec<(usize, f32, f32)> = groups
            .iter()
            .map(|(d, (sc, st, cnt))| (*d, sc / *cnt as f32, st / *cnt as f32))
            .collect();
        dose_means.sort_by_key(|(d, _, _)| *d);

        // E1：每个被激发节点 72h Δmd > 1e-3
        let mut activated_count = 0usize;
        for i in 0..n {
            if dose[i] == 0 {
                continue;
            }
            activated_count += 1;
            let delta = ctrl_series[i][last_idx] - trt_series[i][last_idx];
            if delta <= 1e-3 {
                passed = false;
                detail_lines.push(format!(
                    "E1失败: [{}] dose={} Δmd(72h)={:.4} ≤ 1e-3",
                    ids[i].chars().take(8).collect::<String>(),
                    dose[i],
                    delta
                ));
            }
        }

        // E2：未激发节点无差异（激发无全局副作用）
        for i in 0..n {
            if dose[i] != 0 {
                continue;
            }
            let delta = (ctrl_series[i][last_idx] - trt_series[i][last_idx]).abs();
            if delta > 1e-4 {
                passed = false;
                detail_lines.push(format!(
                    "E2失败: [{}] 未激发但 Δmd(72h)={:.4}",
                    ids[i].chars().take(8).collect::<String>(),
                    delta
                ));
            }
        }

        // E3：剂量-反应单调（允许饱和）；饱和点（50 与 100）效果相同 = 可观察封顶
        for w in dose_means.windows(2) {
            let (da, _, ma) = w[0];
            let (db, _, mb) = w[1];
            if mb > ma + 1e-3 {
                passed = false;
                detail_lines.push(format!(
                    "E3失败: dose{} 平均 md {:.4} > dose{} {:.4}",
                    da, ma, db, mb
                ));
            }
            // 饱和点之前应严格递减（50 与 100 允许相同）
            if db <= 50 && mb >= ma - 1e-3 {
                passed = false;
                detail_lines.push(format!(
                    "E3失败: dose{}→{} 平均 md 未严格递减 {:.4} → {:.4}",
                    da, db, ma, mb
                ));
            }
        }
        if let (Some(g50), Some(g100)) = (groups.get(&50), groups.get(&100)) {
            let (m50, m100) = (g50.1 / g50.2 as f32, g100.1 / g100.2 as f32);
            if (m50 - m100).abs() > 1e-3 {
                passed = false;
                detail_lines.push(format!(
                    "E3失败: 封顶 dose50={:.4} vs dose100={:.4} 应相同",
                    m50, m100
                ));
            }
        }

        // 质量门槛：dose ≥ 3 组平均 Δmd(72h) ≥ 0.05
        for (d, sc, st) in &dose_means {
            if *d >= 3 {
                let delta = sc - st;
                if delta < 0.05 {
                    passed = false;
                    detail_lines.push(format!(
                        "质量门槛失败: dose={} 平均 Δmd={:.4} < 0.05",
                        d, delta
                    ));
                }
            }
        }

        // ── 延缓指标（时间域）：到达 md=0.5 的时间（线性插值），延缓 = 实验 − 对照 ──
        let mut delays: Vec<Option<f32>> = Vec::with_capacity(n);
        let mut trt_crossed: Vec<bool> = Vec::with_capacity(n);
        for i in 0..n {
            let t_ctrl = crossing_time(&ctrl_series[i], &checkpoints, THRESHOLD_MD);
            let t_trt = crossing_time(&trt_series[i], &checkpoints, THRESHOLD_MD);
            trt_crossed.push(t_trt.is_some());
            match (t_ctrl, t_trt) {
                (Some(a), Some(b)) => delays.push(Some(b - a)),
                // 实验组未达阈值：报告窗口内下限（对照已越过阈值）
                (Some(a), None) => delays.push(Some(ELAPSED_HOURS as f32 - a)),
                _ => delays.push(None),
            }
        }

        // ── 汇总与报告 ──
        // metrics 只放**总体指标**：主视觉（对照/激发平均曲线）由观测数据在 UI 端
        // 聚合，这里只出 4 个关键数值；逐剂量组 / 事件研究 / 逐节点明细下沉 detail_lines。
        let mut hist: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        hist.insert("Activated", activated_count);
        hist.insert("Control", n - activated_count);

        // 口径：对照组平均 = 全体参与遗忘节点的未激发 md；激发组平均 = 仅被激发节点。
        let act_idxs: Vec<usize> = (0..n).filter(|&i| dose[i] > 0).collect();
        let avg_ctrl_md = ctrl_series.iter().map(|s| s[last_idx]).sum::<f32>() / n.max(1) as f32;
        let avg_trt_md = act_idxs
            .iter()
            .map(|&i| trt_series[i][last_idx])
            .sum::<f32>()
            / act_idxs.len().max(1) as f32;

        metrics.push((
            "激发测试".into(),
            "72h 平均缺失度 对照/激发".into(),
            format!(
                "{:.3}/{:.3}（Δ{:.3}，{}个被激发节点）",
                avg_ctrl_md,
                avg_trt_md,
                avg_ctrl_md - avg_trt_md,
                act_idxs.len()
            ),
        ));
        let hs: Vec<f32> = act_idxs.iter().filter_map(|&i| delays[i]).collect();
        if !hs.is_empty() {
            let mean = hs.iter().sum::<f32>() / hs.len() as f32;
            let min = hs.iter().cloned().fold(f32::INFINITY, f32::min);
            metrics.push((
                "激发测试".into(),
                "平均延缓(md→0.5，均值/最小)".into(),
                format!("{:.1}h / {:.1}h", mean, min),
            ));
        }
        if let (Some(g50), Some(g100)) = (groups.get(&50), groups.get(&100)) {
            metrics.push((
                "激发测试".into(),
                "封顶 dose50 vs dose100".into(),
                format!(
                    "Δmd={:.4}（应≈0：可观察封顶）",
                    (g50.1 / g50.2 as f32 - g100.1 / g100.2 as f32).abs()
                ),
            ));
        }

        // ── 观测明细（detail_lines）：剂量组摘要 → 事件研究摘要 → 逐节点 ──
        for (d, sc, st) in &dose_means {
            let cnt = groups.get(d).map(|e| e.2).unwrap_or(0);
            detail_lines.push(format!(
                "剂量组 dose={}: 平均缺失度 对照 {:.3} / 激发 {:.3}（Δ{:.3}，{}节点）",
                d, sc, st, sc - st, cnt
            ));
        }
        for d in [1usize, 3, 10, 30, 50, 100] {
            let idxs: Vec<usize> = (0..n).filter(|&i| dose[i] == d).collect();
            if idxs.is_empty() {
                continue;
            }
            let hs: Vec<f32> = idxs.iter().filter_map(|&i| delays[i]).collect();
            if hs.is_empty() {
                continue;
            }
            let mean = hs.iter().sum::<f32>() / hs.len() as f32;
            let min = hs.iter().cloned().fold(f32::INFINITY, f32::min);
            let unreached = idxs.iter().filter(|&&i| !trt_crossed[i]).count();
            detail_lines.push(format!(
                "剂量组 dose={}: 延缓(md→0.5) 均值 {:.1}h / 最小 {:.1}h（{}节点，激发组未达阈值{}个）",
                d, mean, min, idxs.len(), unreached
            ));
        }
        // 事件研究摘要：只展示关键检查点（24/48/72h），避免明细过长
        for &t_hours in KEY_HOURS.iter() {
            let Some(k) = checkpoints.iter().position(|h| *h == t_hours) else {
                continue;
            };
            let mut sum_act = 0.0f32;
            let mut cnt_act = 0usize;
            let mut sum_inact = 0.0f32;
            let mut cnt_inact = 0usize;
            for i in 0..n {
                let delta = (ctrl_series[i][k] - trt_series[i][k]).abs();
                if activated_at[i].map_or(false, |at| at <= t_hours) {
                    sum_act += delta;
                    cnt_act += 1;
                } else {
                    sum_inact += delta;
                    cnt_inact += 1;
                }
            }
            detail_lines.push(format!(
                "事件研究 t={}h：已激发节点平均|Δmd| {:.4}（{}节点）· 未激发节点平均|Δmd| {:.4}（{}节点）",
                t_hours,
                if cnt_act > 0 { sum_act / cnt_act as f32 } else { 0.0 },
                cnt_act,
                if cnt_inact > 0 { sum_inact / cnt_inact as f32 } else { 0.0 },
                cnt_inact,
            ));
        }

        // 逐节点明细：配对对照、激发时刻、延缓时长（md 只展示关键检查点）
        for i in 0..n {
            let act_txt = match activated_at[i] {
                Some(at) => format!("激发@{}h", at),
                None => "未激发".into(),
            };
            let delay_txt = match (delays[i], trt_crossed[i]) {
                (Some(h), true) => format!("延缓{:.1}h", h),
                (Some(h), false) => format!("延缓≥{:.1}h(激发组未达md=0.5)", h),
                _ => "—".into(),
            };
            let fmt_series = |series: &[f32]| {
                KEY_HOURS
                    .iter()
                    .filter_map(|h| {
                        checkpoints
                            .iter()
                            .position(|c| c == h)
                            .map(|idx| format!("{}h:{:.3}", h, series[idx]))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let ctrl_txt = fmt_series(&ctrl_series[i]);
            let trt_txt = fmt_series(&trt_series[i]);
            detail_lines.push(format!(
                "[{}] dose={:<3} {} | md对照 {} | md激发 {} | {}",
                ids[i].chars().take(8).collect::<String>(),
                dose[i],
                act_txt,
                ctrl_txt,
                trt_txt,
                delay_txt
            ));
        }

        let nodes: Vec<NodeForgetStat> = (0..n)
            .map(|i| NodeForgetStat {
                id: ids[i].clone(),
                type_name: type_names[i],
                original: originals[i].clone(),
                md_before: ctrl_series[i][last_idx], // 对照（未激发）作为基线
                md_after: trt_series[i][last_idx],
                // 激发测试不执行遮罩/LLM：动作只区分 激发组(Activated) / 对照组(Control)
                action: if dose[i] > 0 { "Activated" } else { "Control" },
                mask: None,
                masked_text: None,
                llm_reply: None,
                effective: false,
            })
            .collect();

        let node_series: Vec<NodeSeries> = (0..n)
            .map(|i| NodeSeries {
                id: ids[i].clone(),
                type_name: type_names[i],
                original: originals[i].clone(),
                steps: (0..checkpoints.len())
                    .map(|k| NodeStepStat {
                        hours: checkpoints[k],
                        step: k,
                        md: trt_series[i][k], // 激发组（y 主曲线）
                        md_ctrl: Some(ctrl_series[i][k]), // 对照组（配对对照曲线）
                        action: if activated_at[i].map_or(false, |at| at <= checkpoints[k]) {
                            "Activated"
                        } else {
                            "Control"
                        },
                        masked_text: None,
                        llm_reply: None,
                        effective: false,
                    })
                    .collect(),
            })
            .collect();

        let max_md = trt_series.iter().map(|s| s[last_idx]).fold(0.0f32, f32::max);

        ForgetCaseData {
            case_name: format!("excitation-{}", schedule.tag()),
            passed,
            llm_available: false,
            node_count: n,
            edge_count: {
                let g = trt.graph();
                g.edge_count()
            },
            llm_revised: 0,
            effective_revised: 0,
            action_histogram: hist.into_iter().collect(),
            avg_missing_degree: avg_trt_md,
            max_missing_degree: max_md,
            avg_masked_ratio: 0.0,
            avg_edge_intensity: 0.0,
            nodes,
            node_series,
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
            node_series: vec![],
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
        } else if spec.elapsed_hours == -12 {
            self.run_excitation_case(ExcitationSchedule::Early)
        } else if spec.elapsed_hours == -13 {
            self.run_excitation_case(ExcitationSchedule::Spaced)
        } else if spec.elapsed_hours == -14 {
            self.run_excitation_case(ExcitationSchedule::Late)
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
        let mut metrics: Vec<crate::engine::suite::MetricEntry> = Vec::new();
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
                metrics.push(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                ));
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
            metrics.push(chart_metric(
                "遗忘衰减曲线".to_string(),
                "遗忘缺失度".to_string(),
                "时间跨度(小时)".to_string(),
                "平均缺失度".to_string(),
                vec![Series {
                    label: "平均缺失度".to_string(),
                    points: decay_points,
                }],
            ));
        }

        metrics.push(key_value_metric(
            "图".to_string(),
            "图".to_string(),
            format!(
                "{}（节点 {} / 边 {}）",
                self.graph_name, max_node_count, max_edge_count
            ),
        ));
        metrics.push(key_value_metric(
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
        ));
        metrics.push(key_value_metric(
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
        ));

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

    // ── 阶段 2：Revise（无 LLM 时用例失败但不 panic）──

    #[test]
    fn test_revise_suite_loads_fixture_samples() {
        let suite =
            ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        assert!(suite.llm.is_none(), "测试环境不应配置 LLM");
        assert!(!suite.samples.is_empty(), "全量模式应覆盖全部可遗忘节点");
        // 全量覆盖：可遗忘节点（SemMemory/SpecificSituation）全部进入
        let (cluster, _) = load_graph_cluster(&fixture_graph()).expect("加载 fixture 图");
        let maskable = cluster
            .graph()
            .node_weights()
            .filter(|n| is_maskable(n.note()) && !get_summary(n.note()).unwrap_or_default().trim().is_empty())
            .count();
        assert_eq!(
            suite.samples.len(),
            maskable * REVISE_MASK_GRADIENTS.len(),
            "全量模式应覆盖全部可遗忘节点 × 全梯度（{} × {}）",
            maskable,
            REVISE_MASK_GRADIENTS.len()
        );
        let jieba = Jieba::new();
        for s in &suite.samples {
            assert!(!s.original.trim().is_empty(), "样本原文不应为空");
            assert!(
                REVISE_MASK_GRADIENTS.contains(&s.mask_md),
                "样本应带合法遮罩梯度"
            );
            // 极短文本（如单字节点）在低梯度下 round(md×词数)=0，mask 模块不遮罩
            // （返回原文，无占位符）——属正确行为；有遮罩时必含占位符
            let words = mask_word_count(&jieba, &s.original);
            let expect_mask = (s.mask_md * words as f32).round() as usize > 0;
            if expect_mask {
                assert!(s.masked.contains(MASK_WORD.trim()), "样本应含遮罩");
            }
        }
    }

    #[test]
    fn test_revise_sampled_stays_within_budget() {
        // 抽样模式：约 8 个节点 × 全梯度、可复现（固定种子）、每类都有代表
        let a = ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let all_types: std::collections::HashSet<&'static str> =
            a.samples.iter().map(|s| s.type_name).collect();
        let s1 =
            ForgetReviseSuite::load_with_mode(&fixture_graph(), ReviseMode::Sampled(42))
                .expect("加载 fixture 图");
        let s2 =
            ForgetReviseSuite::load_with_mode(&fixture_graph(), ReviseMode::Sampled(42))
                .expect("加载 fixture 图");
        assert!(
            s1.samples.len() <= REVISE_MAX_SAMPLES * REVISE_MASK_GRADIENTS.len()
                && s1.samples.len() > 0,
            "抽样应约 {} 节点 × {} 梯度，实际 {}",
            REVISE_MAX_SAMPLES,
            REVISE_MASK_GRADIENTS.len(),
            s1.samples.len()
        );
        // 每类可遗忘节点至少 1 个代表
        let sampled_types: std::collections::HashSet<&'static str> =
            s1.samples.iter().map(|s| s.type_name).collect();
        for t in &all_types {
            assert!(sampled_types.contains(t), "抽样缺少类型 {t} 的代表");
        }
        // 固定种子可复现（节点 × 梯度 序列一致）
        let ids1: Vec<(String, u32)> = s1
            .samples
            .iter()
            .map(|s| (s.node_id.clone(), s.mask_md.to_bits()))
            .collect();
        let ids2: Vec<(String, u32)> = s2
            .samples
            .iter()
            .map(|s| (s.node_id.clone(), s.mask_md.to_bits()))
            .collect();
        assert_eq!(ids1, ids2, "固定种子抽样应可复现");
    }

    #[test]
    fn test_revise_case_fails_without_llm() {
        let suite =
            ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let outcome = suite.run_case(0); // 第一个样本（无 probe）
        assert!(!outcome.passed, "无 LLM 时补全用例应失败");
        assert_ne!(outcome.case_name, "forget/revise/probe", "probe 应已移除");
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
    fn test_pipeline_excitation_delays_forgetting() {
        // 激发测试（黑盒效果）：三种时机子场景各自独立验证"激发 → 遗忘被延缓"
        let suite =
            ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        for s in [
            ExcitationSchedule::Early,
            ExcitationSchedule::Spaced,
            ExcitationSchedule::Late,
        ] {
            let data = suite.run_excitation_case(s);
            assert!(data.passed, "激发测试 {:?} 失败", s);
            assert!(!data.nodes.is_empty(), "激发测试应覆盖全图节点");
            assert!(!data.detail_lines.is_empty());
            let hist = &data.action_histogram;
            assert!(
                hist.iter().any(|(k, v)| *k == "Activated" && *v > 0),
                "应存在被激发节点"
            );
            assert!(
                hist.iter().any(|(k, v)| *k == "Control" && *v > 0),
                "应存在未激发对照组节点"
            );
            // 延缓指标已产出（时间域：到达 md=0.5）
            assert!(
                data.metrics.iter().any(|(_, label, _)| label.contains("延缓")),
                "应产出延缓指标"
            );
        }
    }

    #[test]
    fn test_pipeline_excitation_deterministic() {
        // E6：同一场景两次运行结果完全一致（固定模拟时钟 + 固定种子）
        let suite =
            ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let a = suite.run_excitation_case(ExcitationSchedule::Spaced);
        let b = suite.run_excitation_case(ExcitationSchedule::Spaced);
        assert_eq!(a.passed, b.passed);
        assert_eq!(a.node_count, b.node_count);
        for (x, y) in a.nodes.iter().zip(b.nodes.iter()) {
            assert!(
                (x.md_after - y.md_after).abs() < 1e-6,
                "非确定性: {} md 两次运行 {:.6} vs {:.6}",
                x.id,
                x.md_after,
                y.md_after
            );
        }
    }

    #[test]
    fn test_pipeline_excitation_only_loads_three_cases() {
        // GUI 独立模式入口（api.rs mode="excitation"）：只加载 excitation-* 三个
        // 时机子场景，不启用 LLM，全部通过
        let suite =
            ForgetPipelineSuite::load_excitation_only(&fixture_graph()).expect("加载 fixture 图");
        assert_eq!(suite.cases.len(), 3, "应只加载 3 个激发用例");
        assert!(
            suite.cases.iter().all(|c| c.name.starts_with("excitation-")),
            "用例应全部为 excitation-*"
        );
        assert!(suite.llm.is_none(), "激发测试不应启用 LLM");
        for i in 0..suite.case_count() {
            let outcome = suite.run_case(i);
            assert!(
                outcome.passed,
                "激发用例 {} 失败: {}",
                outcome.case_name,
                outcome.description
            );
        }
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
            if matches!(
                data.case_name.as_str(),
                "low"
                    | "medium"
                    | "high"
                    | "multi-step"
                    | "excitation-early"
                    | "excitation-spaced"
                    | "excitation-late"
            ) {
                assert!(
                    !data.nodes.is_empty(),
                    "{} 的节点数据为空（观测页将显示 0/0）",
                    o.case_name
                );
            }
        }
    }
}
