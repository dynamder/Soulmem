//! 遗忘算法测试套件。
//!
//! soul-tune 本身是 SoulMem 的测试框架，这里不再复刻算法分支里的单元小测试，
//! 而是**直接驱动具体的遗忘算法管线**，且：
//!
//! - **数据**：从 fixtures 读取真实角色记忆图（`graph.json`），逐节点/逐边驱动遗忘；
//! - **LLM**：复用 soul-tune 自身的 `LlamaServer`（llama.cpp server），
//!   与 playtest 一致的 `SOUL_TUNE_LLAMA_URL`（直连）或
//!   `SOUL_TUNE_CANDLE_MODEL_PATH`（自动拉起）约定；
//! - **输出**：`Revised` 补全结果中直接贴出 LLM 的**原始回复**与遮罩输入。
//!
//! 管线调用（与检索侧的惰性遗忘调用路径一致）：
//! 1. [`compute_all_missing_degrees`] —— 全图批量刷新节点与边的缺失度（增量公式）；
//! 2. [`lazy_forget`] —— 对可遮罩节点（SemMemory / SpecificSituation）执行
//!    衰减 → 分词遮罩 →（LLM 可用时）llama-server 推测修订；不可用时验证降级为 MaskOnly；
//! 3. [`decay_graph_edge`] / [`weight_placeholder`] —— 边独立衰减与检索权重占位；
//! 4. [`update_missing_degree_incremental`] —— 增量缺失度与从头计算的一致性。
//!
//! 5 个场景用例对同一张加载图施加不同的模拟老化时间跨度，
//! 产出逐节点明细（含 LLM 原始回复）与聚合指标（缺失度/遮罩率/动作分布/边强度/衰减曲线）。

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use jieba_rs::Jieba;

use soul_mem_algo::algo::forget::decay_calculator::{
    compute_missing_degree, update_missing_degree_incremental, DEFAULT_MAX_ACTIVATION_CAP,
};
use soul_mem_algo::algo::forget::decay_revise::{
    compute_all_missing_degrees, current_missing_degree, decay_graph_edge, get_summary,
    lazy_forget, weight_placeholder, ForgetAction, DEFAULT_ACTIVE_FACTOR,
    DEFAULT_BASE_HALF_LIFE_HOURS, REVISE_THRESHOLD,
};
use soul_mem_algo::algo::forget::mask::MASK_WORD;
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
// LLM 调用闭包（与算法侧 lazy_forget 的签名对齐，后端为 soul-tune 的 llama-server）
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

/// 每个 LLM 用例最多修订的节点数（按缺失度降序抽样，控制本地推理耗时）
const MAX_LLM_REVISIONS: usize = 8;

/// 使用 soul-tune 的 `LlamaServer`（与 playtest 相同后端）的闭包
///
/// 注意：`LlmBackend::chat` 是阻塞调用（reqwest::blocking），必须通过
/// `spawn_blocking` 移到 tokio 阻塞线程池执行——直接在 `block_on` 的
/// 异步上下文中调用会让 reqwest 内部创建的 runtime 在异步上下文里被
/// drop，触发 "Cannot drop a runtime in a context where blocking is not allowed"。
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

// ========================================================================
// 测试场景定义
// ========================================================================

/// 遗忘场景：对加载的真实角色图施加指定时间跨度（距上次遗忘操作的小时数）后跑真实管线
#[derive(Debug, Clone, Copy)]
pub struct ForgetCaseSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// 模拟老化小时数（距上次遗忘操作/访问），决定遗忘缺失度
    pub elapsed_hours: i64,
    /// 是否允许调用 llama-server（不可用时自动降级为遮罩）
    pub want_llm: bool,
}

/// 内置场景集：低/中/高遗忘强度 + 混合时间跨度 + 增量一致性
pub const BUILTIN_CASES: [ForgetCaseSpec; 5] = [
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
        description: "高遗忘强度（Δt=72h）：llama-server 修订高缺失度节点（抽样）",
        elapsed_hours: 72,
        want_llm: true,
    },
    ForgetCaseSpec {
        name: "mixed",
        description: "混合时间跨度（按节点 8/24/72h）：真实分布场景",
        elapsed_hours: -1, // 特殊标记：按节点使用自定义跨度
        want_llm: true,
    },
    ForgetCaseSpec {
        name: "incremental",
        description: "增量一致性：两次 12h 增量更新 == 一次 24h 全量计算",
        elapsed_hours: -2, // 特殊标记：增量一致性场景
        want_llm: false,
    },
];

// ========================================================================
// 用例可观测数据（放入 TestCaseOutcome.data）
// ========================================================================

/// 单个节点的遗忘观测结果
pub struct NodeForgetStat {
    pub id: String,
    pub type_name: &'static str,
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
}

/// 单个用例（一次完整管线运行）的观测数据
pub struct ForgetCaseData {
    pub case_name: String,
    pub passed: bool,
    pub llm_available: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub llm_revised: usize,
    pub action_histogram: Vec<(&'static str, usize)>,
    pub avg_missing_degree: f32,
    pub max_missing_degree: f32,
    pub avg_masked_ratio: f32,
    pub avg_edge_intensity: f64,
    pub detail_lines: Vec<String>,
    /// 供 build_report 汇总的键值对（组, 标签, 值）
    pub metrics: Vec<(String, String, String)>,
}

// ========================================================================
// 遗忘测试套件
// ========================================================================

/// 遗忘 LLM 修订 system prompt（记忆重建角色，与算法默认一致）
const FORGET_SYSTEM_PROMPT: &str = "You are a memory reconstruction assistant. \
    A segment of memory text has been partially masked, with [masked] placeholders. \
    Based on the context and the remaining fragments, infer and complete the missing parts \
    naturally. Output only the completed text, no explanation.";

pub struct ForgetSuite {
    /// 从 fixtures 加载的真实角色记忆图（蓝图，每个用例克隆后施加老化）
    graph: MemoryCluster,
    graph_name: String,
    jieba: Jieba,
    /// llama-server 后端（与 playtest 一致）；None → 遮罩降级路径
    llm: Option<Arc<Mutex<LlamaServer>>>,
    cases: Vec<ForgetCaseSpec>,
}

impl ForgetSuite {
    /// 从 fixture graph JSON 加载真实角色图，并按环境变量启用 llama-server。
    ///
    /// - `SOUL_TUNE_LLAMA_URL` 已设置 → 直连正在运行的 llama-server；
    /// - 否则使用 `SOUL_TUNE_CANDLE_MODEL_PATH` 指定的 GGUF 自动拉起 llama-server
    ///   （与 playtest 的 `LlamaServer::load` 约定完全一致）；
    /// - 两者皆不可用 → 无 LLM，验证遮罩降级路径。
    pub fn load(path: &Path) -> Result<Self, String> {
        let (graph, _id_map) = load_graph_cluster(path)
            .map_err(|e| format!("加载图 '{}' 失败: {}", path.display(), e))?;
        let graph_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let llm = Self::try_create_llm();
        Ok(Self {
            graph,
            graph_name,
            jieba: Jieba::new(),
            llm,
            cases: BUILTIN_CASES.to_vec(),
        })
    }

    /// 仅加载图、不启用 LLM（测试 / 确定性验证使用）
    #[allow(dead_code)] // 测试与外部调用方使用；非测试构建下无引用
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
            cases: BUILTIN_CASES.to_vec(),
        })
    }

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

    /// 模拟老化：把整张图的节点/边统一回拨 `last_forget_time`（缺失度归零），
    /// 使真实 fixture 图在指定时间跨度下产生遗忘。
    /// `mixed_offsets` 非空时按节点序号 i%3 轮转使用 8/24/72h。
    fn apply_aging(
        &self,
        cluster: &mut MemoryCluster,
        now: DateTime<Utc>,
        hours: i64,
        mixed_offsets: Option<[(usize, i64); 3]>,
    ) {
        let g = cluster.graph_mut();
        let node_indices: Vec<_> = g.node_indices().collect();
        for (i, idx) in node_indices.iter().enumerate() {
            let h = mixed_offsets
                .and_then(|offs| offs.iter().find(|(j, _)| *j == i % 3).map(|(_, h)| *h))
                .unwrap_or(hours.max(1));
            let t = now - ChronoDuration::hours(h);
            let n = g.node_weight_mut(*idx).expect("node");
            n.note.set_last_forget_time(t);
            n.note.set_missing_degree(0.0);
        }
        let edge_hours = if hours > 0 { hours } else { 24 };
        let t_edge = now - ChronoDuration::hours(edge_hours);
        for ei in g.edge_indices().collect::<Vec<_>>() {
            let l = g.edge_weight_mut(ei).expect("edge");
            l.set_last_forget_time(t_edge);
            l.set_missing_degree(0.0);
        }
    }

    /// 运行一次完整的真实遗忘管线（与检索侧的惰性遗忘调用路径一致）：
    /// 1. 克隆加载图并施加老化；
    /// 2. `compute_all_missing_degrees` 全图批量刷新节点+边缺失度；
    /// 3. 逐节点 `lazy_forget`：可遮罩节点中缺失度最高的前
    ///    [`MAX_LLM_REVISIONS`] 个走 llama-server 修订，其余走遮罩/降级；
    /// 4. `decay_graph_edge` 边独立衰减 + `weight_placeholder` 检索权重。
    fn run_pipeline(&self, spec: &ForgetCaseSpec) -> ForgetCaseData {
        let now = Utc::now();
        let mut cluster = self.graph.clone();
        let mixed_offsets = if spec.elapsed_hours == -1 {
            Some([(0usize, 8i64), (1, 24), (2, 72)])
        } else {
            None
        };
        self.apply_aging(&mut cluster, now, spec.elapsed_hours, mixed_offsets);

        let use_llm = self.llm.is_some() && spec.want_llm;

        // ── 步骤 1：全图批量刷新缺失度（节点 + 边）──
        compute_all_missing_degrees(&mut cluster, now);

        // LLM 修订抽样：可遮罩且缺失度 ≥ REVISE_THRESHOLD 的节点，按缺失度降序取前 N
        let revise_set: std::collections::HashSet<_> = {
            let g = cluster.graph();
            let mut candidates: Vec<_> = g
                .node_indices()
                .filter(|idx| {
                    let n = g.node_weight(*idx).expect("node");
                    is_maskable(n.note()) && n.note().missing_degree() >= REVISE_THRESHOLD
                })
                .map(|idx| (idx, g.node_weight(idx).expect("node").note().missing_degree()))
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

            // 惰性遗忘（内部先刷新缺失度，再按阈值执行遮罩 / LLM）
            let (action, after, orig_words) = {
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
                (act, after, orig_words)
            };

            // 动作分类、遮罩统计与 LLM 原始回复
            let (action_name, masked_words_this, mask_ratio_this, llm_reply, masked_text) =
                match &action {
                    ForgetAction::NoAction => ("NoAction", 0usize, None, None, None),
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
                        (
                            "Revised",
                            masked,
                            ratio,
                            Some(new_summary.clone()),
                            Some(masked_text.clone()),
                        )
                    }
                };

            if is_maskable_type(type_name) {
                if orig_words > 0 {
                    masked_words_total += masked_words_this;
                    total_words_total += orig_words;
                }
                // 文本足够长时：遮罩比例应接近缺失度（±15% 容差）；
                // 过短文本取整误差大，跳过比例校验
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
                md_before: before,
                md_after: after,
                action: action_name,
                mask: mask_ratio_this
                    .map(|r| ((r * orig_words as f32).round() as usize, orig_words)),
                masked_text,
                llm_reply,
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
                "{} [{}] md {:.3}→{:.3} 动作={}{}",
                s.type_name, short_id, s.md_before, s.md_after, s.action, mask_txt
            ));
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
                format!("{} llama-server 修订数(抽样)", spec.name),
                llm_revised.to_string(),
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
            action_histogram: hist_vec,
            avg_missing_degree: avg_md,
            max_missing_degree: max_md,
            avg_masked_ratio,
            avg_edge_intensity,
            detail_lines,
            metrics,
        }
    }

    /// 增量一致性场景：两次 12h 增量更新 == 一次 24h 全量计算（同一公式，误差 < 1e-3）
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

        // 全量：一次 24h（起点为 24h 前、初始缺失度 0）
        let mut full = make(24);
        let full_md = compute_and_update(&mut full, now);
        // 增量：先 12h，再 12h（起点与全量完全一致）
        let mut inc = make(24);
        let mid = now - ChronoDuration::hours(12);
        let mid_md = compute_and_update(&mut inc, mid);
        let inc_md = compute_and_update(&mut inc, now);

        // 参考：从创建时间直接计算的全程缺失度（仅展示）
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
            action_histogram: vec![("NoAction", 1)],
            avg_missing_degree: inc_md,
            max_missing_degree: inc_md,
            avg_masked_ratio: 0.0,
            avg_edge_intensity: 0.0,
            detail_lines,
            metrics,
        }
    }
}

impl Default for ForgetSuite {
    fn default() -> Self {
        // 无默认图：测试请通过 load / load_without_llm 加载 fixture 图
        panic!("ForgetSuite 必须通过 load / load_without_llm 加载 fixture 图")
    }
}

impl TestSuite for ForgetSuite {
    fn case_count(&self) -> usize {
        self.cases.len()
    }

    fn run_case(&self, index: usize) -> TestCaseOutcome {
        let spec = &self.cases[index];
        let data = if spec.elapsed_hours == -2 {
            self.run_incremental_case()
        } else {
            self.run_pipeline(spec)
        };
        let passed = data.passed;
        TestCaseOutcome {
            case_name: format!("forget/{}", spec.name),
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
        let mut max_node_count = 0usize;
        let mut max_edge_count = 0usize;

        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<ForgetCaseData>() else {
                continue;
            };
            llm_available |= data.llm_available;
            total_llm_revised += data.llm_revised;
            max_node_count = max_node_count.max(data.node_count);
            max_edge_count = max_edge_count.max(data.edge_count);
            for (group, label, value) in &data.metrics {
                metrics.push(Box::new(key_value_metric(
                    label.clone(),
                    group.clone(),
                    value.clone(),
                )));
            }
            // 衰减曲线：低/中/高场景的 (Δt, 平均缺失度)
            let hours = match data.case_name.as_str() {
                "low" => Some(8.0),
                "medium" => Some(24.0),
                "high" => Some(72.0),
                _ => None,
            };
            if let Some(h) = hours {
                decay_points.push((h, data.avg_missing_degree as f64));
            }
            // 用例汇总行
            let hist_txt: String = data
                .action_histogram
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ");
            detail_rows.push(DetailRow {
                text: format!(
                    "[{}] 节点{} 边{} | 缺失度均值{:.3}/最大{:.3} | 遮罩率{:.1}% | 边强度{:.3} | LLM修订{} | {}",
                    data.case_name,
                    data.node_count,
                    data.edge_count,
                    data.avg_missing_degree,
                    data.max_missing_degree,
                    data.avg_masked_ratio * 100.0,
                    data.avg_edge_intensity,
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
                format!("是（llama-server，修订 {} 节点）", total_llm_revised)
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
                "遗忘管线逐节点明细（图 {}：节点 {} / 边 {}，通过 {}/{}，耗时 {:.2}s）",
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

fn is_maskable(note: &MemoryNote) -> bool {
    matches!(
        note.mem_type(),
        MemoryType::Situation(SituationType::SpecificSituation(_)) | MemoryType::Semantic(_)
    )
}

fn is_maskable_type(type_name: &'static str) -> bool {
    matches!(type_name, "SemMemory" | "SpecificSituation")
}

fn forget_type_name(node: &MemoryNote) -> &'static str {
    match node.mem_type() {
        MemoryType::Semantic(_) => "SemMemory",
        MemoryType::Situation(SituationType::SpecificSituation(_)) => "SpecificSituation",
        MemoryType::Procedure(_) => "Procedure",
        _ => "Other",
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

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_algo::algo::forget::mask::mask_text;
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

    #[test]
    fn test_loads_real_fixture_graph() {
        let suite = ForgetSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        assert!(suite.llm.is_none());
        let node_count = suite.graph.graph().node_count();
        let edge_count = suite.graph.graph().edge_count();
        assert!(node_count > 10, "真实图应有足够节点，实际 {}", node_count);
        assert!(edge_count > 0, "真实图应有边，实际 {}", edge_count);
    }

    #[test]
    fn test_suite_has_five_cases() {
        let suite = ForgetSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        assert_eq!(suite.case_count(), 5);
    }

    #[test]
    fn test_all_cases_pass_without_llm() {
        let suite = ForgetSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        assert!(suite.llm.is_none(), "测试环境不应配置 LLM");
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
    fn test_report_builds_metrics_and_rows() {
        let suite = ForgetSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
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
    fn test_mask_ratio_matches_missing_degree() {
        // 直接驱动真实 mask 模块：遮罩比例应接近缺失度
        let jieba = Jieba::new();
        let text = "红魔馆的女仆长十六夜咲夜擅长投掷银质小刀她害怕烫的食物是众所周知的猫舌";
        for md in [0.2f32, 0.5, 0.87] {
            let r = mask_text(text, md, &jieba);
            let ratio = r.masked_count as f32 / r.total_count.max(1) as f32;
            assert!((ratio - md).abs() < 0.15, "md={} ratio={}", md, ratio);
            assert!(r.masked_text.contains(MASK_WORD.trim()));
        }
    }

    #[test]
    fn test_incremental_consistency() {
        let suite = ForgetSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
        let data = suite.run_incremental_case();
        assert!(data.passed, "增量一致性失败");
    }
}
