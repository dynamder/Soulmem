//! 遗忘算法测试套件。
//!
//! soul-tune 本身是 SoulMem 的测试框架，这里不再复刻算法分支里的单元小测试，
//! 而是**直接驱动具体的遗忘算法管线**：
//!
//! 1. [`compute_all_missing_degrees`] —— 全图批量刷新节点与边的缺失度（增量公式）；
//! 2. [`lazy_forget`] —— 对可遮罩节点（SemMemory / SpecificSituation）执行
//!    衰减 → 分词遮罩 →（可选）LLM 推测修订，LLM 不可用时验证降级为 MaskOnly；
//! 3. [`decay_graph_edge`] / [`weight_placeholder`] —— 边独立衰减与检索权重占位；
//! 4. [`update_missing_degree_incremental`] —— 增量缺失度与从头计算的一致性。
//!
//! 套件以 5 个场景用例驱动同一张记忆图在不同时间跨度下的真实遗忘流程，
//! 产出逐节点明细与聚合指标（缺失度 / 遮罩率 / 动作分布 / 边强度 / 衰减曲线）。

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use jieba_rs::Jieba;

use soul_mem_algo::algo::forget::decay_calculator::{
    compute_missing_degree, update_missing_degree_incremental, DEFAULT_MAX_ACTIVATION_CAP,
};
use soul_mem_algo::algo::forget::decay_revise::{
    compute_all_missing_degrees, current_missing_degree, decay_graph_edge, get_summary,
    lazy_forget, weight_placeholder, ForgetAction, DEFAULT_ACTIVE_FACTOR,
    DEFAULT_BASE_HALF_LIFE_HOURS,
};
use soul_mem_algo::algo::forget::mask::MASK_WORD;
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::{
    Context, Environment, SituationType, SpecificSituation,
};
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant};
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_runtime::cluster::memory_cluster::MemoryCluster;
use soul_mem_runtime::working_memory::llm::client::LlmClient;
use soul_mem_runtime::working_memory::llm::config::LLMConfig;

use crate::engine::suite::{
    chart_metric, key_value_metric, DetailRow, Series, SuiteReport, TestCaseOutcome, TestSuite,
};

// ========================================================================
// LLM 调用闭包（与算法侧 lazy_forget 的签名对齐）
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

/// 使用真实 `LlmClient`（环境变量 API_KEY / API_BASE / MODEL）的闭包
fn real_llm_closure(client: Arc<LlmClient>) -> LlmCall {
    Box::new(move |system: &str, user: &str| {
        let c = client.clone();
        let sys = system.to_string();
        let usr = user.to_string();
        Box::pin(async move {
            use async_openai::types::chat::{
                ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
            };
            let mut resp = c
                .call_llm(vec![
                    ChatCompletionRequestSystemMessage::from(sys).into(),
                    ChatCompletionRequestUserMessage::from(usr).into(),
                ])
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            Ok(resp.remove(0))
        })
    })
}

/// LLM 不可用时传入的错误闭包：算法应优雅降级为 MaskOnly（遮罩不回退）
fn failing_llm_closure() -> LlmCall {
    Box::new(|_system: &str, _user: &str| {
        Box::pin(async {
            Err::<String, Box<dyn std::error::Error + Send + Sync>>(
                "LLM 未配置（缺少 API_KEY / API_BASE / MODEL）".into(),
            )
        })
    })
}

// ========================================================================
// 测试场景定义
// ========================================================================

/// 遗忘场景：同一张记忆图在指定时间跨度（距上次遗忘操作的小时数）下跑真实管线
#[derive(Debug, Clone, Copy)]
pub struct ForgetCaseSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// 距离上次遗忘操作/访问的时间（小时），决定遗忘缺失度
    pub elapsed_hours: i64,
    /// 是否允许调用 LLM（无 key 时自动降级为遮罩）
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
        description: "中遗忘强度（Δt=24h）：缺失度约半衰，遮罩开始触发",
        elapsed_hours: 24,
        want_llm: false,
    },
    ForgetCaseSpec {
        name: "high",
        description: "高遗忘强度（Δt=72h）：触发 LLM 修订或降级遮罩",
        elapsed_hours: 72,
        want_llm: true,
    },
    ForgetCaseSpec {
        name: "mixed",
        description: "混合时间跨度（各节点遗忘历史不同）：真实分布场景",
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
}

/// 单个用例（一次完整管线运行）的观测数据
pub struct ForgetCaseData {
    pub case_name: String,
    pub passed: bool,
    pub llm_available: bool,
    pub node_count: usize,
    pub edge_count: usize,
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

/// 遗忘 LLM 修订 system prompt（与算法默认一致的记忆重建角色）
const FORGET_SYSTEM_PROMPT: &str = "You are a memory reconstruction assistant. \
    A segment of memory text has been partially masked, with [masked] placeholders. \
    Based on the context and the remaining fragments, infer and complete the missing parts \
    naturally. Output only the completed text, no explanation.";

pub struct ForgetSuite {
    jieba: Jieba,
    /// 真实 LLM 客户端（环境变量就绪时）；否则 None → 遮罩降级路径
    llm: Option<Arc<LlmClient>>,
    cases: Vec<ForgetCaseSpec>,
}

impl ForgetSuite {
    /// 新建套件（内置角色化记忆图）。
    ///
    /// 保持无 LLM（幂等）：套件测试与确定性验证使用此构造；
    /// 需要真实 LLM 修订时使用 [`Self::with_llm`]。
    pub fn new() -> Self {
        Self {
            jieba: Jieba::new(),
            llm: None,
            cases: BUILTIN_CASES.to_vec(),
        }
    }

    /// 读取环境变量（API_KEY / API_BASE / MODEL）启用真实 LLM 修订；
    /// 未配置时与 [`Self::new`] 等价（遮罩降级路径）。
    pub fn with_llm() -> Self {
        Self {
            llm: Self::try_create_llm_client(),
            ..Self::new()
        }
    }

    /// 与 retrieve 套件保持一致的加载入口。
    ///
    /// 遗忘算法不依赖嵌入与查询文件，数据集路径当前仅作展示；
    /// 套件使用内置角色化记忆图运行真实管线，保证离线可复现。
    pub fn load(_path: &Path) -> Result<Self, String> {
        Ok(Self::with_llm())
    }

    fn try_create_llm_client() -> Option<Arc<LlmClient>> {
        let key = std::env::var("API_KEY").ok()?;
        let base = std::env::var("API_BASE").ok()?;
        let model = std::env::var("MODEL").ok()?;
        Some(Arc::new(LlmClient::new(LLMConfig::new(
            &key, &base, &model,
        ))))
    }

    /// 构建内置场景图（节点创建于 240h 前，last_forget_time 由场景指定）
    ///
    /// `node_offsets`: 每类节点的距上次遗忘的小时数（None 表示使用场景统一跨度）
    fn build_cluster(
        &self,
        elapsed_hours: i64,
        node_offsets: Option<[(usize, i64); 3]>, // (索引, 小时)，索引 0/1/2 对应 sem/sit/proc
    ) -> MemoryCluster {
        let now = Utc::now();
        let created = now - ChronoDuration::hours(24 * 10);

        let offset_for = |index: usize| -> DateTime<Utc> {
            let h = node_offsets
                .and_then(|offs| offs.iter().find(|(i, _)| *i == index).map(|(_, h)| *h))
                .unwrap_or(elapsed_hours);
            now - ChronoDuration::hours(h)
        };

        let make_embedded = |note: MemoryNote| -> EmbeddedMemoryNote {
            let embedding = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            EmbeddedMemoryNote { note, embedding }
        };

        let id_sem = MemoryId::new();
        let id_sit = MemoryId::new();
        let id_proc = MemoryId::new();

        let sem = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "十六夜咲夜是红魔馆的女仆长拥有操纵时间的能力她可以停止时间在静止的世界中完成所有家务银质小刀是她惯用的武器"
                .to_string(),
            ConceptType::Entity,
            "红魔馆的女仆长".to_string(),
        )))
        .id(id_sem)
        .create_time(created)
        .last_accessed_time(created)
        .last_forget_time(offset_for(0))
        .build()
        .expect("sem note");

        let sit = MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(
            SpecificSituation::new(
                "傍晚我在红魔馆的庭院为大小姐斟茶蕾米莉亚坐在阳台的红伞下望着雾之湖畔天色渐暗四周渐渐安静下来".to_string(),
                created,
                Context::new(
                    None, vec![], vec![], vec![],
                    Environment { atmosphere: "日常".to_string(), tone: "平静".to_string() },
                    vec![],
                ),
            ),
        )))
        .id(id_sit)
        .create_time(created)
        .last_accessed_time(created)
        .last_forget_time(offset_for(1))
        .build()
        .expect("sit note");

        let proc = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(Action::new(
            "红魔馆女仆长每日停止时间打扫洋馆再回收飞刀的工作流程".to_string(),
            ActionType::Think,
        ))))
        .id(id_proc)
        .create_time(created)
        .last_accessed_time(created)
        .last_forget_time(offset_for(2))
        .build()
        .expect("proc note");

        let mut cluster = MemoryCluster::new();
        cluster.add_single_node(make_embedded(sem));
        cluster.add_single_node(make_embedded(sit));
        cluster.add_single_node(make_embedded(proc));

        // 两条边：sem -> sit（关联），sit -> proc（引发）
        // 混合场景（elapsed_hours<=0）下边统一使用 24h 跨度
        let edge_hours = if elapsed_hours > 0 { elapsed_hours } else { 24 };
        let mut link1 = MemoryLink::new(
            id_sem,
            id_sit,
            MemoryLinkType::Sem(SemMemLink::new("关联".to_string(), 1.0)),
        );
        link1.set_last_forget_time(now - ChronoDuration::hours(edge_hours));
        let mut link2 = MemoryLink::new(
            id_sit,
            id_proc,
            MemoryLinkType::Sem(SemMemLink::new("引发".to_string(), 1.0)),
        );
        link2.set_last_forget_time(now - ChronoDuration::hours(edge_hours));

        {
            let graph = cluster.graph_mut();
            for n in graph.node_weights_mut() {
                if n.note().id() == id_sem {
                    n.note.links_mut().push(link1.clone());
                } else if n.note().id() == id_sit {
                    n.note.links_mut().push(link2.clone());
                }
            }
        }
        cluster.refresh_node(&id_sem);
        cluster.refresh_node(&id_sit);
        cluster
    }

    /// 运行一次完整的真实遗忘管线（与检索侧的惰性遗忘调用路径一致）：
    /// 1. `compute_all_missing_degrees` 全图批量刷新节点+边缺失度；
    /// 2. 逐节点 `lazy_forget`（SemMemory / SpecificSituation 触发遮罩/LLM）；
    /// 3. 边强度 = 原始强度 × (1 - 缺失度)，检索权重占位 `weight_placeholder`。
    fn run_pipeline(&self, spec: &ForgetCaseSpec) -> ForgetCaseData {
        let now = Utc::now();
        // 混合场景：三节点分别 8h / 24h / 72h 前遗忘
        let node_offsets = if spec.elapsed_hours == -1 {
            Some([(0usize, 8i64), (1, 24), (2, 72)])
        } else {
            None
        };
        let mut cluster = self.build_cluster(spec.elapsed_hours, node_offsets);

        let use_llm = self.llm.is_some() && spec.want_llm;

        // ── 步骤 1：全图批量刷新缺失度（节点 + 边）──
        compute_all_missing_degrees(&mut cluster, now);

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
                let closure: LlmCall = if use_llm {
                    real_llm_closure(self.llm.as_ref().expect("llm").clone())
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

            // 动作分类与遮罩词数
            let (action_name, masked_words_this, mask_ratio_this) = match &action {
                ForgetAction::NoAction => ("NoAction", 0usize, None),
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
                    let _ = masked_text;
                    ("MaskOnly", *masked_count, ratio)
                }
                ForgetAction::Revised { masked_text, .. } => {
                    let masked = count_masked(masked_text);
                    let ratio = if orig_words > 0 {
                        Some(masked as f32 / orig_words as f32)
                    } else {
                        None
                    };
                    ("Revised", masked, ratio)
                }
            };

            if matches!(type_name, "SemMemory" | "SpecificSituation") {
                if orig_words > 0 {
                    masked_words_total += masked_words_this;
                    total_words_total += orig_words;
                }
                // 纯遮罩路径下：遮罩比例应接近缺失度（±15% 容差）
                if let ForgetAction::MaskOnly { .. } = &action {
                    if let Some(r) = mask_ratio_this {
                        if (r - after).abs() > 0.15 {
                            passed = false;
                        }
                    }
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
                md_before: before,
                md_after: after,
                action: action_name,
                mask: mask_ratio_this.map(|r| {
                    (
                        (r * orig_words as f32).round() as usize,
                        orig_words,
                    )
                }),
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
        Self::new()
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

        for o in &outcomes {
            let Some(data) = o.data.downcast_ref::<ForgetCaseData>() else {
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
            // 用例汇总行（读取全部观测字段）
            let hist_txt: String = data
                .action_histogram
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ");
            detail_rows.push(DetailRow {
                text: format!(
                    "[{}] 节点{} 边{} | 缺失度均值{:.3}/最大{:.3} | 遮罩率{:.1}% | 边强度{:.3} | {}",
                    data.case_name,
                    data.node_count,
                    data.edge_count,
                    data.avg_missing_degree,
                    data.max_missing_degree,
                    data.avg_masked_ratio * 100.0,
                    data.avg_edge_intensity,
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
            "LLM 可用".to_string(),
            "LLM".to_string(),
            if llm_available {
                "是（真实修订）".to_string()
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
                "遗忘管线逐节点明细（通过 {}/{}，耗时 {:.2}s）",
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

    #[test]
    fn test_suite_has_five_cases() {
        let suite = ForgetSuite::new();
        assert_eq!(suite.case_count(), 5);
    }

    #[test]
    fn test_all_cases_pass_without_llm() {
        let suite = ForgetSuite::new();
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
        let suite = ForgetSuite::new();
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
        let suite = ForgetSuite::new();
        let data = suite.run_incremental_case();
        assert!(data.passed, "增量一致性失败");
    }
}
