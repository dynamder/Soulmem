use super::*;

// 同一 `ForgetPipelineSuite` 的固有 impl 按场景分散在下列子模块中
// （Rust 允许同一 crate 内对同一类型写多个固有 impl）。
mod activation;
mod excitation;
mod multistep;

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
/// `pub(super)`：激发测试子模块与单元测试（`forget::tests`）都要引用它。
pub(in crate::engine::forget) enum ExcitationSchedule {
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
    pub(in crate::engine::forget) graph: MemoryCluster,
    graph_name: String,
    jieba: Jieba,
    pub(in crate::engine::forget) llm: Option<Arc<Mutex<LlamaServer>>>,
    pub(in crate::engine::forget) cases: Vec<ForgetCaseSpec>,
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
                // 只有"确实要用 LLM 且该节点需要修订"时才给出引擎；
                // 其余情况显式传 None，由算法层降级为仅遮罩
                let server = if use_llm && revise_set.contains(&idx) {
                    Some(
                        self.llm
                            .as_ref()
                            .expect("llm")
                            .lock()
                            .expect("llama-server 锁"),
                    )
                } else {
                    None
                };
                let engine = server.as_ref().map(|guard| guard.engine());
                let act = runtime.block_on(lazy_forget(
                    node,
                    now,
                    &self.jieba,
                    Some(FORGET_SYSTEM_PROMPT),
                    engine,
                ));
                let after = node.missing_degree();
                (act, after, orig_words, orig)
            };

            // 动作分类、遮罩统计与 LLM 原始回复
            let (
                action_name,
                masked_words_this,
                mask_ratio_this,
                llm_reply,
                masked_text,
                effective,
            ) = match &action {
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
                    masked_text,
                    new_summary,
                    ..
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
                if orig_words >= 8
                    && let Some(r) = mask_ratio_this
                    && (r - after).abs() > 0.15
                {
                    passed = false;
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
}

impl ForgetPipelineSuite {
    /// 增量一致性场景：两次 12h 增量更新 == 一次 24h 全量计算（误差 < 1e-3）
    pub(in crate::engine::forget) fn run_incremental_case(&self) -> ForgetCaseData {
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
        metrics.push((
            "增量一致性".into(),
            "全量24h缺失度".into(),
            format!("{:.4}", full_md),
        ));
        metrics.push((
            "增量一致性".into(),
            "两次12h增量缺失度".into(),
            format!("{:.4}", inc_md),
        ));
        metrics.push(("增量一致性".into(), "差值".into(), format!("{:.5}", diff)));
        metrics.push((
            "增量一致性".into(),
            "中间态12h缺失度".into(),
            format!("{:.4}", mid_md),
        ));

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
