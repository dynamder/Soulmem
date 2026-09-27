use super::*;

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
    /// `pub(super)`：单元测试（`forget::tests`）需要直接检查 LLM 可用性
    pub(in crate::engine::forget) llm: Option<Arc<Mutex<LlamaServer>>>,
    /// 长文本样本（从 fixture 图提取）
    pub(in crate::engine::forget) samples: Vec<ReviseSample>,
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
            Some(server) => {
                // 与算法层 `reconstruct_summary` 完全一致的提示词（user 带占位符计数说明），
                // 保证修订测试 = 真实管线行为。
                //
                // 这里刻意**直接调用**同步外壳而不是走 `reconstruct_summary`：后者的
                // "全遮罩直接返回固定遗忘句"是管线语义，而本套件要测的是 LLM 面对遮罩输入
                // 的真实产出（含 md=1.0 的极端档），不能被短路掉。
                let (system, user) =
                    build_reconstruct_prompt(&sample.masked, Some(FORGET_SYSTEM_PROMPT));
                let mut guard = server.lock().expect("llama-server 锁");
                match guard.chat(&system, &user, LLM_MAX_TOKENS) {
                    Ok(reply) => (reply, None),
                    Err(error) => (String::new(), Some(format!("{error}"))),
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

        let metrics = vec![(
            "补全".into(),
            format!(
                "{} md{:.2} 回复字数",
                sample.node_id.chars().take(8).collect::<String>(),
                sample.mask_md
            ),
            if llm_err.is_none() {
                reply.chars().count().to_string()
            } else {
                "失败".into()
            },
        )];

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
