use super::*;

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
///
/// `pub(super)`：单元测试（`forget::tests`）直接引用它做断言。
pub(in crate::engine::forget) const MASK_TEXTS: [(&str, &str); 3] = [
    ("short", "格蕾修是逐火十三英桀之一也是用画笔说话的画家"),
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
            if spec.missing_degree >= 1.0 && r1.total_count > 0 && r1.masked_count != r1.total_count
            {
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
        if spec.name.ends_with("-determinism") && r1.masked_text != r2.masked_text {
            detail_lines.push("  确定性检查: 两次结果不一致！".into());
        }

        let metrics = vec![
            (
                "遮罩".into(),
                format!("{} 遮罩率", spec.node_id),
                format!(
                    "{:.0}%",
                    if r1.total_count > 0 {
                        r1.masked_count as f32 / r1.total_count as f32 * 100.0
                    } else {
                        0.0
                    }
                ),
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
            description: format!("遮罩验证: md={:.2}", self.cases[index].missing_degree),
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
