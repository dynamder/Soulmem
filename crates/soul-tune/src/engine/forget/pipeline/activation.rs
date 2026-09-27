use super::*;
use rand::Rng;

impl ForgetPipelineSuite {
    /// 激活测试：固定种子随机选节点激活多次（`retrieval_increment`），
    /// 统一老化 72h 后验证**整图**的遗忘状态是否符合设计：
    /// - 激活次数越多 → 半衰期越长 → 缺失度越低（负相关）；
    /// - 每个节点的实测缺失度与理论值一致（±1e-3）。
    pub(in crate::engine::forget) fn run_activation_case(&self) -> ForgetCaseData {
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

            let short_id: String = display_id(&self.id_rev, n.note().id())
                .chars()
                .take(8)
                .collect();
            detail_lines.push(format!(
                "{} 激活{}次 md实测{:.3} 理论{:.3} 偏差{:.5}",
                short_id,
                count,
                md_actual,
                md_theory,
                md_actual - md_theory
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
}
