use super::*;

impl ForgetPipelineSuite {
    /// 多步遗忘：3 轮 × 24h，对图中**每个受遗忘影响的节点**逐步执行
    /// 衰减 → 遮罩 →（LLM 修订，全部满足条件者）补全，收集每一步的输入输出与缺失度轨迹。
    ///
    /// 观测内容：
    /// - 每节点的缺失度轨迹（md0 → md1 → md2 → md3，单调不减）；
    /// - 每轮动作分布（NoAction / MaskOnly / Revised）与有效修订数；
    /// - 内容退化轨迹：原始文本 → 每轮遮罩输入 → LLM 原始回复。
    pub(in crate::engine::forget) fn run_multi_step_case(&self) -> ForgetCaseData {
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
                    // 与上面一致：没有可用 LLM 时显式传 None，而不是伪造失败
                    let server = if use_llm && revise_set.contains(idx) {
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
                    (act, node.missing_degree())
                };

                let (action_name, masked_text, llm_reply, effective) = match &action {
                    ForgetAction::NoAction => ("NoAction", None, None, false),
                    ForgetAction::MaskOnly { masked_text, .. } => {
                        ("MaskOnly", Some(masked_text.clone()), None, false)
                    }
                    ForgetAction::Revised {
                        masked_text,
                        new_summary,
                        ..
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
            ("多步遗忘".into(), "边数".into(), edge_count.to_string()),
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
}
