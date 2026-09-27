use super::*;
use chrono::TimeZone;

impl ForgetPipelineSuite {
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
    pub(in crate::engine::forget) fn run_excitation_case(
        &self,
        schedule: ExcitationSchedule,
    ) -> ForgetCaseData {
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
                get_summary(&trt.graph().node_weight(*idx).expect("node").note).unwrap_or_default()
            })
            .collect();

        // 激发计划：batches = [(激活时刻, [(节点序号, 本次激发次数), ...])]。
        // 激活时刻与观测检查点解耦：Early 全部在 t=0（首个检查点前应用），
        // Spaced 在 24/48/72h 分三批，Late 全部在 t=48h。
        let batches: Vec<(i64, Vec<(usize, usize)>)> = match schedule {
            ExcitationSchedule::Early => vec![(
                0,
                (0..n)
                    .filter(|&i| dose[i] > 0)
                    .map(|i| (i, dose[i]))
                    .collect(),
            )],
            ExcitationSchedule::Spaced => {
                let mut b24 = Vec::new();
                let mut b48 = Vec::new();
                let mut b72 = Vec::new();
                for (i, &d) in dose.iter().enumerate() {
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
                (0..n)
                    .filter(|&i| dose[i] > 0)
                    .map(|i| (i, dose[i]))
                    .collect(),
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
                let activated_by_now = activated_at[i].is_some_and(|at| at <= t_hours);
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
                d,
                sc,
                st,
                sc - st,
                cnt
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
                if activated_at[i].is_some_and(|at| at <= t_hours) {
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
                        md: trt_series[i][k],             // 激发组（y 主曲线）
                        md_ctrl: Some(ctrl_series[i][k]), // 对照组（配对对照曲线）
                        action: if activated_at[i].is_some_and(|at| at <= checkpoints[k]) {
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

        let max_md = trt_series
            .iter()
            .map(|s| s[last_idx])
            .fold(0.0f32, f32::max);

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
}
