use super::*;
use std::path::PathBuf;

use soul_mem_algo::algo::forget::llm_completion::FULLY_MASKED_REPLY;

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
            outcome.case_name, outcome.description
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
            outcome.case_name, outcome.description
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
        assert!((ratio - md).abs() < 0.15, "md={} ratio={}", md, ratio);
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
    let suite = ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
    assert!(suite.llm.is_none(), "测试环境不应配置 LLM");
    assert!(!suite.samples.is_empty(), "全量模式应覆盖全部可遗忘节点");
    // 全量覆盖：可遗忘节点（SemMemory/SpecificSituation）全部进入
    let (cluster, _) = load_graph_cluster(&fixture_graph()).expect("加载 fixture 图");
    let maskable = cluster
        .graph()
        .node_weights()
        .filter(|n| {
            is_maskable(n.note()) && !get_summary(n.note()).unwrap_or_default().trim().is_empty()
        })
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
    let s1 = ForgetReviseSuite::load_with_mode(&fixture_graph(), ReviseMode::Sampled(42))
        .expect("加载 fixture 图");
    let s2 = ForgetReviseSuite::load_with_mode(&fixture_graph(), ReviseMode::Sampled(42))
        .expect("加载 fixture 图");
    assert!(
        s1.samples.len() <= REVISE_MAX_SAMPLES * REVISE_MASK_GRADIENTS.len()
            && !s1.samples.is_empty(),
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
    let suite = ForgetReviseSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
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
            outcome.case_name, outcome.description
        );
    }
}

#[test]
fn test_pipeline_multi_step_without_llm() {
    // 多步遗忘：无 LLM 时全走确定性降级路径且通过（缺失度单调不减）。
    // 注意：全遮罩文本由 llm_completion 短路直接返回固定遗忘句（不调用 LLM），
    // lazy_forget 会将其记为 Revised——因此无 LLM 时不再要求"零修订"，
    // 而是断言 Revised 的内容只能是该固定遗忘句。
    let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
    let data = suite.run_multi_step_case();
    assert!(data.passed, "多步遗忘失败");
    assert!(data.node_count > 10, "应覆盖全图节点");
    assert!(!data.detail_lines.is_empty());
    assert!(
        !data.llm_available,
        "无 LLM 加载时 llm_available 应为 false"
    );
    // 无 LLM：任何 Revised 的内容只能是全遮罩兜底的固定遗忘句
    for stat in &data.nodes {
        if stat.action == "Revised" {
            assert_eq!(
                stat.llm_reply.as_deref(),
                Some(FULLY_MASKED_REPLY),
                "无 LLM 时 Revised 只能来自全遮罩确定性兜底"
            );
        }
    }
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
    let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
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
            data.metrics
                .iter()
                .any(|(_, label, _)| label.contains("延缓")),
            "应产出延缓指标"
        );
    }
}

#[test]
fn test_pipeline_excitation_deterministic() {
    // E6：同一场景两次运行结果完全一致（固定模拟时钟 + 固定种子）
    let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
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
        suite
            .cases
            .iter()
            .all(|c| c.name.starts_with("excitation-")),
        "用例应全部为 excitation-*"
    );
    assert!(suite.llm.is_none(), "激发测试不应启用 LLM");
    for i in 0..suite.case_count() {
        let outcome = suite.run_case(i);
        assert!(
            outcome.passed,
            "激发用例 {} 失败: {}",
            outcome.case_name, outcome.description
        );
    }
}

#[test]
fn test_pipeline_report_builds_metrics_and_rows() {
    let suite = ForgetPipelineSuite::load_without_llm(&fixture_graph()).expect("加载 fixture 图");
    let n = suite.case_count();
    let outcomes: Vec<TestCaseOutcome> = (0..n).map(|i| suite.run_case(i)).collect();
    let passed = outcomes.iter().filter(|o| o.passed).count();
    let report = suite.build_report(outcomes, Duration::from_millis(10), n, passed, n - passed);
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
    let report = suite.build_report(outcomes, Duration::from_millis(10), n, passed, n - passed);
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
