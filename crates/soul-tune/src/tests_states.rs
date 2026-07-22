use std::collections::HashMap;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use soul_mem_core::memory_note::MemoryId;

use crate::base::{AlgoType, TestConfig, TestReport};
use crate::component::Component;
use crate::engine::compare::{CompareAggregate, CompareCaseData, CompareReport};
use crate::engine::retrieve::data::RetrieveCaseData;
use crate::engine::suite::{key_value_metric, DetailRow, SuiteReport, TestCaseOutcome};
use crate::states::compare_mode::SelectAlgoState;
use crate::states::compare_results::CompareResultsState;
use crate::states::main_menu::MainState;
use crate::states::results::ResultsState;
use crate::states::retrieve_mode::RetrieveModeSelectState;

fn render_to_string(state: &dyn Component) -> String {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| state.view(f)).unwrap();
    let buf = terminal.backend().buffer();
    let mut lines = Vec::new();
    for y in 0..buf.area().height {
        let mut line = String::new();
        for x in 0..buf.area().width {
            let cell = buf.get(x, y);
            if cell.symbol() == " " {
                line.push(' ');
            } else {
                line.push_str(cell.symbol());
            }
        }
        let trimmed = line.trim_end().to_string();
        if !trimmed.is_empty() || y < 3 {
            lines.push(trimmed);
        }
    }
    lines.join("\n")
}

fn make_id() -> MemoryId {
    MemoryId::new()
}

fn mock_test_report() -> TestReport {
    let id1 = make_id();
    let id2 = make_id();
    let data = RetrieveCaseData {
        case_name: "test_case".into(),
        description: "mock".into(),
        combined_retrieved_ids: vec![id1],
        combined_ranking_metrics: crate::engine::retrieve::data::RankingMetrics {
            recall_at: vec![(1, 0.5), (3, 1.0)],
            precision_at: vec![(1, 1.0), (3, 0.667)],
            mrr: 1.0,
            ndcg_at: vec![(1, 1.0), (3, 0.8)],
            hit_rate: 1.0,
        },
        per_query_metrics: vec![],
        action_metrics: crate::engine::retrieve::data::ActionMetrics {
            action_hit_rate: 1.0,
            action_recall_at: vec![(1, 1.0)],
        },
        tag_weight: 0.5,
        variant_weight: 0.5,
        id_names: None,
        expected_combined_ranking: vec![id1],
        bonus_combined_ranking: vec![],
        graph_names: None,
        sub_queries: vec![],
    };

    let outcome = TestCaseOutcome {
        case_name: "test_case".into(),
        description: "mock".into(),
        passed: true,
        data: Box::new(data),
    };

    let suite_report = SuiteReport {
        metrics: vec![
            Box::new(key_value_metric("Hit Rate", "准确率", "0.85")),
            Box::new(key_value_metric("MRR", "准确率", "1.00")),
        ],
        detail_header: "用例  MRR  Hit".into(),
        detail_rows: vec![DetailRow {
            text: "test_case  1.00  0.85  ✓".into(),
            has_error: false,
        }],
        outcomes: vec![outcome],
    };

    TestReport {
        config: TestConfig {
            algo: AlgoType::Retrieve(crate::base::RetrieveMode::Embedding),
            dataset_path: "/mock/test.json".into(),
            params: HashMap::new(),
        },
        total: 1,
        passed: 1,
        failed: 0,
        elapsed: std::time::Duration::from_secs(1),
        suite_report,
        error: None,
    }
}

fn mock_compare_report() -> CompareReport {
    let id1 = make_id();
    let id2 = make_id();
    CompareReport {
        cases: vec![CompareCaseData {
            case_name: "case_a".into(),
            description: "compare mock".into(),
            tag_weight: 0.5,
            variant_weight: 0.5,
            embedding_hit: 0.6,
            fullpipeline_hit: 0.9,
            embedding_mrr: 0.5,
            fullpipeline_mrr: 0.8,
            embedding_recall_at: vec![(1, 0.6), (3, 0.8)],
            fullpipeline_recall_at: vec![(1, 0.9), (3, 1.0)],
            embedding_precision_at: vec![(1, 1.0), (3, 0.5)],
            fullpipeline_precision_at: vec![(1, 1.0), (3, 0.667)],
            embedding_retrieved: vec![id1],
            fullpipeline_retrieved: vec![id1, id2],
            expected_combined_ranking: vec![id1, id2],
        }],
        aggregate: CompareAggregate {
            case_count: 1,
            avg_embedding_hit: 0.6,
            avg_fullpipeline_hit: 0.9,
            avg_embedding_mrr: 0.5,
            avg_fullpipeline_mrr: 0.8,
            hit_improvement_count: 1,
            mrr_improvement_count: 1,
        },
    }
}

#[test]
fn test_snapshot_main_menu() {
    let state = MainState;
    let rendered = render_to_string(&state);
    insta::assert_yaml_snapshot!("state_main_menu", rendered);
}

#[test]
fn test_snapshot_retrieve_mode_select() {
    let state = RetrieveModeSelectState::new();
    let rendered = render_to_string(&state);
    insta::assert_yaml_snapshot!("state_retrieve_mode", rendered);
}

#[test]
fn test_snapshot_select_algo() {
    let state = SelectAlgoState::new();
    let rendered = render_to_string(&state);
    insta::assert_yaml_snapshot!("state_select_algo", rendered);
}

#[test]
fn test_snapshot_results_summary() {
    let report = mock_test_report();
    let state = ResultsState::new(report);
    let rendered = render_to_string(&state);
    insta::assert_yaml_snapshot!("state_results_summary", rendered);
}

#[test]
fn test_snapshot_compare_results() {
    let report = mock_compare_report();
    let state = CompareResultsState::new(report);
    let rendered = render_to_string(&state);
    insta::assert_yaml_snapshot!("state_compare_results", rendered);
}
