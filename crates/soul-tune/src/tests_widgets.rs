use std::collections::HashMap;
use std::sync::Arc;

use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

use crate::engine::retrieve::data::{
    build_drilldown_sections, ActionMetrics, PerQueryMetrics, RankingMetrics, RetrieveCaseData,
};
use crate::engine::retrieve::dataset::SubQuery;
use crate::widgets::scroll::{display_width, pad_to_width, ScrollState};

fn make_id() -> MemoryId {
    MemoryId::new()
}

fn mock_case_data() -> RetrieveCaseData {
    let id1 = make_id();
    let id2 = make_id();
    let mut names = HashMap::new();
    names.insert(id1, "节点_A".to_string());
    names.insert(id2, "节点_B".to_string());
    RetrieveCaseData {
        case_name: "快照测试用例".into(),
        description: "mock数据".into(),
        combined_retrieved_ids: vec![id1, id2],
        combined_ranking_metrics: RankingMetrics {
            recall_at: vec![(1, 0.5), (3, 1.0)],
            precision_at: vec![(1, 1.0), (3, 0.667)],
            mrr: 1.0,
            ndcg_at: vec![(1, 1.0), (3, 0.8)],
            hit_rate: 1.0,
        },
        per_query_metrics: vec![PerQueryMetrics {
            query_index: 0,
            ranking_metrics: RankingMetrics {
                recall_at: vec![(1, 1.0)],
                precision_at: vec![(1, 1.0)],
                mrr: 1.0,
                ndcg_at: vec![(1, 1.0)],
                hit_rate: 1.0,
            },
        }],
        action_metrics: ActionMetrics {
            action_hit_rate: 1.0,
            action_recall_at: vec![(1, 1.0)],
        },
        tag_weight: 0.4,
        variant_weight: 0.6,
        id_names: None,
        expected_combined_ranking: vec![id1],
        bonus_combined_ranking: vec![],
        graph_names: Some(Arc::new(names)),
        sub_queries: vec![SubQuery {
            priority: 1,
            tags: vec!["角色".into()],
            variant: MemoryRetrieveQueryVariant::Semantic(vec![]),
        }],
    }
}

#[test]
fn test_scroll_state_defaults() {
    let s = ScrollState::new();
    let result = (s.cursor, s.offset);
    insta::assert_yaml_snapshot!("scroll_defaults", result);
}

#[test]
fn test_scroll_state_move_down() {
    let mut s = ScrollState::new();
    s.move_down(10);
    assert_eq!(s.cursor, 1);
}

#[test]
fn test_scroll_state_move_down_clamped() {
    let mut s = ScrollState::new();
    s.cursor = 0;
    s.move_down(2);
    assert_eq!(s.cursor, 1);
    s.move_down(2);
    assert_eq!(s.cursor, 1);
}

#[test]
fn test_scroll_state_clamp() {
    let mut s = ScrollState::new();
    s.cursor = 100;
    s.clamp_cursor(5);
    assert_eq!(s.cursor, 4);
}

#[test]
fn test_scroll_state_clamp_zero() {
    let mut s = ScrollState::new();
    s.cursor = 100;
    s.clamp_cursor(0);
    assert_eq!(s.cursor, 0);
}

#[test]
fn test_scroll_offset_basic() {
    let offset = ScrollState::offset(10, 50, 15);
    assert_eq!(offset, 6);
}

#[test]
fn test_scroll_offset_at_top() {
    let offset = ScrollState::offset(10, 50, 3);
    assert_eq!(offset, 0);
}

#[test]
fn test_display_width_ascii() {
    assert_eq!(display_width("hello"), 5);
}

#[test]
fn test_display_width_cjk() {
    let w = display_width("你好世界");
    assert_eq!(w, 8);
}

#[test]
fn test_pad_to_width_short() {
    let padded = pad_to_width("ab", 5);
    assert_eq!(padded.len(), 5);
}

#[test]
fn test_pad_to_width_long() {
    let padded = pad_to_width("abcdefghij", 5);
    assert!(padded.len() <= 5);
}

#[test]
fn test_drilldown_sections_non_empty() {
    let data = mock_case_data();
    let sections = build_drilldown_sections(&data);
    assert!(!sections.header_lines.is_empty());
    assert!(!sections.metrics_rows.is_empty());
    assert!(!sections.comparison_rows.is_empty());
}

#[test]
fn test_drilldown_sections_snapshot() {
    let data = mock_case_data();
    let sections = build_drilldown_sections(&data);
    insta::assert_yaml_snapshot!("drilldown_header", &sections.header_lines);
    insta::assert_yaml_snapshot!("drilldown_metrics", &sections.metrics_rows);
}
