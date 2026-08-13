use std::collections::HashMap;
use std::sync::Arc;

use soul_mem_core::memory_note::MemoryId;

use crate::engine::retrieve::dataset::SubQuery;

#[derive(Clone)]
pub struct RankingMetrics {
    pub recall_at: Vec<(usize, f64)>,
    pub precision_at: Vec<(usize, f64)>,
    pub mrr: f64,
    pub ndcg_at: Vec<(usize, f64)>,
    pub hit_rate: f64,
}

#[derive(Clone)]
pub struct ActionMetrics {
    pub action_hit_rate: f64,
    pub action_recall_at: Vec<(usize, f64)>,
    /// 该用例是否带 expected_actions 真值（False 表示占位指标，不应计入统计）
    pub has_expected_actions: bool,
}

#[derive(Clone)]
pub struct PerQueryMetrics {
    pub query_index: usize,
    pub ranking_metrics: RankingMetrics,
}

#[derive(Clone)]
pub struct RetrieveCaseData {
    pub case_name: String,
    pub description: String,
    pub combined_retrieved_ids: Vec<MemoryId>,
    pub combined_ranking_metrics: RankingMetrics,
    pub per_query_metrics: Vec<PerQueryMetrics>,
    pub action_metrics: ActionMetrics,
    /// 该用例的期望结果中是否包含抽象情境节点（有真值才计入抽象指标）。
    pub has_expected_abstract: bool,
    /// 期望抽象节点是否出现在合并结果（相似度+PPR）中。
    pub abstract_detected: Option<bool>,
    /// 期望抽象节点是否仍被相似度直接命中（数据侧泛化是否达标的观测门）。
    pub abstract_direct_hit: Option<bool>,
    pub tag_weight: f32,
    pub variant_weight: f32,
    pub id_names: Option<Arc<HashMap<MemoryId, NodeSummary>>>,
    pub expected_combined_ranking: Vec<MemoryId>,
    pub bonus_combined_ranking: Vec<MemoryId>,
    pub graph_names: Option<Arc<HashMap<MemoryId, String>>>,
    pub sub_queries: Vec<SubQuery>,
}

#[derive(Clone)]
pub struct NodeSummary {
    pub tags: Vec<String>,
    pub type_label: String,
    pub primary: String,
    pub secondary: String,
}

pub struct DrilldownSections {
    pub header_lines: Vec<String>,
    pub metrics_rows: Vec<String>,
    pub subquery_items: Vec<SubQueryItem>,
    pub comparison_rows: Vec<ComparisonRow>,
}

pub struct SubQueryItem {
    pub index: usize,
    pub mrr: f64,
    pub hit_rate: f64,
}

pub struct ComparisonRow {
    pub position: usize,
    pub retrieved: Option<RetrievedEntry>,
    pub expected: Option<String>,
    pub is_hit: bool,
}

pub struct RetrievedEntry {
    pub name: String,
    pub id: MemoryId,
}

pub fn build_drilldown_sections(data: &RetrieveCaseData) -> DrilldownSections {
    let mut sections = DrilldownSections {
        header_lines: Vec::new(),
        metrics_rows: Vec::new(),
        subquery_items: Vec::new(),
        comparison_rows: Vec::new(),
    };

    sections
        .header_lines
        .push(format!(" 用例: {}", data.case_name));
    let passed =
        data.combined_ranking_metrics.hit_rate > 0.0 || data.combined_ranking_metrics.mrr > 0.0;
    sections
        .header_lines
        .push(format!(" 状态: {}", if passed { "通过" } else { "失败" }));

    sections
        .metrics_rows
        .push("  K     Recall    Precision  NDCG".to_string());
    for (k, r) in &data.combined_ranking_metrics.recall_at {
        let p = data
            .combined_ranking_metrics
            .precision_at
            .iter()
            .find(|(pk, _)| pk == k)
            .map(|(_, v)| v)
            .unwrap_or(&0.0);
        let n = data
            .combined_ranking_metrics
            .ndcg_at
            .iter()
            .find(|(nk, _)| nk == k)
            .map(|(_, v)| v)
            .unwrap_or(&0.0);
        sections
            .metrics_rows
            .push(format!("  @{:<2}   {:.4}    {:.4}    {:.4}", k, r, p, n));
    }
    sections.metrics_rows.push(format!(
        "  MRR: {:.4}     Hit: {:.2}",
        data.combined_ranking_metrics.mrr, data.combined_ranking_metrics.hit_rate
    ));

    for m in &data.per_query_metrics {
        sections.subquery_items.push(SubQueryItem {
            index: m.query_index,
            mrr: m.ranking_metrics.mrr,
            hit_rate: m.ranking_metrics.hit_rate,
        });
    }

    let n_max = data
        .combined_retrieved_ids
        .len()
        .min(10)
        .max(data.expected_combined_ranking.len().min(5));
    for pos in 0..n_max {
        let retrieved = data.combined_retrieved_ids.get(pos).map(|id| {
            let name = data
                .graph_names
                .as_ref()
                .and_then(|m| m.get(id))
                .cloned()
                .unwrap_or_default();
            RetrievedEntry { name, id: *id }
        });
        let expected = data.expected_combined_ranking.get(pos).map(|eid| {
            data.graph_names
                .as_ref()
                .and_then(|m| m.get(eid))
                .cloned()
                .unwrap_or_default()
        });
        let is_hit = retrieved
            .as_ref()
            .map(|r| {
                data.expected_combined_ranking
                    .iter()
                    .any(|eid| *eid == r.id)
            })
            .unwrap_or(false);
        sections.comparison_rows.push(ComparisonRow {
            position: pos + 1,
            retrieved,
            expected,
            is_hit,
        });
    }

    sections
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

    fn make_id() -> MemoryId {
        MemoryId::new()
    }

    fn mock_case_data() -> RetrieveCaseData {
        let id1 = make_id();
        let id2 = make_id();
        let mut names = HashMap::new();
        names.insert(id1, "node_1".to_string());
        names.insert(id2, "node_2".to_string());
        RetrieveCaseData {
            case_name: "test_case".into(),
            description: "test".into(),
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
                has_expected_actions: false,
            },
            has_expected_abstract: false,
            abstract_detected: None,
            abstract_direct_hit: None,
            tag_weight: 0.4,
            variant_weight: 0.6,
            id_names: None,
            expected_combined_ranking: vec![id1],
            bonus_combined_ranking: vec![],
            graph_names: Some(Arc::new(names)),
            sub_queries: vec![SubQuery {
                priority: 1,
                tags: vec!["test".into()],
                variant: MemoryRetrieveQueryVariant::Semantic(vec![]),
            }],
        }
    }

    #[test]
    fn test_build_drilldown_sections_normal() {
        let data = mock_case_data();
        let sections = build_drilldown_sections(&data);
        assert!(!sections.header_lines.is_empty());
        assert!(!sections.metrics_rows.is_empty());
        assert_eq!(sections.subquery_items.len(), 1);
        assert!(!sections.comparison_rows.is_empty());
    }

    #[test]
    fn test_build_drilldown_header_contains_name() {
        let data = mock_case_data();
        let sections = build_drilldown_sections(&data);
        let h = sections.header_lines.join(" ");
        assert!(h.contains("test_case"));
    }

    #[test]
    fn test_build_drilldown_edge_no_ids() {
        let mut data = mock_case_data();
        data.combined_retrieved_ids.clear();
        data.expected_combined_ranking.clear();
        let sections = build_drilldown_sections(&data);
        assert!(sections.comparison_rows.is_empty());
    }

    #[test]
    fn test_ranking_metrics_clone() {
        let rm = RankingMetrics {
            recall_at: vec![(1, 0.5)],
            precision_at: vec![(1, 0.5)],
            mrr: 0.5,
            ndcg_at: vec![(1, 0.5)],
            hit_rate: 0.5,
        };
        let c = rm.clone();
        assert!((c.mrr - 0.5).abs() < 1e-6);
    }
}
