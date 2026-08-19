use std::collections::HashMap;

use serde::Serialize;
use soul_mem_core::memory_note::MemoryId;

use crate::engine::retrieve::data::RetrieveCaseData;
use crate::engine::suite::TestCaseOutcome;

#[derive(Clone, Serialize)]
pub struct CompareCaseData {
    pub case_name: String,
    pub description: String,
    pub tag_weight: f32,
    pub variant_weight: f32,

    pub embedding_hit: f64,
    pub fullpipeline_hit: f64,
    pub embedding_mrr: f64,
    pub fullpipeline_mrr: f64,
    pub embedding_recall_at: Vec<(usize, f64)>,
    pub fullpipeline_recall_at: Vec<(usize, f64)>,
    pub embedding_precision_at: Vec<(usize, f64)>,
    pub fullpipeline_precision_at: Vec<(usize, f64)>,

    pub embedding_retrieved: Vec<MemoryId>,
    pub fullpipeline_retrieved: Vec<MemoryId>,
    pub expected_combined_ranking: Vec<MemoryId>,
}

#[derive(Clone, Default, Serialize)]
pub struct CompareAggregate {
    pub case_count: usize,
    pub avg_embedding_hit: f64,
    pub avg_fullpipeline_hit: f64,
    pub avg_embedding_mrr: f64,
    pub avg_fullpipeline_mrr: f64,
    pub hit_improvement_count: usize,
    pub mrr_improvement_count: usize,
}

#[derive(Clone, Serialize)]
pub struct CompareReport {
    pub cases: Vec<CompareCaseData>,
    pub aggregate: CompareAggregate,
}

pub fn build_compare_report(
    emb_outcomes: &[TestCaseOutcome],
    full_outcomes: &[TestCaseOutcome],
) -> CompareReport {
    let emb_map: HashMap<(String, u32, u32), RetrieveCaseData> = emb_outcomes
        .iter()
        .filter_map(|o| {
            o.data.downcast_ref::<RetrieveCaseData>().map(|d| {
                let key = (
                    d.case_name.clone(),
                    (d.tag_weight * 100.0).round() as u32,
                    (d.variant_weight * 100.0).round() as u32,
                );
                (key, d.clone())
            })
        })
        .collect();

    let full_map: HashMap<(String, u32, u32), RetrieveCaseData> = full_outcomes
        .iter()
        .filter_map(|o| {
            o.data.downcast_ref::<RetrieveCaseData>().map(|d| {
                let key = (
                    d.case_name.clone(),
                    (d.tag_weight * 100.0).round() as u32,
                    (d.variant_weight * 100.0).round() as u32,
                );
                (key, d.clone())
            })
        })
        .collect();

    let mut keys: Vec<_> = emb_map.keys().collect();
    keys.sort();

    let mut cases = Vec::new();

    for key in &keys {
        let emb = &emb_map[key];
        let full = full_map.get(*key);

        let case_data = CompareCaseData {
            case_name: emb.case_name.clone(),
            description: emb.description.clone(),
            tag_weight: emb.tag_weight,
            variant_weight: emb.variant_weight,

            embedding_hit: emb.combined_ranking_metrics.hit_rate,
            fullpipeline_hit: full
                .map(|d| d.combined_ranking_metrics.hit_rate)
                .unwrap_or(0.0),
            embedding_mrr: emb.combined_ranking_metrics.mrr,
            fullpipeline_mrr: full.map(|d| d.combined_ranking_metrics.mrr).unwrap_or(0.0),

            embedding_recall_at: emb.combined_ranking_metrics.recall_at.clone(),
            fullpipeline_recall_at: full
                .map(|d| d.combined_ranking_metrics.recall_at.clone())
                .unwrap_or_default(),
            embedding_precision_at: emb.combined_ranking_metrics.precision_at.clone(),
            fullpipeline_precision_at: full
                .map(|d| d.combined_ranking_metrics.precision_at.clone())
                .unwrap_or_default(),

            embedding_retrieved: emb.combined_retrieved_ids.clone(),
            fullpipeline_retrieved: full
                .map(|d| d.combined_retrieved_ids.clone())
                .unwrap_or_default(),
            expected_combined_ranking: emb.expected_combined_ranking.clone(),
        };
        cases.push(case_data);
    }

    let case_count = cases.len();
    if case_count == 0 {
        return CompareReport {
            cases,
            aggregate: CompareAggregate::default(),
        };
    }

    let avg_embedding_hit = cases.iter().map(|c| c.embedding_hit).sum::<f64>() / case_count as f64;
    let avg_fullpipeline_hit =
        cases.iter().map(|c| c.fullpipeline_hit).sum::<f64>() / case_count as f64;
    let avg_embedding_mrr = cases.iter().map(|c| c.embedding_mrr).sum::<f64>() / case_count as f64;
    let avg_fullpipeline_mrr =
        cases.iter().map(|c| c.fullpipeline_mrr).sum::<f64>() / case_count as f64;

    let hit_improvement_count = cases
        .iter()
        .filter(|c| c.fullpipeline_hit > c.embedding_hit)
        .count();
    let mrr_improvement_count = cases
        .iter()
        .filter(|c| c.fullpipeline_mrr > c.embedding_mrr)
        .count();

    CompareReport {
        cases,
        aggregate: CompareAggregate {
            case_count,
            avg_embedding_hit,
            avg_fullpipeline_hit,
            avg_embedding_mrr,
            avg_fullpipeline_mrr,
            hit_improvement_count,
            mrr_improvement_count,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::retrieve::data::{ActionMetrics, RankingMetrics};

    fn metrics(hit: f64, mrr: f64) -> RankingMetrics {
        RankingMetrics {
            recall_at: vec![(5, hit)],
            precision_at: vec![(5, hit)],
            mrr,
            ndcg_at: vec![],
            hit_rate: hit,
        }
    }

    fn case_data(
        name: &str,
        tag_weight: f32,
        variant_weight: f32,
        hit: f64,
        mrr: f64,
    ) -> RetrieveCaseData {
        RetrieveCaseData {
            case_name: name.to_string(),
            description: format!("desc {name}"),
            combined_retrieved_ids: vec![],
            combined_ranking_metrics: metrics(hit, mrr),
            per_query_metrics: vec![],
            action_metrics: ActionMetrics {
                action_hit_rate: 0.0,
                action_recall_at: vec![],
                has_expected_actions: false,
            },
            has_expected_abstract: false,
            abstract_detected: None,
            abstract_direct_hit: None,
            tag_weight,
            variant_weight,
            id_names: None,
            expected_combined_ranking: vec![],
            bonus_combined_ranking: vec![],
            graph_names: None,
            sub_queries: vec![],
        }
    }

    fn outcome(case: RetrieveCaseData) -> TestCaseOutcome {
        let name = case.case_name.clone();
        TestCaseOutcome {
            case_name: name,
            description: String::new(),
            passed: true,
            data: Box::new(case),
        }
    }

    #[test]
    fn test_build_compare_report_basic() {
        let emb = vec![
            outcome(case_data("a", 0.5, 0.5, 0.8, 0.6)),
            outcome(case_data("b", 0.4, 0.6, 0.5, 0.3)),
        ];
        let full = vec![
            outcome(case_data("a", 0.5, 0.5, 0.9, 0.8)),
            outcome(case_data("b", 0.4, 0.6, 0.4, 0.2)),
        ];
        let report = build_compare_report(&emb, &full);
        assert_eq!(report.cases.len(), 2);
        // avg embedding hit = (0.8 + 0.5)/2 = 0.65
        assert!((report.aggregate.avg_embedding_hit - 0.65).abs() < 1e-6);
        // avg fullpipeline hit = (0.9 + 0.4)/2 = 0.65
        assert!((report.aggregate.avg_fullpipeline_hit - 0.65).abs() < 1e-6);
        // avg embedding mrr = (0.6 + 0.3)/2 = 0.45
        assert!((report.aggregate.avg_embedding_mrr - 0.45).abs() < 1e-6);
        // avg fullpipeline mrr = (0.8 + 0.2)/2 = 0.5
        assert!((report.aggregate.avg_fullpipeline_mrr - 0.5).abs() < 1e-6);
        // hit improvement: a (0.9>0.8) yes, b (0.4>0.5) no → 1
        assert_eq!(report.aggregate.hit_improvement_count, 1);
        // mrr improvement: a (0.8>0.6) yes, b (0.2>0.3) no → 1
        assert_eq!(report.aggregate.mrr_improvement_count, 1);
    }

    #[test]
    fn test_build_compare_report_empty() {
        let report = build_compare_report(&[], &[]);
        assert!(report.cases.is_empty());
        assert_eq!(report.aggregate.case_count, 0);
    }

    #[test]
    fn test_build_compare_report_missing_full_entry() {
        // full 缺少某个 case → 对应分数为 0
        let emb = vec![outcome(case_data("a", 0.5, 0.5, 0.8, 0.6))];
        let full = vec![];
        let report = build_compare_report(&emb, &full);
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.cases[0].fullpipeline_hit, 0.0);
        assert_eq!(report.cases[0].fullpipeline_mrr, 0.0);
        assert!((report.aggregate.avg_fullpipeline_hit - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_compare_report_weight_rounding_key() {
        // tag_weight/variant_weight 相同但 case 名不同 → 独立条目
        let emb = vec![
            outcome(case_data("x", 0.5, 0.5, 0.1, 0.1)),
            outcome(case_data("y", 0.5, 0.5, 0.2, 0.2)),
        ];
        let full = vec![];
        let report = build_compare_report(&emb, &full);
        assert_eq!(report.cases.len(), 2);
    }

    #[test]
    fn test_build_compare_report_weight_scaled() {
        // 权重乘以 100 取整作为 key 的一部分：0.55 与 0.55 应归并
        let emb = vec![outcome(case_data("k", 0.55, 0.45, 0.7, 0.5))];
        let full = vec![outcome(case_data("k", 0.55, 0.45, 0.8, 0.6))];
        let report = build_compare_report(&emb, &full);
        assert_eq!(report.cases.len(), 1);
        assert!((report.cases[0].fullpipeline_hit - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_build_compare_report_equal_values_not_counted_as_improvement() {
        // fullpipeline == embedding 时不计入 improvement（区分 > 与 >=）
        let emb = vec![
            outcome(case_data("a", 0.5, 0.5, 0.8, 0.6)),
            outcome(case_data("b", 0.4, 0.6, 0.5, 0.3)),
        ];
        let full = vec![
            outcome(case_data("a", 0.5, 0.5, 0.8, 0.6)), // 相等
            outcome(case_data("b", 0.4, 0.6, 0.6, 0.4)), // 提升
        ];
        let report = build_compare_report(&emb, &full);
        assert_eq!(report.aggregate.hit_improvement_count, 1);
        assert_eq!(report.aggregate.mrr_improvement_count, 1);
    }
}
