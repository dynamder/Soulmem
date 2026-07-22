use std::collections::HashMap;

use soul_mem_core::memory_note::MemoryId;

use crate::engine::retrieve::data::RetrieveCaseData;
use crate::engine::suite::TestCaseOutcome;

#[derive(Clone)]
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

#[derive(Clone, Default)]
pub struct CompareAggregate {
    pub case_count: usize,
    pub avg_embedding_hit: f64,
    pub avg_fullpipeline_hit: f64,
    pub avg_embedding_mrr: f64,
    pub avg_fullpipeline_mrr: f64,
    pub hit_improvement_count: usize,
    pub mrr_improvement_count: usize,
}

#[derive(Clone)]
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
