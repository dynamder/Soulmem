//! 同管线「直接（全量工作记忆）vs 数据库召回」逐用例对比。
//!
//! 同一 question.json、同一 [`RetrieveFlavor`]（embedding/association/full）下，
//! 分别以两种记忆来源各跑一遍检索测试：
//! - 直接：example_data 全量载入工作记忆（`RetrieveMode::*`）；
//! - 数据库：example_data 先写入 mem 数据库，经 DB 召回（候选 + 一跳邻居）
//!   后在工作记忆子图上执行同一条管线（`RetrieveMode::*Db`）。
//!
//! 产出逐用例指标（hit/MRR/recall@k/检索序列/期望排名）与差量，以及聚合汇总，
//! 用于观测数据库召回路径相对全量直接检索的覆盖与精度差异。

use std::collections::HashMap;
use std::path::Path;

use serde::Serialize;
use soul_mem_core::memory_note::MemoryId;

use crate::base::{RetrieveFlavor, RetrieveMode};
use crate::engine::retrieve::data::RetrieveCaseData;
use crate::engine::retrieve::suite::RetrieveSuite;
use crate::engine::suite::{TestCaseOutcome, TestSuite};

/// 逐用例对比数据（两侧分数 + 差量 + 提升/回退/持平标志）。
#[derive(Clone, Serialize)]
pub struct DbCompareCaseData {
    pub case_name: String,
    pub description: String,
    pub tag_weight: f32,
    pub variant_weight: f32,

    pub direct_hit: f64,
    pub db_hit: f64,
    pub direct_mrr: f64,
    pub db_mrr: f64,
    pub direct_recall_at: Vec<(usize, f64)>,
    pub db_recall_at: Vec<(usize, f64)>,
    pub direct_precision_at: Vec<(usize, f64)>,
    pub db_precision_at: Vec<(usize, f64)>,

    pub direct_retrieved: Vec<MemoryId>,
    pub db_retrieved: Vec<MemoryId>,
    pub expected_combined_ranking: Vec<MemoryId>,

    pub hit_delta: f64,
    pub mrr_delta: f64,
    pub improved_hit: bool,
    pub regressed_hit: bool,
    pub improved_mrr: bool,
    pub regressed_mrr: bool,
}

/// 聚合汇总：均值、差量与提升/回退/持平计数。
#[derive(Clone, Default, Serialize)]
pub struct DbCompareAggregate {
    pub case_count: usize,
    pub avg_direct_hit: f64,
    pub avg_db_hit: f64,
    pub hit_delta: f64,
    pub avg_direct_mrr: f64,
    pub avg_db_mrr: f64,
    pub mrr_delta: f64,
    pub avg_direct_recall3: f64,
    pub avg_db_recall3: f64,
    pub recall3_delta: f64,
    pub hit_improved_count: usize,
    pub hit_regressed_count: usize,
    pub hit_equal_count: usize,
    pub mrr_improved_count: usize,
    pub mrr_regressed_count: usize,
    pub mrr_equal_count: usize,
}

#[derive(Clone, Serialize)]
pub struct DbCompareReport {
    pub flavor: String,
    pub cases: Vec<DbCompareCaseData>,
    pub aggregate: DbCompareAggregate,
}

/// 直接模式对应的 RetrieveMode（同管线）。
fn mode_of(flavor: RetrieveFlavor) -> RetrieveMode {
    match flavor {
        RetrieveFlavor::Embedding => RetrieveMode::Embedding,
        RetrieveFlavor::Association => RetrieveMode::Association,
        RetrieveFlavor::FullPipeline => RetrieveMode::FullPipeline,
    }
}

fn recall_at3(recall_at: &[(usize, f64)]) -> f64 {
    recall_at
        .iter()
        .find(|(k, _)| *k == 3)
        .map(|(_, v)| *v)
        .unwrap_or(0.0)
}

/// 依次跑"直接"与"数据库"两套件（同一 question.json、同一管线），产出对比报告。
///
/// 任一套件加载失败都会返回错误说明；单个用例运行不失败（失败以零指标用例呈现）。
pub fn run_db_compare(
    dataset_path: &Path,
    flavor: RetrieveFlavor,
) -> Result<DbCompareReport, String> {
    let direct_mode = mode_of(flavor);
    let db_mode = direct_mode
        .db_mode()
        .expect("direct mode always maps to a db mode");

    let direct_suite = RetrieveSuite::load(dataset_path, direct_mode)
        .map_err(|e| format!("加载直接套件失败: {e}"))?;
    let db_suite = RetrieveSuite::load(dataset_path, db_mode)
        .map_err(|e| format!("加载数据库套件失败: {e}"))?;

    let n = direct_suite.case_count();
    let mut direct_outcomes = Vec::with_capacity(n);
    for i in 0..n {
        direct_outcomes.push(direct_suite.run_case(i));
    }
    let n_db = db_suite.case_count();
    let mut db_outcomes = Vec::with_capacity(n_db);
    for i in 0..n_db {
        db_outcomes.push(db_suite.run_case(i));
    }

    Ok(build_db_compare_report(
        &direct_outcomes,
        &db_outcomes,
        flavor.to_string(),
    ))
}

type CaseKey = (String, u32, u32);

fn key_of(data: &RetrieveCaseData) -> CaseKey {
    (
        data.case_name.clone(),
        (data.tag_weight * 100.0).round() as u32,
        (data.variant_weight * 100.0).round() as u32,
    )
}

/// 由两侧用例结果构建对比报告（两侧套件应来自同一 question.json 与管线）。
pub fn build_db_compare_report(
    direct_outcomes: &[TestCaseOutcome],
    db_outcomes: &[TestCaseOutcome],
    flavor: String,
) -> DbCompareReport {
    let direct_map: HashMap<CaseKey, RetrieveCaseData> = direct_outcomes
        .iter()
        .filter_map(|o| {
            o.data
                .downcast_ref::<RetrieveCaseData>()
                .map(|d| (key_of(d), d.clone()))
        })
        .collect();
    let db_map: HashMap<CaseKey, RetrieveCaseData> = db_outcomes
        .iter()
        .filter_map(|o| {
            o.data
                .downcast_ref::<RetrieveCaseData>()
                .map(|d| (key_of(d), d.clone()))
        })
        .collect();

    let mut keys: Vec<CaseKey> = direct_map.keys().cloned().collect();
    keys.sort();

    let mut cases = Vec::with_capacity(keys.len());
    for key in &keys {
        let direct = &direct_map[key];
        let db = db_map.get(key);
        let direct_hit = direct.combined_ranking_metrics.hit_rate;
        let db_hit = db.map(|d| d.combined_ranking_metrics.hit_rate).unwrap_or(0.0);
        let direct_mrr = direct.combined_ranking_metrics.mrr;
        let db_mrr = db.map(|d| d.combined_ranking_metrics.mrr).unwrap_or(0.0);

        cases.push(DbCompareCaseData {
            case_name: direct.case_name.clone(),
            description: direct.description.clone(),
            tag_weight: direct.tag_weight,
            variant_weight: direct.variant_weight,
            direct_hit,
            db_hit,
            direct_mrr,
            db_mrr,
            direct_recall_at: direct.combined_ranking_metrics.recall_at.clone(),
            db_recall_at: db
                .map(|d| d.combined_ranking_metrics.recall_at.clone())
                .unwrap_or_default(),
            direct_precision_at: direct.combined_ranking_metrics.precision_at.clone(),
            db_precision_at: db
                .map(|d| d.combined_ranking_metrics.precision_at.clone())
                .unwrap_or_default(),
            direct_retrieved: direct.combined_retrieved_ids.clone(),
            db_retrieved: db.map(|d| d.combined_retrieved_ids.clone()).unwrap_or_default(),
            expected_combined_ranking: direct.expected_combined_ranking.clone(),
            hit_delta: db_hit - direct_hit,
            mrr_delta: db_mrr - direct_mrr,
            improved_hit: db_hit > direct_hit,
            regressed_hit: db_hit < direct_hit,
            improved_mrr: db_mrr > direct_mrr,
            regressed_mrr: db_mrr < direct_mrr,
        });
    }

    let case_count = cases.len();
    let mut agg = DbCompareAggregate {
        case_count,
        ..Default::default()
    };
    if case_count > 0 {
        let n = case_count as f64;
        agg.avg_direct_hit = cases.iter().map(|c| c.direct_hit).sum::<f64>() / n;
        agg.avg_db_hit = cases.iter().map(|c| c.db_hit).sum::<f64>() / n;
        agg.hit_delta = agg.avg_db_hit - agg.avg_direct_hit;
        agg.avg_direct_mrr = cases.iter().map(|c| c.direct_mrr).sum::<f64>() / n;
        agg.avg_db_mrr = cases.iter().map(|c| c.db_mrr).sum::<f64>() / n;
        agg.mrr_delta = agg.avg_db_mrr - agg.avg_direct_mrr;
        agg.avg_direct_recall3 = cases.iter().map(|c| recall_at3(&c.direct_recall_at)).sum::<f64>() / n;
        agg.avg_db_recall3 = cases.iter().map(|c| recall_at3(&c.db_recall_at)).sum::<f64>() / n;
        agg.recall3_delta = agg.avg_db_recall3 - agg.avg_direct_recall3;
        agg.hit_improved_count = cases.iter().filter(|c| c.improved_hit).count();
        agg.hit_regressed_count = cases.iter().filter(|c| c.regressed_hit).count();
        agg.hit_equal_count = case_count - agg.hit_improved_count - agg.hit_regressed_count;
        agg.mrr_improved_count = cases.iter().filter(|c| c.improved_mrr).count();
        agg.mrr_regressed_count = cases.iter().filter(|c| c.regressed_mrr).count();
        agg.mrr_equal_count = case_count - agg.mrr_improved_count - agg.mrr_regressed_count;
    }

    DbCompareReport {
        flavor,
        cases,
        aggregate: agg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::retrieve::data::{ActionMetrics, RankingMetrics};
    use crate::engine::retrieve::dataset::SubQuery;
    use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

    fn metrics(hit: f64, mrr: f64, recall3: f64) -> RankingMetrics {
        RankingMetrics {
            recall_at: vec![(3, recall3), (5, hit)],
            precision_at: vec![(3, recall3), (5, hit)],
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
            combined_ranking_metrics: metrics(hit, mrr, hit),
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
            sub_queries: vec![SubQuery {
                priority: 1,
                tags: vec![],
                variant: MemoryRetrieveQueryVariant::Semantic(vec![]),
            }],
        }
    }

    fn outcome(case: RetrieveCaseData) -> TestCaseOutcome {
        let name = case.case_name.clone();
        let description = case.description.clone();
        TestCaseOutcome {
            case_name: name,
            description,
            passed: true,
            data: Box::new(case),
        }
    }

    #[test]
    fn test_build_db_compare_report_basic() {
        let direct = vec![
            outcome(case_data("a", 0.5, 0.5, 0.8, 0.6)),
            outcome(case_data("b", 0.4, 0.6, 0.5, 0.3)),
        ];
        let db = vec![
            outcome(case_data("a", 0.5, 0.5, 0.9, 0.8)),
            outcome(case_data("b", 0.4, 0.6, 0.4, 0.2)),
        ];
        let report = build_db_compare_report(&direct, &db, "full".into());
        assert_eq!(report.cases.len(), 2);
        assert!((report.aggregate.avg_direct_hit - 0.65).abs() < 1e-6);
        assert!((report.aggregate.avg_db_hit - 0.65).abs() < 1e-6);
        assert!((report.aggregate.avg_direct_mrr - 0.45).abs() < 1e-6);
        assert!((report.aggregate.avg_db_mrr - 0.5).abs() < 1e-6);
        // a 提升 hit/mrr；b 回退 hit/mrr → 各 1
        assert_eq!(report.aggregate.hit_improved_count, 1);
        assert_eq!(report.aggregate.hit_regressed_count, 1);
        assert_eq!(report.aggregate.mrr_improved_count, 1);
        assert_eq!(report.aggregate.mrr_regressed_count, 1);
    }

    #[test]
    fn test_build_db_compare_report_empty() {
        let report = build_db_compare_report(&[], &[], "full".into());
        assert!(report.cases.is_empty());
        assert_eq!(report.aggregate.case_count, 0);
    }

    #[test]
    fn test_build_db_compare_report_missing_db_entry_counts_as_regression() {
        let direct = vec![outcome(case_data("a", 0.5, 0.5, 0.8, 0.6))];
        let report = build_db_compare_report(&direct, &[], "embedding".into());
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.cases[0].db_hit, 0.0);
        assert_eq!(report.cases[0].db_mrr, 0.0);
        assert!(report.cases[0].regressed_hit);
        assert!(report.cases[0].regressed_mrr);
    }

    #[test]
    fn test_build_db_compare_report_equal_not_counted() {
        let direct = vec![outcome(case_data("a", 0.5, 0.5, 0.8, 0.6))];
        let db = vec![outcome(case_data("a", 0.5, 0.5, 0.8, 0.6))];
        let report = build_db_compare_report(&direct, &db, "full".into());
        assert_eq!(report.aggregate.hit_improved_count, 0);
        assert_eq!(report.aggregate.hit_regressed_count, 0);
        assert_eq!(report.aggregate.hit_equal_count, 1);
        assert_eq!(report.aggregate.mrr_equal_count, 1);
        assert_eq!(report.aggregate.hit_delta, 0.0);
    }

    #[test]
    fn test_build_db_compare_report_weight_rounding_key() {
        // 权重 ×100 取整归并；case 名不同 → 独立条目
        let direct = vec![
            outcome(case_data("x", 0.55, 0.45, 0.1, 0.1)),
            outcome(case_data("y", 0.55, 0.45, 0.2, 0.2)),
        ];
        let report = build_db_compare_report(&direct, &[], "full".into());
        assert_eq!(report.cases.len(), 2);
    }

    #[test]
    fn test_recall_at3_finder() {
        assert!((recall_at3(&[(1, 0.5), (3, 0.75)]) - 0.75).abs() < 1e-6);
        assert!((recall_at3(&[(1, 0.5)]) - 0.0).abs() < 1e-6);
    }
}
