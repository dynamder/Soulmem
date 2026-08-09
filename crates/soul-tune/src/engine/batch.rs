use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::base::RetrieveMode;

use crate::engine::suite::TestCaseOutcome;
use crate::engine::retrieve::data::RetrieveCaseData;

pub struct BatchResult {
    pub datasets: Vec<DatasetResult>,
    pub total_cases: usize,
    pub total_passed: usize,
    pub total_failed: usize,
    pub elapsed: Duration,
}

pub struct DatasetResult {
    pub name: String,
    pub path: PathBuf,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub elapsed: Duration,
    pub outcomes: Vec<TestCaseOutcome>,
    pub error: Option<String>,
}

pub fn scan_question_jsons(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if !dir.is_dir() {
        return results;
    }
    scan_recursive(dir, &mut results);
    results
}

/// 动作命中汇总：只统计带 expected_actions 真值的用例（占位指标不参与）。
#[derive(Debug, Clone, Copy)]
pub struct ActionSummary {
    pub cases: usize,
    pub hit_cases: usize,
    pub recall_at3: f64,
}

impl ActionSummary {
    pub fn hit_rate(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.hit_cases as f64 / self.cases as f64
        }
    }

    pub fn combine(a: Option<&ActionSummary>, b: Option<&ActionSummary>) -> Option<ActionSummary> {
        match (a, b) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(*x),
            (Some(x), Some(y)) => Some(ActionSummary {
                cases: x.cases + y.cases,
                hit_cases: x.hit_cases + y.hit_cases,
                recall_at3: (x.recall_at3 * x.cases as f64 + y.recall_at3 * y.cases as f64)
                    / (x.cases + y.cases) as f64,
            }),
        }
    }
}

/// 从用例结果中汇总动作命中指标。
pub fn summarize_action_metrics(outcomes: &[TestCaseOutcome]) -> Option<ActionSummary> {
    let mut summary: Option<ActionSummary> = None;
    for o in outcomes {
        if let Some(data) = o.data.downcast_ref::<RetrieveCaseData>() {
            if !data.action_metrics.has_expected_actions {
                continue;
            }
            let recall_at3 = data
                .action_metrics
                .action_recall_at
                .iter()
                .find(|(k, _)| *k == 3)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            let s = ActionSummary {
                cases: 1,
                hit_cases: if data.action_metrics.action_hit_rate > 0.0 { 1 } else { 0 },
                recall_at3,
            };
            summary = ActionSummary::combine(summary.as_ref(), Some(&s));
        }
    }
    summary
}

fn scan_recursive(dir: &Path, results: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_recursive(&path, results);
            } else if path
                .file_name()
                .map(|n| n == "question.json")
                .unwrap_or(false)
            {
                results.push(path);
            }
        }
    }
}

pub fn run_batch(
    datasets: &[PathBuf],
    mode: RetrieveMode,
    processor: impl Fn(&Path, RetrieveMode, Option<&HashMap<String, String>>, Instant) -> DatasetResult
        + Send
        + Sync
        + 'static,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> BatchResult {
    let start = Instant::now();
    let total = datasets.len();
    let mut total_cases = 0;
    let mut total_passed = 0;
    let mut total_failed = 0;

    let counter = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<(usize, DatasetResult)>();
    let n_workers = 4.min(total).max(1);
    let processor = Arc::new(processor);

    for _ in 0..n_workers {
        let datasets = datasets.to_vec();
        let mode = mode;
        let counter = Arc::clone(&counter);
        let tx = tx.clone();
        let processor = Arc::clone(&processor);
        std::thread::Builder::new()
            .name("batch-worker".into())
            .spawn(move || loop {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i >= datasets.len() {
                    break;
                }
                let ds_start = Instant::now();
                let ds = processor(&datasets[i], mode, None, ds_start);
                let _ = tx.send((i, ds));
            })
            .ok();
    }
    drop(tx);

    let mut results = Vec::with_capacity(total);
    let mut placed = 0usize;
    if let Some(cb) = on_progress {
        cb(0, total);
    }
    for (idx, ds) in rx {
        placed += 1;
        total_cases += ds.total;
        total_passed += ds.passed;
        total_failed += ds.failed;
        results.push((idx, ds));
        if let Some(cb) = on_progress {
            cb(placed, total);
        }
    }

    results.sort_by_key(|(idx, _)| *idx);
    let datasets: Vec<DatasetResult> = results.into_iter().map(|(_, ds)| ds).collect();

    BatchResult {
        datasets,
        total_cases,
        total_passed,
        total_failed,
        elapsed: start.elapsed(),
    }
}

pub fn print_batch_result(result: &BatchResult) {
    println!(
        "=== 批量测试结果 ===\n数据集数: {} | 总用例: {} | 通过: {} | 失败: {} | 总耗时: {:.2}s\n",
        result.datasets.len(),
        result.total_cases,
        result.total_passed,
        result.total_failed,
        result.elapsed.as_secs_f64(),
    );

    let mut sorted: Vec<&DatasetResult> = result.datasets.iter().collect();
    sorted.sort_by(|a, b| {
        b.pass_rate
            .partial_cmp(&a.pass_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{:<30} {:>5} {:>5} {:>5} {:>7} {:>8} {:>8}",
        "数据集", "用例", "通过", "失败", "通过率", "动作Hit", "耗时"
    );
    println!("{}", "-".repeat(70));
    let mut total_summary: Option<ActionSummary> = None;
    for ds in &sorted {
        let ok_mark = if ds.error.is_some() {
            "!"
        } else if ds.pass_rate >= 80.0 {
            "+"
        } else {
            "-"
        };
        let action_col = match summarize_action_metrics(&ds.outcomes) {
            Some(s) => {
                total_summary = ActionSummary::combine(total_summary.as_ref(), Some(&s));
                format!("{:>7.1}%", s.hit_rate() * 100.0)
            }
            None => format!("{:>8}", "-"),
        };
        println!(
            "{:<30} {:>5} {:>5} {:>5} {:>6.1}% {} {:>7.2}s {}",
            ds.name.chars().take(28).collect::<String>(),
            ds.total,
            ds.passed,
            ds.failed,
            ds.pass_rate,
            action_col,
            ds.elapsed.as_secs_f64(),
            ok_mark,
        );
    }
    match total_summary {
        Some(s) => println!(
            "动作评测: {} 个带期望动作的用例 | 动作Hit {:.1}% | Recall@3 {:.3}",
            s.cases,
            s.hit_rate() * 100.0,
            s.recall_at3
        ),
        None => println!("动作评测: 无带 expected_actions 的用例"),
    }
    println!("{}", "-".repeat(70));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::engine::retrieve::data::{ActionMetrics, RankingMetrics};

    fn fixtures_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures")
    }

    #[test]
    fn test_scan_finds_question_jsons() {
        let data_dir = fixtures_dir().join("example_data");
        let results = scan_question_jsons(&data_dir);
        assert!(
            !results.is_empty(),
            "example_data dir should contain question.json files in subdirectories"
        );
    }

    #[test]
    fn test_scan_non_existent_dir() {
        let results = scan_question_jsons(Path::new("/nonexistent/dir/_soul_tune_test_"));
        assert!(results.is_empty(), "non-existent dir should return empty");
    }

    fn outcome_with_action(hit: f64, recall3: f64, has_expected: bool) -> TestCaseOutcome {
        let data = RetrieveCaseData {
            case_name: "c".into(),
            description: String::new(),
            combined_retrieved_ids: vec![],
            combined_ranking_metrics: RankingMetrics {
                recall_at: vec![(1, 0.0)],
                precision_at: vec![(1, 0.0)],
                mrr: 0.0,
                ndcg_at: vec![(1, 0.0)],
                hit_rate: 0.0,
            },
            per_query_metrics: vec![],
            action_metrics: ActionMetrics {
                action_hit_rate: hit,
                action_recall_at: vec![(1, recall3), (3, recall3), (5, recall3)],
                has_expected_actions: has_expected,
            },
            tag_weight: 0.3,
            variant_weight: 0.7,
            id_names: None,
            expected_combined_ranking: vec![],
            bonus_combined_ranking: vec![],
            graph_names: None,
            sub_queries: vec![],
        };
        TestCaseOutcome {
            case_name: "c".into(),
            description: String::new(),
            passed: true,
            data: Box::new(data),
        }
    }

    #[test]
    fn test_summarize_action_metrics_empty() {
        assert!(summarize_action_metrics(&[]).is_none());
    }

    #[test]
    fn test_summarize_action_metrics_ignores_placeholder() {
        // 只有占位指标（无 expected_actions）→ None
        let outcomes = vec![outcome_with_action(1.0, 1.0, false)];
        assert!(summarize_action_metrics(&outcomes).is_none());
    }

    #[test]
    fn test_summarize_action_metrics_counts_only_expected_cases() {
        let outcomes = vec![
            outcome_with_action(1.0, 0.5, true),
            outcome_with_action(1.0, 0.5, true),
            outcome_with_action(0.0, 0.0, true),
            outcome_with_action(1.0, 1.0, false), // 占位不计入
        ];
        let s = summarize_action_metrics(&outcomes).expect("should summarize");
        assert_eq!(s.cases, 3);
        assert_eq!(s.hit_cases, 2);
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.recall_at3 - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_summary_combine_weighted_recall() {
        let a = ActionSummary { cases: 2, hit_cases: 2, recall_at3: 0.5 };
        let b = ActionSummary { cases: 1, hit_cases: 0, recall_at3: 0.0 };
        let c = ActionSummary::combine(Some(&a), Some(&b)).unwrap();
        assert_eq!(c.cases, 3);
        assert_eq!(c.hit_cases, 2);
        assert!((c.recall_at3 - 1.0 / 3.0).abs() < 1e-9);
        assert!(ActionSummary::combine(None, None).is_none());
    }

    #[test]
    fn test_scan_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let results = scan_question_jsons(dir.path());
        assert!(results.is_empty(), "empty dir should return empty");
    }

    #[test]
    fn test_dataset_result_defaults() {
        let result = DatasetResult {
            name: "test".into(),
            path: Path::new("/tmp/test.json").into(),
            total: 0,
            passed: 0,
            failed: 0,
            pass_rate: 0.0,
            elapsed: std::time::Duration::ZERO,
            outcomes: vec![],
            error: None,
        };
        assert_eq!(result.name, "test");
        assert!(result.error.is_none());
    }
}
