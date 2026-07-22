use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::base::RetrieveMode;

use crate::engine::suite::TestCaseOutcome;

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
        "{:<30} {:>5} {:>5} {:>5} {:>7} {:>8}",
        "数据集", "用例", "通过", "失败", "通过率", "耗时"
    );
    println!("{}", "-".repeat(70));
    for ds in &sorted {
        let ok_mark = if ds.error.is_some() {
            "!"
        } else if ds.pass_rate >= 80.0 {
            "+"
        } else {
            "-"
        };
        println!(
            "{:<30} {:>5} {:>5} {:>5} {:>6.1}% {:>7.2}s {}",
            ds.name.chars().take(28).collect::<String>(),
            ds.total,
            ds.passed,
            ds.failed,
            ds.pass_rate,
            ds.elapsed.as_secs_f64(),
            ok_mark,
        );
    }
    println!("{}", "-".repeat(70));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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
