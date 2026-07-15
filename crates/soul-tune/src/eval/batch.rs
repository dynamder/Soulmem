use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::base::RetrieveMode;
use crate::eval::retrieve_suite::RetrieveSuite;
use crate::eval::runner::SuiteReport;
use crate::eval::runner::TestSuite;

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
    pub error: Option<String>,
}

/// Recursively find all `question.json` files under `dir`.
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

/// Run a batch of datasets sequentially.
/// Reports progress to `on_progress(done, total)` if provided.
pub fn run_batch(
    datasets: &[PathBuf],
    mode: RetrieveMode,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> BatchResult {
    let start = Instant::now();
    let mut results = Vec::with_capacity(datasets.len());
    let mut total_cases = 0;
    let mut total_passed = 0;
    let mut total_failed = 0;

    for (i, path) in datasets.iter().enumerate() {
        if let Some(cb) = on_progress {
            cb(i, datasets.len());
        }

        let ds_start = Instant::now();
        let name = path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string());

        match RetrieveSuite::load(path, mode) {
            Ok(suite) => {
                let n = suite.case_count();
                let mut passed = 0;
                let mut outcomes = Vec::with_capacity(n);
                for j in 0..n {
                    let outcome = suite.run_case(j);
                    if outcome.passed {
                        passed += 1;
                    }
                    outcomes.push(outcome);
                }
                let ds_elapsed = ds_start.elapsed();
                total_cases += n;
                total_passed += passed;
                total_failed += n - passed;
                results.push(DatasetResult {
                    name,
                    path: path.clone(),
                    total: n,
                    passed,
                    failed: n - passed,
                    pass_rate: if n > 0 {
                        passed as f64 / n as f64 * 100.0
                    } else {
                        0.0
                    },
                    elapsed: ds_start.elapsed(),
                    error: None,
                });
            }
            Err(e) => {
                results.push(DatasetResult {
                    name,
                    path: path.clone(),
                    total: 0,
                    passed: 0,
                    failed: 0,
                    pass_rate: 0.0,
                    elapsed: ds_start.elapsed(),
                    error: Some(format!("{}", e)),
                });
            }
        }
    }

    if let Some(cb) = on_progress {
        cb(datasets.len(), datasets.len());
    }

    BatchResult {
        datasets: results,
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
