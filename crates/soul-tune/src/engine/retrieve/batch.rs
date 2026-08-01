use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::base::RetrieveMode;
use crate::engine::batch::{scan_question_jsons, DatasetResult};
use crate::engine::compare::build_compare_report;
use crate::engine::retrieve::suite::RetrieveSuite;
use crate::engine::suite::{TestCaseOutcome, TestSuite};

#[derive(Clone)]
pub struct CompareDatasetResult {
    pub name: String,
    pub path: PathBuf,
    pub case_count: usize,
    pub emb_passed: usize,
    pub full_passed: usize,
    pub avg_emb_hit: f64,
    pub avg_full_hit: f64,
    pub hit_delta: f64,
    pub avg_emb_mrr: f64,
    pub avg_full_mrr: f64,
    pub mrr_delta: f64,
    pub elapsed: Duration,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct BatchCompareResult {
    pub datasets: Vec<CompareDatasetResult>,
    pub total_datasets: usize,
    pub avg_emb_hit: f64,
    pub avg_full_hit: f64,
    pub hit_delta: f64,
    pub avg_emb_mrr: f64,
    pub avg_full_mrr: f64,
    pub mrr_delta: f64,
    pub elapsed: Duration,
}

pub fn process_one_dataset(
    path: &Path,
    mode: RetrieveMode,
    params: Option<&HashMap<String, String>>,
    ds_start: Instant,
    progress_cb: impl Fn(f64, &str),
    msg: impl Fn(String),
) -> DatasetResult {
    let name = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());

    progress_cb(0.1, "加载图数据...");
    let load_result = match params {
        Some(p) => RetrieveSuite::load_with_params(path, mode, Some(p)),
        None => RetrieveSuite::load(path, mode),
    };
    match load_result {
        Ok(suite) => {
            let n = suite.case_count();
            if n == 0 {
                msg(format!("  {}: 0 用例", name));
                progress_cb(1.0, "0 用例");
                return DatasetResult {
                    name,
                    path: path.to_path_buf(),
                    total: 0,
                    passed: 0,
                    failed: 0,
                    pass_rate: 0.0,
                    elapsed: ds_start.elapsed(),
                    outcomes: Vec::new(),
                    error: None,
                };
            }
            let mut passed = 0;
            let mut outcomes = Vec::with_capacity(n);
            for j in 0..n {
                let outcome = suite.run_case(j);
                if outcome.passed {
                    passed += 1;
                }
                outcomes.push(outcome);
                progress_cb(
                    0.3 + 0.65 * (j as f64 + 1.0) / n as f64,
                    &format!("运行 {}/{} 测试", j + 1, n),
                );
            }
            progress_cb(
                1.0,
                &format!(
                    "{:.1}s 通过 {}/{} ({:.0}%)",
                    ds_start.elapsed().as_secs_f64(),
                    passed,
                    n,
                    if n > 0 {
                        passed as f64 / n as f64 * 100.0
                    } else {
                        0.0
                    },
                ),
            );
            DatasetResult {
                name,
                path: path.to_path_buf(),
                total: n,
                passed,
                failed: n - passed,
                pass_rate: if n > 0 {
                    passed as f64 / n as f64 * 100.0
                } else {
                    0.0
                },
                elapsed: ds_start.elapsed(),
                outcomes,
                error: None,
            }
        }
        Err(e) => {
            progress_cb(1.0, &format!("✗ {}", e));
            msg(format!("  {}: ✗ {}", name, e));
            DatasetResult {
                name,
                path: path.to_path_buf(),
                total: 0,
                passed: 0,
                failed: 0,
                pass_rate: 0.0,
                elapsed: ds_start.elapsed(),
                outcomes: Vec::new(),
                error: Some(format!("{}", e)),
            }
        }
    }
}

pub fn process_one_compare_dataset(
    path: &Path,
    params: Option<&HashMap<String, String>>,
    ds_start: Instant,
    progress_cb: impl Fn(f64, &str),
    msg: impl Fn(String),
) -> CompareDatasetResult {
    let name = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());

    progress_cb(0.05, "加载 Embedding...");
    let load_emb = || -> Result<_, _> {
        match params {
            Some(p) => RetrieveSuite::load_with_params(path, RetrieveMode::Embedding, Some(p)),
            None => RetrieveSuite::load(path, RetrieveMode::Embedding),
        }
    };
    let emb_suite = match load_emb() {
        Ok(s) => s,
        Err(e) => {
            progress_cb(1.0, &format!("✗ {}", e));
            msg(format!("  {}: ✗ {}", name, e));
            return CompareDatasetResult {
                name,
                path: path.to_path_buf(),
                case_count: 0,
                emb_passed: 0,
                full_passed: 0,
                avg_emb_hit: 0.0,
                avg_full_hit: 0.0,
                hit_delta: 0.0,
                avg_emb_mrr: 0.0,
                avg_full_mrr: 0.0,
                mrr_delta: 0.0,
                elapsed: ds_start.elapsed(),
                error: Some(format!("{}", e)),
            };
        }
    };

    let n = emb_suite.case_count();
    if n == 0 {
        progress_cb(1.0, "0 用例");
        return CompareDatasetResult {
            name,
            path: path.to_path_buf(),
            case_count: 0,
            emb_passed: 0,
            full_passed: 0,
            avg_emb_hit: 0.0,
            avg_full_hit: 0.0,
            hit_delta: 0.0,
            avg_emb_mrr: 0.0,
            avg_full_mrr: 0.0,
            mrr_delta: 0.0,
            elapsed: ds_start.elapsed(),
            error: None,
        };
    }

    let mut emb_outcomes = Vec::with_capacity(n);
    for j in 0..n {
        let outcome = emb_suite.run_case(j);
        emb_outcomes.push(outcome);
        progress_cb(
            0.1 + 0.35 * (j as f64 + 1.0) / n as f64,
            &format!("Embedding {}/{}", j + 1, n),
        );
    }

    progress_cb(0.5, "加载 FullPipeline...");
    let load_full = || -> Result<_, _> {
        match params {
            Some(p) => RetrieveSuite::load_with_params(path, RetrieveMode::FullPipeline, Some(p)),
            None => RetrieveSuite::load(path, RetrieveMode::FullPipeline),
        }
    };
    let full_suite = match load_full() {
        Ok(s) => s,
        Err(e) => {
            progress_cb(1.0, &format!("✗ {}", e));
            msg(format!("  {}: FullPipeline ✗ {}", name, e));
            return CompareDatasetResult {
                name,
                path: path.to_path_buf(),
                case_count: n,
                emb_passed: emb_outcomes.iter().filter(|o| o.passed).count(),
                full_passed: 0,
                avg_emb_hit: 0.0,
                avg_full_hit: 0.0,
                hit_delta: 0.0,
                avg_emb_mrr: 0.0,
                avg_full_mrr: 0.0,
                mrr_delta: 0.0,
                elapsed: ds_start.elapsed(),
                error: Some(format!("FullPipeline: {}", e)),
            };
        }
    };

    let n_full = full_suite.case_count();
    let mut full_outcomes = Vec::with_capacity(n_full);
    for j in 0..n_full {
        let outcome = full_suite.run_case(j);
        full_outcomes.push(outcome);
        progress_cb(
            0.55 + 0.4 * (j as f64 + 1.0) / n_full as f64,
            &format!("FullPipeline {}/{}", j + 1, n_full),
        );
    }

    let report = build_compare_report(&emb_outcomes, &full_outcomes);
    let agg = &report.aggregate;

    let emb_passed = emb_outcomes.iter().filter(|o| o.passed).count();
    let full_passed = full_outcomes.iter().filter(|o| o.passed).count();

    progress_cb(1.0, &format!("{:.1}s", ds_start.elapsed().as_secs_f64()));

    CompareDatasetResult {
        name,
        path: path.to_path_buf(),
        case_count: n,
        emb_passed,
        full_passed,
        avg_emb_hit: agg.avg_embedding_hit,
        avg_full_hit: agg.avg_fullpipeline_hit,
        hit_delta: agg.avg_fullpipeline_hit - agg.avg_embedding_hit,
        avg_emb_mrr: agg.avg_embedding_mrr,
        avg_full_mrr: agg.avg_fullpipeline_mrr,
        mrr_delta: agg.avg_fullpipeline_mrr - agg.avg_embedding_mrr,
        elapsed: ds_start.elapsed(),
        error: None,
    }
}

pub fn run_batch_compare(
    datasets: &[PathBuf],
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> BatchCompareResult {
    let start = Instant::now();
    let total = datasets.len();
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (tx, rx) = std::sync::mpsc::channel::<(usize, CompareDatasetResult)>();
    let n_workers = 4.min(total).max(1);

    for _ in 0..n_workers {
        let datasets = datasets.to_vec();
        let counter = std::sync::Arc::clone(&counter);
        let tx = tx.clone();
        std::thread::Builder::new()
            .name("batch-compare-worker".into())
            .spawn(move || loop {
                let i = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= datasets.len() {
                    break;
                }
                //worker内panic（如模型加载失败）会静默杀死线程导致batch挂起，
                //这里捕获panic并报告错误结果，保证每个任务都有产出
                let ds_start = Instant::now();
                let ds = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    process_one_compare_dataset(&datasets[i], None, ds_start, |_, _| {}, |_| {})
                }))
                .unwrap_or_else(|_| CompareDatasetResult {
                    name: datasets[i]
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    path: datasets[i].clone(),
                    case_count: 0,
                    emb_passed: 0,
                    full_passed: 0,
                    avg_emb_hit: 0.0,
                    avg_full_hit: 0.0,
                    hit_delta: 0.0,
                    avg_emb_mrr: 0.0,
                    avg_full_mrr: 0.0,
                    mrr_delta: 0.0,
                    elapsed: ds_start.elapsed(),
                    error: Some("worker panic".to_string()),
                });
                let _ = tx.send((i, ds));
            })
            .ok();
    }
    drop(tx);

    let mut placed = 0usize;
    if let Some(cb) = on_progress {
        cb(0, total);
    }
    let mut results = Vec::new();
    for (idx, ds) in rx {
        placed += 1;
        results.push((idx, ds));
        if let Some(cb) = on_progress {
            cb(placed, total);
        }
    }

    results.sort_by_key(|(idx, _)| *idx);
    let datasets: Vec<CompareDatasetResult> = results.into_iter().map(|(_, ds)| ds).collect();

    let total_datasets = datasets.len();
    let avg_emb_hit = if total_datasets > 0 {
        datasets.iter().map(|d| d.avg_emb_hit).sum::<f64>() / total_datasets as f64
    } else {
        0.0
    };
    let avg_full_hit = if total_datasets > 0 {
        datasets.iter().map(|d| d.avg_full_hit).sum::<f64>() / total_datasets as f64
    } else {
        0.0
    };
    let avg_emb_mrr = if total_datasets > 0 {
        datasets.iter().map(|d| d.avg_emb_mrr).sum::<f64>() / total_datasets as f64
    } else {
        0.0
    };
    let avg_full_mrr = if total_datasets > 0 {
        datasets.iter().map(|d| d.avg_full_mrr).sum::<f64>() / total_datasets as f64
    } else {
        0.0
    };

    BatchCompareResult {
        datasets,
        total_datasets,
        avg_emb_hit,
        avg_full_hit,
        hit_delta: avg_full_hit - avg_emb_hit,
        avg_emb_mrr,
        avg_full_mrr,
        mrr_delta: avg_full_mrr - avg_emb_mrr,
        elapsed: start.elapsed(),
    }
}
