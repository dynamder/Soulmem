pub(crate) mod app;
pub(crate) mod base;
pub(crate) mod cmd;
pub(crate) mod eval;
pub(crate) mod metric;
pub(crate) mod reporter;
pub(crate) mod state;
pub(crate) mod tui;
pub(crate) mod utils;

use std::path::PathBuf;

use base::{AlgoType, TestReport};
use eval::retrieve_suite::RetrieveSuite;
use eval::runner::TestSuite;

fn main() -> color_eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 4 && args[1] == "run" {
        // Headless mode: soul-tune run <algo> <dataset_path>
        let algo_str = &args[2];
        let dataset_path = PathBuf::from(&args[3]);

        let algo = match algo_str.as_str() {
            "retrieve" | "r" => AlgoType::Retrieve,
            "consolidate" | "c" => AlgoType::Consolidate,
            "forget" | "f" => AlgoType::Forget,
            _ => {
                eprintln!(
                    "未知算法: {} (可选: retrieve, consolidate, forget)",
                    algo_str
                );
                std::process::exit(1);
            }
        };

        run_headless(algo, dataset_path)?;
    } else {
        // TUI mode
        color_eyre::install()?;
        let mut app = app::App::new()?;
        app.run()?;
    }

    Ok(())
}

fn run_headless(algo: AlgoType, dataset_path: PathBuf) -> color_eyre::Result<()> {
    let dataset_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    println!("=== 测试加载中 ===");
    println!("算法: {}", algo);
    println!("数据集: {}\n", dataset_name);

    let suite = match algo {
        AlgoType::Retrieve => {
            RetrieveSuite::load(&dataset_path).map_err(|e| color_eyre::eyre::eyre!("{}", e))?
        }
        _ => {
            eprintln!("{} 尚未支持 headless 模式", algo);
            std::process::exit(1);
        }
    };

    let n = suite.case_count();
    println!("共 {} 个测试用例\n", n);

    let start = std::time::Instant::now();
    let mut outcomes = Vec::with_capacity(n);
    let mut passed = 0;
    let mut failed = 0;

    for i in 0..n {
        let outcome = suite.run_case(i);
        if outcome.passed {
            passed += 1;
        } else {
            failed += 1;
        }
        outcomes.push(outcome);
    }

    let elapsed = start.elapsed();

    let report = suite.build_report(outcomes, elapsed, n, passed, failed);

    print_report(&TestReport {
        config: base::TestConfig {
            algo,
            dataset_path,
            params: std::collections::HashMap::new(),
        },
        total: n,
        passed,
        failed,
        elapsed,
        suite_report: report,
    });

    Ok(())
}

fn print_report(report: &TestReport) {
    let n = report.total;
    let pass_rate = if n > 0 {
        report.passed as f64 / n as f64 * 100.0
    } else {
        0.0
    };

    println!(
        "=== 测试结果 ===\n总用例: {} | 通过: {} | 失败: {} | 通过率: {:.1}% | 耗时: {:.2}s\n",
        n,
        report.passed,
        report.failed,
        pass_rate,
        report.elapsed.as_secs_f64(),
    );

    // Summary groups
    for group in &report.suite_report.summary_groups {
        println!("--- {} ---", group.label);
        for (k, v) in &group.items {
            println!("  {}: {}", k, v);
        }
        println!();
    }

    // Detail
    if !report.suite_report.detail_header.is_empty() {
        println!("--- 详细 ---");
        println!("{}", report.suite_report.detail_header);
        for row in &report.suite_report.detail_rows {
            let marker = if row.has_error { "✗" } else { "✓" };
            println!("{} {}", marker, row.text);
        }
    }
}
