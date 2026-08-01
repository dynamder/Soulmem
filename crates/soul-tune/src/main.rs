mod app;
mod base;
mod cmd;
mod component;
mod engine;
mod states;
mod widgets;
mod utils;

#[cfg(test)]
mod tests_state_machine;
#[cfg(test)]
mod tests_widgets;
#[cfg(test)]
mod tests_states;
#[cfg(test)]
mod tests_playtest_mock;
#[cfg(test)]
mod tests_cli;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base::{AlgoType, RetrieveMode, TestReport};
use engine::batch::{print_batch_result, run_batch, scan_question_jsons};
use engine::llm::LlamaServer;
use engine::playtest::{DialogueFile, PlayTestRunner};
use engine::retrieve::batch::process_one_dataset;
use engine::retrieve::RetrieveSuite;
use engine::suite::{MetricFormat, ReportMetric, TestSuite};
use states::inspect::{InspectFileType, InspectState};

fn main() -> color_eyre::Result<()> {
    dotenvy::dotenv().ok();
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 3 && args[1] == "inspect" {
        let path_str = &args[2];
        let path = PathBuf::from(path_str);
        if !path.exists() {
            eprintln!("路径不存在: {}", path_str);
            std::process::exit(1);
        }
        let state = InspectState::new(path);
        println!("=== 检视数据集 ===");
        println!("文件: {}", state.file_path.display());
        println!(
            "类型: {}",
            match state.file_type {
                InspectFileType::Graph => "图 (Graph)",
                InspectFileType::Query => "查询 (Query)",
            }
        );
        println!("条目数: {}", state.entries.len());
        if let Some(ref stats) = state.stats {
            println!("图统计:");
            for line in stats {
                println!("  {}", line);
            }
        }
        println!();
        for (i, entry) in state.entries.iter().enumerate() {
            println!("[{:>3}] {}", i, entry.summary);
            for line in &entry.detail_lines {
                println!("      {}", line);
            }
            if !entry.links.is_empty() {
                println!("      连接:");
                for l in &entry.links {
                    let dir = if l.is_outgoing { "→" } else { "←" };
                    println!(
                        "        {} {}  {}  [{:.2}] {}",
                        dir,
                        if l.is_outgoing {
                            l.to_id.clone()
                        } else {
                            l.from_id.clone()
                        },
                        l.link_type_desc,
                        l.intensity,
                        if l.is_outgoing { "(出)" } else { "(入)" },
                    );
                }
            }
            println!();
        }
        return Ok(());
    }

    if args.len() >= 3 && args[1] == "playtest" {
        return run_headless_playtest(&args);
    }

    if args.len() >= 4 && args[1] == "run" {
        let is_batch = args.iter().any(|a| a == "--batch");
        //固定位置解析：run <algo> <dataset> [--batch]，--batch只是开关，不影响位置
        let algo_str = &args[2];
        let path_str = &args[3];
        let dataset_path = PathBuf::from(path_str);

        let algo = match algo_str.as_str() {
            "retrieve" | "r" | "retrieve/embedding" | "re" => {
                AlgoType::Retrieve(RetrieveMode::Embedding)
            }
            "retrieve/association" | "ra" => AlgoType::Retrieve(RetrieveMode::Association),
            "retrieve/full" | "rf" => AlgoType::Retrieve(RetrieveMode::FullPipeline),
            "consolidate" | "c" => AlgoType::Consolidate,
            "forget" | "f" => AlgoType::Forget,
            _ => {
                eprintln!("未知算法: {} (可选: retrieve/embedding, retrieve/association, retrieve/full, consolidate, forget)", algo_str);
                std::process::exit(1);
            }
        };

        let mode = match algo {
            AlgoType::Retrieve(m) => m,
            _ => {
                eprintln!("{} 尚未支持 headless 模式", algo);
                std::process::exit(1);
            }
        };

        if is_batch {
            run_headless_batch(&dataset_path, mode);
        } else {
            run_headless_single(algo, dataset_path)?;
        }
    } else {
        color_eyre::install()?;
        let mut app = app::App::new()?;
        app.run()?;
    }

    Ok(())
}

fn run_headless_single(algo: AlgoType, dataset_path: PathBuf) -> color_eyre::Result<()> {
    let dataset_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    println!("=== 测试加载中 ===");
    println!("算法: {}", algo);
    println!("数据集: {}\n", dataset_name);

    let mode = match algo {
        AlgoType::Retrieve(m) => m,
        _ => {
            eprintln!("{} 尚未支持 headless 模式", algo);
            std::process::exit(1);
        }
    };

    let suite =
        RetrieveSuite::load(&dataset_path, mode).map_err(|e| color_eyre::eyre::eyre!("{}", e))?;

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
        error: None,
    });

    Ok(())
}

fn run_headless_batch(dir: &Path, mode: RetrieveMode) {
    println!("=== 批量扫描 ===");
    println!("模式: retrieve/{}", mode);
    println!("目录: {}\n", dir.display());

    let datasets = scan_question_jsons(dir);
    if datasets.is_empty() {
        eprintln!("在 {} 下未找到任何 question.json 文件", dir.display());
        std::process::exit(1);
    }
    println!("找到 {} 个数据集\n", datasets.len());

    let result = run_batch(&datasets, mode, |path, mode, params, start| {
        process_one_dataset(path, mode, params, start, |_, _| {}, |_| {})
    }, None);
    print_batch_result(&result);
}

fn run_headless_playtest(args: &[String]) -> color_eyre::Result<()> {
    if args.len() < 4 {
        eprintln!("用法: soul-tune playtest <graph_dir> <dialogue_file>");
        eprintln!("环境变量: SOUL_TUNE_CANDLE_MODEL_PATH 必须设置");
        std::process::exit(1);
    }

    let graph_dir = PathBuf::from(&args[2]);
    let dialogue_path = PathBuf::from(&args[3]);

    if !graph_dir.exists() {
        eprintln!("图目录不存在: {}", graph_dir.display());
        std::process::exit(1);
    }
    if !dialogue_path.exists() {
        eprintln!("对话文件不存在: {}", dialogue_path.display());
        std::process::exit(1);
    }

    let model_path = match std::env::var("SOUL_TUNE_CANDLE_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("请设置环境变量 SOUL_TUNE_CANDLE_MODEL_PATH");
            std::process::exit(1);
        }
    };

    println!("=== 角色扮演测试 (CLI) ===");
    println!("图目录: {}", graph_dir.display());
    println!("对话: {}", dialogue_path.display());
    println!("模型: {}\n", model_path);

    println!("[1/3] 加载角色图...");
    let runner = PlayTestRunner::load(&graph_dir)
        .map_err(|e| color_eyre::eyre::eyre!("加载图失败: {}", e))?;
    let runner = Arc::new(runner);
    println!("  ✓ 图加载完成");

    println!("[2/3] 启动 LLM 服务...");
    let mut llm = LlamaServer::load(&model_path)
        .map_err(|e| color_eyre::eyre::eyre!("启动 LLM 失败: {}", e))?;
    println!("  ✓ LLM 服务就绪");

    println!("[3/3] 运行对话...\n");
    let dialogue: DialogueFile = serde_json::from_str(
        &std::fs::read_to_string(&dialogue_path)
            .map_err(|e| color_eyre::eyre::eyre!("读取对话文件失败: {}", e))?,
    )
    .map_err(|e| color_eyre::eyre::eyre!("解析对话文件失败: {}", e))?;

    let n = dialogue.conversations.len();
    let mut results = Vec::with_capacity(n);

    for (i, entry) in dialogue.conversations.iter().enumerate() {
        println!("--- 第 {}/{} 轮 ---", i + 1, n);
        println!("用户: {}", entry.user_message);

        let turn = runner.process_turn(entry, i, &mut llm);
        results.push(turn);

        let last = results.last().unwrap();

        if let Some(ref err) = last.runs.first().and_then(|r| r.error.as_ref()) {
            println!("  错误: {}", err);
        }

        println!(
            "  查询: {}",
            &last
                .generated_queries_json
                .chars()
                .take(120)
                .collect::<String>()
        );

        if let Some(ref think) = last.query_think_content {
            println!("  思考: {}", think);
        }

        if let Some(ref resp) = last.runs.first().and_then(|r| r.embedding_response.as_ref()) {
            println!(
                "  Embedding 响应: {}",
                resp.chars().take(80).collect::<String>()
            );
        }
        if let Some(ref resp) = last.runs.first().and_then(|r| r.fullpipeline_response.as_ref()) {
            println!(
                "  FullPipeline 响应: {}",
                resp.chars().take(80).collect::<String>()
            );
        }
        println!();
    }

    println!(
        "调试输出已写入: {}",
        std::env::temp_dir()
            .join("soul_tune_llm_output.txt")
            .display()
    );

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

    let mut groups: std::collections::BTreeMap<String, Vec<&dyn ReportMetric>> = std::collections::BTreeMap::new();
    for metric in &report.suite_report.metrics {
        groups.entry(metric.group().to_string())
            .or_default()
            .push(metric.as_ref());
    }

    for (group, items) in &groups {
        println!("--- {} ---", group);
        for m in items {
            match m.format() {
                MetricFormat::KeyValue { value } => {
                    println!("  {}: {}", m.label(), value);
                }
                MetricFormat::Chart { .. } => {
                    println!("  {}: [图表数据]", m.label());
                }
            }
        }
        println!();
    }

    if !report.suite_report.detail_header.is_empty() {
        println!("--- 详细 ---");
        println!("{}", report.suite_report.detail_header);
        for row in &report.suite_report.detail_rows {
            let marker = if row.has_error { "✗" } else { "✓" };
            println!("{} {}", marker, row.text);
        }
    }
}
