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
use engine::batch::{print_batch_result, run_batch, scan_question_jsons, summarize_action_metrics};
use engine::llm::LlamaServer;
use engine::playtest::trace::RetrievalTrace;
use engine::playtest::{DialogueFile, PlayTestRunner, PlayTurnResult};
use engine::retrieve::batch::process_one_dataset;
use engine::retrieve::data::RetrieveCaseData;
use engine::retrieve::RetrieveSuite;
use engine::suite::{MetricFormat, ReportMetric, TestCaseOutcome, TestSuite};
use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;
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
        //--batch只是开关，位置不固定：同时兼容 run <algo> <dataset> [--batch] 与 run <algo> --batch <dataset>
        let is_batch = args.iter().any(|a| a == "--batch");
        let positional: Vec<&str> = args
            .iter()
            .skip(1)
            .filter(|a| a.as_str() != "--batch")
            .map(|s| s.as_str())
            .collect();
        if positional.len() < 3 {
            eprintln!("用法: soul-tune run <algo> <dataset> [--batch]");
            std::process::exit(1);
        }
        let algo_str = positional[1];
        let path_str = positional[2];
        let dataset_path = PathBuf::from(path_str);

        let algo = match algo_str {
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

    // 文件日志（观测用，不影响结果）
    write_retrieve_log(&outcomes);

    let action_summary = summarize_action_metrics(&outcomes);
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

    if let Some(s) = action_summary {
        println!(
            "\n动作评测: {} 个带期望动作的用例 | 动作Hit {:.1}% | Recall@3 {:.3}",
            s.cases,
            s.hit_rate() * 100.0,
            s.recall_at3
        );
    } else {
        println!("\n动作评测: 无带 expected_actions 的用例");
    }

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

// ===== 文件日志（仅观测用，不改动任何功能逻辑）=====

fn fmt_variant(v: &MemoryRetrieveQueryVariant) -> String {
    match v {
        MemoryRetrieveQueryVariant::Semantic(units) => {
            let parts: Vec<String> = units
                .iter()
                .map(|u| {
                    let mut p = format!("concept={:?}", u.concept_identifier().unwrap_or(""));
                    if let Some(d) = u.description() {
                        p.push_str(&format!(" desc={:?}", d));
                    }
                    p
                })
                .collect();
            format!("Semantic[{}]", parts.join(" | "))
        }
        MemoryRetrieveQueryVariant::Situation(units) => {
            let parts: Vec<String> = units
                .iter()
                .map(|u| {
                    let mut p: Vec<String> = Vec::new();
                    if let Some(n) = u.narrative() {
                        p.push(format!("narrative={:?}", n));
                    }
                    if let Some(l) = u.location() {
                        p.push(format!(
                            "location={:?}",
                            l.iter().map(|x| x.name()).collect::<Vec<_>>()
                        ));
                    }
                    if let Some(ps) = u.participants() {
                        p.push(format!(
                            "participants={:?}",
                            ps.iter().map(|x| x.name().unwrap_or("")).collect::<Vec<_>>()
                        ));
                    }
                    if let Some(e) = u.environment() {
                        p.push(format!("env={e:?}"));
                    }
                    if let Some(ev) = u.event() {
                        p.push(format!(
                            "event={:?}",
                            ev.iter().map(|x| x.action()).collect::<Vec<_>>()
                        ));
                    }
                    p.join(", ")
                })
                .collect();
            format!("Situation[{}]", parts.join(" | "))
        }
    }
}

fn fmt_trace(kind: &str, t: &RetrievalTrace) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "[{}] 合并 {} 节点, 总耗时 {:.2}s\n",
        kind,
        t.merged_nodes.len(),
        t.total_elapsed.as_secs_f64()
    ));
    for (qi, qt) in t.per_query.iter().enumerate() {
        if qt.dropped {
            out.push_str(&format!(
                "  Q{}: (dropped) {}\n",
                qi,
                fmt_variant(qt.query.variant())
            ));
            continue;
        }
        out.push_str(&format!(
            "  Q{}: {}\n      sim={} ppr={} action={} (查询耗时 {:.2}s)\n",
            qi,
            fmt_variant(qt.query.variant()),
            qt.sim_nodes.len(),
            qt.ppr_nodes.len(),
            qt.action_nodes.len(),
            qt.total_elapsed.as_secs_f64()
        ));
    }
    for n in t.merged_nodes.iter().take(12) {
        let content: String = n.content.chars().take(40).collect();
        out.push_str(&format!(
            "  - [{}] {:?} score={:.4} | {}\n",
            n.name, n.stage, n.score, content
        ));
    }
    out
}

fn write_playtest_log(results: &[PlayTurnResult]) {
    let mut out = String::new();
    for turn in results {
        out.push_str(&format!(
            "########## 第 {} 轮 ##########\n用户: {}\n",
            turn.index + 1, turn.user_message
        ));
        out.push_str(&format!("查询JSON: {}\n", turn.generated_queries_json));
        match (&turn.embedding_trace, &turn.fullpipeline_trace) {
            (Some(e), Some(f)) => {
                out.push_str(&fmt_trace("Embedding", e));
                out.push_str(&fmt_trace("FullPipeline", f));
            }
            (e, f) => {
                if e.is_none() {
                    out.push_str("[Embedding] trace=None（无有效查询或全部嵌入失败）\n");
                }
                if f.is_none() {
                    out.push_str("[FullPipeline] trace=None（无有效查询或全部嵌入失败）\n");
                }
            }
        }
        if let Some(err) = turn.runs.first().and_then(|r| r.error.as_ref()) {
            out.push_str(&format!("错误: {}\n", err));
        }
        out.push('\n');
    }
    let path = std::env::temp_dir().join("soul_tune_playtest_log.txt");
    if std::fs::write(&path, out).is_ok() {
        println!("Playtest 日志已写入: {}", path.display());
    }
}

fn write_retrieve_log(outcomes: &[TestCaseOutcome]) {
    let mut out = String::new();
    let mut passed = 0;
    for (i, o) in outcomes.iter().enumerate() {
        let ok = o.passed;
        if ok {
            passed += 1;
        }
        out.push_str(&format!(
            "===== 用例 {}: {} [{}] =====\n",
            i,
            o.case_name,
            if ok { "通过" } else { "失败" }
        ));
        if let Some(data) = o.data.downcast_ref::<RetrieveCaseData>() {
            let m = &data.combined_ranking_metrics;
            out.push_str(&format!("  MRR={:.4} Hit={:.2}\n", m.mrr, m.hit_rate));
            for pm in &data.per_query_metrics {
                out.push_str(&format!(
                    "  Q{}: MRR={:.4} Hit={:.2}\n",
                    pm.query_index, pm.ranking_metrics.mrr, pm.ranking_metrics.hit_rate
                ));
            }
            if data.action_metrics.has_expected_actions {
                let r3 = data
                    .action_metrics
                    .action_recall_at
                    .iter()
                    .find(|(k, _)| *k == 3)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0);
                out.push_str(&format!(
                    "  动作: hit={:.2} recall@3={:.3}\n",
                    data.action_metrics.action_hit_rate, r3
                ));
            } else {
                out.push_str("  动作: N/A（无 expected_actions）\n");
            }
            out.push_str(&format!(
                "  检索到 {} 节点:\n",
                data.combined_retrieved_ids.len()
            ));
            for (pos, id) in data.combined_retrieved_ids.iter().enumerate() {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(id))
                    .cloned()
                    .unwrap_or_default();
                out.push_str(&format!("    #{:<3} {}\n", pos + 1, name));
            }
            out.push_str("  期望命中(前5):\n");
            for (pos, id) in data.expected_combined_ranking.iter().take(5).enumerate() {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(id))
                    .cloned()
                    .unwrap_or_default();
                out.push_str(&format!("    E#{:<3} {}\n", pos + 1, name));
            }
        }
        out.push('\n');
    }
    out.push_str(&format!("共 {} 用例, 通过 {}\n", outcomes.len(), passed));
    let path = std::env::temp_dir().join("soul_tune_retrieve_log.txt");
    if std::fs::write(&path, out).is_ok() {
        println!("Retrieve 日志已写入: {}", path.display());
    }
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

    let dialogue: DialogueFile = serde_json::from_str(
        &std::fs::read_to_string(&dialogue_path)
            .map_err(|e| color_eyre::eyre::eyre!("读取对话文件失败: {}", e))?,
    )
    .map_err(|e| color_eyre::eyre::eyre!("解析对话文件失败: {}", e))?;

    println!("=== 角色扮演测试 (CLI) ===");
    println!("图目录: {}", graph_dir.display());
    println!("对话: {}", dialogue_path.display());
    println!(
        "自身角色: {}",
        dialogue.role.as_deref().unwrap_or("（未设置）")
    );
    println!("模型: {}\n", model_path);

    println!("[1/3] 加载角色图...");
    let mut runner = PlayTestRunner::load(&graph_dir)
        .map_err(|e| color_eyre::eyre::eyre!("加载图失败: {}", e))?;
    if let Some(ref cfg) = dialogue.config {
        runner = runner.with_config(cfg.clone());
    }
    if let Some(ref role) = dialogue.role {
        runner = runner.with_human_role(Some(role.clone()));
    }
    let runner = Arc::new(runner);
    println!("  ✓ 图加载完成");

    println!("[2/3] 启动 LLM 服务...");
    let mut llm = LlamaServer::load(&model_path)
        .map_err(|e| color_eyre::eyre::eyre!("启动 LLM 失败: {}", e))?;
    println!("  ✓ LLM 服务就绪");

    println!("[3/3] 运行对话...\n");
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

    // 文件日志（观测用，不影响结果）
    write_playtest_log(&results);

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
