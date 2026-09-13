//! FRB 接口（JSON-over-FRB）。
//!
//! 约定：
//! - 所有返回值与流事件均为 JSON 字符串（serde_json 序列化）。
//! - 事件流通过 `StreamSink<String>` 推流，事件统一为内部标签枚举（见 [`RunEvent`] / [`BatchEvent`]）。
//! - 取消：全局 [`CANCEL`] 标志，运行循环在用例之间检查；Dart 侧每次运行前调 [`reset_cancel`]。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use flutter_rust_bridge::frb;
use serde::Serialize;

// FRB 2.12：StreamSink 由 frb_generated_boilerplate! 宏生成在 crate 的 frb_generated 模块中
// （第二个类型参数默认 SseCodec），crate 根不导出。
use crate::frb_generated::StreamSink;

use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;
use soul_tune::base::{AlgoType, RetrieveMode};
use soul_tune::engine::batch::{BatchResult, scan_question_jsons};
use soul_tune::engine::compare::{CompareReport, build_compare_report};
use soul_tune::engine::forget::{
    ForgetCaseData, ForgetMaskSuite, ForgetPipelineSuite, ForgetReviseSuite, MaskCaseData,
    NodeForgetStat, NodeSeries, ReviseCaseData, ideal_ebbinghaus_curve,
};
use soul_tune::engine::llm::LlamaServer;
use soul_tune::engine::playtest::runner::{ConversationEntry, PlayTestRunner};
use soul_tune::engine::playtest::trace::{HitStage, RetrievalTrace, TracedNode};
use soul_tune::engine::retrieve::RetrieveSuite;
use soul_tune::engine::retrieve::batch::process_one_dataset;
use soul_tune::engine::retrieve::data::RetrieveCaseData;
use soul_tune::engine::retrieve::db_compare::{DbCompareReport, build_db_compare_report};
use soul_tune::engine::suite::{DetailRow, MetricEntry, TestCaseOutcome, TestSuite};

/// 全局取消标志（单跑/批量共享）。
static CANCEL: AtomicBool = AtomicBool::new(false);

// ======================= 元数据接口 =======================

#[derive(Serialize)]
struct DatasetEntryJson {
    name: String,
    path: String,
}

/// 递归扫描目录下的全部 question.json 数据集文件。
#[frb]
pub fn scan_datasets_json(dir: String) -> String {
    let entries: Vec<DatasetEntryJson> = scan_question_jsons(&PathBuf::from(dir))
        .iter()
        .map(|p| DatasetEntryJson {
            name: p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            path: p.to_string_lossy().to_string(),
        })
        .collect();
    serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
}

#[derive(Serialize)]
struct DatasetMetaJson {
    name: String,
    description: String,
    case_count: usize,
    graph_path: String,
    algo_type: String,
    error: Option<String>,
}

/// 解析 question.json 头部元数据（供数据集预览卡）。
#[frb]
pub fn dataset_meta_json(path: String) -> String {
    let content = std::fs::read_to_string(&path);
    let v: Option<serde_json::Value> = content.ok().and_then(|c| serde_json::from_str(&c).ok());
    let Some(v) = v else {
        return serde_json::to_string(&DatasetMetaJson {
            name: String::new(),
            description: String::new(),
            case_count: 0,
            graph_path: String::new(),
            algo_type: String::new(),
            error: Some(format!("无法解析: {path}")),
        })
        .unwrap_or_default();
    };
    let meta = DatasetMetaJson {
        name: v
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        description: v
            .get("description")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        case_count: v
            .get("test_cases")
            .and_then(|x| x.as_array())
            .map(|a| a.len())
            .unwrap_or(0),
        graph_path: v
            .get("graph_path")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        algo_type: v
            .get("algo_type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        error: None,
    };
    serde_json::to_string(&meta).unwrap_or_else(|_| "{}".to_string())
}

#[derive(Serialize)]
struct ParamSpecJson {
    name: String,
    default: String,
    description: String,
}

/// 可调参数规格（参数表单用；与 engine 读取的 key 一一对应）。
#[frb]
pub fn default_params_json() -> String {
    let specs = vec![
        ParamSpecJson {
            name: "top_k".into(),
            default: "10".into(),
            description: "检索返回数量上限".into(),
        },
        ParamSpecJson {
            name: "threshold".into(),
            default: "0.7".into(),
            description: "相似度阈值".into(),
        },
    ];
    serde_json::to_string(&specs).unwrap_or_else(|_| "[]".to_string())
}

/// 清空取消标志（每次运行前调用）。
#[frb]
pub fn reset_cancel() {
    CANCEL.store(false, Ordering::SeqCst);
}

// ======================= 单跑 =======================

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RunEvent {
    Loading {
        message: String,
    },
    Progress {
        done: usize,
        total: usize,
        passed: usize,
        failed: usize,
        elapsed_ms: u64,
        case_name: String,
    },
    Done {
        report: ReportJson,
    },
    Error {
        message: String,
    },
    Cancelled,
}

#[derive(Serialize)]
struct ReportJson {
    algo: String,
    dataset_name: String,
    dataset_path: String,
    total: usize,
    passed: usize,
    failed: usize,
    elapsed_secs: f64,
    metrics: Vec<MetricEntry>,
    detail_header: String,
    detail_rows: Vec<DetailRow>,
    outcomes: Vec<OutcomeJson>,
}

#[derive(Serialize)]
struct OutcomeJson {
    case_name: String,
    description: String,
    passed: bool,
    /// 套件具体的用例数据（retrieve 为 RetrieveCaseData 序列化结果）
    data: serde_json::Value,
}

/// 运行单个检索套件，进度与结果通过流推送。
#[frb]
pub fn run_suite(algo: String, dataset: String, params_json: String, sink: StreamSink<String>) {
    std::thread::spawn(move || {
        run_stream(
            &sink,
            || run_suite_impl(&algo, &dataset, &params_json, &sink),
            |message| RunEvent::Error { message },
        );
    });
}

fn emit<T: Serialize>(sink: &StreamSink<String>, ev: &T) {
    if let Ok(json) = serde_json::to_string(ev) {
        let _ = sink.add(json);
    }
}

/// 在 spawn 出来的线程里执行一次运行，失败时把错误推成对应的 `*Event::Error`。
///
/// 四处入口此前都是 `let _ = run_*_impl(...)`：算法名写错、数据集缺失、参数非法等错误
/// 被直接丢弃，Dart 侧只会看到流静静关闭，没有任何事件可读——界面表现为"转圈之后没反应"，
/// 调试时也没有任何线索。
///
/// 补发事件是在兑现既有契约而非新增契约：Dart 的 `RunEvent` / `BatchEvent` /
/// `CompareEvent` / `ForgetEvent` 都已经实现了 `error` 分支
/// （见 `soul-tune-ui/lib/src/models.dart`），只是 Rust 侧从未发送过。
fn run_stream<E: Serialize>(
    sink: &StreamSink<String>,
    run: impl FnOnce() -> anyhow::Result<()>,
    on_error: impl FnOnce(String) -> E,
) {
    if let Err(err) = run() {
        // `{:#}` 展开 anyhow 的完整错误链，便于定位失败发生在哪一层
        emit(sink, &on_error(format!("{err:#}")));
    }
}

fn run_suite_impl(
    algo: &str,
    dataset: &str,
    params_json: &str,
    sink: &StreamSink<String>,
) -> anyhow::Result<()> {
    let params: HashMap<String, String> = serde_json::from_str(params_json).unwrap_or_default();
    let algo = parse_algo(algo)?;
    let dataset_path = PathBuf::from(dataset);
    let dataset_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    CANCEL.store(false, Ordering::SeqCst);
    emit(
        sink,
        &RunEvent::Loading {
            message: "正在加载数据集与嵌入模型...".into(),
        },
    );

    let suite: Box<dyn TestSuite> = match algo {
        AlgoType::Retrieve(mode) => Box::new(
            RetrieveSuite::load_with_params(&dataset_path, mode, Some(&params))
                .map_err(|e| anyhow::anyhow!("加载数据集失败: {e}"))?,
        ),
        other => return Err(anyhow::anyhow!("暂不支持该算法: {other}")),
    };
    let total = suite.case_count();
    emit(
        sink,
        &RunEvent::Loading {
            message: format!("准备就绪，共 {total} 个测试用例"),
        },
    );

    let start = Instant::now();
    let mut outcomes = Vec::with_capacity(total);
    let mut passed = 0usize;
    let mut failed = 0usize;
    for i in 0..total {
        if CANCEL.load(Ordering::SeqCst) {
            emit(sink, &RunEvent::Cancelled);
            return Ok(());
        }
        let outcome = suite.run_case(i);
        let case_name = outcome.case_name.clone();
        if outcome.passed {
            passed += 1;
        } else {
            failed += 1;
        }
        outcomes.push(outcome);
        emit(
            sink,
            &RunEvent::Progress {
                done: i + 1,
                total,
                passed,
                failed,
                elapsed_ms: start.elapsed().as_millis() as u64,
                case_name,
            },
        );
    }
    let elapsed = start.elapsed();
    let report = suite.build_report(outcomes, elapsed, total, passed, failed);
    emit(
        sink,
        &RunEvent::Done {
            report: build_report_json(algo, &dataset_path, &dataset_name, report, elapsed),
        },
    );
    Ok(())
}

fn build_report_json(
    algo: AlgoType,
    dataset_path: &Path,
    dataset_name: &str,
    report: soul_tune::engine::suite::SuiteReport,
    elapsed: Duration,
) -> ReportJson {
    let outcomes: Vec<OutcomeJson> = report
        .outcomes
        .into_iter()
        .map(|o| {
            let data = o
                .data
                .downcast_ref::<RetrieveCaseData>()
                .and_then(|d| serde_json::to_value(d).ok())
                .unwrap_or(serde_json::Value::Null);
            OutcomeJson {
                case_name: o.case_name,
                description: o.description,
                passed: o.passed,
                data,
            }
        })
        .collect();
    let total = outcomes.len();
    let passed = outcomes.iter().filter(|o| o.passed).count();
    let failed = total - passed;
    ReportJson {
        algo: algo.to_string(),
        dataset_name: dataset_name.to_string(),
        dataset_path: dataset_path.to_string_lossy().to_string(),
        total,
        passed,
        failed,
        elapsed_secs: elapsed.as_secs_f64(),
        metrics: report.metrics,
        detail_header: report.detail_header,
        detail_rows: report.detail_rows,
        outcomes,
    }
}

fn parse_algo(s: &str) -> anyhow::Result<AlgoType> {
    match s {
        "retrieve/embedding" | "re" => Ok(AlgoType::Retrieve(RetrieveMode::Embedding)),
        "retrieve/association" | "ra" => Ok(AlgoType::Retrieve(RetrieveMode::Association)),
        "retrieve/full" | "retrieve" | "rf" => Ok(AlgoType::Retrieve(RetrieveMode::FullPipeline)),
        "retrieve/db/embedding" | "rde" => Ok(AlgoType::Retrieve(RetrieveMode::EmbeddingDb)),
        "retrieve/db/association" | "rda" => Ok(AlgoType::Retrieve(RetrieveMode::AssociationDb)),
        "retrieve/db" | "retrieve/db/full" | "rd" => {
            Ok(AlgoType::Retrieve(RetrieveMode::FullPipelineDb))
        }
        other => Err(anyhow::anyhow!("未知算法: {other}")),
    }
}

// ======================= 批量 =======================

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BatchEvent {
    Scanning {
        dir: String,
    },
    Progress {
        done: usize,
        total: usize,
    },
    DatasetDone {
        index: usize,
        name: String,
        total: usize,
        passed: usize,
        failed: usize,
        pass_rate: f64,
        elapsed_ms: u64,
        error: Option<String>,
    },
    Done {
        result: BatchReportJson,
    },
    Error {
        message: String,
    },
    Cancelled,
}

#[derive(Serialize)]
struct DatasetResultJson {
    name: String,
    path: String,
    total: usize,
    passed: usize,
    failed: usize,
    pass_rate: f64,
    elapsed_ms: u64,
    error: Option<String>,
    outcomes: Vec<OutcomeJson>,
}

#[derive(Serialize)]
struct BatchReportJson {
    total_datasets: usize,
    total_cases: usize,
    total_passed: usize,
    total_failed: usize,
    elapsed_secs: f64,
    datasets: Vec<DatasetResultJson>,
}

/// 批量运行目录下全部检索数据集（4 worker 并发，逐数据集推流）。
///
/// **注意**：`params_json` 目前被忽略。`soul-tune` 的批量执行路径
/// （`engine/batch.rs`）对每个数据集固定传 `None` 参数，因此批量运行始终使用
/// `RetrieveSuite::load` 的默认参数。单数据集入口（`run_suite` / `run_compare`）
/// 是认这个参数的——不要从这里的签名推断 batch 也认。
#[frb]
pub fn run_batch(dir: String, mode: String, params_json: String, sink: StreamSink<String>) {
    std::thread::spawn(move || {
        run_stream(
            &sink,
            || run_batch_impl(&dir, &mode, &params_json, &sink),
            |message| BatchEvent::Error { message },
        );
    });
}

fn run_batch_impl(
    dir: &str,
    mode: &str,
    _params_json: &str,
    sink: &StreamSink<String>,
) -> anyhow::Result<()> {
    let mode = match mode {
        "embedding" => RetrieveMode::Embedding,
        "association" => RetrieveMode::Association,
        "full" | "fullpipeline" => RetrieveMode::FullPipeline,
        "db/embedding" | "dbe" => RetrieveMode::EmbeddingDb,
        "db/association" | "dba" => RetrieveMode::AssociationDb,
        "db" | "db/full" | "db/fullpipeline" => RetrieveMode::FullPipelineDb,
        other => return Err(anyhow::anyhow!("未知检索模式: {other}")),
    };
    let dir_path = PathBuf::from(dir);

    CANCEL.store(false, Ordering::SeqCst);
    emit(
        sink,
        &BatchEvent::Scanning {
            dir: dir_path.to_string_lossy().to_string(),
        },
    );

    let datasets = scan_question_jsons(&dir_path);
    if datasets.is_empty() {
        return Err(anyhow::anyhow!(
            "目录下未找到 question.json: {}",
            dir_path.display()
        ));
    }

    let result: BatchResult = soul_tune::engine::batch::run_batch(
        &datasets,
        mode,
        move |path, mode, params, start| {
            process_one_dataset(path, mode, params, start, |_, _| {}, |_| {})
        },
        Some(&|done, total| {
            emit(sink, &BatchEvent::Progress { done, total });
            !CANCEL.load(Ordering::SeqCst)
        }),
    );

    if CANCEL.load(Ordering::SeqCst) {
        emit(sink, &BatchEvent::Cancelled);
        return Ok(());
    }

    for (i, ds) in result.datasets.iter().enumerate() {
        emit(
            sink,
            &BatchEvent::DatasetDone {
                index: i,
                name: ds.name.clone(),
                total: ds.total,
                passed: ds.passed,
                failed: ds.failed,
                pass_rate: ds.pass_rate,
                elapsed_ms: ds.elapsed.as_millis() as u64,
                error: ds.error.clone(),
            },
        );
    }

    let datasets_json: Vec<DatasetResultJson> = result
        .datasets
        .iter()
        .map(|ds| DatasetResultJson {
            name: ds.name.clone(),
            path: ds.path.to_string_lossy().to_string(),
            total: ds.total,
            passed: ds.passed,
            failed: ds.failed,
            pass_rate: ds.pass_rate,
            elapsed_ms: ds.elapsed.as_millis() as u64,
            error: ds.error.clone(),
            outcomes: ds
                .outcomes
                .iter()
                .map(|o| {
                    let data = o
                        .data
                        .downcast_ref::<RetrieveCaseData>()
                        .and_then(|d| serde_json::to_value(d).ok())
                        .unwrap_or(serde_json::Value::Null);
                    OutcomeJson {
                        case_name: o.case_name.clone(),
                        description: o.description.clone(),
                        passed: o.passed,
                        data,
                    }
                })
                .collect(),
        })
        .collect();

    emit(
        sink,
        &BatchEvent::Done {
            result: BatchReportJson {
                total_datasets: datasets_json.len(),
                total_cases: result.total_cases,
                total_passed: result.total_passed,
                total_failed: result.total_failed,
                elapsed_secs: result.elapsed.as_secs_f64(),
                datasets: datasets_json,
            },
        },
    );
    Ok(())
}

// ======================= 对比（embedding vs full） =======================

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CompareEvent {
    Loading {
        message: String,
    },
    Progress {
        phase: String,
        done: usize,
        total: usize,
        passed: usize,
        failed: usize,
        elapsed_ms: u64,
        case_name: String,
    },
    Done {
        report: CompareReportJson,
    },
    Error {
        message: String,
    },
    Cancelled,
}

#[derive(Serialize)]
struct CompareAggregateJson {
    case_count: usize,
    avg_embedding_hit: f64,
    avg_fullpipeline_hit: f64,
    avg_embedding_mrr: f64,
    avg_fullpipeline_mrr: f64,
    hit_improvement_count: usize,
    mrr_improvement_count: usize,
}

#[derive(Serialize)]
struct CompareCaseJson {
    case_name: String,
    description: String,
    tag_weight: f32,
    variant_weight: f32,
    embedding_hit: f64,
    fullpipeline_hit: f64,
    embedding_mrr: f64,
    fullpipeline_mrr: f64,
    embedding_recall_at: Vec<(usize, f64)>,
    fullpipeline_recall_at: Vec<(usize, f64)>,
    embedding_precision_at: Vec<(usize, f64)>,
    fullpipeline_precision_at: Vec<(usize, f64)>,
    embedding_ndcg_at: Vec<(usize, f64)>,
    fullpipeline_ndcg_at: Vec<(usize, f64)>,
    embedding_retrieved: Vec<String>,
    fullpipeline_retrieved: Vec<String>,
    expected_combined_ranking: Vec<String>,
    improved_hit: bool,
    improved_mrr: bool,
    /// 各子查询双侧指标（详情页钻取用；两侧按 query_index 对齐）
    per_query: Vec<ComparePerQueryJson>,
}

/// 单个子查询的双侧指标（embedding_* 承载侧 A、fullpipeline_* 承载侧 B，
/// 与用例字段的命名约定一致：direct_db 对比时 A=直接、B=数据库）。
#[derive(Serialize)]
struct ComparePerQueryJson {
    query_index: usize,
    embedding_mrr: f64,
    fullpipeline_mrr: f64,
    embedding_hit: f64,
    fullpipeline_hit: f64,
    embedding_recall_at: Vec<(usize, f64)>,
    fullpipeline_recall_at: Vec<(usize, f64)>,
    embedding_precision_at: Vec<(usize, f64)>,
    fullpipeline_precision_at: Vec<(usize, f64)>,
    embedding_ndcg_at: Vec<(usize, f64)>,
    fullpipeline_ndcg_at: Vec<(usize, f64)>,
}

/// 从每侧用例结果中抽取详情用扩展数据（Precision/NDCG/逐子查询），
/// 统一由 [`retrieve::data::RankingMetrics`] 提供，无需改动引擎对比报告结构。
fn find_query_metrics(
    idx: usize,
    list: &[soul_tune::engine::retrieve::data::PerQueryMetrics],
) -> Option<&soul_tune::engine::retrieve::data::RankingMetrics> {
    list.iter()
        .find(|p| p.query_index == idx)
        .map(|p| &p.ranking_metrics)
}

/// [`case_extras_json`] 的多通道返回值。
///
/// 这里刻意用**命名结构体**而不是元组：返回值是多通道指标的聚合，元组会让调用方
/// 依赖位置记忆（原先只能靠 `// emb precision` 这类行内注释区分），而且会让
/// "整体替换返回值" 这一类变异测试发生组合爆炸——cargo-mutants 曾为该元组单函数
/// 生成约 4800 个变异体，占全仓库变异体总数的约 46%，把杀灭率指标完全淹掉。
struct CaseExtras {
    /// embedding 侧的 precision@k 曲线
    emb_precision: Vec<(usize, f64)>,
    /// full pipeline 侧的 precision@k 曲线
    full_precision: Vec<(usize, f64)>,
    /// embedding 侧的 ndcg@k 曲线
    emb_ndcg: Vec<(usize, f64)>,
    /// full pipeline 侧的 ndcg@k 曲线
    full_ndcg: Vec<(usize, f64)>,
    /// 逐查询明细
    per_query: Vec<ComparePerQueryJson>,
}

fn case_extras_json(
    key: &(String, u32, u32),
    emb_map: &HashMap<(String, u32, u32), &RetrieveCaseData>,
    full_map: &HashMap<(String, u32, u32), &RetrieveCaseData>,
) -> CaseExtras {
    use soul_tune::engine::retrieve::data::RankingMetrics;
    fn metrics_of<'a>(
        map: &'a HashMap<(String, u32, u32), &'a RetrieveCaseData>,
        key: &(String, u32, u32),
    ) -> Option<&'a RankingMetrics> {
        map.get(key).map(|d| &d.combined_ranking_metrics)
    }
    let zero = || Vec::<(usize, f64)>::new();
    let emb_m = metrics_of(emb_map, key);
    let full_m = metrics_of(full_map, key);
    let emb_precision = emb_m.map(|m| m.precision_at.clone()).unwrap_or_else(zero);
    let full_precision = full_m.map(|m| m.precision_at.clone()).unwrap_or_else(zero);
    let emb_ndcg = emb_m.map(|m| m.ndcg_at.clone()).unwrap_or_else(zero);
    let full_ndcg = full_m.map(|m| m.ndcg_at.clone()).unwrap_or_else(zero);

    // 逐子查询：取两侧 query_index 并集（按序），缺失侧给零指标
    let emb_pq = emb_map
        .get(key)
        .map(|d| &d.per_query_metrics)
        .cloned()
        .unwrap_or_default();
    let full_pq = full_map
        .get(key)
        .map(|d| &d.per_query_metrics)
        .cloned()
        .unwrap_or_default();
    let mut indices: Vec<usize> = emb_pq
        .iter()
        .map(|p| p.query_index)
        .chain(full_pq.iter().map(|p| p.query_index))
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let empty_metrics = RankingMetrics {
        recall_at: Vec::new(),
        precision_at: Vec::new(),
        mrr: 0.0,
        ndcg_at: Vec::new(),
        hit_rate: 0.0,
    };
    let per_query = indices
        .into_iter()
        .map(|idx| {
            let e = find_query_metrics(idx, &emb_pq).unwrap_or(&empty_metrics);
            let f = find_query_metrics(idx, &full_pq).unwrap_or(&empty_metrics);
            ComparePerQueryJson {
                query_index: idx,
                embedding_mrr: e.mrr,
                fullpipeline_mrr: f.mrr,
                embedding_hit: e.hit_rate,
                fullpipeline_hit: f.hit_rate,
                embedding_recall_at: e.recall_at.clone(),
                fullpipeline_recall_at: f.recall_at.clone(),
                embedding_precision_at: e.precision_at.clone(),
                fullpipeline_precision_at: f.precision_at.clone(),
                embedding_ndcg_at: e.ndcg_at.clone(),
                fullpipeline_ndcg_at: f.ndcg_at.clone(),
            }
        })
        .collect();

    CaseExtras {
        emb_precision,
        full_precision,
        emb_ndcg,
        full_ndcg,
        per_query,
    }
}

fn compare_key(name: &str, tag_weight: f32, variant_weight: f32) -> (String, u32, u32) {
    (
        name.to_string(),
        (tag_weight * 100.0).round() as u32,
        (variant_weight * 100.0).round() as u32,
    )
}

/// 按用例键索引一侧的 RetrieveCaseData（用于抽取 Precision/NDCG/逐子查询等扩展数据）。
fn index_side_data(outcomes: &[TestCaseOutcome]) -> HashMap<(String, u32, u32), &RetrieveCaseData> {
    let mut map: HashMap<(String, u32, u32), &RetrieveCaseData> = HashMap::new();
    for o in outcomes {
        if let Some(d) = o.data.downcast_ref::<RetrieveCaseData>() {
            let key = compare_key(&d.case_name, d.tag_weight, d.variant_weight);
            map.insert(key, d);
        }
    }
    map
}

#[derive(Serialize)]
struct CompareReportJson {
    dataset_name: String,
    dataset_path: String,
    aggregate: CompareAggregateJson,
    cases: Vec<CompareCaseJson>,
}

/// 对比运行：同一数据集依次跑 embedding 与 full pipeline，产出逐用例对比报告。
#[frb]
pub fn run_compare(dataset: String, params_json: String, sink: StreamSink<String>) {
    std::thread::spawn(move || {
        run_stream(
            &sink,
            || run_compare_impl(&dataset, &params_json, &sink),
            |message| CompareEvent::Error { message },
        );
    });
}

fn run_compare_impl(
    dataset: &str,
    params_json: &str,
    sink: &StreamSink<String>,
) -> anyhow::Result<()> {
    let params: HashMap<String, String> = serde_json::from_str(params_json).unwrap_or_default();
    let dataset_path = PathBuf::from(dataset);
    let dataset_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    // 对比类型：embedding_full（embedding vs full pipeline，默认）| direct_db（同管线 直接 vs 数据库）
    let kind = params
        .get("kind")
        .map(String::as_str)
        .unwrap_or("embedding_full");
    if kind == "direct_db" {
        return run_compare_db_impl(&params, &dataset_name, &dataset_path, sink);
    }

    CANCEL.store(false, Ordering::SeqCst);

    // 阶段 1：embedding
    emit(
        sink,
        &CompareEvent::Loading {
            message: "正在加载 Embedding 套件...".into(),
        },
    );
    let emb_suite: Box<dyn TestSuite> = Box::new(
        RetrieveSuite::load_with_params(&dataset_path, RetrieveMode::Embedding, Some(&params))
            .map_err(|e| anyhow::anyhow!("加载 Embedding 套件失败: {e}"))?,
    );
    let emb_total = emb_suite.case_count();
    emit(
        sink,
        &CompareEvent::Loading {
            message: format!("Embedding 就绪，共 {emb_total} 个用例"),
        },
    );
    let (emb_outcomes, _, _) = run_compare_phase(emb_suite.as_ref(), "embedding", emb_total, sink)?;

    // 阶段 2：full pipeline
    emit(
        sink,
        &CompareEvent::Loading {
            message: "正在加载 FullPipeline 套件...".into(),
        },
    );
    let full_suite: Box<dyn TestSuite> = Box::new(
        RetrieveSuite::load_with_params(&dataset_path, RetrieveMode::FullPipeline, Some(&params))
            .map_err(|e| anyhow::anyhow!("加载 FullPipeline 套件失败: {e}"))?,
    );
    let full_total = full_suite.case_count();
    emit(
        sink,
        &CompareEvent::Loading {
            message: format!("FullPipeline 就绪，共 {full_total} 个用例"),
        },
    );
    let (full_outcomes, _, _) = run_compare_phase(full_suite.as_ref(), "full", full_total, sink)?;

    let report = build_compare_report(&emb_outcomes, &full_outcomes);
    emit(
        sink,
        &CompareEvent::Done {
            report: build_compare_json(
                report,
                &emb_outcomes,
                &full_outcomes,
                &dataset_name,
                &dataset_path,
            ),
        },
    );
    Ok(())
}

fn run_compare_phase(
    suite: &dyn TestSuite,
    phase: &str,
    total: usize,
    sink: &StreamSink<String>,
) -> anyhow::Result<(Vec<TestCaseOutcome>, usize, usize)> {
    let start = Instant::now();
    let mut outcomes = Vec::with_capacity(total);
    let mut passed = 0usize;
    let mut failed = 0usize;
    for i in 0..total {
        if CANCEL.load(Ordering::SeqCst) {
            emit(sink, &CompareEvent::Cancelled);
            return Err(anyhow::anyhow!("已取消"));
        }
        let outcome = suite.run_case(i);
        let case_name = outcome.case_name.clone();
        if outcome.passed {
            passed += 1;
        } else {
            failed += 1;
        }
        outcomes.push(outcome);
        emit(
            sink,
            &CompareEvent::Progress {
                phase: phase.to_string(),
                done: i + 1,
                total,
                passed,
                failed,
                elapsed_ms: start.elapsed().as_millis() as u64,
                case_name,
            },
        );
    }
    Ok((outcomes, passed, failed))
}

fn build_compare_json(
    report: CompareReport,
    emb_outcomes: &[TestCaseOutcome],
    full_outcomes: &[TestCaseOutcome],
    dataset_name: &str,
    dataset_path: &Path,
) -> CompareReportJson {
    // 从一侧用例数据中收集节点名映射（graph_names），两侧共用同一张图
    let name_map = collect_graph_names(emb_outcomes);
    let names = |ids: &[soul_mem_core::memory_note::MemoryId]| -> Vec<String> {
        ids.iter()
            .map(|id| {
                name_map
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("{id:?}"))
            })
            .collect()
    };
    // 两侧 RetrieveCaseData 索引（供 Precision/NDCG/逐子查询扩展字段）
    let emb_map = index_side_data(emb_outcomes);
    let full_map = index_side_data(full_outcomes);
    let agg = &report.aggregate;
    CompareReportJson {
        dataset_name: dataset_name.to_string(),
        dataset_path: dataset_path.to_string_lossy().to_string(),
        aggregate: CompareAggregateJson {
            case_count: agg.case_count,
            avg_embedding_hit: agg.avg_embedding_hit,
            avg_fullpipeline_hit: agg.avg_fullpipeline_hit,
            avg_embedding_mrr: agg.avg_embedding_mrr,
            avg_fullpipeline_mrr: agg.avg_fullpipeline_mrr,
            hit_improvement_count: agg.hit_improvement_count,
            mrr_improvement_count: agg.mrr_improvement_count,
        },
        cases: report
            .cases
            .iter()
            .map(|c| {
                let key = compare_key(&c.case_name, c.tag_weight, c.variant_weight);
                let CaseExtras {
                    emb_precision,
                    full_precision,
                    emb_ndcg,
                    full_ndcg,
                    per_query,
                } = case_extras_json(&key, &emb_map, &full_map);
                CompareCaseJson {
                    case_name: c.case_name.clone(),
                    description: c.description.clone(),
                    tag_weight: c.tag_weight,
                    variant_weight: c.variant_weight,
                    embedding_hit: c.embedding_hit,
                    fullpipeline_hit: c.fullpipeline_hit,
                    embedding_mrr: c.embedding_mrr,
                    fullpipeline_mrr: c.fullpipeline_mrr,
                    embedding_recall_at: c.embedding_recall_at.clone(),
                    fullpipeline_recall_at: c.fullpipeline_recall_at.clone(),
                    embedding_precision_at: emb_precision,
                    fullpipeline_precision_at: full_precision,
                    embedding_ndcg_at: emb_ndcg,
                    fullpipeline_ndcg_at: full_ndcg,
                    embedding_retrieved: names(&c.embedding_retrieved),
                    fullpipeline_retrieved: names(&c.fullpipeline_retrieved),
                    expected_combined_ranking: names(&c.expected_combined_ranking),
                    improved_hit: c.fullpipeline_hit > c.embedding_hit,
                    improved_mrr: c.fullpipeline_mrr > c.embedding_mrr,
                    per_query,
                }
            })
            .collect(),
    }
}

/// 从用例结果中收集 graph_names（MemoryId → 图内可读名），供检索列表展示。
fn collect_graph_names(
    outcomes: &[TestCaseOutcome],
) -> HashMap<soul_mem_core::memory_note::MemoryId, String> {
    let mut name_map: HashMap<soul_mem_core::memory_note::MemoryId, String> = HashMap::new();
    for o in outcomes {
        if let Some(d) = o.data.downcast_ref::<RetrieveCaseData>()
            && let Some(names) = d.graph_names.as_ref()
        {
            for (id, n) in names.iter() {
                name_map.insert(*id, n.clone());
            }
        }
    }
    name_map
}

// ======================= 对比（同管线 直接 vs 数据库） =======================

/// params flavor（"full"/"embedding"/"association"，缺省 full）→ (直接模式, 数据库模式)。
fn compare_db_modes(flavor: &str) -> (RetrieveMode, RetrieveMode) {
    let direct = match flavor {
        "embedding" => RetrieveMode::Embedding,
        "association" => RetrieveMode::Association,
        _ => RetrieveMode::FullPipeline,
    };
    let db = direct
        .db_mode()
        .expect("direct retrieve mode always maps to a db mode");
    (direct, db)
}

/// 直接 vs 数据库 对比：同一 question.json、同一管线（flavor），两阶段执行后配对报告。
fn run_compare_db_impl(
    params: &HashMap<String, String>,
    dataset_name: &str,
    dataset_path: &Path,
    sink: &StreamSink<String>,
) -> anyhow::Result<()> {
    let flavor = params.get("flavor").map(String::as_str).unwrap_or("full");
    let (direct_mode, db_mode) = compare_db_modes(flavor);

    CANCEL.store(false, Ordering::SeqCst);

    // 阶段 1：直接（全量工作记忆）
    emit(
        sink,
        &CompareEvent::Loading {
            message: format!("正在加载直接套件（{direct_mode}）..."),
        },
    );
    let direct_suite: Box<dyn TestSuite> = Box::new(
        RetrieveSuite::load_with_params(dataset_path, direct_mode, Some(params))
            .map_err(|e| anyhow::anyhow!("加载直接套件失败: {e}"))?,
    );
    let direct_total = direct_suite.case_count();
    emit(
        sink,
        &CompareEvent::Loading {
            message: format!("直接套件就绪，共 {direct_total} 个用例"),
        },
    );
    let (direct_outcomes, _, _) =
        run_compare_phase(direct_suite.as_ref(), "direct", direct_total, sink)?;

    // 阶段 2：数据库（example_data 先入 mem 数据库，再 DB 召回）
    emit(
        sink,
        &CompareEvent::Loading {
            message: format!("正在加载数据库套件（{db_mode}）..."),
        },
    );
    let db_suite: Box<dyn TestSuite> = Box::new(
        RetrieveSuite::load_with_params(dataset_path, db_mode, Some(params))
            .map_err(|e| anyhow::anyhow!("加载数据库套件失败: {e}"))?,
    );
    let db_total = db_suite.case_count();
    emit(
        sink,
        &CompareEvent::Loading {
            message: format!("数据库套件就绪，共 {db_total} 个用例"),
        },
    );
    let (db_outcomes, _, _) = run_compare_phase(db_suite.as_ref(), "db", db_total, sink)?;

    let report = build_db_compare_report(&direct_outcomes, &db_outcomes, flavor.to_string());
    emit(
        sink,
        &CompareEvent::Done {
            report: build_db_compare_json(
                report,
                &direct_outcomes,
                &db_outcomes,
                dataset_name,
                dataset_path,
            ),
        },
    );
    Ok(())
}

/// 把引擎的 DB 对比报告映射为 UI 既有 CompareReportJson 契约：
/// embedding_* 字段承载「直接」侧，fullpipeline_* 字段承载「数据库」侧
/// （UI 依据 kind/flavor 参数决定展示标签，无需改动 FRB 绑定）。
fn build_db_compare_json(
    report: DbCompareReport,
    direct_outcomes: &[TestCaseOutcome],
    db_outcomes: &[TestCaseOutcome],
    dataset_name: &str,
    dataset_path: &Path,
) -> CompareReportJson {
    let name_map = collect_graph_names(direct_outcomes);
    let names = |ids: &[soul_mem_core::memory_note::MemoryId]| -> Vec<String> {
        ids.iter()
            .map(|id| {
                name_map
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("{id:?}"))
            })
            .collect()
    };
    let direct_map = index_side_data(direct_outcomes);
    let db_map = index_side_data(db_outcomes);
    let agg = &report.aggregate;
    CompareReportJson {
        dataset_name: dataset_name.to_string(),
        dataset_path: dataset_path.to_string_lossy().to_string(),
        aggregate: CompareAggregateJson {
            case_count: agg.case_count,
            avg_embedding_hit: agg.avg_direct_hit,
            avg_fullpipeline_hit: agg.avg_db_hit,
            avg_embedding_mrr: agg.avg_direct_mrr,
            avg_fullpipeline_mrr: agg.avg_db_mrr,
            hit_improvement_count: agg.hit_improved_count,
            mrr_improvement_count: agg.mrr_improved_count,
        },
        cases: report
            .cases
            .iter()
            .map(|c| {
                let key = compare_key(&c.case_name, c.tag_weight, c.variant_weight);
                let CaseExtras {
                    emb_precision,
                    full_precision,
                    emb_ndcg,
                    full_ndcg,
                    per_query,
                } = case_extras_json(&key, &direct_map, &db_map);
                CompareCaseJson {
                    case_name: c.case_name.clone(),
                    description: c.description.clone(),
                    tag_weight: c.tag_weight,
                    variant_weight: c.variant_weight,
                    embedding_hit: c.direct_hit,
                    fullpipeline_hit: c.db_hit,
                    embedding_mrr: c.direct_mrr,
                    fullpipeline_mrr: c.db_mrr,
                    embedding_recall_at: c.direct_recall_at.clone(),
                    fullpipeline_recall_at: c.db_recall_at.clone(),
                    embedding_precision_at: emb_precision,
                    fullpipeline_precision_at: full_precision,
                    embedding_ndcg_at: emb_ndcg,
                    fullpipeline_ndcg_at: full_ndcg,
                    embedding_retrieved: names(&c.direct_retrieved),
                    fullpipeline_retrieved: names(&c.db_retrieved),
                    expected_combined_ranking: names(&c.expected_combined_ranking),
                    improved_hit: c.improved_hit,
                    improved_mrr: c.improved_mrr,
                    per_query,
                }
            })
            .collect(),
    }
}

// ======================= 检视数据集 =======================

#[derive(Serialize)]
struct InspectLinkJson {
    from_id: String,
    to_id: String,
    link_type_desc: String,
    intensity: f64,
    is_outgoing: bool,
    /// 邻居节点在条目列表中的索引（GUI 点击链接跳转）
    target_idx: usize,
}

#[derive(Serialize)]
struct InspectEntryJson {
    id: String,
    summary: String,
    preview_lines: Vec<String>,
    detail_lines: Vec<String>,
    links: Vec<InspectLinkJson>,
}

#[derive(Serialize)]
struct InspectEntriesJson {
    file_type: String,
    file_path: String,
    stats: Option<Vec<String>>,
    entries: Vec<InspectEntryJson>,
}

/// 检视数据集（结构化条目）：复用 engine/inspect 的解析（与 TUI inspect 一致），
/// 输出图节点/查询用例的分层条目供 GUI 卡片渲染。
#[frb]
pub fn inspect_entries_json(path: String) -> String {
    let data = soul_tune::engine::inspect::inspect_data(PathBuf::from(&path));
    let file_type = match data.file_type {
        soul_tune::engine::inspect::InspectFileType::Graph => "graph",
        soul_tune::engine::inspect::InspectFileType::Query => "question",
    };
    let entries: Vec<InspectEntryJson> = data
        .entries
        .into_iter()
        .map(|e| InspectEntryJson {
            id: e.id,
            summary: e.summary,
            preview_lines: e.preview_lines,
            detail_lines: e.detail_lines,
            links: e
                .links
                .into_iter()
                .map(|l| InspectLinkJson {
                    from_id: l.from_id,
                    to_id: l.to_id,
                    link_type_desc: l.link_type_desc,
                    intensity: l.intensity,
                    is_outgoing: l.is_outgoing,
                    target_idx: l.target_idx,
                })
                .collect(),
        })
        .collect();
    serde_json::to_string(&InspectEntriesJson {
        file_type: file_type.to_string(),
        file_path: path,
        stats: data.stats,
        entries,
    })
    .unwrap_or_else(|_| "{}".to_string())
}

/// 检视任意 JSON 数据集文件：检测类型（question/graph/json）并返回完整解析数据。
#[frb]
pub fn inspect_file_json(path: String) -> String {
    let meta = |file_type: &str, error: Option<&str>, data: serde_json::Value| {
        serde_json::json!({
            "path": path,
            "file_type": file_type,
            "error": error,
            "data": data,
        })
        .to_string()
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return meta("error", Some("无法读取文件"), serde_json::Value::Null);
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) else {
        return meta("error", Some("JSON 解析失败"), serde_json::Value::Null);
    };
    let file_type = if v.is_array() {
        // 顶层数组视为图节点列表（部分 graph.json 以数组形式存储）
        "graph"
    } else if v.get("test_cases").is_some() {
        "question"
    } else if v.get("nodes").is_some() || v.get("links").is_some() {
        "graph"
    } else {
        "json"
    };
    meta(file_type, None, v)
}

// ======================= 遗忘测试（mask / revise / pipeline） =======================

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ForgetEvent {
    Loading {
        message: String,
    },
    Progress {
        done: usize,
        total: usize,
        passed: usize,
        failed: usize,
        elapsed_ms: u64,
        case_name: String,
    },
    Done {
        report: ForgetReportJson,
    },
    Error {
        message: String,
    },
    Cancelled,
}

/// 遗忘观测页数据：pipeline 为节点列表，mask/revise 为指标+明细文本。
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ForgetObserverCaseJson {
    Nodes {
        case_name: String,
        passed: bool,
        llm_available: bool,
        node_count: usize,
        edge_count: usize,
        llm_revised: usize,
        effective_revised: usize,
        action_histogram: Vec<(String, usize)>,
        avg_missing_degree: f32,
        max_missing_degree: f32,
        avg_masked_ratio: f32,
        avg_edge_intensity: f64,
        /// 用例代表的时间跨度（low=8h / medium=24h / high=72h，供节点演变曲线 x 轴）
        hours: Option<f64>,
        nodes: Vec<NodeForgetStat>,
        /// 逐节点时间步长序列：遗忘以节点为单位，节点内容按时间步变化
        node_series: Vec<NodeSeries>,
        /// 理想艾宾浩斯曲线采样（x=小时, y=缺失度），与实测叠加对比
        ideal_points: Vec<(f64, f64)>,
        /// 该用例的指标（激发测试按 case 聚合三时机对比时使用）
        metrics: Vec<(String, String, String)>,
    },
    Text {
        case_name: String,
        /// 源记忆节点 id（mask/revise 按此以节点为单位展示）
        node_id: Option<String>,
        passed: bool,
        llm_available: bool,
        /// 原文（mask/revise 均提供，供原文对照展示）
        original: Option<String>,
        /// mask：遮罩结果文本；revise：遮罩输入
        masked: Option<String>,
        /// 遮罩率（mask 模式）
        mask_ratio: Option<f64>,
        llm_reply: Option<String>,
        metrics: Vec<(String, String, String)>,
        detail_lines: Vec<String>,
    },
}

#[derive(Serialize)]
struct ForgetReportJson {
    mode: String,
    dataset_name: String,
    dataset_path: String,
    total: usize,
    passed: usize,
    failed: usize,
    elapsed_secs: f64,
    metrics: Vec<MetricEntry>,
    detail_header: String,
    detail_rows: Vec<DetailRow>,
    cases: Vec<ForgetObserverCaseJson>,
}

/// 运行遗忘套件（mask=遮罩 / revise[=/full|=/sample[:seed]]=遮罩补全 /
/// pipeline=全管线 / excitation=激发测试）。
#[frb]
pub fn run_forget(mode: String, dataset: String, sink: StreamSink<String>) {
    std::thread::spawn(move || {
        run_stream(
            &sink,
            || run_forget_impl(&mode, &dataset, &sink),
            |message| ForgetEvent::Error { message },
        );
    });
}

fn run_forget_impl(mode: &str, dataset: &str, sink: &StreamSink<String>) -> anyhow::Result<()> {
    let dataset_path = PathBuf::from(dataset);
    let dataset_name = dataset_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    CANCEL.store(false, Ordering::SeqCst);
    emit(
        sink,
        &ForgetEvent::Loading {
            message: "正在加载遗忘套件...".into(),
        },
    );

    let suite: Box<dyn TestSuite> = match mode {
        "mask" => Box::new(
            ForgetMaskSuite::load(&dataset_path)
                .map_err(|e| anyhow::anyhow!("加载遮罩套件失败: {e}"))?,
        ),
        "revise" | "revise/full" => Box::new(
            ForgetReviseSuite::load(&dataset_path)
                .map_err(|e| anyhow::anyhow!("加载遮罩补全套件失败: {e}"))?,
        ),
        // revise/sample[:seed]：分层抽样（默认种子 20260820），启用 LLM
        m if m.starts_with("revise/sample") => {
            let seed = m
                .split(':')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(20260820u64);
            Box::new(
                ForgetReviseSuite::load_sampled(&dataset_path, seed)
                    .map_err(|e| anyhow::anyhow!("加载遮罩补全套件失败: {e}"))?,
            )
        }
        "pipeline" | "full" => Box::new(
            ForgetPipelineSuite::load(&dataset_path)
                .map_err(|e| anyhow::anyhow!("加载全管线套件失败: {e}"))?,
        ),
        "excitation" => Box::new(
            ForgetPipelineSuite::load_excitation_only(&dataset_path)
                .map_err(|e| anyhow::anyhow!("加载激发测试套件失败: {e}"))?,
        ),
        other => return Err(anyhow::anyhow!("未知遗忘模式: {other}")),
    };
    let total = suite.case_count();
    emit(
        sink,
        &ForgetEvent::Loading {
            message: format!("准备就绪，共 {total} 个用例"),
        },
    );

    let start = Instant::now();
    let mut outcomes = Vec::with_capacity(total);
    let mut passed = 0usize;
    let mut failed = 0usize;
    for i in 0..total {
        if CANCEL.load(Ordering::SeqCst) {
            emit(sink, &ForgetEvent::Cancelled);
            return Ok(());
        }
        let outcome = suite.run_case(i);
        let case_name = outcome.case_name.clone();
        if outcome.passed {
            passed += 1;
        } else {
            failed += 1;
        }
        outcomes.push(outcome);
        emit(
            sink,
            &ForgetEvent::Progress {
                done: i + 1,
                total,
                passed,
                failed,
                elapsed_ms: start.elapsed().as_millis() as u64,
                case_name,
            },
        );
    }
    let elapsed = start.elapsed();
    let report = suite.build_report(outcomes, elapsed, total, passed, failed);

    let cases: Vec<ForgetObserverCaseJson> = report
        .outcomes
        .iter()
        .filter_map(|o| {
            if let Some(d) = o.data.downcast_ref::<ForgetCaseData>() {
                // 理想艾宾浩斯曲线：以该用例节点序列的最大时间步为横轴范围
                let max_hours = d
                    .node_series
                    .iter()
                    .flat_map(|ns| ns.steps.iter().map(|s| s.hours))
                    .max()
                    .unwrap_or(0);
                // 激发测试的基线是"未激发对照组"（md_ctrl），理想艾宾浩斯曲线
                // （零激活理论公式）对它无意义且有误导，故不输出。
                let ideal_points = if d.case_name.starts_with("excitation-") {
                    vec![]
                } else {
                    ideal_ebbinghaus_curve(max_hours)
                };
                Some(ForgetObserverCaseJson::Nodes {
                    case_name: d.case_name.clone(),
                    passed: d.passed,
                    llm_available: d.llm_available,
                    node_count: d.node_count,
                    edge_count: d.edge_count,
                    llm_revised: d.llm_revised,
                    effective_revised: d.effective_revised,
                    action_histogram: d
                        .action_histogram
                        .iter()
                        .map(|(k, v)| (k.to_string(), *v))
                        .collect(),
                    avg_missing_degree: d.avg_missing_degree,
                    max_missing_degree: d.max_missing_degree,
                    avg_masked_ratio: d.avg_masked_ratio,
                    avg_edge_intensity: d.avg_edge_intensity,
                    hours: match d.case_name.as_str() {
                        "low" => Some(8.0),
                        "medium" => Some(24.0),
                        "high" => Some(72.0),
                        _ => None,
                    },
                    nodes: d.nodes.clone(),
                    node_series: d.node_series.clone(),
                    ideal_points,
                    metrics: d.metrics.clone(),
                })
            } else if let Some(d) = o.data.downcast_ref::<ReviseCaseData>() {
                Some(ForgetObserverCaseJson::Text {
                    case_name: d.case_name.clone(),
                    node_id: Some(d.node_id.clone()),
                    passed: d.passed,
                    llm_available: d.llm_available,
                    original: Some(d.original.clone()),
                    masked: Some(d.masked_text.clone()),
                    mask_ratio: None,
                    llm_reply: Some(d.llm_reply.clone()),
                    metrics: d.metrics.clone(),
                    detail_lines: d.detail_lines.clone(),
                })
            } else {
                o.data
                    .downcast_ref::<MaskCaseData>()
                    .map(|d| ForgetObserverCaseJson::Text {
                        case_name: d.case_name.clone(),
                        node_id: Some(d.node_id.clone()),
                        passed: d.passed,
                        llm_available: false,
                        original: Some(d.original.clone()),
                        masked: Some(d.masked.clone()),
                        mask_ratio: if d.total_count > 0 {
                            Some(d.masked_count as f64 / d.total_count as f64)
                        } else {
                            None
                        },
                        llm_reply: None,
                        metrics: d.metrics.clone(),
                        detail_lines: d.detail_lines.clone(),
                    })
            }
        })
        .collect();

    emit(
        sink,
        &ForgetEvent::Done {
            report: ForgetReportJson {
                mode: mode.to_string(),
                dataset_name: dataset_name.clone(),
                dataset_path: dataset_path.to_string_lossy().to_string(),
                total,
                passed,
                failed,
                elapsed_secs: elapsed.as_secs_f64(),
                metrics: report.metrics,
                detail_header: report.detail_header,
                detail_rows: report.detail_rows,
                cases,
            },
        },
    );
    Ok(())
}

// ======================= 模型来源（llama-server） =======================

/// 模型来源状态（只探测、不启动）：
/// - `running`：检测到**已运行**的 llama-server，直接复用；
/// - `spawned`：无运行服务，将自动拉起本地缓存模型（`model_path`）；
/// - `unavailable`：都没有，报错或降级（`reason` 说明）。
#[derive(Serialize)]
struct ModelStatusJson {
    available: bool,
    source: String,
    url: Option<String>,
    model_path: Option<String>,
    reason: Option<String>,
}

/// 查询模型可用性：先探测运行中的 llama-server，无则查找本地缓存模型（不启动）。
///
/// 所有需要模型的地方共用同一套来源决策（见 `soul_tune::engine::llm::resolver`）。
#[frb]
pub fn model_status_json() -> String {
    let s = soul_tune::engine::llm::probe_status();
    serde_json::to_string(&ModelStatusJson {
        available: s.available,
        source: s.source,
        url: s.url,
        model_path: s.model_path,
        reason: s.reason,
    })
    .unwrap_or_else(|_| "{}".to_string())
}

// ======================= 角色扮演（playtest） =======================

struct PlaytestSession {
    runner: PlayTestRunner,
    llm: LlamaServer,
    turn_index: usize,
    character_name: String,
    /// 人工投票记录：turn_index → 0=embedding 更好 / 1=full 更好 / 2=持平
    votes: HashMap<usize, u8>,
}

static PLAYTEST_SESSION: OnceLock<Mutex<Option<PlaytestSession>>> = OnceLock::new();

fn playtest_session() -> &'static Mutex<Option<PlaytestSession>> {
    PLAYTEST_SESSION.get_or_init(|| Mutex::new(None))
}

/// 启动 playtest：加载角色图 + LLM。
/// 参数可以是图目录（须含 graph.json）或 graph.json 文件本身（取其父目录）。
/// `user_role` 非空时作为对话中对方（人类）的身份注入。
/// LLM 来源统一解析：先复用运行中的 llama-server，无则自动拉起本地缓存模型，
/// 都没有则返回明确错误（见 `soul_tune::engine::llm::resolver`）。
#[frb]
pub fn playtest_start(graph_dir: String, user_role: String) -> String {
    let err = |msg: String| {
        serde_json::json!({ "ok": false, "character_name": "", "error": msg }).to_string()
    };

    // 归一化：去除 \\?\ 前缀与首尾空白（file_picker 某些版本可能返回扩展路径）
    let graph_dir = graph_dir.trim().trim_start_matches("\\\\?\\");
    let graph_path = Path::new(graph_dir);
    let (resolved_dir, picked_file) = if graph_path.is_dir() {
        (graph_path.to_path_buf(), None)
    } else {
        let parent = graph_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| graph_path.to_path_buf());
        (
            parent,
            graph_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string()),
        )
    };
    let graph_file = resolved_dir.join("graph.json");
    if !graph_file.exists() {
        let hint = match &picked_file {
            Some(f) if f != "graph.json" => {
                format!("所选文件不是 graph.json（实际为 {f}）。请选择角色图目录下的 graph.json")
            }
            _ => format!("未找到角色图: {}", graph_file.display()),
        };
        return err(hint);
    }
    let runner = match PlayTestRunner::load(&resolved_dir) {
        Ok(r) => r,
        Err(e) => return err(format!("加载角色图失败（{}）: {e}", resolved_dir.display())),
    };
    let runner = if user_role.trim().is_empty() {
        runner
    } else {
        runner.with_human_role(Some(user_role.trim().to_string()))
    };
    let character_name = resolved_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "角色".to_string());

    // 统一模型来源解析：复用运行中的 llama-server → 自动拉起本地缓存模型 → 报错
    let resolution = soul_tune::engine::llm::resolve_llm();
    let llm = match resolution.server {
        Some(l) => l,
        None => {
            let reason = resolution
                .status
                .reason
                .unwrap_or_else(|| "LLM 不可用".to_string());
            return err(format!(
                "LLM 不可用: {reason}（playtest 需要 LLM 生成查询与回复）"
            ));
        }
    };

    *playtest_session().lock().unwrap() = Some(PlaytestSession {
        runner,
        llm,
        turn_index: 0,
        character_name: character_name.clone(),
        votes: HashMap::new(),
    });
    serde_json::json!({ "ok": true, "character_name": character_name, "error": serde_json::Value::Null })
        .to_string()
}

/// 记录一轮对话的人工投票：0=embedding 更好 / 1=full 更好 / 2=持平。
#[frb]
pub fn playtest_vote(turn_index: usize, pick: u8) {
    if let Ok(mut guard) = playtest_session().lock() {
        if let Some(s) = guard.as_mut() {
            s.votes.insert(turn_index, pick);
        }
    }
}

/// 结束 playtest 会话（关闭 llama-server 子进程）。
#[frb]
pub fn playtest_finish() {
    *playtest_session().lock().unwrap() = None;
}

#[derive(Serialize)]
struct TraceNodeJson {
    name: String,
    stage: String,
    score: f64,
    content: String,
}

#[derive(Serialize)]
struct PerQueryJson {
    dropped: bool,
    preview: String,
    sim: usize,
    ppr: usize,
    action: usize,
    elapsed_ms: u64,
}

#[derive(Serialize)]
struct TraceJson {
    mode: String,
    total_elapsed_ms: u64,
    merged: Vec<TraceNodeJson>,
    actions: Vec<TraceNodeJson>,
    speech: Vec<TraceNodeJson>,
    think: Vec<TraceNodeJson>,
    per_query: Vec<PerQueryJson>,
}

#[derive(Serialize)]
struct RunJson {
    response: Option<String>,
    trace: Option<TraceJson>,
}

#[derive(Serialize)]
struct TurnJson {
    index: usize,
    user_message: String,
    error: Option<String>,
    generated_queries_json: String,
    query_think_content: Option<String>,
    embedding: RunJson,
    full: RunJson,
}

/// 处理一轮对话：生成查询 → 双管线检索 → LLM 回复，结果以单条 JSON 推流。
#[frb]
pub fn playtest_turn(user_message: String, sink: StreamSink<String>) {
    std::thread::spawn(move || {
        let session = playtest_session();
        let Ok(mut guard) = session.lock() else {
            return;
        };
        let Some(s) = guard.as_mut() else {
            let _ = sink.add(
                serde_json::json!({ "index": 0, "user_message": "", "error": "playtest 尚未启动，请先选择角色图", "generated_queries_json": "", "query_think_content": null, "embedding": null, "full": null }).to_string(),
            );
            return;
        };
        let entry = ConversationEntry {
            user_message: user_message.clone(),
        };
        let idx = s.turn_index;
        s.turn_index += 1;
        let result = s.runner.process_turn(&entry, idx, &mut s.llm);
        drop(guard);
        let _ = sink.add(build_turn_json(&result));
    });
}

fn stage_name(stage: HitStage) -> &'static str {
    match stage {
        HitStage::Similarity => "similarity",
        HitStage::Ppr => "ppr",
        HitStage::Action => "action",
        HitStage::Both => "both",
    }
}

fn trace_node(n: &TracedNode) -> TraceNodeJson {
    TraceNodeJson {
        name: n.name.clone(),
        stage: stage_name(n.stage).to_string(),
        score: n.score,
        content: n.content.clone(),
    }
}

fn query_preview(v: &MemoryRetrieveQueryVariant) -> String {
    match v {
        MemoryRetrieveQueryVariant::Semantic(units) => units
            .iter()
            .filter_map(|u| u.concept_identifier())
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" | "),
        MemoryRetrieveQueryVariant::Situation(units) => units
            .iter()
            .map(|u| {
                let mut p = Vec::new();
                if let Some(n) = u.narrative() {
                    p.push(format!("故事:{n}"));
                }
                if let Some(l) = u.location() {
                    p.push(format!(
                        "地点:{}",
                        l.iter().map(|x| x.name()).collect::<Vec<_>>().join(",")
                    ));
                }
                if let Some(ps) = u.participants() {
                    p.push(format!(
                        "人物:{}",
                        ps.iter()
                            .filter_map(|x| x.name())
                            .collect::<Vec<_>>()
                            .join(",")
                    ));
                }
                p.join(" ")
            })
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

fn trace_json(t: &Option<RetrievalTrace>) -> Option<TraceJson> {
    t.as_ref().map(|tr| TraceJson {
        mode: tr.mode.to_string(),
        total_elapsed_ms: tr.total_elapsed.as_millis() as u64,
        merged: tr.merged_nodes.iter().map(trace_node).collect(),
        actions: tr.action_nodes.iter().map(trace_node).collect(),
        speech: tr.speech_nodes.iter().map(trace_node).collect(),
        think: tr.think_nodes.iter().map(trace_node).collect(),
        per_query: tr
            .per_query
            .iter()
            .map(|q| PerQueryJson {
                dropped: q.dropped,
                preview: query_preview(q.query.variant()),
                sim: q.sim_nodes.len(),
                ppr: q.ppr_nodes.len(),
                action: q.action_nodes.len(),
                elapsed_ms: q.total_elapsed.as_millis() as u64,
            })
            .collect(),
    })
}

fn build_turn_json(t: &soul_tune::engine::playtest::runner::PlayTurnResult) -> String {
    let first = t.runs.first();
    serde_json::json!({
        "index": t.index,
        "user_message": t.user_message,
        "error": first.and_then(|r| r.error.as_ref()),
        "generated_queries_json": t.generated_queries_json,
        "query_think_content": t.query_think_content,
        "embedding": {
            "response": first.and_then(|r| r.embedding_response.as_ref()),
            "trace": trace_json(&t.embedding_trace),
        },
        "full": {
            "response": first.and_then(|r| r.fullpipeline_response.as_ref()),
            "trace": trace_json(&t.fullpipeline_trace),
        },
    })
    .to_string()
}

// 供 build_report_json 之外复用的小工具（保留，供将来扩展）
#[allow(dead_code)]
fn outcome_json(o: &TestCaseOutcome) -> OutcomeJson {
    let data = o
        .data
        .downcast_ref::<RetrieveCaseData>()
        .and_then(|d| serde_json::to_value(d).ok())
        .unwrap_or(serde_json::Value::Null);
    OutcomeJson {
        case_name: o.case_name.clone(),
        description: o.description.clone(),
        passed: o.passed,
        data,
    }
}
