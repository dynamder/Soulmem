/// JSON-over-FRB 数据模型：与 crates/soul-tune-api 的事件/报告 JSON 一一对应。
///
/// 约定：Rust 侧 serde 输出（`#[serde(tag="type", rename_all="snake_case")]` 等），
/// 此处 fromJson 严格按字段名解析。
library;

class MetricEntry {
  final String label;
  final String group;
  final String kind; // 'key_value' | 'chart'
  final String? value;
  final String? xLabel;
  final String? yLabel;
  final List<ChartSeries>? datasets;

  MetricEntry({
    required this.label,
    required this.group,
    required this.kind,
    this.value,
    this.xLabel,
    this.yLabel,
    this.datasets,
  });

  factory MetricEntry.fromJson(Map<String, dynamic> j) => MetricEntry(
        label: j['label'] as String? ?? '',
        group: j['group'] as String? ?? '',
        kind: j['kind'] as String? ?? 'key_value',
        value: j['value'] as String?,
        xLabel: j['x_label'] as String?,
        yLabel: j['y_label'] as String?,
        datasets: (j['datasets'] as List?)
            ?.map((e) => ChartSeries.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}

class ChartSeries {
  final String label;
  final List<(double, double)> points;

  ChartSeries({required this.label, required this.points});

  factory ChartSeries.fromJson(Map<String, dynamic> j) => ChartSeries(
        label: j['label'] as String? ?? '',
        points: (j['points'] as List?)
                ?.map((e) {
                  final p = e as List;
                  if (p.length >= 2) {
                    return ((p[0] as num).toDouble(), (p[1] as num).toDouble());
                  }
                  return (0.0, 0.0);
                })
                .toList() ??
            [],
      );
}

class DetailRow {
  final String text;
  final bool hasError;

  DetailRow({required this.text, required this.hasError});

  factory DetailRow.fromJson(Map<String, dynamic> j) => DetailRow(
        text: j['text'] as String? ?? '',
        hasError: j['has_error'] as bool? ?? false,
      );
}

class Outcome {
  final String caseName;
  final String description;
  final bool passed;
  final Map<String, dynamic> data;

  Outcome({
    required this.caseName,
    required this.description,
    required this.passed,
    required this.data,
  });

  factory Outcome.fromJson(Map<String, dynamic> j) => Outcome(
        caseName: j['case_name'] as String? ?? '',
        description: j['description'] as String? ?? '',
        passed: j['passed'] as bool? ?? false,
        data: (j['data'] as Map?)?.cast<String, dynamic>() ?? const {},
      );
}

class Report {
  final String algo;
  final String datasetName;
  final String datasetPath;
  final int total;
  final int passed;
  final int failed;
  final double elapsedSecs;
  final List<MetricEntry> metrics;
  final String detailHeader;
  final List<DetailRow> detailRows;
  final List<Outcome> outcomes;

  Report({
    required this.algo,
    required this.datasetName,
    required this.datasetPath,
    required this.total,
    required this.passed,
    required this.failed,
    required this.elapsedSecs,
    required this.metrics,
    required this.detailHeader,
    required this.detailRows,
    required this.outcomes,
  });

  double get passRate => total == 0 ? 0 : passed / total;

  factory Report.fromJson(Map<String, dynamic> j) => Report(
        algo: j['algo'] as String? ?? '',
        datasetName: j['dataset_name'] as String? ?? '',
        datasetPath: j['dataset_path'] as String? ?? '',
        total: j['total'] as int? ?? 0,
        passed: j['passed'] as int? ?? 0,
        failed: j['failed'] as int? ?? 0,
        elapsedSecs: (j['elapsed_secs'] as num?)?.toDouble() ?? 0,
        metrics: (j['metrics'] as List?)
                ?.map((e) => MetricEntry.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        detailHeader: j['detail_header'] as String? ?? '',
        detailRows: (j['detail_rows'] as List?)
                ?.map((e) => DetailRow.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        outcomes: (j['outcomes'] as List?)
                ?.map((e) => Outcome.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

// ── 单跑事件流 ──

sealed class RunEvent {
  const RunEvent();

  factory RunEvent.fromJson(Map<String, dynamic> j) {
    return switch (j['type']) {
      'loading' => RunLoading(message: j['message'] as String? ?? ''),
      'progress' => RunProgress(
          done: j['done'] as int? ?? 0,
          total: j['total'] as int? ?? 0,
          passed: j['passed'] as int? ?? 0,
          failed: j['failed'] as int? ?? 0,
          elapsedMs: j['elapsed_ms'] as int? ?? 0,
          caseName: j['case_name'] as String? ?? '',
        ),
      'done' => RunDone(report: Report.fromJson(j['report'] as Map<String, dynamic>)),
      'error' => RunError(message: j['message'] as String? ?? ''),
      'cancelled' => const RunCancelled(),
      _ => const RunError(message: '未知事件'),
    };
  }
}

class RunLoading extends RunEvent {
  final String message;
  const RunLoading({required this.message});
}

class RunProgress extends RunEvent {
  final int done;
  final int total;
  final int passed;
  final int failed;
  final int elapsedMs;
  final String caseName;
  const RunProgress({
    required this.done,
    required this.total,
    required this.passed,
    required this.failed,
    required this.elapsedMs,
    required this.caseName,
  });
}

class RunDone extends RunEvent {
  final Report report;
  const RunDone({required this.report});
}

class RunError extends RunEvent {
  final String message;
  const RunError({required this.message});
}

class RunCancelled extends RunEvent {
  const RunCancelled();
}

// ── 批量事件流 ──

sealed class BatchEvent {
  const BatchEvent();

  factory BatchEvent.fromJson(Map<String, dynamic> j) {
    return switch (j['type']) {
      'scanning' => BatchScanning(dir: j['dir'] as String? ?? ''),
      'progress' => BatchProgress(
          done: j['done'] as int? ?? 0,
          total: j['total'] as int? ?? 0,
        ),
      'dataset_done' => BatchDatasetDone(
          index: j['index'] as int? ?? 0,
          name: j['name'] as String? ?? '',
          total: j['total'] as int? ?? 0,
          passed: j['passed'] as int? ?? 0,
          failed: j['failed'] as int? ?? 0,
          passRate: (j['pass_rate'] as num?)?.toDouble() ?? 0,
          elapsedMs: j['elapsed_ms'] as int? ?? 0,
          error: j['error'] as String?,
        ),
      'done' => BatchDone(result: BatchReport.fromJson(j['result'] as Map<String, dynamic>)),
      'error' => BatchError(message: j['message'] as String? ?? ''),
      'cancelled' => const BatchCancelled(),
      _ => const BatchError(message: '未知事件'),
    };
  }
}

class BatchScanning extends BatchEvent {
  final String dir;
  const BatchScanning({required this.dir});
}

class BatchProgress extends BatchEvent {
  final int done;
  final int total;
  const BatchProgress({required this.done, required this.total});
}

class BatchDatasetDone extends BatchEvent {
  final int index;
  final String name;
  final int total;
  final int passed;
  final int failed;
  final double passRate;
  final int elapsedMs;
  final String? error;
  const BatchDatasetDone({
    required this.index,
    required this.name,
    required this.total,
    required this.passed,
    required this.failed,
    required this.passRate,
    required this.elapsedMs,
    this.error,
  });
}

class BatchDone extends BatchEvent {
  final BatchReport result;
  const BatchDone({required this.result});
}

class BatchError extends BatchEvent {
  final String message;
  const BatchError({required this.message});
}

class BatchCancelled extends BatchEvent {
  const BatchCancelled();
}

class DatasetResultJson {
  final String name;
  final String path;
  final int total;
  final int passed;
  final int failed;
  final double passRate;
  final int elapsedMs;
  final String? error;
  final List<Outcome> outcomes;

  DatasetResultJson({
    required this.name,
    required this.path,
    required this.total,
    required this.passed,
    required this.failed,
    required this.passRate,
    required this.elapsedMs,
    this.error,
    required this.outcomes,
  });

  factory DatasetResultJson.fromJson(Map<String, dynamic> j) => DatasetResultJson(
        name: j['name'] as String? ?? '',
        path: j['path'] as String? ?? '',
        total: j['total'] as int? ?? 0,
        passed: j['passed'] as int? ?? 0,
        failed: j['failed'] as int? ?? 0,
        passRate: (j['pass_rate'] as num?)?.toDouble() ?? 0,
        elapsedMs: j['elapsed_ms'] as int? ?? 0,
        error: j['error'] as String?,
        outcomes: (j['outcomes'] as List?)
                ?.map((e) => Outcome.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

class BatchReport {
  final int totalDatasets;
  final int totalCases;
  final int totalPassed;
  final int totalFailed;
  final double elapsedSecs;
  final List<DatasetResultJson> datasets;

  BatchReport({
    required this.totalDatasets,
    required this.totalCases,
    required this.totalPassed,
    required this.totalFailed,
    required this.elapsedSecs,
    required this.datasets,
  });

  factory BatchReport.fromJson(Map<String, dynamic> j) => BatchReport(
        totalDatasets: j['total_datasets'] as int? ?? 0,
        totalCases: j['total_cases'] as int? ?? 0,
        totalPassed: j['total_passed'] as int? ?? 0,
        totalFailed: j['total_failed'] as int? ?? 0,
        elapsedSecs: (j['elapsed_secs'] as num?)?.toDouble() ?? 0,
        datasets: (j['datasets'] as List?)
                ?.map((e) => DatasetResultJson.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

// ── 元数据 ──

class DatasetEntry {
  final String name;
  final String path;
  DatasetEntry({required this.name, required this.path});

  factory DatasetEntry.fromJson(Map<String, dynamic> j) =>
      DatasetEntry(name: j['name'] as String? ?? '', path: j['path'] as String? ?? '');
}

class DatasetMeta {
  final String name;
  final String description;
  final int caseCount;
  final String graphPath;
  final String algoType;
  final String? error;

  DatasetMeta({
    required this.name,
    required this.description,
    required this.caseCount,
    required this.graphPath,
    required this.algoType,
    this.error,
  });

  factory DatasetMeta.fromJson(Map<String, dynamic> j) => DatasetMeta(
        name: j['name'] as String? ?? '',
        description: j['description'] as String? ?? '',
        caseCount: j['case_count'] as int? ?? 0,
        graphPath: j['graph_path'] as String? ?? '',
        algoType: j['algo_type'] as String? ?? '',
        error: j['error'] as String?,
      );
}

class ParamSpec {
  final String name;
  final String defaultValue;
  final String description;
  ParamSpec({required this.name, required this.defaultValue, required this.description});

  factory ParamSpec.fromJson(Map<String, dynamic> j) => ParamSpec(
        name: j['name'] as String? ?? '',
        defaultValue: j['default'] as String? ?? '',
        description: j['description'] as String? ?? '',
      );
}

// ── 对比（embedding vs full）──

sealed class CompareEvent {
  const CompareEvent();

  factory CompareEvent.fromJson(Map<String, dynamic> j) {
    return switch (j['type']) {
      'loading' => CompareLoading(message: j['message'] as String? ?? ''),
      'progress' => CompareProgress(
          phase: j['phase'] as String? ?? '',
          done: j['done'] as int? ?? 0,
          total: j['total'] as int? ?? 0,
          passed: j['passed'] as int? ?? 0,
          failed: j['failed'] as int? ?? 0,
          elapsedMs: j['elapsed_ms'] as int? ?? 0,
          caseName: j['case_name'] as String? ?? '',
        ),
      'done' => CompareDone(report: CompareReport.fromJson(j['report'] as Map<String, dynamic>)),
      'error' => CompareError(message: j['message'] as String? ?? ''),
      'cancelled' => const CompareCancelled(),
      _ => const CompareError(message: '未知事件'),
    };
  }
}

class CompareLoading extends CompareEvent {
  final String message;
  const CompareLoading({required this.message});
}

class CompareProgress extends CompareEvent {
  final String phase;
  final int done;
  final int total;
  final int passed;
  final int failed;
  final int elapsedMs;
  final String caseName;
  const CompareProgress({
    required this.phase,
    required this.done,
    required this.total,
    required this.passed,
    required this.failed,
    required this.elapsedMs,
    required this.caseName,
  });
}

class CompareDone extends CompareEvent {
  final CompareReport report;
  const CompareDone({required this.report});
}

class CompareError extends CompareEvent {
  final String message;
  const CompareError({required this.message});
}

class CompareCancelled extends CompareEvent {
  const CompareCancelled();
}

class CompareAggregate {
  final int caseCount;
  final double avgEmbeddingHit;
  final double avgFullpipelineHit;
  final double avgEmbeddingMrr;
  final double avgFullpipelineMrr;
  final int hitImprovementCount;
  final int mrrImprovementCount;

  CompareAggregate({
    required this.caseCount,
    required this.avgEmbeddingHit,
    required this.avgFullpipelineHit,
    required this.avgEmbeddingMrr,
    required this.avgFullpipelineMrr,
    required this.hitImprovementCount,
    required this.mrrImprovementCount,
  });

  factory CompareAggregate.fromJson(Map<String, dynamic> j) => CompareAggregate(
        caseCount: j['case_count'] as int? ?? 0,
        avgEmbeddingHit: (j['avg_embedding_hit'] as num?)?.toDouble() ?? 0,
        avgFullpipelineHit: (j['avg_fullpipeline_hit'] as num?)?.toDouble() ?? 0,
        avgEmbeddingMrr: (j['avg_embedding_mrr'] as num?)?.toDouble() ?? 0,
        avgFullpipelineMrr: (j['avg_fullpipeline_mrr'] as num?)?.toDouble() ?? 0,
        hitImprovementCount: j['hit_improvement_count'] as int? ?? 0,
        mrrImprovementCount: j['mrr_improvement_count'] as int? ?? 0,
      );
}

class CompareCase {
  final String caseName;
  final String description;
  final double tagWeight;
  final double variantWeight;
  final double embeddingHit;
  final double fullpipelineHit;
  final double embeddingMrr;
  final double fullpipelineMrr;
  final List<(int, double)> embeddingRecallAt;
  final List<(int, double)> fullpipelineRecallAt;
  final List<String> embeddingRetrieved;
  final List<String> fullpipelineRetrieved;
  final List<String> expected;
  final bool improvedHit;
  final bool improvedMrr;

  CompareCase({
    required this.caseName,
    required this.description,
    required this.tagWeight,
    required this.variantWeight,
    required this.embeddingHit,
    required this.fullpipelineHit,
    required this.embeddingMrr,
    required this.fullpipelineMrr,
    required this.embeddingRecallAt,
    required this.fullpipelineRecallAt,
    required this.embeddingRetrieved,
    required this.fullpipelineRetrieved,
    required this.expected,
    required this.improvedHit,
    required this.improvedMrr,
  });

  double get hitDelta => fullpipelineHit - embeddingHit;
  double get mrrDelta => fullpipelineMrr - embeddingMrr;

  factory CompareCase.fromJson(Map<String, dynamic> j) => CompareCase(
        caseName: j['case_name'] as String? ?? '',
        description: j['description'] as String? ?? '',
        tagWeight: (j['tag_weight'] as num?)?.toDouble() ?? 0,
        variantWeight: (j['variant_weight'] as num?)?.toDouble() ?? 0,
        embeddingHit: (j['embedding_hit'] as num?)?.toDouble() ?? 0,
        fullpipelineHit: (j['fullpipeline_hit'] as num?)?.toDouble() ?? 0,
        embeddingMrr: (j['embedding_mrr'] as num?)?.toDouble() ?? 0,
        fullpipelineMrr: (j['fullpipeline_mrr'] as num?)?.toDouble() ?? 0,
        embeddingRecallAt: _parsePairs(j['embedding_recall_at']),
        fullpipelineRecallAt: _parsePairs(j['fullpipeline_recall_at']),
        embeddingRetrieved: (j['embedding_retrieved'] as List?)?.cast<String>() ?? const [],
        fullpipelineRetrieved: (j['fullpipeline_retrieved'] as List?)?.cast<String>() ?? const [],
        expected: (j['expected_combined_ranking'] as List?)?.cast<String>() ?? const [],
        improvedHit: j['improved_hit'] as bool? ?? false,
        improvedMrr: j['improved_mrr'] as bool? ?? false,
      );

  static List<(int, double)> _parsePairs(dynamic raw) => (raw as List?)
          ?.map((e) {
            final p = e as List;
            if (p.length >= 2) return (p[0] as int, (p[1] as num).toDouble());
            return (0, 0.0);
          })
          .toList() ??
      [];
}

class CompareReport {
  final String datasetName;
  final String datasetPath;
  final CompareAggregate aggregate;
  final List<CompareCase> cases;

  CompareReport({
    required this.datasetName,
    required this.datasetPath,
    required this.aggregate,
    required this.cases,
  });

  factory CompareReport.fromJson(Map<String, dynamic> j) => CompareReport(
        datasetName: j['dataset_name'] as String? ?? '',
        datasetPath: j['dataset_path'] as String? ?? '',
        aggregate: CompareAggregate.fromJson(
            j['aggregate'] as Map<String, dynamic>? ?? const {}),
        cases: (j['cases'] as List?)
                ?.map((e) => CompareCase.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

// ── 检视数据集 ──

class InspectFile {
  final String path;
  final String fileType; // question | graph | json | error
  final String? error;
  /// 解析后的 JSON 数据：顶层可能是 Map（对象）或 List（节点数组）。
  final dynamic data;

  InspectFile({
    required this.path,
    required this.fileType,
    this.error,
    required this.data,
  });

  factory InspectFile.fromJson(Map<String, dynamic> j) => InspectFile(
        path: j['path'] as String? ?? '',
        fileType: j['file_type'] as String? ?? 'json',
        error: j['error'] as String?,
        data: j['data'],
      );
}

// ── 遗忘测试 ──

sealed class ForgetEvent {
  const ForgetEvent();

  factory ForgetEvent.fromJson(Map<String, dynamic> j) {
    return switch (j['type']) {
      'loading' => ForgetLoading(message: j['message'] as String? ?? ''),
      'progress' => ForgetProgress(
          done: j['done'] as int? ?? 0,
          total: j['total'] as int? ?? 0,
          passed: j['passed'] as int? ?? 0,
          failed: j['failed'] as int? ?? 0,
          elapsedMs: j['elapsed_ms'] as int? ?? 0,
          caseName: j['case_name'] as String? ?? '',
        ),
      'done' => ForgetDone(report: ForgetReport.fromJson(j['report'] as Map<String, dynamic>)),
      'error' => ForgetError(message: j['message'] as String? ?? ''),
      'cancelled' => const ForgetCancelled(),
      _ => const ForgetError(message: '未知事件'),
    };
  }
}

class ForgetLoading extends ForgetEvent {
  final String message;
  const ForgetLoading({required this.message});
}

class ForgetProgress extends ForgetEvent {
  final int done;
  final int total;
  final int passed;
  final int failed;
  final int elapsedMs;
  final String caseName;
  const ForgetProgress({
    required this.done,
    required this.total,
    required this.passed,
    required this.failed,
    required this.elapsedMs,
    required this.caseName,
  });
}

class ForgetDone extends ForgetEvent {
  final ForgetReport report;
  const ForgetDone({required this.report});
}

class ForgetError extends ForgetEvent {
  final String message;
  const ForgetError({required this.message});
}

class ForgetCancelled extends ForgetEvent {
  const ForgetCancelled();
}

class NodeForgetStat {
  final String id;
  final String typeName;
  final String original;
  final double mdBefore;
  final double mdAfter;
  final String action;
  final (int, int)? mask;
  final String? maskedText;
  final String? llmReply;
  final bool effective;

  NodeForgetStat({
    required this.id,
    required this.typeName,
    required this.original,
    required this.mdBefore,
    required this.mdAfter,
    required this.action,
    this.mask,
    this.maskedText,
    this.llmReply,
    required this.effective,
  });

  factory NodeForgetStat.fromJson(Map<String, dynamic> j) {
    final maskRaw = j['mask'] as List?;
    return NodeForgetStat(
      id: j['id'] as String? ?? '',
      typeName: j['type_name'] as String? ?? '',
      original: j['original'] as String? ?? '',
      mdBefore: (j['md_before'] as num?)?.toDouble() ?? 0,
      mdAfter: (j['md_after'] as num?)?.toDouble() ?? 0,
      action: j['action'] as String? ?? '',
      mask: maskRaw != null && maskRaw.length >= 2
          ? (maskRaw[0] as int, maskRaw[1] as int)
          : null,
      maskedText: j['masked_text'] as String?,
      llmReply: j['llm_reply'] as String?,
      effective: j['effective'] as bool? ?? false,
    );
  }
}

/// 单个节点在某时间步的遗忘观测（节点 × 时间步曲线的一个数据点）。
class NodeStepStat {
  final int hours; // x 轴：累计小时数（多步 24/48/72；单步为用例时间跨度）
  final int step; // 步序号（0 起始）
  final double md; // 该步后的缺失度（y 轴主指标；激发测试为激发组 md）
  final double? mdCtrl; // 对照组（未激发）同刻缺失度：仅激发测试填充，用于"对照 vs 激发"双曲线
  final String action; // NoAction / MaskOnly / Revised；激发测试为 Activated / Control
  final String? maskedText; // LLM 补全的遮罩输入
  final String? llmReply; // LLM 原始回复
  final bool effective; // 是否有效修订

  NodeStepStat({
    required this.hours,
    required this.step,
    required this.md,
    this.mdCtrl,
    required this.action,
    this.maskedText,
    this.llmReply,
    required this.effective,
  });

  factory NodeStepStat.fromJson(Map<String, dynamic> j) => NodeStepStat(
        hours: j['hours'] as int? ?? 0,
        step: j['step'] as int? ?? 0,
        md: (j['md'] as num?)?.toDouble() ?? 0,
        mdCtrl: (j['md_ctrl'] as num?)?.toDouble(),
        action: j['action'] as String? ?? '',
        maskedText: j['masked_text'] as String?,
        llmReply: j['llm_reply'] as String?,
        effective: j['effective'] as bool? ?? false,
      );
}

/// 单个节点的完整时间步长序列：遗忘以节点为单位，对节点内容按时间步变化。
class NodeSeries {
  final String id;
  final String typeName;
  final String original; // 图节点原文（遗忘前）
  final List<NodeStepStat> steps; // 按 hours 升序

  NodeSeries({
    required this.id,
    required this.typeName,
    required this.original,
    required this.steps,
  });

  factory NodeSeries.fromJson(Map<String, dynamic> j) => NodeSeries(
        id: j['id'] as String? ?? '',
        typeName: j['type_name'] as String? ?? '',
        original: j['original'] as String? ?? '',
        steps: (j['steps'] as List?)
                ?.map((e) => NodeStepStat.fromJson(e as Map<String, dynamic>))
                .toList() ??
            const [],
      );
}

sealed class ForgetObserverCase {
  const ForgetObserverCase();

  String get caseName;

  bool get passed;

  factory ForgetObserverCase.fromJson(Map<String, dynamic> j) {
    if (j['kind'] == 'nodes') {
      return ForgetObserverNodes(
        caseName: j['case_name'] as String? ?? '',
        passed: j['passed'] as bool? ?? false,
        llmAvailable: j['llm_available'] as bool? ?? false,
        nodeCount: j['node_count'] as int? ?? 0,
        edgeCount: j['edge_count'] as int? ?? 0,
        llmRevised: j['llm_revised'] as int? ?? 0,
        effectiveRevised: j['effective_revised'] as int? ?? 0,
        actionHistogram: ((j['action_histogram'] as List?) ?? const [])
            .map((e) {
              final p = e as List;
              return (p[0] as String, p[1] as int);
            })
            .toList(),
        avgMissingDegree: (j['avg_missing_degree'] as num?)?.toDouble() ?? 0,
        maxMissingDegree: (j['max_missing_degree'] as num?)?.toDouble() ?? 0,
        avgMaskedRatio: (j['avg_masked_ratio'] as num?)?.toDouble() ?? 0,
        avgEdgeIntensity: (j['avg_edge_intensity'] as num?)?.toDouble() ?? 0,
        hours: (j['hours'] as num?)?.toDouble(),
        nodes: (j['nodes'] as List?)
                ?.map((e) => NodeForgetStat.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        nodeSeries: (j['node_series'] as List?)
                ?.map((e) => NodeSeries.fromJson(e as Map<String, dynamic>))
                .toList() ??
            const [],
        idealPoints: ((j['ideal_points'] as List?) ?? const [])
            .map((e) {
              final p = e as List;
              if (p.length >= 2) {
                return ((p[0] as num).toDouble(), (p[1] as num).toDouble());
              }
              return (0.0, 0.0);
            })
            .toList(),
        metrics: ((j['metrics'] as List?) ?? const [])
            .map((e) {
              final m = e as List;
              return (m[0] as String, m[1] as String, m[2] as String);
            })
            .toList(),
      );
    }
    return ForgetObserverText(
      caseName: j['case_name'] as String? ?? '',
      nodeId: j['node_id'] as String?,
      passed: j['passed'] as bool? ?? false,
      llmAvailable: j['llm_available'] as bool? ?? false,
      original: j['original'] as String?,
      masked: j['masked'] as String?,
      maskRatio: (j['mask_ratio'] as num?)?.toDouble(),
      llmReply: j['llm_reply'] as String?,
      metrics: ((j['metrics'] as List?) ?? const [])
          .map((e) {
            final m = e as List;
            return (m[0] as String, m[1] as String, m[2] as String);
          })
          .toList(),
      detailLines: (j['detail_lines'] as List?)?.cast<String>() ?? const [],
    );
  }
}

class ForgetObserverNodes extends ForgetObserverCase {
  @override
  final String caseName;
  @override
  final bool passed;
  final bool llmAvailable;
  final int nodeCount;
  final int edgeCount;
  final int llmRevised;
  final int effectiveRevised;
  final List<(String, int)> actionHistogram;
  final double avgMissingDegree;
  final double maxMissingDegree;
  final double avgMaskedRatio;
  final double avgEdgeIntensity;
  /// 用例代表的时间跨度（low=8h / medium=24h / high=72h）
  final double? hours;
  final List<NodeForgetStat> nodes;
  /// 逐节点时间步长序列：遗忘以节点为单位，节点内容按时间步变化
  final List<NodeSeries> nodeSeries;
  /// 理想艾宾浩斯曲线采样（x=小时, y=缺失度），与实测叠加对比
  final List<(double, double)> idealPoints;
  /// 该用例的指标（激发测试按 case 聚合三时机对比时使用）
  final List<(String, String, String)> metrics;

  ForgetObserverNodes({
    required this.caseName,
    required this.passed,
    required this.llmAvailable,
    required this.nodeCount,
    required this.edgeCount,
    required this.llmRevised,
    required this.effectiveRevised,
    required this.actionHistogram,
    required this.avgMissingDegree,
    required this.maxMissingDegree,
    required this.avgMaskedRatio,
    required this.avgEdgeIntensity,
    this.hours,
    required this.nodes,
    this.nodeSeries = const [],
    this.idealPoints = const [],
    this.metrics = const [],
  });
}

class ForgetObserverText extends ForgetObserverCase {
  @override
  final String caseName;
  /// 源记忆节点 id（mask/revise 按此以节点为单位展示）
  final String? nodeId;
  @override
  final bool passed;
  final bool llmAvailable;
  /// 原文（原文对照展示）
  final String? original;
  /// mask：遮罩结果文本；revise：遮罩输入
  final String? masked;
  /// 遮罩率（mask 模式）
  final double? maskRatio;
  final String? llmReply;
  final List<(String, String, String)> metrics;
  final List<String> detailLines;

  ForgetObserverText({
    required this.caseName,
    this.nodeId,
    required this.passed,
    required this.llmAvailable,
    this.original,
    this.masked,
    this.maskRatio,
    this.llmReply,
    required this.metrics,
    required this.detailLines,
  });
}

class ForgetReport {
  final String mode;
  final String datasetName;
  final String datasetPath;
  final int total;
  final int passed;
  final int failed;
  final double elapsedSecs;
  final List<MetricEntry> metrics;
  final String detailHeader;
  final List<DetailRow> detailRows;
  final List<ForgetObserverCase> cases;

  ForgetReport({
    required this.mode,
    required this.datasetName,
    required this.datasetPath,
    required this.total,
    required this.passed,
    required this.failed,
    required this.elapsedSecs,
    required this.metrics,
    required this.detailHeader,
    required this.detailRows,
    required this.cases,
  });

  factory ForgetReport.fromJson(Map<String, dynamic> j) => ForgetReport(
        mode: j['mode'] as String? ?? '',
        datasetName: j['dataset_name'] as String? ?? '',
        datasetPath: j['dataset_path'] as String? ?? '',
        total: j['total'] as int? ?? 0,
        passed: j['passed'] as int? ?? 0,
        failed: j['failed'] as int? ?? 0,
        elapsedSecs: (j['elapsed_secs'] as num?)?.toDouble() ?? 0,
        metrics: (j['metrics'] as List?)
                ?.map((e) => MetricEntry.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        detailHeader: j['detail_header'] as String? ?? '',
        detailRows: (j['detail_rows'] as List?)
                ?.map((e) => DetailRow.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
        cases: (j['cases'] as List?)
                ?.map((e) => ForgetObserverCase.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

// ── 检视数据集（结构化条目） ──

class InspectLink {
  final String fromId;
  final String toId;
  final String linkTypeDesc;
  final double intensity;
  final bool isOutgoing;
  /// 邻居节点在条目列表中的索引（点击链接可跳转）
  final int targetIdx;

  InspectLink({
    required this.fromId,
    required this.toId,
    required this.linkTypeDesc,
    required this.intensity,
    required this.isOutgoing,
    required this.targetIdx,
  });

  factory InspectLink.fromJson(Map<String, dynamic> j) => InspectLink(
        fromId: j['from_id'] as String? ?? '',
        toId: j['to_id'] as String? ?? '',
        linkTypeDesc: j['link_type_desc'] as String? ?? '',
        intensity: (j['intensity'] as num?)?.toDouble() ?? 0,
        isOutgoing: j['is_outgoing'] as bool? ?? false,
        targetIdx: j['target_idx'] as int? ?? 0,
      );
}

class InspectEntryItem {
  final String id;
  final String summary;
  final List<String> previewLines;
  final List<String> detailLines;
  final List<InspectLink> links;

  InspectEntryItem({
    required this.id,
    required this.summary,
    required this.previewLines,
    required this.detailLines,
    required this.links,
  });

  factory InspectEntryItem.fromJson(Map<String, dynamic> j) => InspectEntryItem(
        id: j['id'] as String? ?? '',
        summary: j['summary'] as String? ?? '',
        previewLines: (j['preview_lines'] as List?)?.cast<String>() ?? const [],
        detailLines: (j['detail_lines'] as List?)?.cast<String>() ?? const [],
        links: (j['links'] as List?)
                ?.map((e) => InspectLink.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

class InspectEntries {
  final String fileType; // graph | question
  final String filePath;
  final List<String> stats;
  final List<InspectEntryItem> entries;

  InspectEntries({
    required this.fileType,
    required this.filePath,
    required this.stats,
    required this.entries,
  });

  factory InspectEntries.fromJson(Map<String, dynamic> j) => InspectEntries(
        fileType: j['file_type'] as String? ?? 'json',
        filePath: j['file_path'] as String? ?? '',
        stats: (j['stats'] as List?)?.cast<String>() ?? const [],
        entries: (j['entries'] as List?)
                ?.map((e) => InspectEntryItem.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );
}

// ── 角色扮演（playtest）──

class PlaytestStartResult {
  final bool ok;
  final String characterName;
  final String? error;

  PlaytestStartResult({
    required this.ok,
    required this.characterName,
    this.error,
  });

  factory PlaytestStartResult.fromJson(Map<String, dynamic> j) => PlaytestStartResult(
        ok: j['ok'] as bool? ?? false,
        characterName: j['character_name'] as String? ?? '',
        error: j['error'] as String?,
      );
}

class PlayTraceNode {
  final String name;
  final String stage;
  final double score;
  final String content;

  PlayTraceNode({
    required this.name,
    required this.stage,
    required this.score,
    required this.content,
  });

  factory PlayTraceNode.fromJson(Map<String, dynamic> j) => PlayTraceNode(
        name: j['name'] as String? ?? '',
        stage: j['stage'] as String? ?? '',
        score: (j['score'] as num?)?.toDouble() ?? 0,
        content: j['content'] as String? ?? '',
      );
}

class PlayPerQuery {
  final bool dropped;
  final String preview;
  final int sim;
  final int ppr;
  final int action;
  final int elapsedMs;

  PlayPerQuery({
    required this.dropped,
    required this.preview,
    required this.sim,
    required this.ppr,
    required this.action,
    required this.elapsedMs,
  });

  factory PlayPerQuery.fromJson(Map<String, dynamic> j) => PlayPerQuery(
        dropped: j['dropped'] as bool? ?? false,
        preview: j['preview'] as String? ?? '',
        sim: j['sim'] as int? ?? 0,
        ppr: j['ppr'] as int? ?? 0,
        action: j['action'] as int? ?? 0,
        elapsedMs: j['elapsed_ms'] as int? ?? 0,
      );
}

class PlayTrace {
  final String mode;
  final int totalElapsedMs;
  final List<PlayTraceNode> merged;
  final List<PlayTraceNode> actions;
  final List<PlayTraceNode> speech;
  final List<PlayTraceNode> think;
  final List<PlayPerQuery> perQuery;

  PlayTrace({
    required this.mode,
    required this.totalElapsedMs,
    required this.merged,
    required this.actions,
    required this.speech,
    required this.think,
    required this.perQuery,
  });

  factory PlayTrace.fromJson(Map<String, dynamic> j) => PlayTrace(
        mode: j['mode'] as String? ?? '',
        totalElapsedMs: j['total_elapsed_ms'] as int? ?? 0,
        merged: _nodes(j['merged']),
        actions: _nodes(j['actions']),
        speech: _nodes(j['speech']),
        think: _nodes(j['think']),
        perQuery: (j['per_query'] as List?)
                ?.map((e) => PlayPerQuery.fromJson(e as Map<String, dynamic>))
                .toList() ??
            [],
      );

  static List<PlayTraceNode> _nodes(dynamic raw) => (raw as List?)
          ?.map((e) => PlayTraceNode.fromJson(e as Map<String, dynamic>))
          .toList() ??
      [];
}

class PlayRun {
  final String? response;
  final PlayTrace? trace;

  PlayRun({this.response, this.trace});

  factory PlayRun.fromJson(Map<String, dynamic> j) => PlayRun(
        response: j['response'] as String?,
        trace: j['trace'] == null ? null : PlayTrace.fromJson(j['trace'] as Map<String, dynamic>),
      );
}

class PlayTurn {
  final int index;
  final String userMessage;
  final String? error;
  final String generatedQueriesJson;
  final String? thinkContent;
  final PlayRun? embedding;
  final PlayRun? full;

  PlayTurn({
    required this.index,
    required this.userMessage,
    this.error,
    required this.generatedQueriesJson,
    this.thinkContent,
    this.embedding,
    this.full,
  });

  factory PlayTurn.fromJson(Map<String, dynamic> j) => PlayTurn(
        index: j['index'] as int? ?? 0,
        userMessage: j['user_message'] as String? ?? '',
        error: j['error'] as String?,
        generatedQueriesJson: j['generated_queries_json'] as String? ?? '',
        thinkContent: j['query_think_content'] as String?,
        embedding: j['embedding'] == null
            ? null
            : PlayRun.fromJson(j['embedding'] as Map<String, dynamic>),
        full: j['full'] == null
            ? null
            : PlayRun.fromJson(j['full'] as Map<String, dynamic>),
      );
}

// ── 模型来源（llama-server） ──

/// 模型可用性状态：所有需要模型的地方共用同一套来源决策——
/// 1. 复用运行中的 llama-server；2. 自动拉起本地缓存模型；3. 报错或降级。
class ModelStatus {
  final bool available;
  /// running（复用运行中服务）| spawned（将自动拉起本地模型）| unavailable
  final String source;
  final String? url;
  final String? modelPath;
  final String? reason;

  ModelStatus({
    required this.available,
    required this.source,
    this.url,
    this.modelPath,
    this.reason,
  });

  factory ModelStatus.fromJson(Map<String, dynamic> j) => ModelStatus(
        available: j['available'] as bool? ?? false,
        source: j['source'] as String? ?? 'unavailable',
        url: j['url'] as String?,
        modelPath: j['model_path'] as String?,
        reason: j['reason'] as String?,
      );
}
