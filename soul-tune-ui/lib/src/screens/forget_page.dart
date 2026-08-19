import 'dart:async';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/metric_panel.dart';
import '../widgets/mini_bar_chart.dart';
import '../widgets/model_status_banner.dart';
import '../widgets/results_rail.dart';

/// 遗忘测试配置页：模式（mask/revise/pipeline）+ 图数据集路径。
class ForgetConfigPage extends StatefulWidget {
  const ForgetConfigPage({super.key});

  @override
  State<ForgetConfigPage> createState() => _ForgetConfigPageState();
}

class _ForgetConfigPageState extends State<ForgetConfigPage> {
  String _mode = 'pipeline';
  final _pathCtrl = TextEditingController();

  @override
  void dispose() {
    _pathCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      dialogTitle: '选择角色图（graph.json 或图目录）',
    );
    if (result != null && result.files.single.path != null) {
      setState(() => _pathCtrl.text = result.files.single.path!);
    }
  }

  bool get _canStart => _pathCtrl.text.trim().isNotEmpty;

  void _start() {
    if (!_canStart) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ForgetRunPage(mode: _mode, dataset: _pathCtrl.text.trim()),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final modeDesc = switch (_mode) {
      'mask' => '只验证遮罩模块（纯算法、确定性，无需 LLM）',
      'revise' => '只验证遮罩补全（LLM 来源见下方状态）',
      _ => '全管线：衰减 → 遮罩 → LLM 补全 → 边衰减（LLM 可用时逐节点修订）',
    };
    return Scaffold(
      appBar: AppBar(title: const Text('遗忘测试')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              const ModelStatusBanner(),
              const SizedBox(height: 20),
              Text('测试模式', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'mask', label: Text('遮罩 mask')),
                  ButtonSegment(value: 'revise', label: Text('遮罩补全 revise')),
                  ButtonSegment(value: 'pipeline', label: Text('全管线 pipeline')),
                ],
                selected: {_mode},
                onSelectionChanged: (s) => setState(() => _mode = s.first),
              ),
              const SizedBox(height: 8),
              Text(modeDesc, style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
              const SizedBox(height: 24),
              Text('角色图数据集', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _pathCtrl,
                      decoration: const InputDecoration(
                        labelText: '图路径',
                        hintText: 'fixtures 下的 graph.json 或图目录',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _pickFile,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('选择…'),
                  ),
                ],
              ),
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: _canStart ? _start : null,
                icon: const Icon(Icons.delete_outline),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('开始测试', style: TextStyle(fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 遗忘运行页：进度流。
class ForgetRunPage extends StatefulWidget {
  final String mode;
  final String dataset;
  const ForgetRunPage({super.key, required this.mode, required this.dataset});

  @override
  State<ForgetRunPage> createState() => _ForgetRunPageState();
}

class _ForgetRunPageState extends State<ForgetRunPage> {
  StreamSubscription<ForgetEvent>? _sub;
  String? _loadingMsg;
  int _done = 0, _total = 0, _passed = 0, _failed = 0;
  int _elapsedMs = 0;
  String _caseName = '';
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _start();
  }

  void _start() {
    resetCancel();
    _sub = runForget(mode: widget.mode, dataset: widget.dataset).listen((e) {
      if (!mounted) return;
      setState(() {
        switch (e) {
          case ForgetLoading(:final message):
            _loading = true;
            _loadingMsg = message;
          case ForgetProgress(:final done, :final total, :final passed, :final failed,
              :final elapsedMs, :final caseName):
            _loading = false;
            _done = done;
            _total = total;
            _passed = passed;
            _failed = failed;
            _elapsedMs = elapsedMs;
            _caseName = caseName;
          case ForgetDone(:final report):
            Navigator.of(context).pushReplacement(
              MaterialPageRoute(builder: (_) => ForgetResultsPage(report: report)),
            );
          case ForgetError(:final message):
            _showError(message);
          case ForgetCancelled():
            Navigator.of(context).pop();
        }
      });
    }, onError: (Object e) => _showError('桥接错误: $e'));
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message), backgroundColor: AppColors.fail));
    Navigator.of(context).pop();
  }

  void _cancel() {
    resetCancel();
    Navigator.of(context).pop();
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ratio = _total == 0 ? 0.0 : _done / _total;
    return Scaffold(
      appBar: AppBar(title: Text('遗忘测试 · ${widget.mode}')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (_loading)
                  Column(
                    children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(_loadingMsg ?? '正在加载…',
                          style: const TextStyle(color: AppColors.subtle)),
                    ],
                  )
                else ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: ratio,
                      minHeight: 14,
                      backgroundColor: Colors.grey.shade800,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text('$_done / $_total  (${(ratio * 100).toStringAsFixed(0)}%)',
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 18)),
                  const SizedBox(height: 16),
                  Text('通过 $_passed · 失败 $_failed · 耗时 ${(_elapsedMs / 1000).toStringAsFixed(1)}s',
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontFamily: 'monospace', color: AppColors.subtle)),
                  const SizedBox(height: 12),
                  Text('当前: $_caseName',
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: AppColors.subtle),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 24),
                  OutlinedButton.icon(
                    onPressed: _cancel,
                    icon: const Icon(Icons.stop_circle_outlined),
                    style: OutlinedButton.styleFrom(foregroundColor: AppColors.fail),
                    label: const Text('取消'),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ========================================================================
// 遗忘结果页
// ========================================================================

/// 曲线上的一个数据点（三模式通用）。
///
/// 遗忘算法**以节点为单位**，对节点内容按时间步长变化；不同模式的时间步维度：
/// - pipeline：x = 累计小时（8/24/48/72h），y = 缺失度，理想对照 = 艾宾浩斯曲线；
/// - mask：x = 缺失度梯度（0.0→1.0，模拟"时间越长遮罩越多"），y = 遮罩率，理想对照 = y=x 对角线；
/// - revise：无时间轴，按节点展示 原文/遮罩输入/LLM 回复 对照。
class _CurvePoint {
  final double x;
  final double y;
  /// 数据点标签（如 "24h" / "md0.50"）
  final String label;
  final String action;
  final bool effective;
  final String? original; // 该点的原文（mask 模式每点有自己的原文）
  final String? maskedText;
  final String? llmReply;

  _CurvePoint({
    required this.x,
    required this.y,
    required this.label,
    required this.action,
    this.effective = false,
    this.original,
    this.maskedText,
    this.llmReply,
  });
}

/// 单个"记忆节点"的趋势曲线（三模式通用）。
class _NodeCurve {
  final String id;
  final String typeName;
  final String original; // 节点原文（遗忘前）
  final List<_CurvePoint> points; // 按 x 升序
  final List<(double, double)> ideal; // 理想曲线采样（pipeline: 艾宾浩斯；mask: y=x）
  final String xLabel; // 横轴名（时间步长（小时）/ 缺失度）
  final String yLabel; // 纵轴名（缺失度 / 遮罩率）

  _NodeCurve({
    required this.id,
    required this.typeName,
    required this.original,
    required this.points,
    required this.ideal,
    required this.xLabel,
    required this.yLabel,
  });

  _CurvePoint get last => points.last;
  double get firstY => points.first.y;
  double get lastY => points.last.y;
  double get firstX => points.first.x;
  double get lastX => points.last.x;

  /// 是否参与了遗忘（任一数据点触发过遮罩/修订/激活；NoAction 仅更新缺失度）
  bool get participated => points.any((p) => p.action != 'NoAction');

  String get actionSummary {
    final counts = <String, int>{};
    for (final p in points) {
      counts[p.action] = (counts[p.action] ?? 0) + 1;
    }
    return counts.entries.map((e) => '${e.key}×${e.value}').join(' ');
  }
}

/// 遗忘结果页：汇总（图表+关键指标） / 观测（以节点为单位看时间步演变）。
class ForgetResultsPage extends StatefulWidget {
  final ForgetReport report;
  const ForgetResultsPage({super.key, required this.report});

  @override
  State<ForgetResultsPage> createState() => _ForgetResultsPageState();
}

class _ForgetResultsPageState extends State<ForgetResultsPage> {
  int _tab = 0; // 0=汇总 1=观测
  String? _selectedNodeId;
  int _selectedStep = -1; // 选中的数据点（-1=自动选时间步最小的第一个点）
  bool _showAllNodes = false; // 观测列表：仅显示参与遗忘的节点 / 显示全部

  ForgetReport get report => widget.report;

  /// pipeline：合并全部用例的逐节点时间步序列（按 id 聚合、按小时排序去重）。
  /// 每个节点**一个条目**，内部一条折线展示趋势（x=时间步长，y=缺失度）。
  List<_NodeCurve> get _curves {
    final byId = <String, List<({NodeStepStat stat, int sourceLen})>>{};
    final meta = <String, (String, String)>{};
    var bestIdeal = <(double, double)>[];
    var foundSeries = false;

    for (final c in report.cases) {
      if (c is! ForgetObserverNodes) continue;
      if (c.idealPoints.length > bestIdeal.length) bestIdeal = c.idealPoints;
      for (final ns in c.nodeSeries) {
        foundSeries = true;
        meta[ns.id] = (ns.typeName, ns.original);
        final list = byId.putIfAbsent(ns.id, () => []);
        for (final s in ns.steps) {
          final idx = list.indexWhere((e) => e.stat.hours == s.hours);
          if (idx < 0) {
            list.add((stat: s, sourceLen: ns.steps.length));
          } else if (list[idx].sourceLen < ns.steps.length) {
            // 同小时多来源时，取轨迹更完整（步数更多）的来源
            list[idx] = (stat: s, sourceLen: ns.steps.length);
          }
        }
      }
    }

    if (!foundSeries) {
      // 兼容旧数据：无 node_series 时退化为跨用例单点聚合
      return _legacyCurves(bestIdeal);
    }

    final curves = <_NodeCurve>[];
    for (final entry in byId.entries) {
      final steps = entry.value.map((e) => e.stat).toList()
        ..sort((a, b) => a.hours.compareTo(b.hours));
      if (steps.isEmpty) continue;
      final (typeName, original) = meta[entry.key] ?? ('?', '');
      curves.add(_NodeCurve(
        id: entry.key,
        typeName: typeName,
        original: original,
        points: [
          for (final s in steps)
            _CurvePoint(
              x: s.hours.toDouble(),
              y: s.md,
              label: '${s.hours}h',
              action: s.action,
              effective: s.effective,
              maskedText: s.maskedText,
              llmReply: s.llmReply,
            ),
        ],
        ideal: bestIdeal,
        xLabel: '时间步长（小时）',
        yLabel: '缺失度',
      ));
    }
    curves.sort((a, b) => a.id.compareTo(b.id));
    return curves;
  }

  /// 旧数据兼容：按节点 id 聚合各用例的 nodes 列表，x=用例 hours。
  List<_NodeCurve> _legacyCurves(List<(double, double)> bestIdeal) {
    final byId = <String, List<(double, NodeForgetStat)>>{};
    final meta = <String, (String, String)>{};
    for (final c in report.cases) {
      if (c is! ForgetObserverNodes) continue;
      final h = c.hours;
      for (final n in c.nodes) {
        meta[n.id] = (n.typeName, n.original);
        if (h != null) byId.putIfAbsent(n.id, () => []).add((h, n));
      }
    }
    final curves = <_NodeCurve>[];
    for (final entry in byId.entries) {
      final pts = entry.value..sort((a, b) => a.$1.compareTo(b.$1));
      final (typeName, original) = meta[entry.key] ?? ('?', '');
      curves.add(_NodeCurve(
        id: entry.key,
        typeName: typeName,
        original: original,
        points: [
          for (final (h, n) in pts)
            _CurvePoint(
              x: h,
              y: n.mdAfter,
              label: '${h.round()}h',
              action: n.action,
              effective: n.effective,
              maskedText: n.maskedText,
              llmReply: n.llmReply,
            ),
        ],
        ideal: bestIdeal,
        xLabel: '时间步长（小时）',
        yLabel: '缺失度',
      ));
    }
    curves.sort((a, b) => a.id.compareTo(b.id));
    return curves;
  }

  /// mask：把遮罩用例按**记忆节点**（node_id）分组——
  /// 时间越长 → 缺失度梯度越高 → 遮罩越多。每个节点一个条目，
  /// x=缺失度梯度（0.0→1.0），y=遮罩率，理想对照 = y=x 对角线。
  List<_NodeCurve> get _maskCurves {
    final textCases = report.cases.whereType<ForgetObserverText>().toList();
    final byNode = <String, List<ForgetObserverText>>{};
    for (final tc in textCases) {
      // 仅保留带缺失度梯度的用例（排除 determinism 等非梯度用例）
      if (!tc.caseName.contains('-md')) continue;
      final key = tc.nodeId ?? _maskTextKey(tc.caseName);
      byNode.putIfAbsent(key, () => []).add(tc);
    }
    // 理想对角线：遮罩率应 ≈ 缺失度
    final ideal = <(double, double)>[
      for (var v = 0.0; v <= 1.0 + 1e-9; v += 0.05) (v, v),
    ];
    final curves = <_NodeCurve>[];
    for (final entry in byNode.entries) {
      final list = entry.value
        ..sort((a, b) => _maskMd(a.caseName).compareTo(_maskMd(b.caseName)));
      if (list.isEmpty) continue;
      final first = list.first;
      curves.add(_NodeCurve(
        id: entry.key,
        typeName: '遮罩节点',
        original: first.original ?? '',
        points: [
          for (final tc in list)
            _CurvePoint(
              x: _maskMd(tc.caseName),
              y: tc.maskRatio ?? 0,
              label: 'md${_maskMd(tc.caseName).toStringAsFixed(2)}',
              action: 'MaskOnly',
              original: tc.original,
              maskedText: tc.masked,
            ),
        ],
        ideal: ideal,
        xLabel: '缺失度梯度（模拟时间）',
        yLabel: '遮罩率',
      ));
    }
    curves.sort((a, b) => a.id.compareTo(b.id));
    return curves;
  }

  /// 从遮罩用例名（如 "medium-md0.50"）提取源文本 key（去掉 -md 后缀）。
  static String _maskTextKey(String caseName) {
    final i = caseName.indexOf('-md');
    return i < 0 ? caseName : caseName.substring(0, i);
  }

  /// 从用例名提取缺失度梯度（如 "medium-md0.50" → 0.5）。
  static double _maskMd(String caseName) {
    final m = RegExp(r'md(\d+(?:\.\d+)?)').firstMatch(caseName);
    return m == null ? 0.0 : double.parse(m.group(1)!);
  }

  /// x 轴数值的展示格式（小时去小数，其余保留两位）。
  static String _fmtX(double x) => x == x.roundToDouble() ? '${x.round()}' : x.toStringAsFixed(2);

  @override
  Widget build(BuildContext context) {
    final r = report;
    return Scaffold(
      appBar: AppBar(title: Text('遗忘结果 · ${r.mode} · ${r.datasetName}')),
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ResultsRail(
            passRate: r.total == 0 ? 0 : r.passed / r.total,
            passed: r.passed,
            failed: r.failed,
            elapsedSecs: r.elapsedSecs,
            tab: _tab,
            tabs: const [
              (label: '汇总', icon: Icons.dashboard_outlined),
              (label: '观测', icon: Icons.visibility_outlined),
            ],
            onTab: (i) => setState(() => _tab = i),
          ),
          Expanded(child: _tab == 0 ? _buildSummary() : _buildObserver()),
        ],
      ),
    );
  }

  Widget _buildSummary() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('关键指标', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        MetricPanel(metrics: report.metrics),
        const SizedBox(height: 8),
        Text('逐用例 通过/失败', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        MiniBarChart(passFlags: report.cases.map((c) => c.passed).toList()),
      ],
    );
  }

  Widget _buildObserver() {
    // 三种模式统一"以记忆节点为单位"：
    // - pipeline：节点 × 时间步长（小时）曲线，理想 = 艾宾浩斯
    // - mask：文本（源节点）× 缺失度梯度（模拟时间）曲线，理想 = y=x 对角线
    // - revise：按节点展示 原文/遮罩输入/LLM 回复 对照
    switch (report.mode) {
      case 'mask':
        final curves = _maskCurves;
        if (curves.isNotEmpty) return _buildNodeCentric(curves);
      case 'revise':
        final textCases = report.cases.whereType<ForgetObserverText>().toList();
        if (textCases.isNotEmpty) return _buildReviseObserver(textCases);
      default:
        final curves = _curves;
        if (curves.isNotEmpty) return _buildNodeCentric(curves);
    }
    final hasNodeCases =
        report.cases.any((c) => c is ForgetObserverNodes && c.nodeCount > 0);
    return Center(
      child: Text(
        hasNodeCases ? '该模式为整图级观测（激活/增量一致性），无逐节点时间步数据' : '（无观测数据）',
        style: const TextStyle(color: AppColors.subtle),
      ),
    );
  }

  // ── revise：左节点列表 + 右 原文/遮罩输入/LLM 回复 对照 ──
  Widget _buildReviseObserver(List<ForgetObserverText> textCases) {
    var idx = 0;
    final parsed = int.tryParse(_selectedNodeId ?? '');
    if (parsed != null && parsed >= 0 && parsed < textCases.length) idx = parsed;
    final c = textCases[idx];
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          width: 300,
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerLow,
            border: Border(
              right: BorderSide(
                  color: Theme.of(context).colorScheme.outlineVariant, width: 0.5),
            ),
          ),
          child: ListView.builder(
            itemCount: textCases.length,
            itemBuilder: (context, i) {
              final tc = textCases[i];
              final sub = tc.maskRatio != null
                  ? '遮罩率 ${(tc.maskRatio! * 100).toStringAsFixed(0)}%'
                  : (tc.llmReply != null
                      ? '回复 ${tc.llmReply!.runes.length} 字'
                      : '');
              return ListTile(
                dense: true,
                selected: i == idx,
                selectedTileColor:
                    Theme.of(context).colorScheme.primary.withValues(alpha: 0.12),
                leading: Icon(
                  tc.passed ? Icons.check_circle_outline : Icons.cancel_outlined,
                  size: 16,
                  color: tc.passed ? AppColors.pass : AppColors.fail,
                ),
                title: Text(tc.caseName,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 12, fontFamily: 'monospace')),
                subtitle: Text(sub, style: const TextStyle(fontSize: 10)),
                onTap: () => setState(() => _selectedNodeId = '$i'),
              );
            },
          ),
        ),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (!c.llmAvailable && c.masked != null)
                const Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: Text('LLM 不可用（未检测到 llama-server / 本地模型，已降级）',
                      style: TextStyle(color: AppColors.warn, fontSize: 12)),
                ),
              if (c.metrics.isNotEmpty)
                Card(
                  elevation: 0,
                  color: Theme.of(context).colorScheme.surfaceContainerHigh,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Wrap(
                      spacing: 16,
                      runSpacing: 6,
                      children: [
                        for (final (_, l, v) in c.metrics)
                          Text('$l: $v',
                              style:
                                  const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                      ],
                    ),
                  ),
                ),
              if (c.original != null) ...[
                const SizedBox(height: 10),
                _CompareBlock(
                  title: '图节点原文',
                  text: c.original!,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ],
              if (c.masked != null) ...[
                const SizedBox(height: 10),
                _CompareBlock(
                  title: c.llmReply != null ? '遮罩输入' : '遮罩结果',
                  text: c.masked!,
                  color: AppColors.warn,
                  ratio: c.maskRatio,
                ),
              ],
              if (c.llmReply != null) ...[
                const SizedBox(height: 10),
                _CompareBlock(
                  title: 'LLM 原始回复',
                  text: c.llmReply!,
                  color: AppColors.pass,
                ),
              ],
              if (c.detailLines.isNotEmpty) ...[
                const SizedBox(height: 10),
                Card(
                  elevation: 0,
                  color: Theme.of(context).colorScheme.surfaceContainerHigh,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        for (final l in c.detailLines)
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 1),
                            child: Text(l,
                                style: const TextStyle(
                                    fontFamily: 'monospace', fontSize: 11, height: 1.4)),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  // ── 节点为中心：左节点列表 + 右"节点 × 时间步"曲线与数据点展开（pipeline/mask 通用）──
  Widget _buildNodeCentric(List<_NodeCurve> curves) {
    // 默认只显示"参与遗忘"的节点（触发过遮罩/修订/激活）；可切换显示全部
    final visible = _showAllNodes
        ? curves
        : (curves.where((c) => c.participated).toList().isNotEmpty
            ? curves.where((c) => c.participated).toList()
            : curves);
    var selected = visible.first;
    for (final a in visible) {
      if (a.id == _selectedNodeId) {
        selected = a;
        break;
      }
    }
    // 切换节点后自动选中**时间步最小**（第一个）的数据点
    if (_selectedStep < 0 || _selectedStep >= selected.points.length) {
      _selectedStep = 0;
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 左：节点列表
        Container(
          width: 300,
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerLow,
            border: Border(
              right: BorderSide(color: Theme.of(context).colorScheme.outlineVariant, width: 0.5),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 过滤开关
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 6, 8, 2),
                child: Row(
                  children: [
                    Text('节点 ${visible.length}/${curves.length}',
                        style: const TextStyle(
                            fontFamily: 'monospace', fontSize: 11, color: AppColors.subtle)),
                    const Spacer(),
                    FilterChip(
                      label: const Text('显示全部节点', style: TextStyle(fontSize: 11)),
                      visualDensity: VisualDensity.compact,
                      selected: _showAllNodes,
                      onSelected: (v) => setState(() => _showAllNodes = v),
                    ),
                  ],
                ),
              ),
              Expanded(
                child: ListView.builder(
                  itemCount: visible.length,
                  itemBuilder: (context, i) {
                    final a = visible[i];
                    final fg = switch (a.last.action) {
                      'Revised' => a.last.effective ? AppColors.pass : AppColors.fail,
                      'MaskOnly' => AppColors.warn,
                      _ => AppColors.subtle,
                    };
                    final short = a.id.length > 12 ? a.id.substring(0, 12) : a.id;
                    return ListTile(
                      dense: true,
                      selected: a.id == selected.id,
                      selectedTileColor:
                          Theme.of(context).colorScheme.primary.withValues(alpha: 0.12),
                      leading: Icon(
                        switch (a.last.action) {
                          'Revised' => Icons.healing_outlined,
                          'MaskOnly' => Icons.visibility_off_outlined,
                          _ => Icons.check_circle_outline,
                        },
                        size: 16,
                        color: fg,
                      ),
                      title: Text('${a.typeName} [$short]',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontSize: 12, fontFamily: 'monospace')),
                      subtitle: Text(
                        '${_fmtX(a.firstX)}→${_fmtX(a.lastX)} · ${a.yLabel} ${a.firstY.toStringAsFixed(2)}→${a.lastY.toStringAsFixed(2)} · ${a.points.length}点',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 10),
                      ),
                      onTap: () => setState(() {
                        _selectedNodeId = a.id;
                        _selectedStep = -1; // 触发自动选时间步最小的第一个点
                      }),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        // 右：节点 × 时间步 曲线 + 数据点展开
        Expanded(
          child: _NodeEvolutionView(
            curve: selected,
            selectedStep: _selectedStep,
            onSelectStep: (i) => setState(() => _selectedStep = i),
          ),
        ),
      ],
    );
  }
}

/// 节点演变视图：x（时间步长/缺失度）× y（指标）曲线 + 理想曲线叠加 +
/// 可点击数据点 → 展开该数据点的原始输出与节点原文（pipeline/mask 通用）。
class _NodeEvolutionView extends StatelessWidget {
  final _NodeCurve curve;
  final int selectedStep;
  final ValueChanged<int> onSelectStep;
  const _NodeEvolutionView({
    required this.curve,
    required this.selectedStep,
    required this.onSelectStep,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final points = curve.points;
    final pt = points[selectedStep.clamp(0, points.length - 1)];
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Row(
          children: [
            Expanded(
              child: Text('${curve.typeName} [${curve.id}]',
                  style: Theme.of(context).textTheme.titleMedium),
            ),
            Text(curve.actionSummary,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 11, color: AppColors.subtle)),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          '${curve.xLabel} ${_fmtX(curve.firstX)} → ${_fmtX(curve.lastX)} · ${curve.yLabel} '
          '${curve.firstY.toStringAsFixed(3)} → ${curve.lastY.toStringAsFixed(3)} · 共 ${points.length} 个数据点',
          style: const TextStyle(fontFamily: 'monospace', fontSize: 12, color: AppColors.subtle),
        ),
        const SizedBox(height: 12),
        // ── 曲线卡片 ──
        Card(
          elevation: 0,
          color: scheme.surfaceContainerHigh,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        '${curve.id} · 实测 vs 理想曲线',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                      ),
                    ),
                    const _LegendDot(color: AppColors.running, label: '实测'),
                    const SizedBox(width: 12),
                    const _LegendDot(color: AppColors.warn, label: '理想'),
                    const SizedBox(width: 12),
                    _LegendDot(color: scheme.primary, label: '点击数据点展开'),
                  ],
                ),
                const SizedBox(height: 10),
                SizedBox(
                  height: 300,
                  child: _TrendChart(
                    curve: curve,
                    selected: selectedStep,
                    onSelect: onSelectStep,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        // ── 选中数据点的展开详情 ──
        _PointDetailCard(point: pt, curveOriginal: curve.original, yLabel: curve.yLabel),
        const SizedBox(height: 8),
        Text('提示：点击曲线上任意数据点，可展开该点的遮罩结果/LLM 原始输出与图节点原文',
            style: const TextStyle(fontSize: 11, color: AppColors.subtle)),
      ],
    );
  }

  static String _fmtX(double x) =>
      x == x.roundToDouble() ? '${x.round()}' : x.toStringAsFixed(2);
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 11)),
      ],
    );
  }
}

/// 趋势曲线图表：x = 时间步长（小时）或缺失度，y = 指标；
/// 理想曲线虚线叠加（pipeline=艾宾浩斯 / mask=对角线）；每个实测点可点击。
class _TrendChart extends StatelessWidget {
  final _NodeCurve curve;
  final int selected;
  final ValueChanged<int> onSelect;
  const _TrendChart({
    required this.curve,
    required this.selected,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    final points = curve.points;
    if (points.isEmpty) {
      return const Center(child: Text('（无数据）', style: TextStyle(color: AppColors.subtle)));
    }
    final maxX = points.map((p) => p.x).reduce(math.max);
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = Size(constraints.maxWidth, constraints.maxHeight);
        final geo = _ChartGeo(size, maxX: maxX);
        return ClipRect(
          child: Stack(
            children: [
              CustomPaint(
                size: size,
                painter: _TrendPainter(geo: geo, curve: curve, selected: selected),
              ),
              // 实测数据点（可点击）
              for (var i = 0; i < points.length; i++)
                Positioned(
                  left: geo.x(points[i].x) - 9,
                  top: geo.y(points[i].y) - 9,
                  width: 18,
                  height: 18,
                  child: GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onTap: () => onSelect(i),
                    child: Center(
                      child: Container(
                        width: selected == i ? 14 : 10,
                        height: selected == i ? 14 : 10,
                        decoration: BoxDecoration(
                          color: _pointColor(points[i]),
                          shape: BoxShape.circle,
                          border: selected == i
                              ? Border.all(
                                  color: Theme.of(context).colorScheme.primary, width: 2.5)
                              : null,
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  static Color _pointColor(_CurvePoint p) {
    return switch (p.action) {
      'Revised' => p.effective ? AppColors.pass : AppColors.fail,
      'MaskOnly' => AppColors.warn,
      _ => AppColors.subtle,
    };
  }
}

/// 图表几何：把 数据值 映射为像素坐标（与 painter、可点 widget 共用）。
/// x ∈ [0, maxX]，y ∈ [0, 1]。
class _ChartGeo {
  final double left = 46, right = 12, top = 14, bottom = 30;
  final double maxX;
  final Size size;
  _ChartGeo(this.size, {required double maxX}) : maxX = maxX <= 0 ? 1.0 : maxX;

  double get plotW => size.width - left - right;
  double get plotH => size.height - top - bottom;

  double x(double v) => left + v / maxX * plotW;
  double y(double v) => top + (1.0 - v) * plotH; // v ∈ [0,1]
}

/// 背景绘制：网格、轴标签、理想虚线、实测折线、选中点标注。
class _TrendPainter extends CustomPainter {
  final _ChartGeo geo;
  final _NodeCurve curve;
  final int selected;
  _TrendPainter({required this.geo, required this.curve, required this.selected});

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = const Color(0x22FFFFFF)
      ..strokeWidth = 1;

    // ── y 网格 + 标签（0..1）──
    final labelStyle = TextStyle(
        color: const Color(0xAA9E9E9E), fontSize: 9.5, fontFamily: 'monospace');
    for (var v = 0.0; v <= 1.0 + 1e-9; v += 0.25) {
      final y = geo.y(v);
      canvas.drawLine(Offset(geo.left, y), Offset(size.width - geo.right, y), gridPaint);
      final tp = TextPainter(
        text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(2, y - tp.height / 2));
    }

    // ── x 网格 + 标签（pipeline: 每 24h / 6h；mask: 每 0.25）──
    final isMaskLike = geo.maxX <= 1.0 + 1e-9;
    if (isMaskLike) {
      for (var v = 0.0; v <= 1.0 + 1e-9; v += 0.25) {
        final x = geo.x(v);
        canvas.drawLine(Offset(x, geo.top), Offset(x, size.height - geo.bottom), gridPaint);
        final tp = TextPainter(
          text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, Offset(x - tp.width / 2, size.height - geo.bottom + 6));
      }
    } else {
      final tickEvery = geo.maxX <= 24 ? 6.0 : 24.0;
      for (var h = 0.0; h <= geo.maxX + 1e-9; h += tickEvery) {
        final x = geo.x(h);
        canvas.drawLine(Offset(x, geo.top), Offset(x, size.height - geo.bottom), gridPaint);
        final tp = TextPainter(
          text: TextSpan(text: '${h.round()}h', style: labelStyle),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, Offset(x - tp.width / 2, size.height - geo.bottom + 6));
      }
    }

    // 坐标轴标题
    final axisStyle = const TextStyle(
        color: Color(0xAA9E9E9E), fontSize: 10, fontFamily: 'monospace');
    final yTitle = TextPainter(
      text: TextSpan(text: curve.yLabel, style: axisStyle),
      textDirection: TextDirection.ltr,
    )..layout();
    yTitle.paint(canvas, Offset(8, 2));
    final xTitle = TextPainter(
      text: TextSpan(text: curve.xLabel, style: axisStyle),
      textDirection: TextDirection.ltr,
    )..layout();
    xTitle.paint(canvas, Offset(size.width - geo.right - xTitle.width, size.height - geo.bottom + 18));

    // ── 理想曲线（虚线）──
    if (curve.ideal.length >= 2) {
      final dash = Paint()
        ..color = AppColors.warn.withValues(alpha: 0.75)
        ..strokeWidth = 1.6
        ..style = PaintingStyle.stroke;
      final path = Path();
      for (var i = 0; i < curve.ideal.length; i++) {
        final (ix, iy) = curve.ideal[i];
        if (ix > geo.maxX) break;
        final p = Offset(geo.x(ix), geo.y(iy));
        if (i == 0) {
          path.moveTo(p.dx, p.dy);
        } else {
          path.lineTo(p.dx, p.dy);
        }
      }
      _drawDashed(canvas, path, dash);
    }

    // ── 实测折线 ──
    final line = Paint()
      ..color = AppColors.running
      ..strokeWidth = 2.2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    final points = curve.points;
    final path = Path();
    for (var i = 0; i < points.length; i++) {
      final p = Offset(geo.x(points[i].x), geo.y(points[i].y));
      if (i == 0) {
        path.moveTo(p.dx, p.dy);
      } else {
        path.lineTo(p.dx, p.dy);
      }
    }
    canvas.drawPath(path, line);

    // 选中点标注（标签 · y值）
    if (selected >= 0 && selected < points.length) {
      final s = points[selected];
      final p = Offset(geo.x(s.x), geo.y(s.y));
      final halo = Paint()..color = AppColors.running.withValues(alpha: 0.25);
      canvas.drawCircle(p, 12, halo);
      final label = '${s.label} · ${curve.yLabel} ${s.y.toStringAsFixed(3)}';
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
              color: Colors.black87, fontSize: 10, fontWeight: FontWeight.w700, fontFamily: 'monospace'),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final bgRect = Rect.fromLTWH(
        p.dx + 12,
        p.dy - tp.height - 4,
        tp.width + 10,
        tp.height + 6,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
        Paint()..color = AppColors.running,
      );
      tp.paint(canvas, Offset(bgRect.left + 5, bgRect.top + 3));
    }
  }

  void _drawDashed(Canvas canvas, Path path, Paint paint) {
    for (final metric in path.computeMetrics()) {
      var dist = 0.0;
      const dashLen = 7.0, gapLen = 5.0;
      while (dist < metric.length) {
        final end = math.min(dist + dashLen, metric.length);
        canvas.drawPath(metric.extractPath(dist, end), paint);
        dist = end + gapLen;
      }
    }
  }

  @override
  bool shouldRepaint(_TrendPainter old) =>
      old.curve != curve || old.selected != selected;
}

/// 数据点展开详情：该点的遮罩结果/LLM 原始输出 + 图节点原文。
class _PointDetailCard extends StatelessWidget {
  final _CurvePoint point;
  final String curveOriginal; // 节点原文（pipeline 模式）；mask 模式用 point.original
  final String yLabel;
  const _PointDetailCard({
    required this.point,
    required this.curveOriginal,
    required this.yLabel,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final actionColor = switch (point.action) {
      'Revised' => point.effective ? AppColors.pass : AppColors.fail,
      'MaskOnly' => AppColors.warn,
      _ => AppColors.subtle,
    };
    final original = (point.original != null && point.original!.trim().isNotEmpty)
        ? point.original!
        : curveOriginal;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: actionColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(point.label,
                      style: TextStyle(
                          color: actionColor, fontWeight: FontWeight.w700, fontSize: 12)),
                ),
                const SizedBox(width: 10),
                Text(point.action,
                    style: TextStyle(
                        color: actionColor, fontWeight: FontWeight.w700, fontSize: 12)),
                if (point.action == 'Revised') ...[
                  const SizedBox(width: 8),
                  Text(point.effective ? '（有效修订）' : '（无效修订）',
                      style: TextStyle(
                          fontSize: 11,
                          color: point.effective ? AppColors.pass : AppColors.fail)),
                ],
                const Spacer(),
                Text('$yLabel ${point.y.toStringAsFixed(3)}',
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
              ],
            ),
            const SizedBox(height: 10),
            _DetailSection(
              title: '图节点原文',
              text: original,
              color: scheme.primary,
              showEmpty: false,
            ),
            if (point.maskedText != null) ...[
              const SizedBox(height: 10),
              _DetailSection(
                title: point.action == 'MaskOnly' ? '遮罩结果' : '遮罩输入',
                text: point.maskedText!,
                color: AppColors.warn,
              ),
            ],
            if (point.llmReply != null) ...[
              const SizedBox(height: 10),
              _DetailSection(title: 'LLM 原始输出', text: point.llmReply!, color: AppColors.pass),
            ],
            if (point.maskedText == null && point.llmReply == null)
              const Padding(
                padding: EdgeInsets.only(top: 4),
                child: Text('该数据点未触发遮罩/修订（NoAction），仅指标更新。',
                    style: TextStyle(fontSize: 11, color: AppColors.subtle)),
              ),
          ],
        ),
      ),
    );
  }
}

class _DetailSection extends StatelessWidget {
  final String title;
  final String text;
  final Color color;
  final bool showEmpty;
  const _DetailSection({
    required this.title,
    required this.text,
    required this.color,
    this.showEmpty = true,
  });

  @override
  Widget build(BuildContext context) {
    if (!showEmpty && text.trim().isEmpty) {
      return const SizedBox.shrink();
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(width: 3, height: 12, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
            const SizedBox(width: 8),
            Text(title,
                style: TextStyle(color: color, fontWeight: FontWeight.w600, fontSize: 12)),
          ],
        ),
        const SizedBox(height: 5),
        SelectableText(
          text.trim().isEmpty ? '（空）' : text,
          style: const TextStyle(fontSize: 12, height: 1.5),
        ),
      ],
    );
  }
}

/// 原文对照块：标题（带色）+ 可选遮罩率 + 文本。
class _CompareBlock extends StatelessWidget {
  final String title;
  final String text;
  final Color color;
  final double? ratio;
  const _CompareBlock({
    required this.title,
    required this.text,
    required this.color,
    this.ratio,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 3,
                  height: 12,
                  decoration: BoxDecoration(
                      color: color, borderRadius: BorderRadius.circular(2)),
                ),
                const SizedBox(width: 8),
                Text(title,
                    style:
                        TextStyle(color: color, fontWeight: FontWeight.w600, fontSize: 13)),
                if (ratio != null) ...[
                  const SizedBox(width: 10),
                  Text('遮罩率 ${(ratio! * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(
                          fontFamily: 'monospace', fontSize: 12, color: AppColors.subtle)),
                ],
              ],
            ),
            const SizedBox(height: 8),
            SelectableText(text, style: const TextStyle(height: 1.5, fontSize: 13)),
          ],
        ),
      ),
    );
  }
}
