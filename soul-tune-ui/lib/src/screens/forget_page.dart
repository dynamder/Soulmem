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

/// 遗忘模式 → 中文显示名（mask/revise[/full|/sample]/pipeline/excitation）。
String forgetModeLabel(String mode) {
  if (mode.startsWith('revise/sample')) return '遮罩补全·分层抽样';
  if (mode.startsWith('revise/full')) return '遮罩补全·全量';
  return switch (mode) {
    'mask' => '遮罩 mask',
    'revise' => '遮罩补全 revise',
    'pipeline' => '全管线 pipeline',
    'excitation' => '激发测试 excitation',
    _ => mode,
  };
}

/// 遗忘测试配置页：模式（mask/revise/pipeline/excitation）+ 图数据集路径。
class ForgetConfigPage extends StatefulWidget {
  const ForgetConfigPage({super.key});

  @override
  State<ForgetConfigPage> createState() => _ForgetConfigPageState();
}

class _ForgetConfigPageState extends State<ForgetConfigPage> {
  String _mode = 'pipeline';
  // 修订测试：全量 / 分层抽样（固定种子可复现）
  String _reviseScope = 'sample';
  final _seedCtrl = TextEditingController(text: '20260820');
  final _pathCtrl = TextEditingController();

  @override
  void dispose() {
    _seedCtrl.dispose();
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
    // 修订测试：全量 → revise/full；抽样 → revise/sample:<seed>
    var finalMode = _mode;
    if (_mode == 'revise') {
      finalMode = _reviseScope == 'full'
          ? 'revise/full'
          : 'revise/sample:${_seedCtrl.text.trim()}';
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ForgetRunPage(mode: finalMode, dataset: _pathCtrl.text.trim()),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final modeDesc = switch (_mode) {
      'mask' => '只验证遮罩模块（纯算法、确定性，无需 LLM）',
      'revise' => '只验证遮罩补全（LLM 来源见下方状态）',
      'excitation' => '激发测试：配对对照验证"激发 → 遗忘被延缓"（纯效果 E1~E6、确定性、无需 LLM）',
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
                  ButtonSegment(value: 'excitation', label: Text('激发测试 excitation')),
                ],
                selected: {_mode},
                onSelectionChanged: (s) => setState(() => _mode = s.first),
              ),
              const SizedBox(height: 8),
              Text(modeDesc, style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
              // 修订测试：全量 / 分层抽样（种子）
              if (_mode == 'revise') ...[
                const SizedBox(height: 16),
                Text('修订采样', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 8),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(value: 'sample', label: Text('分层抽样（约8个）', style: TextStyle(fontSize: 11))),
                    ButtonSegment(value: 'full', label: Text('全量（全部可遗忘节点）', style: TextStyle(fontSize: 11))),
                  ],
                  selected: {_reviseScope},
                  showSelectedIcon: false,
                  style: const ButtonStyle(
                      visualDensity: VisualDensity.compact,
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap),
                  onSelectionChanged: (s) => setState(() => _reviseScope = s.first),
                ),
                if (_reviseScope == 'sample') ...[
                  const SizedBox(height: 8),
                  TextField(
                    controller: _seedCtrl,
                    keyboardType: TextInputType.number,
                    decoration: const InputDecoration(
                      labelText: '抽样种子',
                      hintText: '固定种子保证抽样可复现',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                ],
              ],
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
      appBar: AppBar(title: Text('遗忘测试 · ${forgetModeLabel(widget.mode)}')),
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
  /// 对照组（未激发）同刻 y 值：激发测试的双曲线对比用（对照虚线），其余模式为 null
  final double? yCtrl;
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
    this.yCtrl,
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
  /// 来源用例（如 excitation-early），用于三时机叠加时区分颜色/图例
  final String sourceCase;

  _NodeCurve({
    required this.id,
    required this.typeName,
    required this.original,
    required this.points,
    required this.ideal,
    required this.xLabel,
    required this.yLabel,
    this.sourceCase = '',
  });

  _CurvePoint get last => points.last;
  double get firstY => points.first.y;
  double get lastY => points.last.y;
  double get firstX => points.first.x;
  double get lastX => points.last.x;

  /// 是否参与了遗忘（任一数据点触发过遮罩/修订/激活；NoAction 仅更新缺失度）
  bool get participated => points.any((p) => p.action != 'NoAction');

  /// 是否携带对照组曲线（激发测试：对照 vs 激发双线）
  bool get hasCtrl => points.any((p) => p.yCtrl != null);

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
  String? _selectedCase; // 激发测试三时机：当前观测的用例（excitation-early/spaced/late）
  bool _overlayTiming = false; // 三时机视图模式：false=分图（每时机一张），true=叠加（同一节点 4 线）

  ForgetReport get report => widget.report;

  /// pipeline：合并（或按指定用例过滤）逐节点时间步序列（按 id 聚合、按小时排序去重）。
  /// `caseName == null` 合并全部用例；激发测试三时机（early/spaced/late）按用例
  /// 分别观测——同一节点在不同时机下的曲线需要能区分（次数制下应重合，语义
  /// 升级后可能分化）。
  List<_NodeCurve> _curvesFor(String? caseName) {
    final byId = <String, List<({NodeStepStat stat, int sourceLen})>>{};
    final meta = <String, (String, String)>{};
    final nodeCase = <String, String>{}; // 节点 id → 来源用例（三时机叠加时区分）
    final excitationIds = <String>{}; // 来自激发测试用例的节点：不叠加理想曲线
    var bestIdeal = <(double, double)>[];
    var foundSeries = false;

    for (final c in report.cases) {
      if (c is! ForgetObserverNodes) continue;
      if (caseName != null && c.caseName != caseName) continue;
      if (c.idealPoints.length > bestIdeal.length) bestIdeal = c.idealPoints;
      final isExcitation = c.caseName.startsWith('excitation-');
      for (final ns in c.nodeSeries) {
        foundSeries = true;
        meta[ns.id] = (ns.typeName, ns.original);
        nodeCase[ns.id] = c.caseName;
        if (isExcitation) excitationIds.add(ns.id);
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
        sourceCase: nodeCase[entry.key] ?? '',
        points: [
          for (final s in steps)
            _CurvePoint(
              x: s.hours.toDouble(),
              y: s.md,
              yCtrl: s.mdCtrl,
              label: '${s.hours}h',
              action: s.action,
              effective: s.effective,
              maskedText: s.maskedText,
              llmReply: s.llmReply,
            ),
        ],
        ideal: excitationIds.contains(entry.key) ? const [] : bestIdeal,
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

  /// revise：按**记忆节点**分组，x=缺失度梯度（低/中/高），y=字符 n-gram 保留率
  /// （回复 vs 原文）。每个节点一条折线，直观展示"遮罩越深 → 信息保留越少"；
  /// 点展开可见该梯度下的 原文/遮罩输入/LLM 回复。
  List<_NodeCurve> get _reviseCurves {
    final byNode = <String, List<ForgetObserverText>>{};
    final meta = <String, (String, String)>{};
    for (final tc in report.cases.whereType<ForgetObserverText>()) {
      // 仅保留带缺失度梯度的修订用例（caseName 形如 "{id}-md0.50"）
      if (!tc.caseName.contains('-md')) continue;
      final key = tc.nodeId ?? _maskTextKey(tc.caseName);
      byNode.putIfAbsent(key, () => []).add(tc);
      meta[key] = ('遮罩修订', tc.original ?? '');
    }
    final curves = <_NodeCurve>[];
    for (final entry in byNode.entries) {
      final list = entry.value
        ..sort((a, b) => _maskMd(a.caseName).compareTo(_maskMd(b.caseName)));
      if (list.isEmpty) continue;
      final (typeName, original) = meta[entry.key] ?? ('遮罩修订', '');
      curves.add(_NodeCurve(
        id: entry.key,
        typeName: typeName,
        original: original,
        points: [
          for (final tc in list)
            _CurvePoint(
              x: _maskMd(tc.caseName),
              y: (tc.llmReply != null && (tc.original?.isNotEmpty ?? false))
                  ? charNgramOverlap(tc.original!, tc.llmReply!)
                  : 0,
              label: 'md${_maskMd(tc.caseName).toStringAsFixed(2)}',
              action:
                  _ReviseSummaryCard._isEffective(tc.llmReply) ? 'Revised' : 'NoAction',
              effective: _ReviseSummaryCard._isEffective(tc.llmReply),
              original: tc.original,
              maskedText: tc.masked,
              llmReply: tc.llmReply,
            ),
        ],
        ideal: const [],
        xLabel: '缺失度梯度',
        yLabel: 'n-gram 保留率',
      ));
    }
    curves.sort((a, b) => a.id.compareTo(b.id));
    return curves;
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
      appBar: AppBar(title: Text('遗忘结果 · ${forgetModeLabel(r.mode)} · ${r.datasetName}')),
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
    switch (report.mode) {
      case 'excitation':
        return _excitationSummary();
      case 'mask':
        return _maskSummary();
      case 'revise':
      case 'revise/full':
      case 'revise/sample':
        return _reviseSummary();
      default:
        return report.mode.startsWith('revise')
            ? _reviseSummary()
            : _pipelineSummary();
    }
  }

  /// 激发测试汇总：说明卡 + 三时机对比 + 对照/激发平均曲线（主视觉）+ 关键数值。
  Widget _excitationSummary() {
    final avg = _excitationAvgCurves();
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('激发测试 · 总体指标', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        const _ExcitationIntroCard(),
        const SizedBox(height: 12),
        _ExcitationTimingTable(cases: report.cases),
        const SizedBox(height: 12),
        _ExcitationCurveChart(ctrlPts: avg.ctrl, trtPts: avg.trt),
        const SizedBox(height: 12),
        _ExcitationMetricList(metrics: report.metrics),
      ],
    );
  }

  /// 遮罩测试汇总：主视觉 = 遮罩率 vs 缺失度曲线（含对角线参考）。
  Widget _maskSummary() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('遮罩测试 · 总体', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        _MaskRatioChart(cases: report.cases),
        const SizedBox(height: 12),
        MetricPanel(metrics: report.metrics),
      ],
    );
  }

  /// 遮罩补全汇总：n-gram 保留率 vs 缺失度（主视觉）+ 补全质量卡 + 指标。
  Widget _reviseSummary() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('遮罩补全 · 总体', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        _ReviseNgramChart(cases: report.cases),
        const SizedBox(height: 12),
        _ReviseSummaryCard(cases: report.cases),
        const SizedBox(height: 12),
        MetricPanel(metrics: report.metrics),
      ],
    );
  }

  /// 全管线汇总（现状）：关键指标 + 逐用例通过/失败。
  Widget _pipelineSummary() {
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

  /// 激发测试：对照组 / 激发组的**平均缺失度曲线**（按检查点小时聚合所有参与节点）。
  /// 数据来自观测层的 node_series（含 md_ctrl 对照组），UI 端直接聚合，无需改 Rust 结构。
  ({List<(double, double)> ctrl, List<(double, double)> trt}) _excitationAvgCurves() {
    final sumCtrl = <int, double>{};
    final sumTrt = <int, double>{};
    final cnts = <int, int>{};
    for (final c in report.cases.whereType<ForgetObserverNodes>()) {
      for (final ns in c.nodeSeries) {
        for (final s in ns.steps) {
          sumCtrl[s.hours] = (sumCtrl[s.hours] ?? 0) + (s.mdCtrl ?? s.md);
          sumTrt[s.hours] = (sumTrt[s.hours] ?? 0) + s.md;
          cnts[s.hours] = (cnts[s.hours] ?? 0) + 1;
        }
      }
    }
    final hours = cnts.keys.toList()..sort();
    return (
      ctrl: [for (final h in hours) (h.toDouble(), sumCtrl[h]! / cnts[h]!)],
      trt: [for (final h in hours) (h.toDouble(), sumTrt[h]! / cnts[h]!)],
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
      case 'revise/full':
      case 'revise/sample':
        final curves = _reviseCurves;
        if (curves.isNotEmpty) return _buildNodeCentric(curves);
      default:
        if (report.mode.startsWith('revise')) {
          final curves = _reviseCurves;
          if (curves.isNotEmpty) return _buildNodeCentric(curves);
        }
        return _buildNodeObserver();
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

  /// 激发测试时机后缀 → 中文标签。
  static String _timingLabel(String suffix) => switch (suffix) {
        'early' => '前置',
        'spaced' => '均布',
        'late' => '后置',
        _ => suffix,
      };

  /// 三时机激发线的颜色：early=靛蓝（主），spaced=绿，late=琥珀，对照组=青绿虚线。
  static Color _timingColor(String caseName) => switch (caseName) {
        'excitation-spaced' => AppColors.pass,
        'excitation-late' => AppColors.warn,
        _ => AppColors.running,
      };

  /// 激发测试三时机视图：分图（每时机一张，左栏切换）或叠加（同一节点 4 线）。
  Widget _buildNodeObserver() {
    final timingCases = report.cases
        .whereType<ForgetObserverNodes>()
        .where((c) => c.caseName.startsWith('excitation-'))
        .toList();
    final hasTiming = timingCases.length > 1;
    if (!hasTiming) {
      return _buildNodeCentric(_curvesFor(null));
    }
    // 分图模式：单时机（左栏切换器选择）
    if (!_overlayTiming) {
      if (_selectedCase == null || !timingCases.any((c) => c.caseName == _selectedCase)) {
        _selectedCase = timingCases.first.caseName;
      }
      return _buildNodeCentric(_curvesFor(_selectedCase));
    }
    // 叠加模式：同一节点 4 条曲线（对照虚线 + 三时机激发实线）
    final byCase = <String, List<_NodeCurve>>{};
    for (final c in timingCases) {
      byCase[c.caseName] = _curvesFor(c.caseName);
    }
    final mainCase = timingCases.first.caseName;
    final overlays = <_NodeCurve>[];
    for (final c in timingCases.skip(1)) {
      overlays.addAll(byCase[c.caseName] ?? const []);
    }
    return _buildNodeCentric(byCase[mainCase] ?? const [], overlays: overlays);
  }

  // ── 节点为中心：左节点列表 + 右"节点 × 时间步"曲线与数据点展开（pipeline/mask 通用）──
  Widget _buildNodeCentric(List<_NodeCurve> curves,
      {List<_NodeCurve> overlays = const []}) {
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
              // 激发测试三时机视图：模式切换（分图/叠加）+ 分图时的时机切换
              if (curves.any((c) => c.sourceCase.startsWith('excitation-'))) ...[
                Padding(
                  padding: const EdgeInsets.fromLTRB(8, 8, 8, 2),
                  child: SegmentedButton<bool>(
                    segments: const [
                      ButtonSegment(value: false, label: Text('分图·单时机', style: TextStyle(fontSize: 11))),
                      ButtonSegment(value: true, label: Text('叠加·4曲线', style: TextStyle(fontSize: 11))),
                    ],
                    selected: {_overlayTiming},
                    showSelectedIcon: false,
                    style: const ButtonStyle(
                        visualDensity: VisualDensity.compact,
                        tapTargetSize: MaterialTapTargetSize.shrinkWrap),
                    onSelectionChanged: (s) => setState(() {
                      _overlayTiming = s.first;
                      _selectedNodeId = null;
                      _selectedStep = -1;
                    }),
                  ),
                ),
                // 分图模式：时机切换
                if (!_overlayTiming && _selectedCase != null)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(8, 4, 8, 2),
                    child: SegmentedButton<String>(
                      segments: [
                        for (final suffix in ['early', 'spaced', 'late'])
                          if (report.cases.any((c) =>
                              c is ForgetObserverNodes &&
                              c.caseName == 'excitation-$suffix'))
                            ButtonSegment(
                                value: 'excitation-$suffix',
                                label: Text(_timingLabel(suffix),
                                    style: const TextStyle(fontSize: 11))),
                      ],
                      selected: {_selectedCase!},
                      showSelectedIcon: false,
                      style: const ButtonStyle(
                          visualDensity: VisualDensity.compact,
                          tapTargetSize: MaterialTapTargetSize.shrinkWrap),
                      onSelectionChanged: (s) => setState(() {
                        _selectedCase = s.first;
                        _selectedNodeId = null;
                        _selectedStep = -1;
                      }),
                    ),
                  ),
              ],
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
                      'Activated' => AppColors.pass,
                      'Control' => AppColors.ctrl,
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
                          'Activated' => Icons.bolt,
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
            overlays: [
              for (final o in overlays)
                if (o.id == selected.id) o,
            ],
            selectedStep: _selectedStep,
            onSelectStep: (i) => setState(() => _selectedStep = i),
          ),
        ),
      ],
    );
  }
}

/// 三时机对比表：把 excitation-early/spaced/late 三个用例的关键数值并列，
/// 观测"同一批剂量、不同激发时机"的效果是否分化（当前次数制应一致；
/// 若将来算法引入"回鲜"语义，三列将开始分化——本表即语义变化观测窗）。
class _ExcitationTimingTable extends StatelessWidget {
  final List<ForgetObserverCase> cases;
  const _ExcitationTimingTable({required this.cases});

  /// 列定义：case 后缀 → 中文标签
  static const _cols = <String, String>{
    'early': '前置 t=0',
    'spaced': '均布 24/48/72h',
    'late': '后置 t=48h',
  };

  /// 从用例 metrics（三元组 group/label/value）按 label 关键词取值
  static String? _metricOf(ForgetObserverNodes c, String key) {
    for (final (_, label, value) in c.metrics) {
      if (label.contains(key)) return value;
    }
    return null;
  }

  static String? _delayOf(ForgetObserverNodes c) {
    final v = _metricOf(c, '延缓');
    if (v == null) return null;
    return v.split(' / ').first.trim(); // "31.5h / 2.4h" → "31.5h"
  }

  static String? _deltaOf(ForgetObserverNodes c) {
    final v = _metricOf(c, '平均缺失度');
    final m = v == null ? null : RegExp(r'Δ([\d.]+)').firstMatch(v);
    return m?.group(1);
  }

  static String? _capOf(ForgetObserverNodes c) {
    final v = _metricOf(c, '封顶');
    final m = v == null ? null : RegExp(r'Δmd=([\d.]+)').firstMatch(v);
    return m?.group(1);
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final byCol = <String, ForgetObserverNodes>{};
    for (final c in cases.whereType<ForgetObserverNodes>()) {
      for (final suffix in _cols.keys) {
        if (c.caseName == 'excitation-$suffix') byCol[suffix] = c;
      }
    }
    if (byCol.isEmpty) return const SizedBox.shrink();

    // 行定义：(标题, 提取函数)
    final rows = <(String, String? Function(ForgetObserverNodes))>[
      ('平均延缓(至md=0.5)', _delayOf),
      ('72h 平均缺失度 Δ', _deltaOf),
      ('封顶 dose50/100 Δmd', _capOf),
    ];

    Widget cell(String text, {bool bold = false, bool header = false}) {
      final style = header
          ? Theme.of(context).textTheme.labelSmall
          : Theme.of(context).textTheme.bodySmall?.copyWith(
                fontWeight: bold ? FontWeight.w600 : FontWeight.normal,
                fontFamily: bold ? 'monospace' : null,
              );
      return Expanded(
        child: Text(
          text,
          textAlign: TextAlign.center,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: style,
        ),
      );
    }

    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('三时机对比（同一批剂量，激发时机不同）',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 10),
            Row(
              children: [
                SizedBox(width: 150, child: Text('指标', style: Theme.of(context).textTheme.labelSmall)),
                for (final suffix in _cols.keys) cell(_cols[suffix]!, header: true),
              ],
            ),
            const SizedBox(height: 6),
            for (final (title, extract) in rows) ...[
              Divider(height: 1, color: scheme.outlineVariant.withValues(alpha: 0.4)),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Row(
                  children: [
                    SizedBox(width: 150, child: Text(title, style: Theme.of(context).textTheme.bodySmall)),
                    for (final suffix in _cols.keys)
                      cell(byCol[suffix] == null ? '—' : (extract(byCol[suffix]!) ?? '—'),
                          bold: true),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// 遮罩测试汇总主视觉：平均遮罩率 vs 缺失度（x=梯度 0→1，y=平均遮罩率），
/// 叠加 y=x 对角线作参考（当前实现遮罩率≈缺失度；实现演化后仅作参考，不设断言）。
class _MaskRatioChart extends StatelessWidget {
  final List<ForgetObserverCase> cases;
  const _MaskRatioChart({required this.cases});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    // 聚合：按缺失度梯度分组求平均遮罩率
    final sumRatio = <double, double>{};
    final counts = <double, int>{};
    for (final c in cases.whereType<ForgetObserverText>()) {
      if (!c.caseName.contains('-md')) continue; // 排除 determinism 等非梯度用例
      final md = _maskMd(c.caseName);
      sumRatio[md] = (sumRatio[md] ?? 0) + (c.maskRatio ?? 0);
      counts[md] = (counts[md] ?? 0) + 1;
    }
    final mds = sumRatio.keys.toList()..sort();
    if (mds.isEmpty) return const SizedBox.shrink();
    final pts = <(double, double)>[
      for (final md in mds) (md, sumRatio[md]! / counts[md]!),
    ];
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
                const Expanded(
                  child: Text('平均遮罩率 vs 缺失度',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                ),
                const _LegendDot(color: AppColors.running, label: '实测'),
                const SizedBox(width: 12),
                const _LegendDot(color: AppColors.warn, label: '参考 y=x'),
              ],
            ),
            const SizedBox(height: 10),
            SizedBox(
              height: 220,
              child: CustomPaint(
                size: Size.infinite,
                painter: _MaskRatioPainter(points: pts),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 从用例名提取缺失度梯度（如 "medium-md0.50" → 0.5）。
  static double _maskMd(String caseName) {
    final m = RegExp(r'md(\d+(?:\.\d+)?)').firstMatch(caseName);
    return m == null ? 0.0 : double.parse(m.group(1)!);
  }
}

class _MaskRatioPainter extends CustomPainter {
  final List<(double, double)> points;
  _MaskRatioPainter({required this.points});

  static const _left = 46.0, _right = 12.0, _top = 14.0, _bottom = 30.0;

  @override
  void paint(Canvas canvas, Size size) {
    final plotW = size.width - _left - _right;
    final plotH = size.height - _top - _bottom;
    double x(double v) => _left + v.clamp(0.0, 1.0) * plotW;
    double y(double v) => _top + (1.0 - v.clamp(0.0, 1.0)) * plotH;

    final gridPaint = Paint()..color = const Color(0x22FFFFFF)..strokeWidth = 1;
    final labelStyle =
        TextStyle(color: const Color(0xAA9E9E9E), fontSize: 9.5, fontFamily: 'monospace');
    for (var v = 0.0; v <= 1.0 + 1e-9; v += 0.25) {
      final yy = y(v);
      canvas.drawLine(Offset(_left, yy), Offset(size.width - _right, yy), gridPaint);
      final tp = TextPainter(
          text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
          textDirection: TextDirection.ltr)
        ..layout();
      tp.paint(canvas, Offset(2, yy - tp.height / 2));
      final xx = x(v);
      canvas.drawLine(Offset(xx, _top), Offset(xx, size.height - _bottom), gridPaint);
      final tp2 = TextPainter(
          text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
          textDirection: TextDirection.ltr)
        ..layout();
      tp2.paint(canvas, Offset(xx - tp2.width / 2, size.height - _bottom + 6));
    }

    // 参考对角线 y=x（虚线）
    final diag = Path()
      ..moveTo(x(0), y(0))
      ..lineTo(x(1), y(1));
    _drawDashed(canvas, diag,
        Paint()
          ..color = AppColors.warn.withValues(alpha: 0.75)
          ..strokeWidth = 1.6
          ..style = PaintingStyle.stroke);

    // 实测折线 + 点
    if (points.length >= 2) {
      final line = Paint()
        ..color = AppColors.running
        ..strokeWidth = 2.2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final path = Path();
      for (var i = 0; i < points.length; i++) {
        final p = Offset(x(points[i].$1), y(points[i].$2));
        if (i == 0) {
          path.moveTo(p.dx, p.dy);
        } else {
          path.lineTo(p.dx, p.dy);
        }
      }
      canvas.drawPath(path, line);
    }
    final dot = Paint()..color = AppColors.running;
    for (final (mx, my) in points) {
      canvas.drawCircle(Offset(x(mx), y(my)), 4, dot);
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
  bool shouldRepaint(_MaskRatioPainter old) => old.points != points;
}

/// 修订测试汇总主视觉：平均字符 n-gram 保留率（回复 vs 原文）随缺失度梯度变化折线。
/// 遮罩越深 → 上下文越少 → 信息保留率越低（同义改写会低估，仅作趋势参考）。
class _ReviseNgramChart extends StatelessWidget {
  final List<ForgetObserverCase> cases;
  const _ReviseNgramChart({required this.cases});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final sumRatio = <double, double>{};
    final counts = <double, int>{};
    for (final c in cases.whereType<ForgetObserverText>()) {
      if (!c.caseName.contains('-md')) continue;
      if (c.llmReply == null || (c.original?.isEmpty ?? true)) continue;
      final md = _ForgetResultsPageState._maskMd(c.caseName);
      sumRatio[md] = (sumRatio[md] ?? 0) + charNgramOverlap(c.original!, c.llmReply!);
      counts[md] = (counts[md] ?? 0) + 1;
    }
    final mds = sumRatio.keys.toList()..sort();
    if (mds.isEmpty) return const SizedBox.shrink();
    final pts = <(double, double)>[
      for (final md in mds) (md, sumRatio[md]! / counts[md]!),
    ];
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
                const Expanded(
                  child: Text('平均 n-gram 保留率 vs 缺失度',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                ),
                const _LegendDot(color: AppColors.running, label: 'n-gram 保留率'),
              ],
            ),
            const SizedBox(height: 10),
            SizedBox(
              height: 220,
              child: CustomPaint(
                size: Size.infinite,
                painter: _ReviseNgramPainter(points: pts),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ReviseNgramPainter extends CustomPainter {
  final List<(double, double)> points;
  _ReviseNgramPainter({required this.points});

  static const _left = 46.0, _right = 12.0, _top = 14.0, _bottom = 30.0;

  @override
  void paint(Canvas canvas, Size size) {
    final plotW = size.width - _left - _right;
    final plotH = size.height - _top - _bottom;
    double x(double v) => _left + v.clamp(0.0, 1.0) * plotW;
    double y(double v) => _top + (1.0 - v.clamp(0.0, 1.0)) * plotH;

    final gridPaint = Paint()..color = const Color(0x22FFFFFF)..strokeWidth = 1;
    final labelStyle =
        TextStyle(color: const Color(0xAA9E9E9E), fontSize: 9.5, fontFamily: 'monospace');
    for (var v = 0.0; v <= 1.0 + 1e-9; v += 0.25) {
      final yy = y(v);
      canvas.drawLine(Offset(_left, yy), Offset(size.width - _right, yy), gridPaint);
      final tp = TextPainter(
          text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
          textDirection: TextDirection.ltr)
        ..layout();
      tp.paint(canvas, Offset(2, yy - tp.height / 2));
      final xx = x(v);
      canvas.drawLine(Offset(xx, _top), Offset(xx, size.height - _bottom), gridPaint);
      final tp2 = TextPainter(
          text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
          textDirection: TextDirection.ltr)
        ..layout();
      tp2.paint(canvas, Offset(xx - tp2.width / 2, size.height - _bottom + 6));
    }

    if (points.length >= 2) {
      final line = Paint()
        ..color = AppColors.running
        ..strokeWidth = 2.2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final path = Path();
      for (var i = 0; i < points.length; i++) {
        final p = Offset(x(points[i].$1), y(points[i].$2));
        if (i == 0) {
          path.moveTo(p.dx, p.dy);
        } else {
          path.lineTo(p.dx, p.dy);
        }
      }
      canvas.drawPath(path, line);
    }
    final dot = Paint()..color = AppColors.running;
    for (final (mx, my) in points) {
      canvas.drawCircle(Offset(x(mx), y(my)), 4, dot);
    }
  }

  @override
  bool shouldRepaint(_ReviseNgramPainter old) => old.points != points;
}

/// 遮罩补全汇总：有效修订率 + 回复长度对比（原文 vs 回复，字符数）。
class _ReviseSummaryCard extends StatelessWidget {
  final List<ForgetObserverCase> cases;
  const _ReviseSummaryCard({required this.cases});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final samples = cases.whereType<ForgetObserverText>().toList();
    final effective = samples.where((c) => _isEffective(c.llmReply)).length;
    final withReply = samples.where((c) => c.llmReply != null).length;
    final avgReplyLen = withReply == 0
        ? 0.0
        : samples.map((c) => (c.llmReply?.runes.length ?? 0)).reduce((a, b) => a + b) /
            withReply;
    // 平均字符 n-gram 重合率（回复 vs 原文，参考指标）
    final ngramSum = samples
        .where((c) => c.llmReply != null && (c.original?.isNotEmpty ?? false))
        .map((c) => charNgramOverlap(c.original!, c.llmReply!))
        .fold<double>(0, (a, b) => a + b);
    final ngramCount = samples
        .where((c) => c.llmReply != null && (c.original?.isNotEmpty ?? false))
        .length;

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
                Expanded(
                  child: Text('补全质量',
                      style: Theme.of(context).textTheme.titleSmall),
                ),
                Text(
                  '有效修订 $effective/${samples.length} · 平均回复 ${avgReplyLen.toStringAsFixed(0)} 字'
                  '${ngramCount > 0 ? ' · n-gram 重合 ${(ngramSum / ngramCount * 100).toStringAsFixed(0)}%' : ''}',
                  style: const TextStyle(
                      fontFamily: 'monospace', fontSize: 12, color: AppColors.subtle),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (samples.isEmpty)
              const Text('（无样本）', style: TextStyle(color: AppColors.subtle))
            else
              for (final c in samples)
                _ReviseLengthRow(sample: c),
          ],
        ),
      ),
    );
  }

  static bool _isEffective(String? reply) {
    if (reply == null) return false;
    final t = reply.trim();
    return t.isNotEmpty && !t.contains('[masked]');
  }
}

/// 字符 n-gram 重合率（参考指标）：回复 vs 原文的表层保留程度。
/// 同义改写会低估重合率，仅作趋势参考，不设断言。
double charNgramOverlap(String a, String b, {int n = 3}) {
  Set<String> grams(String s) {
    final chars = s.runes.toList();
    if (chars.length < n) return {s};
    return {
      for (var i = 0; i + n <= chars.length; i++)
        String.fromCharCodes(chars.sublist(i, i + n))
    };
  }

  final ga = grams(a);
  if (ga.isEmpty) return 0;
  return ga.intersection(grams(b)).length / ga.length;
}

class _ReviseLengthRow extends StatelessWidget {
  final ForgetObserverText sample;
  const _ReviseLengthRow({required this.sample});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final originalLen = sample.original?.runes.length ?? 0;
    final replyLen = sample.llmReply?.runes.length ?? 0;
    final effective = _ReviseSummaryCard._isEffective(sample.llmReply);
    final maxLen = math.max(originalLen, replyLen).clamp(1, double.infinity).toDouble();

    Widget bar(double len, Color color) => Expanded(
          child: FractionallySizedBox(
            alignment: Alignment.centerLeft,
            widthFactor: len / maxLen,
            child: Container(
                height: 6,
                decoration:
                    BoxDecoration(color: color, borderRadius: BorderRadius.circular(3))),
          ),
        );

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  '${sample.caseName} · 原文 $originalLen 字 → 回复 $replyLen 字',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontSize: 11, fontFamily: 'monospace'),
                ),
              ),
              Icon(
                effective ? Icons.check_circle_outline : Icons.cancel_outlined,
                size: 14,
                color: effective ? AppColors.pass : AppColors.fail,
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              SizedBox(width: 40, child: Text('原文', style: const TextStyle(fontSize: 10, color: AppColors.subtle))),
              bar(originalLen.toDouble(), AppColors.subtle.withValues(alpha: 0.6)),
              const SizedBox(width: 8),
              SizedBox(width: 40, child: Text('回复', style: const TextStyle(fontSize: 10, color: AppColors.subtle))),
              bar(replyLen.toDouble(), effective ? AppColors.pass : AppColors.fail),
            ],
          ),
        ],
      ),
    );
  }
}

/// 激发测试说明卡：测试目的 + 读图方法（简洁）。
class _ExcitationIntroCard extends StatelessWidget {
  const _ExcitationIntroCard();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        child: Text(
          '验证"激发 → 遗忘被延缓"：同一角色图克隆两份，一份按设计剂量激发、一份不激发，72h 内逐节点配对观测。'
          '实线（激发）低于虚线（对照）= 延缓生效。',
          style: Theme.of(context)
              .textTheme
              .bodySmall
              ?.copyWith(color: scheme.onSurfaceVariant, height: 1.5),
        ),
      ),
    );
  }
}

/// 激发测试主视觉：对照组 / 激发组平均缺失度曲线（36 点，双系列）。
class _ExcitationCurveChart extends StatelessWidget {
  final List<(double, double)> ctrlPts;
  final List<(double, double)> trtPts;
  const _ExcitationCurveChart({required this.ctrlPts, required this.trtPts});

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
                const Expanded(
                  child: Text('平均缺失度 · 激发 vs 对照',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                ),
                const _LegendDot(color: AppColors.running, label: '激发'),
                const SizedBox(width: 12),
                const _LegendDot(color: AppColors.ctrl, label: '对照'),
              ],
            ),
            const SizedBox(height: 10),
            SizedBox(
              height: 220,
              child: CustomPaint(
                size: Size.infinite,
                painter: _ExcitationCurvePainter(ctrlPts: ctrlPts, trtPts: trtPts),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 双系列曲线绘制：y = 平均缺失度（0..1），x = 小时（0..72）。
class _ExcitationCurvePainter extends CustomPainter {
  final List<(double, double)> ctrlPts;
  final List<(double, double)> trtPts;
  _ExcitationCurvePainter({required this.ctrlPts, required this.trtPts});

  static const _left = 46.0, _right = 12.0, _top = 14.0, _bottom = 30.0;
  static const _maxX = 72.0;

  @override
  void paint(Canvas canvas, Size size) {
    final plotW = size.width - _left - _right;
    final plotH = size.height - _top - _bottom;
    double x(double v) => _left + v / _maxX * plotW;
    double y(double v) => _top + (1.0 - v.clamp(0.0, 1.0)) * plotH;

    final gridPaint = Paint()..color = const Color(0x22FFFFFF)..strokeWidth = 1;
    final labelStyle =
        TextStyle(color: const Color(0xAA9E9E9E), fontSize: 9.5, fontFamily: 'monospace');
    for (var v = 0.0; v <= 1.0 + 1e-9; v += 0.25) {
      final yy = y(v);
      canvas.drawLine(Offset(_left, yy), Offset(size.width - _right, yy), gridPaint);
      final tp = TextPainter(
          text: TextSpan(text: v.toStringAsFixed(2), style: labelStyle),
          textDirection: TextDirection.ltr)
        ..layout();
      tp.paint(canvas, Offset(2, yy - tp.height / 2));
    }
    for (var h = 0.0; h <= _maxX + 1e-9; h += 24) {
      final xx = x(h);
      canvas.drawLine(Offset(xx, _top), Offset(xx, size.height - _bottom), gridPaint);
      final tp = TextPainter(
          text: TextSpan(text: '${h.round()}h', style: labelStyle),
          textDirection: TextDirection.ltr)
        ..layout();
      tp.paint(canvas, Offset(xx - tp.width / 2, size.height - _bottom + 6));
    }
    // 轴标题
    final axisStyle = const TextStyle(
        color: Color(0xAA9E9E9E), fontSize: 10, fontFamily: 'monospace');
    final yTitle = TextPainter(
        text: TextSpan(text: '平均缺失度', style: axisStyle), textDirection: TextDirection.ltr)
      ..layout();
    yTitle.paint(canvas, const Offset(6, 2));
    final xTitle = TextPainter(
        text: TextSpan(text: '时间（小时）', style: axisStyle), textDirection: TextDirection.ltr)
      ..layout();
    xTitle.paint(
        canvas, Offset(size.width - _right - xTitle.width, size.height - _bottom + 18));

    void drawSeries(List<(double, double)> pts, Color color, double width, {bool dash = false}) {
      if (pts.length < 2) return;
      final paint = Paint()
        ..color = color
        ..strokeWidth = width
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final path = Path();
      for (var i = 0; i < pts.length; i++) {
        final p = Offset(x(pts[i].$1), y(pts[i].$2));
        if (i == 0) {
          path.moveTo(p.dx, p.dy);
        } else {
          path.lineTo(p.dx, p.dy);
        }
      }
      if (dash) {
        for (final metric in path.computeMetrics()) {
          var dist = 0.0;
          const dashLen = 7.0, gapLen = 5.0;
          while (dist < metric.length) {
            final end = math.min(dist + dashLen, metric.length);
            canvas.drawPath(metric.extractPath(dist, end), paint);
            dist = end + gapLen;
          }
        }
      } else {
        canvas.drawPath(path, paint);
      }
    }

    drawSeries(ctrlPts, AppColors.ctrl.withValues(alpha: 0.9), 2.0, dash: true);
    drawSeries(trtPts, AppColors.running, 2.4);
  }

  @override
  bool shouldRepaint(_ExcitationCurvePainter old) =>
      old.ctrlPts != ctrlPts || old.trtPts != trtPts;
}

/// 激发测试指标的悬停释义（按 label 关键词匹配）。
String? _excitationMetricTip(String label) {
  if (label.contains('平均缺失度')) {
    return '缺失度 md：0=新鲜，1=完全遗忘。对照=不激发时的自然遗忘，激发=被激发后；Δ 越大延缓越明显。';
  }
  if (label.contains('延缓')) {
    return '激发组比对照组晚多少小时到达 md=0.5（遗忘到一半），即半衰期延长的实际效果。';
  }
  if (label.contains('封顶')) {
    return '激活次数超过 50 次后不再额外延缓遗忘（算法设计边界）：50 次与 100 次效果应相同。';
  }
  return null;
}

/// 激发测试汇总：以"指标一行"的紧凑列表展示总体指标（label 左、value 右，悬停释义）。
class _ExcitationMetricList extends StatelessWidget {
  final List<MetricEntry> metrics;
  const _ExcitationMetricList({super.key, required this.metrics});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    if (metrics.isEmpty) {
      return const Text('（无指标）', style: TextStyle(color: AppColors.subtle));
    }
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        child: Column(
          children: [
            for (var i = 0; i < metrics.length; i++) ...[
              if (i > 0)
                Divider(height: 1, color: scheme.outlineVariant.withValues(alpha: 0.5)),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    Expanded(
                      child: Tooltip(
                        message: _excitationMetricTip(metrics[i].label) ?? metrics[i].label,
                        waitDuration: const Duration(milliseconds: 300),
                        child: Text(
                          metrics[i].label,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: Theme.of(context)
                              .textTheme
                              .bodyMedium
                              ?.copyWith(color: scheme.onSurfaceVariant),
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Text(
                      metrics[i].value ?? '',
                      textAlign: TextAlign.right,
                      style: const TextStyle(
                          fontFamily: 'monospace', fontSize: 15, fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// 节点演变视图：x（时间步长/缺失度）× y（指标）曲线 + 理想曲线叠加 +
/// 可点击数据点 → 展开该数据点的原始输出与节点原文（pipeline/mask 通用）。
class _NodeEvolutionView extends StatelessWidget {
  final _NodeCurve curve;
  /// 同一节点的其他时机激发曲线（三时机叠加视图：对照 + 3 条激发线）
  final List<_NodeCurve> overlays;
  final int selectedStep;
  final ValueChanged<int> onSelectStep;
  const _NodeEvolutionView({
    required this.curve,
    this.overlays = const [],
    required this.selectedStep,
    required this.onSelectStep,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final points = curve.points;
    final pt = points[selectedStep.clamp(0, points.length - 1)];
    final timingSuffix = (String? caseName) {
      if (caseName == null) return '';
      return caseName.startsWith('excitation-')
          ? caseName.substring('excitation-'.length)
          : '';
    };
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
                        overlays.isNotEmpty
                            ? '${curve.id} · 对照 vs 三时机激发'
                            : (curve.hasCtrl
                                ? '${curve.id} · 激发 vs 对照'
                                : '${curve.id} · 实测 vs 理想曲线'),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                      ),
                    ),
                    if (curve.hasCtrl) ...[
                      const _LegendDot(color: AppColors.ctrl, label: '对照'),
                      const SizedBox(width: 12),
                      _LegendDot(
                          color: _ForgetResultsPageState._timingColor(curve.sourceCase),
                          label: '激发·${timingSuffix(curve.sourceCase)}'),
                      for (final o in overlays) ...[
                        const SizedBox(width: 12),
                        _LegendDot(
                            color: _ForgetResultsPageState._timingColor(o.sourceCase),
                            label: '激发·${timingSuffix(o.sourceCase)}'),
                      ],
                    ] else ...[
                      const _LegendDot(color: AppColors.running, label: '实测'),
                      if (curve.ideal.isNotEmpty) ...[
                        const SizedBox(width: 12),
                        const _LegendDot(color: AppColors.warn, label: '理想'),
                      ],
                    ],
                    const SizedBox(width: 12),
                    _LegendDot(color: scheme.primary, label: '点击数据点展开'),
                  ],
                ),
                const SizedBox(height: 10),
                SizedBox(
                  height: 300,
                  child: _TrendChart(
                    curve: curve,
                    overlays: overlays,
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
  /// 同一节点的其他时机激发曲线（三时机叠加视图）
  final List<_NodeCurve> overlays;
  final int selected;
  final ValueChanged<int> onSelect;
  const _TrendChart({
    required this.curve,
    this.overlays = const [],
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
                painter: _TrendPainter(
                    geo: geo, curve: curve, overlays: overlays, selected: selected),
              ),
              // 对照组数据点（虚线：激发测试的未激发基线，落在 yCtrl 上）
              if (curve.hasCtrl)
                for (var i = 0; i < points.length; i++)
                  if (points[i].yCtrl != null)
                    Positioned(
                      left: geo.x(points[i].x) - 6,
                      top: geo.y(points[i].yCtrl!) - 6,
                      width: 12,
                      height: 12,
                      child: GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTap: () => onSelect(i),
                        child: Center(
                          child: Container(
                            width: selected == i ? 10 : 8,
                            height: selected == i ? 10 : 8,
                            decoration: BoxDecoration(
                              color: AppColors.ctrl,
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                      ),
                    ),
              // 激发组/实测数据点（可点击）
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
                          color: curve.hasCtrl ? AppColors.running : _pointColor(points[i]),
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
              // 选中点标注（最顶层 widget，避免被数据点圆点遮挡）
              if (selected >= 0 && selected < points.length)
                Positioned(
                  left: geo.x(points[selected].x) + 12,
                  top: (geo.y(points[selected].y) - 22).clamp(0.0, size.height - 40.0),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                    decoration: BoxDecoration(
                      color: AppColors.running,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      '${points[selected].label} · ${curve.yLabel} ${points[selected].y.toStringAsFixed(3)}',
                      style: const TextStyle(
                          color: Colors.black87,
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          fontFamily: 'monospace'),
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
      'Activated' => AppColors.pass,
      'Control' => AppColors.ctrl,
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

/// 背景绘制：网格、轴标签、理想虚线、对照组虚线、多时机激发折线。
class _TrendPainter extends CustomPainter {
  final _ChartGeo geo;
  final _NodeCurve curve;
  /// 同一节点的其他时机激发曲线（三时机叠加视图）
  final List<_NodeCurve> overlays;
  final int selected;
  _TrendPainter({
    required this.geo,
    required this.curve,
    this.overlays = const [],
    required this.selected,
  });

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

    // ── 对照组曲线（虚线，激发测试：未激发基线）──
    if (curve.hasCtrl) {
      final ctrlDash = Paint()
        ..color = AppColors.ctrl.withValues(alpha: 0.9)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final ctrlPath = Path();
      final ctrlPoints = curve.points;
      for (var i = 0; i < ctrlPoints.length; i++) {
        final yc = ctrlPoints[i].yCtrl;
        if (yc == null) continue;
        final p = Offset(geo.x(ctrlPoints[i].x), geo.y(yc));
        if (i == 0 || ctrlPoints[i - 1].yCtrl == null) {
          ctrlPath.moveTo(p.dx, p.dy);
        } else {
          ctrlPath.lineTo(p.dx, p.dy);
        }
      }
      _drawDashed(canvas, ctrlPath, ctrlDash);
    }

    // ── 其他时机激发曲线（三时机叠加视图，主激发线之前的底层）──
    for (final ov in overlays) {
      final ovLine = Paint()
        ..color =
            _ForgetResultsPageState._timingColor(ov.sourceCase).withValues(alpha: 0.85)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      final ovPath = Path();
      for (var i = 0; i < ov.points.length; i++) {
        final p = Offset(geo.x(ov.points[i].x), geo.y(ov.points[i].y));
        if (i == 0) {
          ovPath.moveTo(p.dx, p.dy);
        } else {
          ovPath.lineTo(p.dx, p.dy);
        }
      }
      canvas.drawPath(ovPath, ovLine);
    }

    // ── 实测折线（主时机/主曲线）──
    final line = Paint()
      ..color = _ForgetResultsPageState._timingColor(curve.sourceCase)
      ..strokeWidth = 2.4
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
      old.curve != curve ||
      old.selected != selected ||
      old.overlays.length != overlays.length;
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
      'Activated' => AppColors.pass,
      'Control' => AppColors.ctrl,
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
                Text(
                  point.yCtrl != null
                      ? '$yLabel 激发 ${point.y.toStringAsFixed(3)} / 对照 ${point.yCtrl!.toStringAsFixed(3)}'
                      : '$yLabel ${point.y.toStringAsFixed(3)}',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
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
