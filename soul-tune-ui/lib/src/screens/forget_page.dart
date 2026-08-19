import 'dart:async';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/metric_panel.dart';
import '../widgets/mini_bar_chart.dart';
import '../widgets/pass_rate_donut.dart';
import '../widgets/stat_card.dart';

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
      'revise' => '只验证遮罩补全（需 llama-server / SOUL_TUNE_LLAMA_URL）',
      _ => '全管线：衰减 → 遮罩 → LLM 补全 → 边衰减（需 LLM）',
    };
    return Scaffold(
      appBar: AppBar(title: const Text('遗忘测试')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
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

/// 节点跨用例聚合：同一记忆节点在各时间跨度用例下的演变。
class _NodeAggregate {
  final String id;
  final String typeName;
  final String original;
  final List<(ForgetObserverNodes, NodeForgetStat)> points;

  _NodeAggregate({
    required this.id,
    required this.typeName,
    required this.original,
    required this.points,
  });

  String get action => points.last.$2.action;
  bool get effective => points.last.$2.effective;
  double get finalMd => points.last.$2.mdAfter;
}

/// 遗忘结果页：汇总（图表+关键指标） / 观测（以记忆节点为单位看时间演变）。
class ForgetResultsPage extends StatefulWidget {
  final ForgetReport report;
  const ForgetResultsPage({super.key, required this.report});

  @override
  State<ForgetResultsPage> createState() => _ForgetResultsPageState();
}

class _ForgetResultsPageState extends State<ForgetResultsPage> {
  int _tab = 0; // 0=汇总 1=观测
  String? _selectedNodeId;

  ForgetReport get report => widget.report;

  /// 聚合全部 pipeline 用例的节点（按 id）
  List<_NodeAggregate> get _aggregates {
    final map = <String, _NodeAggregate>{};
    for (final c in report.cases) {
      if (c is! ForgetObserverNodes) continue;
      for (final n in c.nodes) {
        final agg = map.putIfAbsent(
            n.id,
            () => _NodeAggregate(
                id: n.id, typeName: n.typeName, original: n.original, points: []));
        agg.points.add((c, n));
      }
    }
    // 按用例顺序排序
    for (final a in map.values) {
      a.points.sort((a, b) => (a.$1.hours ?? a.$1.nodeCount).compareTo(b.$1.hours ?? b.$1.nodeCount));
    }
    return map.values.toList();
  }

  @override
  Widget build(BuildContext context) {
    final r = report;
    return Scaffold(
      appBar: AppBar(title: Text('遗忘结果 · ${r.mode} · ${r.datasetName}')),
      body: Column(
        children: [
          // 紧凑头部：小圆环 + 统计
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 10, 16, 2),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 92,
                  height: 92,
                  child: PassRateDonut(rate: r.total == 0 ? 0 : r.passed / r.total, size: 92),
                ),
                const SizedBox(width: 18),
                Flexible(
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      StatCard(label: '通过', value: '${r.passed}', valueColor: AppColors.pass),
                      StatCard(label: '失败', value: '${r.failed}', valueColor: AppColors.fail),
                      StatCard(label: '耗时', value: r.elapsedSecs.toStringAsFixed(2), unit: 's'),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            child: SegmentedButton<int>(
              segments: const [
                ButtonSegment(value: 0, label: Text('汇总')),
                ButtonSegment(value: 1, label: Text('观测')),
              ],
              selected: {_tab},
              onSelectionChanged: (s) => setState(() => _tab = s.first),
            ),
          ),
          const Divider(height: 1),
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
    final aggregates = _aggregates;
    if (aggregates.isNotEmpty) {
      return _buildNodeCentric(aggregates);
    }
    // mask/revise：文本视图
    final textCases = report.cases.whereType<ForgetObserverText>().toList();
    if (textCases.isEmpty) {
      return const Center(child: Text('（无观测数据）', style: TextStyle(color: AppColors.subtle)));
    }
    return _buildTextCases(textCases);
  }

  // ── 节点为中心：左节点列表 + 右演变详情 ──
  Widget _buildNodeCentric(List<_NodeAggregate> aggregates) {
    var selected = aggregates.first;
    for (final a in aggregates) {
      if (a.id == _selectedNodeId) {
        selected = a;
        break;
      }
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 左：节点列表
        Container(
          width: 280,
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerLow,
            border: Border(
              right: BorderSide(color: Theme.of(context).colorScheme.outlineVariant, width: 0.5),
            ),
          ),
          child: ListView.builder(
            itemCount: aggregates.length,
            itemBuilder: (context, i) {
              final a = aggregates[i];
              final fg = switch (a.action) {
                'Revised' => a.effective ? AppColors.pass : AppColors.fail,
                'MaskOnly' => AppColors.warn,
                _ => AppColors.subtle,
              };
              final short = a.id.length > 10 ? a.id.substring(0, 10) : a.id;
              return ListTile(
                dense: true,
                selected: a.id == selected.id,
                selectedTileColor: Theme.of(context).colorScheme.primary.withValues(alpha: 0.12),
                leading: Icon(
                  switch (a.action) {
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
                subtitle: Text('md ${a.points.first.$2.mdBefore.toStringAsFixed(2)} → ${a.finalMd.toStringAsFixed(2)}',
                    style: const TextStyle(fontSize: 10)),
                onTap: () => setState(() => _selectedNodeId = a.id),
              );
            },
          ),
        ),
        // 右：选中节点的演变
        Expanded(child: _NodeEvolutionView(agg: selected)),
      ],
    );
  }

  // ── mask/revise 文本视图 ──
  Widget _buildTextCases(List<ForgetObserverText> textCases) {
    var idx = 0;
    if (_selectedNodeId != null) {
      final parsed = int.tryParse(_selectedNodeId!);
      if (parsed != null && parsed >= 0 && parsed < textCases.length) idx = parsed;
    }
    final c = textCases[idx];
    final scheme = Theme.of(context).colorScheme;
    return Column(
      children: [
        SizedBox(
          height: 44,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            children: [
              for (var i = 0; i < textCases.length; i++)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label: Text(textCases[i].caseName,
                        maxLines: 1, overflow: TextOverflow.ellipsis),
                    selected: i == idx,
                    onSelected: (_) => setState(() => _selectedNodeId = '$i'),
                  ),
                ),
            ],
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (!c.llmAvailable && c.maskedText != null)
                const Padding(
                  padding: EdgeInsets.only(bottom: 8),
                  child: Text('LLM 不可用（未设置 SOUL_TUNE_LLAMA_URL / 模型路径）',
                      style: TextStyle(color: AppColors.warn, fontSize: 12)),
                ),
              Card(
                elevation: 0,
                color: scheme.surfaceContainerHigh,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      for (final (_, l, v) in c.metrics)
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 2),
                          child: Text('$l: $v',
                              style:
                                  const TextStyle(fontFamily: 'monospace', fontSize: 13)),
                        ),
                    ],
                  ),
                ),
              ),
              if (c.maskedText != null) ...[
                const SizedBox(height: 12),
                _TextBlock(title: 'LLM 遮罩输入', text: c.maskedText!, color: AppColors.warn),
              ],
              if (c.llmReply != null) ...[
                const SizedBox(height: 12),
                _TextBlock(title: 'LLM 原始回复', text: c.llmReply!, color: scheme.primary),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

/// 节点演变视图：时间曲线（缺失度/遮罩率）+ 各时间跨度用例详情。
class _NodeEvolutionView extends StatelessWidget {
  final _NodeAggregate agg;
  const _NodeEvolutionView({required this.agg});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('${agg.typeName} [${agg.id}]',
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 4),
        Text('动作: ${agg.action}${agg.action == 'Revised' ? (agg.effective ? '（有效修订）' : '（无效修订）') : ''}'
            '   缺失度 ${agg.points.first.$2.mdBefore.toStringAsFixed(3)} → ${agg.finalMd.toStringAsFixed(3)}',
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12, color: AppColors.subtle)),
        const SizedBox(height: 14),
        _EvolutionChart(points: agg.points),
        const SizedBox(height: 14),
        for (final (caseData, n) in agg.points) _CaseNodeCard(caseData: caseData, node: n),
      ],
    );
  }
}

/// 节点随时间的演变曲线：x=时间跨度(小时)，y=缺失度；副线=遮罩率。
class _EvolutionChart extends StatelessWidget {
  final List<(ForgetObserverNodes, NodeForgetStat)> points;
  const _EvolutionChart({required this.points});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('时间演变', style: TextStyle(fontSize: 13)),
            const SizedBox(height: 8),
            SizedBox(
              height: 140,
              child: CustomPaint(
                painter: _EvolutionPainter(points),
                child: const SizedBox.expand(),
              ),
            ),
            const SizedBox(height: 6),
            const Row(
              children: [
                _LegendDot(color: AppColors.running, label: '缺失度'),
                SizedBox(width: 14),
                _LegendDot(color: AppColors.warn, label: '遮罩率'),
              ],
            ),
          ],
        ),
      ),
    );
  }
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
        Container(width: 10, height: 10, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 11)),
      ],
    );
  }
}

class _EvolutionPainter extends CustomPainter {
  final List<(ForgetObserverNodes, NodeForgetStat)> points;
  _EvolutionPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    // x：小时（无 hours 的用例用索引）
    final xs = <double>[];
    for (var i = 0; i < points.length; i++) {
      xs.add(points[i].$1.hours ?? (i + 1).toDouble());
    }
    if (xs.isEmpty) return;
    final xMin = xs.reduce(math.min);
    final xMax = xs.reduce(math.max);
    final xSpan = (xMax - xMin).abs() < 1e-9 ? 1.0 : xMax - xMin;

    double xPos(double x) => 10 + ((x - xMin) / xSpan) * (size.width - 20);
    double yPos(double v, double maxV) => size.height - 8 - (v / maxV) * (size.height - 16);

    // 缺失度曲线
    final mdMax = points
        .map((p) => p.$2.mdAfter)
        .fold<double>(0, (a, b) => a > b ? a : b)
        .clamp(1e-6, 1.0)
        .toDouble();
    _stroke(canvas, xs, points.map((p) => p.$2.mdAfter).toList(), mdMax,
        AppColors.running, xPos, yPos);

    // 遮罩率曲线（存在遮罩数据时）
    final ratios = points.map((p) {
      final m = p.$2.mask;
      if (m == null || m.$2 == 0) return 0.0;
      return m.$1 / m.$2;
    }).toList();
    final anyMask = points.any((p) => p.$2.mask != null && p.$2.mask!.$2 > 0);
    if (anyMask) {
      _stroke(canvas, xs, ratios, 1.0, AppColors.warn, xPos, yPos);
    }

    // 网格基线
    final grid = Paint()
      ..color = const Color(0x22FFFFFF)
      ..strokeWidth = 1;
    for (var i = 1; i < 4; i++) {
      final y = size.height * i / 4;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }
  }

  void _stroke(
    Canvas canvas,
    List<double> xs,
    List<double> ys,
    double maxV,
    Color color,
    double Function(double) xPos,
    double Function(double, double) yPos,
  ) {
    if (xs.isEmpty) return;
    final path = Path();
    for (var i = 0; i < xs.length; i++) {
      final x = xPos(xs[i]);
      final y = yPos(ys[i], maxV);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );
    final dot = Paint()..color = color;
    for (var i = 0; i < xs.length; i++) {
      canvas.drawCircle(Offset(xPos(xs[i]), yPos(ys[i], maxV)), 3, dot);
    }
  }

  @override
  bool shouldRepaint(_EvolutionPainter old) => old.points != points;
}

/// 单时间跨度下该节点的详情卡：缺失度/遮罩 + 可展开原文/输入/回复。
class _CaseNodeCard extends StatelessWidget {
  final ForgetObserverNodes caseData;
  final NodeForgetStat node;
  const _CaseNodeCard({required this.caseData, required this.node});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final maskRatio = node.mask == null || node.mask!.$2 == 0
        ? null
        : node.mask!.$1 / node.mask!.$2;
    final title = caseData.hours != null
        ? '${caseData.hours!.toStringAsFixed(0)}h  ${caseData.caseName}'
        : caseData.caseName;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      margin: const EdgeInsets.only(bottom: 10),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          leading: Icon(
            switch (node.action) {
              'Revised' => Icons.healing_outlined,
              'MaskOnly' => Icons.visibility_off_outlined,
              _ => Icons.check_circle_outline,
            },
            size: 18,
            color: switch (node.action) {
              'Revised' => node.effective ? AppColors.pass : AppColors.fail,
              'MaskOnly' => AppColors.warn,
              _ => AppColors.subtle,
            },
          ),
          title: Row(
            children: [
              Text(title, style: const TextStyle(fontSize: 13)),
              const Spacer(),
              Text('md ${node.mdBefore.toStringAsFixed(2)} → ${node.mdAfter.toStringAsFixed(2)}',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
            ],
          ),
          subtitle: Text(
            [
              if (maskRatio != null) '遮罩率 ${(maskRatio * 100).toStringAsFixed(0)}%',
              node.action,
              if (node.action == 'Revised' && node.effective) '有效',
              if (node.action == 'Revised' && !node.effective) '无效',
            ].join(' · '),
            style: const TextStyle(fontSize: 11),
          ),
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (node.original.isNotEmpty) ...[
                    Text('图原文', style: TextStyle(color: scheme.primary, fontSize: 12)),
                    const SizedBox(height: 4),
                    SelectableText(node.original,
                        style: const TextStyle(fontSize: 12, height: 1.5)),
                  ],
                  if (node.maskedText != null) ...[
                    const SizedBox(height: 10),
                    Text('遮罩输入', style: const TextStyle(color: AppColors.warn, fontSize: 12)),
                    const SizedBox(height: 4),
                    SelectableText(node.maskedText!,
                        style: const TextStyle(fontSize: 12, height: 1.5)),
                  ],
                  if (node.llmReply != null) ...[
                    const SizedBox(height: 10),
                    Text('LLM 原始回复',
                        style: TextStyle(color: scheme.primary, fontSize: 12)),
                    const SizedBox(height: 4),
                    SelectableText(node.llmReply!,
                        style: const TextStyle(fontSize: 12, height: 1.5)),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 文本块卡片（遮罩输入 / LLM 回复）。
class _TextBlock extends StatelessWidget {
  final String title;
  final String text;
  final Color color;
  const _TextBlock({required this.title, required this.text, required this.color});

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
            Text(title,
                style: TextStyle(color: color, fontWeight: FontWeight.w600, fontSize: 13)),
            const SizedBox(height: 6),
            SelectableText(text, style: const TextStyle(height: 1.5, fontSize: 13)),
          ],
        ),
      ),
    );
  }
}
