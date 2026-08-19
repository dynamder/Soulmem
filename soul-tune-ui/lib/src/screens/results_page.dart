import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';
import '../widgets/metric_panel.dart';
import '../widgets/mini_bar_chart.dart';
import '../widgets/pass_rate_donut.dart';
import '../widgets/stat_card.dart';
import 'case_detail_page.dart';

/// 结果页（GUI 化）：
/// - 总览：通过率环形图 + 统计卡 + 指标卡片网格 + 逐用例状态条
/// - 明细：可排序表格（搜索/筛选/列头排序/点击钻取）
class ResultsPage extends StatefulWidget {
  final Report report;
  const ResultsPage({super.key, required this.report});

  @override
  State<ResultsPage> createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  int _tab = 0; // 0=总览 1=明细
  String _query = '';
  String _filter = 'all'; // all | passed | failed
  int? _sortColumn; // 0=用例 1=MRR 2=Hit 3=状态
  bool _sortAsc = true;

  Report get report => widget.report;

  double _mrr(Outcome o) =>
      ((o.data['combined_ranking_metrics'] as Map?)?['mrr'] as num?)?.toDouble() ?? 0;
  double _hit(Outcome o) =>
      ((o.data['combined_ranking_metrics'] as Map?)?['hit_rate'] as num?)?.toDouble() ?? 0;

  List<Outcome> get _filtered {
    var list = report.outcomes.where((o) {
      if (_filter == 'passed' && !o.passed) return false;
      if (_filter == 'failed' && o.passed) return false;
      if (_query.isNotEmpty && !o.caseName.toLowerCase().contains(_query.toLowerCase())) {
        return false;
      }
      return true;
    }).toList();
    list.sort((a, b) {
      final cmp = switch (_sortColumn) {
        1 => _mrr(b).compareTo(_mrr(a)),
        2 => _hit(b).compareTo(_hit(a)),
        3 => (a.passed ? 1 : 0).compareTo(b.passed ? 1 : 0),
        _ => a.caseName.compareTo(b.caseName),
      };
      return _sortAsc ? cmp : -cmp;
    });
    return list;
  }

  void _onSort(int columnIndex) {
    setState(() {
      if (_sortColumn == columnIndex) {
        _sortAsc = !_sortAsc;
      } else {
        _sortColumn = columnIndex;
        _sortAsc = true;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final r = report;
    return Scaffold(
      appBar: AppBar(
        title: Text(r.datasetName.isEmpty ? r.algo : '${r.algo} · ${r.datasetName}'),
      ),
      body: Column(
        children: [
          // 摘要条：环形图 + 统计卡（紧凑）
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 2),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 100,
                  height: 100,
                  child: PassRateDonut(rate: r.passRate, size: 100),
                ),
                const SizedBox(width: 20),
                Flexible(
                  child: Wrap(
                    spacing: 10,
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
                ButtonSegment(value: 0, label: Text('总览')),
                ButtonSegment(value: 1, label: Text('明细')),
              ],
              selected: {_tab},
              onSelectionChanged: (s) => setState(() => _tab = s.first),
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: _tab == 0
                ? _buildSummary()
                : _buildDetail(),
          ),
        ],
      ),
    );
  }

  Widget _buildSummary() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('指标', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        MetricPanel(metrics: report.metrics),
        const SizedBox(height: 8),
        Text('逐用例 通过/失败', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        MiniBarChart(passFlags: report.outcomes.map((o) => o.passed).toList()),
        if (report.detailHeader.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(report.detailHeader, style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
        ],
      ],
    );
  }

  Widget _buildDetail() {
    final items = _filtered;
    final scheme = Theme.of(context).colorScheme;

    Widget headerCell(String label, int col, {double flex = 1}) => Expanded(
          flex: flex ~/ 1,
          child: InkWell(
            onTap: () => _onSort(col),
            borderRadius: BorderRadius.circular(6),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Flexible(
                    child: Text(
                      label,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                            color: _sortColumn == col ? scheme.primary : scheme.onSurfaceVariant,
                            fontWeight: _sortColumn == col ? FontWeight.w700 : FontWeight.w500,
                          ),
                    ),
                  ),
                  if (_sortColumn == col)
                    Icon(
                      _sortAsc ? Icons.arrow_drop_up : Icons.arrow_drop_down,
                      size: 18,
                      color: scheme.primary,
                    ),
                ],
              ),
            ),
          ),
        );

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: const InputDecoration(
                    hintText: '搜索用例…',
                    prefixIcon: Icon(Icons.search),
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                  onChanged: (v) => setState(() => _query = v),
                ),
              ),
              const SizedBox(width: 10),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'all', label: Text('全部')),
                  ButtonSegment(value: 'passed', label: Text('通过')),
                  ButtonSegment(value: 'failed', label: Text('失败')),
                ],
                selected: {_filter},
                onSelectionChanged: (s) => setState(() => _filter = s.first),
              ),
            ],
          ),
        ),
        Expanded(
          child: items.isEmpty
              ? const Center(child: Text('（无匹配用例）', style: TextStyle(color: AppColors.subtle)))
              : Card(
                  elevation: 0,
                  color: scheme.surfaceContainerLow,
                  margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                  clipBehavior: Clip.antiAlias,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  child: Column(
                    children: [
                      // 表头
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        child: Row(
                          children: [
                            headerCell('用例', 0, flex: 4),
                            headerCell('MRR', 1, flex: 2),
                            headerCell('Hit', 2, flex: 2),
                            headerCell('状态', 3, flex: 2),
                          ],
                        ),
                      ),
                      const Divider(height: 1),
                      // 行
                      Expanded(
                        child: ListView.builder(
                          itemCount: items.length,
                          itemBuilder: (context, i) {
                            final o = items[i];
                            final stripe = i.isOdd ? scheme.surfaceContainerHigh : Colors.transparent;
                            return InkWell(
                              onTap: () => Navigator.push(
                                context,
                                MaterialPageRoute(builder: (_) => CaseDetailPage(outcome: o)),
                              ),
                              child: Container(
                                color: stripe,
                                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                                child: Row(
                                  children: [
                                    Expanded(
                                      flex: 4,
                                      child: Text(o.caseName,
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                          style: Theme.of(context).textTheme.bodyMedium),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Text(_mrr(o).toStringAsFixed(4),
                                          textAlign: TextAlign.right,
                                          style: const TextStyle(fontFamily: 'monospace')),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Text(_hit(o).toStringAsFixed(2),
                                          textAlign: TextAlign.right,
                                          style: const TextStyle(fontFamily: 'monospace')),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Row(
                                        mainAxisAlignment: MainAxisAlignment.end,
                                        children: [
                                          _StatusChip(passed: o.passed),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
        ),
      ],
    );
  }
}

/// 柔和状态徽章：低饱和底色 + 状态文字。
class _StatusChip extends StatelessWidget {
  final bool passed;
  const _StatusChip({required this.passed});

  @override
  Widget build(BuildContext context) {
    final fg = passed ? AppColors.pass : AppColors.fail;
    final bg = passed ? AppColors.passBg : AppColors.failBg;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(passed ? Icons.check : Icons.close, size: 14, color: fg),
          const SizedBox(width: 4),
          Text(passed ? '通过' : '失败',
              style: TextStyle(fontSize: 12, color: fg, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}
