import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';
import '../widgets/stat_card.dart';

/// 对比结果页：聚合指标卡 + 逐用例对比表（可排序，点击展开详情）。
class CompareResultsPage extends StatefulWidget {
  final CompareReport report;
  const CompareResultsPage({super.key, required this.report});

  @override
  State<CompareResultsPage> createState() => _CompareResultsPageState();
}

class _CompareResultsPageState extends State<CompareResultsPage> {
  int? _sortColumn; // 0=用例 1=EmbHit 2=FullHit 3=ΔHit 4=EmbMRR 5=FullMRR 6=ΔMRR
  bool _sortAsc = true;

  CompareReport get report => widget.report;

  List<CompareCase> get _sorted {
    final list = [...report.cases];
    list.sort((a, b) {
      final cmp = switch (_sortColumn) {
        1 => a.embeddingHit.compareTo(b.embeddingHit),
        2 => a.fullpipelineHit.compareTo(b.fullpipelineHit),
        3 => a.hitDelta.compareTo(b.hitDelta),
        4 => a.embeddingMrr.compareTo(b.embeddingMrr),
        5 => a.fullpipelineMrr.compareTo(b.fullpipelineMrr),
        6 => a.mrrDelta.compareTo(b.mrrDelta),
        _ => a.caseName.compareTo(b.caseName),
      };
      return _sortAsc ? cmp : -cmp;
    });
    return list;
  }

  void _onSort(int col) {
    setState(() {
      if (_sortColumn == col) {
        _sortAsc = !_sortAsc;
      } else {
        _sortColumn = col;
        _sortAsc = true;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final agg = report.aggregate;
    final hitDelta = agg.avgFullpipelineHit - agg.avgEmbeddingHit;
    final mrrDelta = agg.avgFullpipelineMrr - agg.avgEmbeddingMrr;
    return Scaffold(
      appBar: AppBar(title: Text('对比结果 · ${report.datasetName}')),
      body: Column(
        children: [
          // 聚合卡
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 4),
            child: Wrap(
              alignment: WrapAlignment.center,
              spacing: 12,
              runSpacing: 10,
              children: [
                _DeltaStatCard(
                  label: '平均 Hit',
                  base: agg.avgEmbeddingHit,
                  full: agg.avgFullpipelineHit,
                  fmt: (v) => v.toStringAsFixed(2),
                ),
                _DeltaStatCard(
                  label: '平均 MRR',
                  base: agg.avgEmbeddingMrr,
                  full: agg.avgFullpipelineMrr,
                  fmt: (v) => v.toStringAsFixed(4),
                ),
                StatCard(
                  label: 'Hit 提升用例',
                  value: '${agg.hitImprovementCount}/${agg.caseCount}',
                  valueColor: AppColors.pass,
                ),
                StatCard(
                  label: 'MRR 提升用例',
                  value: '${agg.mrrImprovementCount}/${agg.caseCount}',
                  valueColor: AppColors.pass,
                ),
              ],
            ),
          ),
          if (hitDelta != 0 || mrrDelta != 0)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Text(
                'FullPipeline 整体：Hit ${_delta(hitDelta)} · MRR ${_delta(mrrDelta)}',
                style: TextStyle(
                  color: (hitDelta >= 0 && mrrDelta >= 0) ? AppColors.pass : AppColors.warn,
                  fontSize: 12,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          const Divider(height: 16),
          Expanded(child: _buildTable()),
        ],
      ),
    );
  }

  String _delta(double v) =>
      '${v >= 0 ? '+' : ''}${v.toStringAsFixed(v.abs() < 0.01 ? 4 : 2)}';

  Widget _buildTable() {
    final scheme = Theme.of(context).colorScheme;
    final items = _sorted;

    Widget headerCell(String label, int col, {double flex = 1}) => Expanded(
          flex: flex ~/ 1,
          child: InkWell(
            onTap: () => _onSort(col),
            borderRadius: BorderRadius.circular(6),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 6),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Flexible(
                    child: Text(
                      label,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                            color: _sortColumn == col ? scheme.primary : scheme.onSurfaceVariant,
                            fontWeight: _sortColumn == col ? FontWeight.w700 : FontWeight.w500,
                          ),
                    ),
                  ),
                  if (_sortColumn == col)
                    Icon(
                      _sortAsc ? Icons.arrow_drop_up : Icons.arrow_drop_down,
                      size: 16,
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
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Row(
            children: [
              headerCell('用例', 0, flex: 4),
              headerCell('Emb Hit', 1, flex: 2),
              headerCell('Full Hit', 2, flex: 2),
              headerCell('ΔHit', 3, flex: 2),
              headerCell('Emb MRR', 4, flex: 2),
              headerCell('Full MRR', 5, flex: 2),
              headerCell('ΔMRR', 6, flex: 2),
            ],
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: items.isEmpty
              ? const Center(child: Text('（无数据）', style: TextStyle(color: AppColors.subtle)))
              : ListView.builder(
                  itemCount: items.length,
                  itemBuilder: (context, i) {
                    final c = items[i];
                    final improved = c.improvedHit || c.improvedMrr;
                    return InkWell(
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => CompareCaseDetailPage(caseData: c)),
                      ),
                      child: Container(
                        color: improved
                            ? AppColors.passBg.withValues(alpha: 0.5)
                            : i.isOdd
                                ? scheme.surfaceContainerHigh
                                : Colors.transparent,
                        padding:
                            const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                        child: Row(
                          children: [
                            Expanded(
                              flex: 4,
                              child: Text(c.caseName,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: Theme.of(context).textTheme.bodySmall),
                            ),
                            Expanded(flex: 2, child: _num(c.embeddingHit.toStringAsFixed(2))),
                            Expanded(flex: 2, child: _num(c.fullpipelineHit.toStringAsFixed(2))),
                            Expanded(flex: 2, child: _deltaCell(c.hitDelta, c.improvedHit)),
                            Expanded(flex: 2, child: _num(c.embeddingMrr.toStringAsFixed(4))),
                            Expanded(flex: 2, child: _num(c.fullpipelineMrr.toStringAsFixed(4))),
                            Expanded(flex: 2, child: _deltaCell(c.mrrDelta, c.improvedMrr)),
                          ],
                        ),
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }

  Widget _num(String s) => Text(s,
      textAlign: TextAlign.right,
      style: const TextStyle(fontFamily: 'monospace', fontSize: 12));

  Widget _deltaCell(double delta, bool improved) {
    final color = delta > 0.0001
        ? AppColors.pass
        : delta < -0.0001
            ? AppColors.fail
            : AppColors.subtle;
    return Text(
      '${delta > 0 ? '▲' : delta < 0 ? '▼' : '—'} ${delta.abs().toStringAsFixed(2)}',
      textAlign: TextAlign.right,
      style: TextStyle(fontFamily: 'monospace', fontSize: 12, color: color),
    );
  }
}

/// 双模式对比统计卡：base(embedding) → full，含提升徽章。
class _DeltaStatCard extends StatelessWidget {
  final String label;
  final double base;
  final double full;
  final String Function(double) fmt;
  const _DeltaStatCard({required this.label, required this.base, required this.full, required this.fmt});

  @override
  Widget build(BuildContext context) {
    final delta = full - base;
    final improved = delta > 0.0001;
    final regressed = delta < -0.0001;
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(label, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant)),
            const SizedBox(height: 6),
            Row(
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(fmt(full),
                    style: const TextStyle(
                        fontFamily: 'monospace', fontSize: 24, fontWeight: FontWeight.w700)),
                const SizedBox(width: 6),
                Text('(${fmt(base)})',
                    style: TextStyle(fontFamily: 'monospace', fontSize: 13, color: scheme.onSurfaceVariant)),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: improved
                        ? AppColors.passBg
                        : regressed
                            ? AppColors.failBg
                            : AppColors.runningBg,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${improved ? '▲' : regressed ? '▼' : '—'} ${delta.abs().toStringAsFixed(2)}',
                    style: TextStyle(
                      fontSize: 12,
                      fontFamily: 'monospace',
                      color: improved
                          ? AppColors.pass
                          : regressed
                              ? AppColors.fail
                              : AppColors.subtle,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// 单用例对比详情：两种模式的检索列表 vs 期望。
class CompareCaseDetailPage extends StatelessWidget {
  final CompareCase caseData;
  const CompareCaseDetailPage({super.key, required this.caseData});

  @override
  Widget build(BuildContext context) {
    final c = caseData;
    return Scaffold(
      appBar: AppBar(title: Text(c.caseName, maxLines: 1, overflow: TextOverflow.ellipsis)),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _Section(title: '指标对比', child: _kvRows([
            ('Emb Hit / Full Hit', '${c.embeddingHit.toStringAsFixed(2)} / ${c.fullpipelineHit.toStringAsFixed(2)}'),
            ('Emb MRR / Full MRR', '${c.embeddingMrr.toStringAsFixed(4)} / ${c.fullpipelineMrr.toStringAsFixed(4)}'),
            ('Emb Recall@K', _fmtPairs(c.embeddingRecallAt)),
            ('Full Recall@K', _fmtPairs(c.fullpipelineRecallAt)),
          ])),
          _Section(title: '检索列表（embedding）', child: _list(c.embeddingRetrieved, c.expected)),
          _Section(title: '检索列表（full pipeline）', child: _list(c.fullpipelineRetrieved, c.expected)),
          _Section(title: '期望命中', child: _list(c.expected, c.expected)),
          Text('tag_weight=${c.tagWeight}  variant_weight=${c.variantWeight}',
              style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
        ],
      ),
    );
  }

  String _fmtPairs(List<(int, double)> pairs) =>
      pairs.map((p) => '${p.$1}→${p.$2.toStringAsFixed(2)}').join('  ');

  Widget _list(List<String> items, List<String> expected) {
    if (items.isEmpty) {
      return const Text('（空）', style: TextStyle(color: AppColors.subtle));
    }
    return Column(
      children: [
        for (var i = 0; i < items.length; i++)
          Container(
            margin: const EdgeInsets.symmetric(vertical: 2),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: expected.contains(items[i]) ? AppColors.passBg : Colors.transparent,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 36,
                  child: Text('#${i + 1}',
                      style: const TextStyle(fontSize: 11, color: AppColors.subtle)),
                ),
                Expanded(
                  child: Text(items[i],
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 13)),
                ),
                if (expected.contains(items[i]))
                  const Icon(Icons.check_circle_outline, size: 14, color: AppColors.pass),
              ],
            ),
          ),
      ],
    );
  }

  Widget _kvRows(List<(String, String)> rows) => Column(
        children: [
          for (final (k, v) in rows)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 3),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    width: 150,
                    child: Text(k, style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
                  ),
                  Expanded(
                    child: Text(v,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                  ),
                ],
              ),
            ),
        ],
      );
}

class _Section extends StatelessWidget {
  final String title;
  final Widget child;
  const _Section({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      margin: const EdgeInsets.only(bottom: 14),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 3,
                  height: 14,
                  decoration: BoxDecoration(
                      color: scheme.primary, borderRadius: BorderRadius.circular(2)),
                ),
                const SizedBox(width: 8),
                Text(title, style: Theme.of(context).textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 10),
            child,
          ],
        ),
      ),
    );
  }
}
