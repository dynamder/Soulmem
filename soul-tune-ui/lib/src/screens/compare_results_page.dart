import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';
import '../widgets/stat_card.dart';

/// 对比结果页：聚合指标卡 + 逐用例对比表（可排序、横向滚动，点击展开详情）。
/// 侧标签由 kind 决定：embedding_full = Emb/Full；direct_db = 直接/DB。
class CompareResultsPage extends StatefulWidget {
  final CompareReport report;
  final String kind; // embedding_full | direct_db
  final String flavor; // full | embedding | association（direct_db 生效）
  const CompareResultsPage({
    super.key,
    required this.report,
    this.kind = 'embedding_full',
    this.flavor = 'full',
  });

  @override
  State<CompareResultsPage> createState() => _CompareResultsPageState();
}

class _CompareResultsPageState extends State<CompareResultsPage> {
  // 0=用例 1=A Hit 2=B Hit 3=ΔHit 4=A MRR 5=B MRR 6=ΔMRR 7=ΔR@3 8=ΔN@3
  int? _sortColumn;
  bool _sortAsc = true;

  CompareReport get report => widget.report;

  bool get _isDb => widget.kind == 'direct_db';
  String get _labelA => _isDb ? '直接' : 'Emb';
  String get _labelB => _isDb ? 'DB' : 'Full';
  String get _subtitle =>
      _isDb ? '同管线「直接(全量工作记忆) vs 数据库(DB 召回)」· ${widget.flavor}' : 'Embedding vs FullPipeline';

  List<CompareCase> get _sorted {
    final list = [...report.cases];
    list.sort((a, b) {
      double val(CompareCase c, int col) => switch (col) {
            1 => c.embeddingHit,
            2 => c.fullpipelineHit,
            3 => c.hitDelta,
            4 => c.embeddingMrr,
            5 => c.fullpipelineMrr,
            6 => c.mrrDelta,
            7 => c.sideBRecallAt(3) - c.sideARecallAt(3),
            8 => _valueAt(c.fullpipelineNdcgAt, 3) - _valueAt(c.embeddingNdcgAt, 3),
            _ => 0,
          };
      final cmp = _sortColumn == null || _sortColumn == 0
          ? a.caseName.compareTo(b.caseName)
          : val(a, _sortColumn!).compareTo(val(b, _sortColumn!));
      return _sortAsc ? cmp : -cmp;
    });
    return list;
  }

  static double _valueAt(List<(int, double)> pairs, int k) {
    for (final (kk, v) in pairs) {
      if (kk == k) return v;
    }
    return 0;
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
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Text(_subtitle,
                style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
          ),
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
                '$_labelB 相对 $_labelA：Hit ${_delta(hitDelta)} · MRR ${_delta(mrrDelta)}',
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

  // ── 逐用例对比表（固定列宽 + 横向滚动）──
  static const _colWidths = [250, 84, 84, 78, 92, 92, 82, 78, 78];

  List<String> _headers() => [
        '用例',
        '$_labelA Hit',
        '$_labelB Hit',
        'ΔHit',
        '$_labelA MRR',
        '$_labelB MRR',
        'ΔMRR',
        'ΔR@3',
        'ΔN@3',
      ];

  Widget _headerCell(String label, int col, int width) => SizedBox(
        width: width.toDouble(),
        child: InkWell(
          onTap: () => _onSort(col),
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
                          color: _sortColumn == col
                              ? Theme.of(context).colorScheme.primary
                              : Theme.of(context).colorScheme.onSurfaceVariant,
                          fontWeight:
                              _sortColumn == col ? FontWeight.w700 : FontWeight.w500,
                        ),
                  ),
                ),
                if (_sortColumn == col)
                  Icon(
                    _sortAsc ? Icons.arrow_drop_up : Icons.arrow_drop_down,
                    size: 16,
                    color: Theme.of(context).colorScheme.primary,
                  ),
              ],
            ),
          ),
        ),
      );

  Widget _buildTable() {
    final scheme = Theme.of(context).colorScheme;
    final headers = _headers();
    final items = _sorted;
    final totalWidth =
        _colWidths.fold<int>(0, (a, w) => a + w) + 24.0; // 两侧留白

    Widget table() => SizedBox(
          width: totalWidth,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: Row(
                  children: [
                    for (var col = 0; col < headers.length; col++)
                      _headerCell(headers[col], col, _colWidths[col]),
                  ],
                ),
              ),
              const Divider(height: 1),
              if (items.isEmpty)
                const Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('（无数据）', style: TextStyle(color: AppColors.subtle)),
                ),
              for (var i = 0; i < items.length; i++) _caseRow(items[i], i, scheme),
            ],
          ),
        );

    return SingleChildScrollView(
      scrollDirection: Axis.vertical,
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: table(),
      ),
    );
  }

  Widget _cell(Widget child, int width, {Color? bg}) => Container(
        width: width.toDouble(),
        color: bg,
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 10),
        alignment: Alignment.centerRight,
        child: child,
      );

  Widget _caseRow(CompareCase c, int i, ColorScheme scheme) {
    final improved = c.improvedHit || c.improvedMrr;
    final regressed = c.regressedHit || c.regressedMrr;
    final rowBg = improved
        ? AppColors.passBg.withValues(alpha: 0.45)
        : regressed
            ? AppColors.failBg.withValues(alpha: 0.35)
            : i.isOdd
                ? scheme.surfaceContainerHigh
                : Colors.transparent;
    final cells = <Widget>[
      Container(
        width: _colWidths[0].toDouble(),
        color: rowBg,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        alignment: Alignment.centerLeft,
        child: Text(c.caseName,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.bodySmall),
      ),
      _cell(_num(c.embeddingHit.toStringAsFixed(2)), _colWidths[1], bg: rowBg),
      _cell(_num(c.fullpipelineHit.toStringAsFixed(2)), _colWidths[2], bg: rowBg),
      _cell(_deltaCell(c.hitDelta), _colWidths[3], bg: rowBg),
      _cell(_num(c.embeddingMrr.toStringAsFixed(4)), _colWidths[4], bg: rowBg),
      _cell(_num(c.fullpipelineMrr.toStringAsFixed(4)), _colWidths[5], bg: rowBg),
      _cell(_deltaCell(c.mrrDelta), _colWidths[6], bg: rowBg),
      _cell(
          _deltaCell(c.sideBRecallAt(3) - c.sideARecallAt(3)), _colWidths[7],
          bg: rowBg),
      _cell(
          _deltaCell(_valueAt(c.fullpipelineNdcgAt, 3) -
              _valueAt(c.embeddingNdcgAt, 3)),
          _colWidths[8],
          bg: rowBg),
    ];

    return InkWell(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
            builder: (_) => CompareCaseDetailPage(
                caseData: c, labelA: _labelA, labelB: _labelB)),
      ),
      child: Row(children: cells),
    );
  }

  Widget _num(String s) => Text(s,
      textAlign: TextAlign.right,
      style: const TextStyle(fontFamily: 'monospace', fontSize: 12));

  Widget _deltaCell(double delta) {
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

/// 双模式对比统计卡：base(侧 A) → side(侧 B)，含提升徽章。
class _DeltaStatCard extends StatelessWidget {
  final String label;
  final double base;
  final double full;
  final String Function(double) fmt;
  const _DeltaStatCard(
      {required this.label, required this.base, required this.full, required this.fmt});

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
            Text(label,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: scheme.onSurfaceVariant)),
            const SizedBox(height: 6),
            Row(
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(fmt(full),
                    style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 24,
                        fontWeight: FontWeight.w700)),
                const SizedBox(width: 6),
                Text('(${fmt(base)})',
                    style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 13,
                        color: scheme.onSurfaceVariant)),
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

/// 单用例对比详情（对标检索钻取 CaseDetailPage 的分节组织）：
/// 状态徽章 + 综合指标矩阵(双侧+Δ) + 各子查询双侧指标 + 两侧检索列表 vs 期望命中高亮。
class CompareCaseDetailPage extends StatelessWidget {
  final CompareCase caseData;
  final String labelA;
  final String labelB;
  const CompareCaseDetailPage({
    super.key,
    required this.caseData,
    required this.labelA,
    required this.labelB,
  });

  static double _pairAt(List<(int, double)> pairs, int k) {
    for (final (kk, v) in pairs) {
      if (kk == k) return v;
    }
    return 0;
  }

  static List<int> _unionKs(List<(int, double)> a, List<(int, double)> b) {
    final ks = <int>{};
    for (final (k, _) in a) {
      ks.add(k);
    }
    for (final (k, _) in b) {
      ks.add(k);
    }
    final list = ks.toList()..sort();
    return list;
  }

  // 用例级综合指标：双值行 + Δ
  List<Widget> _combinedRows(BuildContext context, CompareCase c) {
    final ks = _unionKs(c.embeddingRecallAt, c.fullpipelineRecallAt);
    return [
      _metricRow(context, 'Hit', c.embeddingHit, c.fullpipelineHit, digits: 2),
      _metricRow(context, 'MRR', c.embeddingMrr, c.fullpipelineMrr, digits: 4),
      for (final k in ks) ...[
        _metricRow(
            context,
            'Recall@$k',
            _pairAt(c.embeddingRecallAt, k),
            _pairAt(c.fullpipelineRecallAt, k),
            digits: 2),
        _metricRow(
            context,
            'Precision@$k',
            _pairAt(c.embeddingPrecisionAt, k),
            _pairAt(c.fullpipelinePrecisionAt, k),
            digits: 2),
        _metricRow(
            context,
            'NDCG@$k',
            _pairAt(c.embeddingNdcgAt, k),
            _pairAt(c.fullpipelineNdcgAt, k),
            digits: 2),
      ],
    ];
  }

  List<Widget> _perQueryRows(BuildContext context, CompareCasePerQuery pq) {
    final ks = _unionKs(pq.embeddingRecallAt, pq.fullpipelineRecallAt);
    return [
      _metricRow(context, 'MRR', pq.embeddingMrr, pq.fullpipelineMrr, digits: 4),
      _metricRow(context, 'Hit', pq.embeddingHit, pq.fullpipelineHit, digits: 2),
      for (final k in ks) ...[
        _metricRow(
            context,
            'Recall@$k',
            _pairAt(pq.embeddingRecallAt, k),
            _pairAt(pq.fullpipelineRecallAt, k),
            digits: 2),
        _metricRow(
            context,
            'Precision@$k',
            _pairAt(pq.embeddingPrecisionAt, k),
            _pairAt(pq.fullpipelinePrecisionAt, k),
            digits: 2),
        _metricRow(
            context,
            'NDCG@$k',
            _pairAt(pq.embeddingNdcgAt, k),
            _pairAt(pq.fullpipelineNdcgAt, k),
            digits: 2),
      ],
    ];
  }

  /// 单行双值指标：label | A 值 | B 值 | Δ（右对齐 monospace，Δ 着色）
  Widget _metricRow(BuildContext context, String label, double a, double b,
      {int digits = 2}) {
    final delta = b - a;
    final dColor = delta > 0.0001
        ? AppColors.pass
        : delta < -0.0001
            ? AppColors.fail
            : AppColors.subtle;
    String fmt(double v) => v.toStringAsFixed(digits);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          SizedBox(
            width: 96,
            child: Text(label, style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
          ),
          Expanded(
            flex: 1,
            child: Text('$labelA ${fmt(a)}',
                textAlign: TextAlign.right,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 6),
            child: Text('|', style: TextStyle(color: AppColors.subtle)),
          ),
          Expanded(
            flex: 1,
            child: Text('$labelB ${fmt(b)}',
                textAlign: TextAlign.right,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
          SizedBox(
            width: 76,
            child: Text(
              '${delta > 0 ? '▲' : delta < 0 ? '▼' : '—'} ${delta.abs().toStringAsFixed(2)}',
              textAlign: TextAlign.right,
              style: TextStyle(
                  fontFamily: 'monospace', fontSize: 12, color: dColor),
            ),
          ),
        ],
      ),
    );
  }

  /// 命中高亮列表：命中期望的项绿底 + ✓；顶部给命中计数。
  Widget _hitList(BuildContext context, List<String> items, List<String> expected) {
    if (items.isEmpty) {
      return const Text('（空）', style: TextStyle(color: AppColors.subtle));
    }
    final hits = items.where(expected.contains).length;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('命中 $hits/${items.length} · 期望 ${expected.length}',
            style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
        const SizedBox(height: 6),
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

  Widget _expectedChips(List<String> expected) {
    if (expected.isEmpty) {
      return const Text('（无期望命中）', style: TextStyle(color: AppColors.subtle));
    }
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        for (final e in expected)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: AppColors.runningBg,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(e,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = caseData;
    final netDelta = c.hitDelta + c.mrrDelta;
    return Scaffold(
      appBar: AppBar(
        title: Text(c.caseName, maxLines: 1, overflow: TextOverflow.ellipsis),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: _ResultChip(
                improved: netDelta > 0.0001,
                regressed: netDelta < -0.0001,
              ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('$labelA vs $labelB · ${c.description.isEmpty ? c.caseName : c.description}',
              style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
          const SizedBox(height: 12),
          _Section(
            title: '综合指标对比',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: _combinedRows(context, c),
            ),
          ),
          if (c.perQuery.isNotEmpty)
            _Section(
              title: '各子查询对比',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final pq in c.perQuery) ...[
                    Text('Q${pq.queryIndex}',
                        style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: Theme.of(context).colorScheme.primary)),
                    const SizedBox(height: 2),
                    ..._perQueryRows(context, pq),
                    const SizedBox(height: 10),
                  ],
                ],
              ),
            ),
          _Section(
            title: '检索列表（$labelA）vs 期望',
            child: _hitList(context, c.embeddingRetrieved, c.expected),
          ),
          _Section(
            title: '检索列表（$labelB）vs 期望',
            child: _hitList(context, c.fullpipelineRetrieved, c.expected),
          ),
          _Section(title: '期望命中', child: _expectedChips(c.expected)),
          Text(
            'tag_weight=${c.tagWeight}  variant_weight=${c.variantWeight}',
            style: const TextStyle(color: AppColors.subtle, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

/// 用例整体结果徽章：按 Hit/MRR 净差量给出 提升/回退/持平。
class _ResultChip extends StatelessWidget {
  final bool improved;
  final bool regressed;
  const _ResultChip({required this.improved, required this.regressed});

  @override
  Widget build(BuildContext context) {
    final (fg, bg, label, icon) = improved
        ? (AppColors.pass, AppColors.passBg, '提升', Icons.trending_up)
        : regressed
            ? (AppColors.fail, AppColors.failBg, '回退', Icons.trending_down)
            : (AppColors.subtle, AppColors.runningBg, '持平', Icons.arrow_right_alt);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(20)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: fg),
          const SizedBox(width: 4),
          Text(label,
              style: TextStyle(fontSize: 12, color: fg, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
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
