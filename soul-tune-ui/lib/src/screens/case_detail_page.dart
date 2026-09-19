import 'package:flutter/material.dart';

import '../models.dart';
import '../theme.dart';

/// 单用例钻取详情（GUI 化）：
/// 分节卡片 + 柔和状态徽章；检索 vs 期望对比中命中行柔和高亮。
/// 数据源为 RetrieveCaseData 的 JSON（Rust serde 序列化输出）。
class CaseDetailPage extends StatelessWidget {
  final Outcome outcome;
  const CaseDetailPage({super.key, required this.outcome});

  @override
  Widget build(BuildContext context) {
    final d = outcome.data;
    final ranking = (d['combined_ranking_metrics'] as Map?)?.cast<String, dynamic>() ?? const {};
    final perQuery =
        (d['per_query_metrics'] as List?)?.cast<Map<String, dynamic>>() ?? const [];
    final action = (d['action_metrics'] as Map?)?.cast<String, dynamic>();
    final retrieved = (d['combined_retrieved_ids'] as List?)?.cast<dynamic>() ?? const [];
    final expected = (d['expected_combined_ranking'] as List?)?.cast<dynamic>() ?? const [];
    final names = (d['graph_names'] as Map?)?.cast<String, dynamic>() ?? const {};
    final expectedNames = (d['id_names'] as Map?)?.cast<String, dynamic>() ?? const {};

    String nameOf(Object? id) {
      if (id == null) return '—';
      final s = id.toString();
      return (names[s] ?? expectedNames[s] ?? s).toString();
    }

    String fmtKList(List? pairs) => (pairs ?? [])
        .map((p) => p is List && p.length >= 2 ? '${p[0]} → ${p[1]}' : '')
        .join('    ');

    return Scaffold(
      appBar: AppBar(
        title: Text(outcome.caseName, maxLines: 1, overflow: TextOverflow.ellipsis),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(child: _StatusChip(passed: outcome.passed)),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (outcome.description.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Text(outcome.description, style: const TextStyle(color: AppColors.subtle)),
            ),
          _SectionCard(
            title: '综合排序指标',
            children: _kvRows([
              ('MRR', (ranking['mrr'] as num?)?.toStringAsFixed(4) ?? '—'),
              ('Hit Rate', (ranking['hit_rate'] as num?)?.toStringAsFixed(2) ?? '—'),
              ('Recall@K', fmtKList(ranking['recall_at'] as List?)),
              ('Precision@K', fmtKList(ranking['precision_at'] as List?)),
              ('NDCG@K', fmtKList(ranking['ndcg_at'] as List?)),
            ]),
          ),
          if (perQuery.isNotEmpty)
            _SectionCard(
              title: '各子查询',
              children: _kvRows([
                for (final pq in perQuery)
                  (
                    'Q${pq['query_index']}',
                    'MRR ${((pq['ranking_metrics'] as Map?)?['mrr'] as num?)?.toStringAsFixed(4) ?? '—'}'
                        ' · Hit ${((pq['ranking_metrics'] as Map?)?['hit_rate'] as num?)?.toStringAsFixed(2) ?? '—'}',
                  ),
              ]),
            ),
          _SectionCard(
            title: '检索结果 vs 期望',
            children: [
              for (var i = 0; i < retrieved.length || i < expected.length; i++)
                _CompareRow(
                  position: i + 1,
                  retrieved: i < retrieved.length ? nameOf(retrieved[i]) : null,
                  expected: i < expected.length ? nameOf(expected[i]) : null,
                  hit: i < retrieved.length && expected.contains(retrieved[i]),
                ),
            ],
          ),
          if (action != null && action['has_expected_actions'] == true)
            _SectionCard(
              title: '动作指标',
              children: _kvRows([
                ('动作 Hit', (action['action_hit_rate'] as num?)?.toStringAsFixed(2) ?? '—'),
                ('动作 Recall@K', fmtKList(action['action_recall_at'] as List?)),
              ]),
            ),
          if (d['has_expected_abstract'] == true)
            _SectionCard(
              title: '抽象检出',
              children: _kvRows([
                ('检出（合并结果）', d['abstract_detected'] == true ? '是' : '否'),
                ('相似度直接命中', d['abstract_direct_hit'] == true ? '是' : '否'),
              ]),
            ),
          const SizedBox(height: 8),
          Text(
            'tag_weight=${d['tag_weight']}  variant_weight=${d['variant_weight']}',
            style: const TextStyle(color: AppColors.subtle, fontSize: 12),
          ),
        ],
      ),
    );
  }

  List<Widget> _kvRows(List<(String, String)> rows) => [
        for (final (k, v) in rows)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(
                  width: 120,
                  child: Text(k, style: const TextStyle(color: AppColors.subtle)),
                ),
                Expanded(
                  child: Text(v, style: const TextStyle(fontFamily: 'monospace', fontSize: 13)),
                ),
              ],
            ),
          ),
      ];
}

/// 分节卡片：左侧强调条 + 标题 + 内容。
class _SectionCard extends StatelessWidget {
  final String title;
  final List<Widget> children;
  const _SectionCard({required this.title, required this.children});

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
                    color: scheme.primary,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(width: 8),
                Text(title, style: Theme.of(context).textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 10),
            ...children,
          ],
        ),
      ),
    );
  }
}

/// 检索 vs 期望对比行：命中行柔和绿底。
class _CompareRow extends StatelessWidget {
  final int position;
  final String? retrieved;
  final String? expected;
  final bool hit;
  const _CompareRow({
    required this.position,
    required this.retrieved,
    required this.expected,
    required this.hit,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 3),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: hit ? AppColors.passBg : Colors.transparent,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 40,
            child: Text('#$position', style: const TextStyle(color: AppColors.subtle, fontSize: 12)),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.arrow_back, size: 12, color: AppColors.subtle),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(retrieved ?? '—',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontFamily: 'monospace', fontSize: 13)),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                Row(
                  children: [
                    const Icon(Icons.arrow_forward, size: 12, color: AppColors.subtle),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(expected ?? '—',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 13,
                            color: hit ? AppColors.pass : null,
                          )),
                    ),
                  ],
                ),
              ],
            ),
          ),
          if (hit)
            const Icon(Icons.check_circle_outline, size: 16, color: AppColors.pass),
        ],
      ),
    );
  }
}

/// 柔和状态徽章。
class _StatusChip extends StatelessWidget {
  final bool passed;
  const _StatusChip({required this.passed});

  @override
  Widget build(BuildContext context) {
    final fg = passed ? AppColors.pass : AppColors.fail;
    final bg = passed ? AppColors.passBg : AppColors.failBg;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(20)),
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
