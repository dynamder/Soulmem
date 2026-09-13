import 'package:flutter/material.dart';

import '../models.dart';

/// 指标分组卡片网格（GUI 化，替代 TUI 的平铺 kv 文本列表）：
/// 每个指标组一张卡片，内部指标以圆角磁贴展示；图表指标绘制迷你折线。
class MetricPanel extends StatelessWidget {
  final List<MetricEntry> metrics;
  const MetricPanel({super.key, required this.metrics});

  @override
  Widget build(BuildContext context) {
    final groups = <String, List<MetricEntry>>{};
    for (final m in metrics) {
      groups.putIfAbsent(m.group, () => []).add(m);
    }
    final keys = groups.keys.toList()..sort();
    if (keys.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(16),
        child: Text('（无指标）', style: TextStyle(color: Colors.grey)),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (final g in keys) ...[
          _GroupCard(group: g, entries: groups[g]!),
          const SizedBox(height: 12),
        ],
      ],
    );
  }
}

class _GroupCard extends StatelessWidget {
  final String group;
  final List<MetricEntry> entries;
  const _GroupCard({required this.group, required this.entries});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
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
                  width: 3,
                  height: 14,
                  decoration: BoxDecoration(
                    color: scheme.primary,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(width: 8),
                Text(group, style: Theme.of(context).textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                for (final m in entries)
                  if (m.kind == 'chart')
                    _ChartTile(entry: m)
                  else
                    _KvTile(entry: m),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _KvTile extends StatelessWidget {
  final MetricEntry entry;
  const _KvTile({required this.entry});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: 190,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            entry.label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
          ),
          const SizedBox(height: 4),
          Text(
            entry.value ?? '',
            style: const TextStyle(fontFamily: 'monospace', fontSize: 17, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

class _ChartTile extends StatelessWidget {
  final MetricEntry entry;
  const _ChartTile({required this.entry});

  @override
  Widget build(BuildContext context) {
    final series = entry.datasets ?? const [];
    final points = series.isEmpty ? const <(double, double)>[] : series.first.points;
    return Container(
      width: 320,
      height: 96,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(entry.label,
              style: Theme.of(context).textTheme.bodySmall,
              maxLines: 1,
              overflow: TextOverflow.ellipsis),
          const SizedBox(height: 6),
          Expanded(child: _MiniLineChart(points: points)),
        ],
      ),
    );
  }
}

class _MiniLineChart extends StatelessWidget {
  final List<(double, double)> points;
  const _MiniLineChart({required this.points});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _LinePainter(points, Theme.of(context).colorScheme.primary),
      child: const SizedBox.expand(),
    );
  }
}

class _LinePainter extends CustomPainter {
  final List<(double, double)> points;
  final Color color;
  _LinePainter(this.points, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 2) return;
    final minX = points.map((p) => p.$1).reduce((a, b) => a < b ? a : b);
    final maxX = points.map((p) => p.$1).reduce((a, b) => a > b ? a : b);
    final maxY = points.map((p) => p.$2).reduce((a, b) => a > b ? a : b).clamp(1e-6, double.infinity);
    final dx = (maxX - minX).abs() < 1e-9 ? 1.0 : maxX - minX;
    Offset toOffset((double, double) p) => Offset(
          ((p.$1 - minX) / dx) * size.width,
          size.height - (p.$2 / maxY) * size.height,
        );
    final path = Path()..moveTo(toOffset(points[0]).dx, toOffset(points[0]).dy);
    for (var i = 1; i < points.length; i++) {
      path.lineTo(toOffset(points[i]).dx, toOffset(points[i]).dy);
    }
    final line = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    canvas.drawPath(path, line);
  }

  @override
  bool shouldRepaint(_LinePainter old) => old.points != points || old.color != color;
}
