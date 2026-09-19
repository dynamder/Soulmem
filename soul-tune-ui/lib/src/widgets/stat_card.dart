import 'package:flutter/material.dart';

/// 大数字统计卡：中性底色，数值默认用主题前景色；可通过 [valueColor] 柔和着色。
class StatCard extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;
  final String? unit;

  const StatCard({super.key, required this.label, required this.value, this.valueColor, this.unit});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
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
                Text(
                  value,
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    color: valueColor ?? scheme.onSurface,
                    height: 1,
                  ),
                ),
                if (unit != null) ...[
                  const SizedBox(width: 4),
                  Text(unit!, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant)),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}
