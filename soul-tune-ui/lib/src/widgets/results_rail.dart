import 'package:flutter/material.dart';

import '../theme.dart';
import 'pass_rate_donut.dart';

/// 结果页侧栏（instrument rail）：
/// 通过率圆环是栏内**唯一**的彩色元素，其余为等宽计数与纵向页签——
/// 占用窄条空间，把主区域留给密集数据。
class ResultsRail extends StatelessWidget {
  final double passRate;
  final int passed;
  final int failed;
  final double elapsedSecs;
  final int tab;
  final List<({String label, IconData icon})> tabs;
  final ValueChanged<int> onTab;

  const ResultsRail({
    super.key,
    required this.passRate,
    required this.passed,
    required this.failed,
    required this.elapsedSecs,
    required this.tab,
    required this.tabs,
    required this.onTab,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: 188,
      decoration: BoxDecoration(
        color: scheme.surfaceContainerLow,
        border: Border(
          right: BorderSide(color: scheme.outlineVariant, width: 0.5),
        ),
      ),
      child: Column(
        children: [
          const SizedBox(height: 18),
          SizedBox(width: 84, height: 84, child: PassRateDonut(rate: passRate, size: 84)),
          const SizedBox(height: 18),
          _StatRow(label: '通过', value: '$passed', color: AppColors.pass),
          _StatRow(label: '失败', value: '$failed', color: AppColors.fail),
          _StatRow(label: '耗时', value: '${elapsedSecs.toStringAsFixed(2)}s'),
          const Divider(height: 26),
          for (var i = 0; i < tabs.length; i++)
            _RailTab(
              icon: tabs[i].icon,
              label: tabs[i].label,
              selected: i == tab,
              onTap: () => onTab(i),
            ),
          const Spacer(),
        ],
      ),
    );
  }
}

class _StatRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? color;
  const _StatRow({required this.label, required this.value, this.color});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 4),
      child: Row(
        children: [
          Text(label,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Theme.of(context).colorScheme.onSurfaceVariant)),
          const Spacer(),
          Text(value,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: color,
              )),
        ],
      ),
    );
  }
}

class _RailTab extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _RailTab({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return InkWell(
      onTap: onTap,
      child: Container(
        color: selected ? scheme.primary.withValues(alpha: 0.14) : Colors.transparent,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 11),
        child: Row(
          children: [
            Icon(icon, size: 17, color: selected ? scheme.primary : scheme.onSurfaceVariant),
            const SizedBox(width: 10),
            Text(
              label,
              style: TextStyle(
                fontSize: 13,
                color: selected ? scheme.primary : scheme.onSurfaceVariant,
                fontWeight: selected ? FontWeight.w700 : FontWeight.w400,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
