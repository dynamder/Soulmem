import 'package:flutter/material.dart';

import '../theme.dart';

/// 柔和柱状图（CustomPaint 手绘）：逐用例 通过/失败 状态条。
/// 使用低饱和状态色，圆角柱体，底部浅基线。
class MiniBarChart extends StatelessWidget {
  final List<bool> passFlags;
  const MiniBarChart({super.key, required this.passFlags});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 120,
      padding: const EdgeInsets.fromLTRB(8, 12, 8, 4),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(12),
      ),
      child: passFlags.isEmpty
          ? const Center(
              child: Text('（无数据）', style: TextStyle(color: AppColors.subtle)))
          : CustomPaint(
              painter: _BarPainter(passFlags),
              child: const SizedBox.expand(),
            ),
    );
  }
}

class _BarPainter extends CustomPainter {
  final List<bool> passFlags;
  _BarPainter(this.passFlags);

  @override
  void paint(Canvas canvas, Size size) {
    // 浅基线
    final basePaint = Paint()
      ..color = const Color(0x33FFFFFF)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(0, size.height), Offset(size.width, size.height), basePaint);

    final gap = 2.0;
    final barW = (size.width - gap * (passFlags.length - 1)) / passFlags.length;
    for (var i = 0; i < passFlags.length; i++) {
      final paint = Paint()
        ..color = passFlags[i] ? AppColors.pass : AppColors.fail;
      final left = i * (barW + gap);
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(left, 0, barW, size.height),
          const Radius.circular(3),
        ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(_BarPainter old) => old.passFlags != passFlags;
}
