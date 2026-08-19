import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

/// 通过率环形图：低饱和状态色圆环 + 中央百分比。
/// 尺寸紧凑（默认 104），满环（100%）时使用平头帽避免圆帽鼓包与文字重叠。
class PassRateDonut extends StatelessWidget {
  final double rate; // 0~1
  final double size;
  const PassRateDonut({super.key, required this.rate, this.size = 104});

  @override
  Widget build(BuildContext context) {
    final color = AppColors.passRate(rate);
    final bg = Theme.of(context).colorScheme.surfaceContainerHighest;
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: Size(size, size),
            painter: _DonutPainter(rate: rate, color: color, trackColor: bg),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '${(rate * 100).toStringAsFixed(1)}%',
                style: TextStyle(
                  fontFamily: 'monospace',
                  // '100.0%' 六字符最宽，按内孔动态收缩字号
                  fontSize: (rate * 100) >= 100 ? size * 0.17 : size * 0.2,
                  fontWeight: FontWeight.w700,
                  color: color,
                  height: 1,
                ),
              ),
              const SizedBox(height: 2),
              Text('通过率',
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: AppColors.subtle, fontSize: 11)),
            ],
          ),
        ],
      ),
    );
  }
}

class _DonutPainter extends CustomPainter {
  final double rate;
  final Color color;
  final Color trackColor;
  _DonutPainter({required this.rate, required this.color, required this.trackColor});

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.shortestSide / 2 - 5;
    const stroke = 9.0;

    final track = Paint()
      ..color = trackColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke;
    canvas.drawCircle(center, radius, track);

    final sweep = math.pi * 2 * rate.clamp(0.0, 1.0);
    if (sweep <= 0) return;
    // 满环用平头帽，避免 round 帽在接缝处鼓包（视觉上与中央文字冲突）
    final arc = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke
      ..strokeCap = rate >= 0.999 ? StrokeCap.butt : StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2,
      sweep,
      false,
      arc,
    );
  }

  @override
  bool shouldRepaint(_DonutPainter old) =>
      old.rate != rate || old.color != color || old.trackColor != trackColor;
}
