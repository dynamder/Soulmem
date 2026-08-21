import 'package:flutter/material.dart';

/// 全局状态色板：柔和、低对比度的状态色（避免刺眼的红绿）。
/// 通过/失败均使用 Material 300 级色调 + 低透明度底色。
class AppColors {
  // 通过（柔和绿）
  static const pass = Color(0xFF81C784);
  static const passBg = Color(0x1A81C784);

  // 失败（柔和红）
  static const fail = Color(0xFFE57373);
  static const failBg = Color(0x1AE57373);

  // 警告（柔和琥珀）
  static const warn = Color(0xFFFFD54F);
  static const warnBg = Color(0x1AFFD54F);

  // 运行中（柔和靛蓝，与主题强调色一致）
  static const running = Color(0xFF9FA8DA);
  static const runningBg = Color(0x1A9FA8DA);

  // 对照组（柔和青绿：激发测试中"未激发基线"曲线）
  static const ctrl = Color(0xFF80CBC4);
  static const ctrlBg = Color(0x1A80CBC4);

  // 中性灰（次级文本/占位）
  static const subtle = Color(0xFF9E9E9E);

  /// 按通过率取状态色：≥80% 绿，50%~80% 琥珀，<50% 红。
  static Color passRate(double rate) =>
      rate >= 0.8 ? pass : (rate >= 0.5 ? warn : fail);

  static Color passRateBg(double rate) =>
      rate >= 0.8 ? passBg : (rate >= 0.5 ? warnBg : failBg);
}
