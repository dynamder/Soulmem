import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';

/// 模型来源状态横幅：所有需要模型的地方共用同一套来源决策——
/// 1. 复用**已运行**的 llama-server；
/// 2. 无运行服务 → 自动拉起本地缓存模型；
/// 3. 都没有 → 显示不可用原因（运行时会报错或降级）。
///
/// 进入页面时自动探测一次；可手动刷新。
class ModelStatusBanner extends StatefulWidget {
  const ModelStatusBanner({super.key});

  @override
  State<ModelStatusBanner> createState() => _ModelStatusBannerState();
}

class _ModelStatusBannerState extends State<ModelStatusBanner> {
  ModelStatus? _status;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final s = await modelStatus();
      if (!mounted) return;
      setState(() {
        _status = s;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '模型状态查询失败: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = _status;

    final (Color color, IconData icon, String title, String detail) =
        _render(status);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: _loading
                ? const Text('正在探测模型来源…',
                    style: TextStyle(color: AppColors.subtle, fontSize: 12))
                : _error != null
                    ? Text(_error!,
                        style: const TextStyle(color: AppColors.fail, fontSize: 12))
                    : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(title,
                              style: TextStyle(
                                  color: color,
                                  fontSize: 12.5,
                                  fontWeight: FontWeight.w700)),
                          const SizedBox(height: 3),
                          Text(detail,
                              maxLines: 4,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(fontSize: 11.5, height: 1.4)),
                        ],
                      ),
          ),
          IconButton(
            tooltip: '重新探测',
            onPressed: _loading ? null : _refresh,
            icon: const Icon(Icons.refresh, size: 16),
            color: AppColors.subtle,
          ),
        ],
      ),
    );
  }

  (Color, IconData, String, String) _render(ModelStatus? s) {
    if (s == null) {
      return (AppColors.subtle, Icons.help_outline, '模型来源未知', '尚未完成探测');
    }
    switch (s.source) {
      case 'running':
        return (
          AppColors.pass,
          Icons.dns_outlined,
          '模型可用 · 复用运行中的 llama-server',
          '已检测到健康的 llama-server：${s.url ?? '（地址未知）'}，直接复用，无需启动。',
        );
      case 'spawned':
        return (
          AppColors.running,
          Icons.rocket_launch_outlined,
          '模型可用 · 将自动拉起本地缓存模型',
          '未检测到运行中的 llama-server，运行时会自动启动并加载：\n${s.modelPath ?? '（路径未知）'}',
        );
      default:
        return (
          AppColors.warn,
          Icons.warning_amber_outlined,
          '模型不可用 · 将报错或降级',
          s.reason ?? '未发现运行中的 llama-server，也未找到本地缓存模型。',
        );
    }
  }
}
