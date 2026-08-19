import 'package:flutter/material.dart';

import 'batch_page.dart';
import 'compare_page.dart';
import 'forget_page.dart';
import 'inspect_page.dart';
import 'playtest_page.dart';
import 'run_config_page.dart';

/// 首页：动作卡片入口（GUI 直觉式，替代 TUI 命令面板）。
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Soul-Tune 测试框架')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 760),
          child: GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            padding: const EdgeInsets.all(24),
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
            childAspectRatio: 1.6,
            children: [
              _ActionCard(
                icon: Icons.science_outlined,
                title: '运行检索测试',
                subtitle: '单数据集 · 实时进度 · 结果钻取',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const RunConfigPage()),
                ),
              ),
              _ActionCard(
                icon: Icons.playlist_play_outlined,
                title: '批量测试',
                subtitle: '扫描目录下全部数据集 · 并发执行',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const BatchConfigPage()),
                ),
              ),
              _ActionCard(
                icon: Icons.compare_arrows,
                title: '对比测试',
                subtitle: '同数据集 embedding vs full pipeline',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CompareConfigPage()),
                ),
              ),
              _ActionCard(
                icon: Icons.visibility_outlined,
                title: '检视数据集',
                subtitle: '查看 question.json / graph.json 结构',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const InspectPage()),
                ),
              ),
              _ActionCard(
                icon: Icons.delete_outline,
                title: '遗忘测试',
                subtitle: '遮罩 / 遮罩补全 / 全管线 · 逐节点观测',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const ForgetConfigPage()),
                ),
              ),
              _ActionCard(
                icon: Icons.psychology_outlined,
                title: '角色扮演测试',
                subtitle: 'LLM 对话 · 双管线检索 · 逐轮轨迹',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const PlaytestConfigPage()),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback? onTap;

  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 30, color: scheme.primary),
              const Spacer(),
              Text(title, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(
                subtitle,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Colors.grey),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
