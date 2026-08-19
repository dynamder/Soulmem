import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/json_tree.dart';

/// 检视数据集：结构化条目卡片（图节点 / 查询用例），
/// 信息架构沿用 TUI inspect（摘要 → 详情 → 连接），原始 JSON 树降为次要。
class InspectPage extends StatefulWidget {
  const InspectPage({super.key});

  @override
  State<InspectPage> createState() => _InspectPageState();
}

class _InspectPageState extends State<InspectPage> {
  String? _path;
  InspectEntries? _data;
  InspectFile? _raw;
  String? _error;
  bool _loading = false;

  Future<void> _pick() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['json'],
      dialogTitle: '选择要检视的 JSON 文件',
    );
    if (result != null && result.files.single.path != null) {
      await _load(result.files.single.path!);
    }
  }

  Future<void> _load(String path) async {
    setState(() {
      _path = path;
      _loading = true;
      _error = null;
    });
    try {
      final results = await Future.wait([
        inspectEntries(path),
        inspectFile(path),
      ]);
      if (!mounted) return;
      setState(() {
        _data = results[0] as InspectEntries;
        _raw = results[1] as InspectFile;
        _error = _raw?.error;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '读取失败: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('检视数据集')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 860),
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    FilledButton.icon(
                      onPressed: _loading ? null : _pick,
                      icon: const Icon(Icons.folder_open),
                      label: const Text('打开文件…'),
                    ),
                    if (_path != null) ...[
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(_path!,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style:
                                const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                      ),
                    ],
                  ],
                ),
              ),
              if (_loading)
                const Expanded(child: Center(child: CircularProgressIndicator()))
              else if (_error != null)
                Expanded(
                  child: Center(
                    child: Text(_error!, style: const TextStyle(color: AppColors.fail)),
                  ),
                )
              else if (_data != null)
                Expanded(child: _buildContent())
              else
                const Expanded(
                  child: Center(
                    child: Text('选择一个 question.json 或 graph.json 开始检视',
                        style: TextStyle(color: AppColors.subtle)),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildContent() {
    final data = _data!;
    final isGraph = data.fileType == 'graph';
    final scheme = Theme.of(context).colorScheme;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      children: [
        // 类型/统计条
        Card(
          elevation: 0,
          color: scheme.surfaceContainerHigh,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Wrap(
              spacing: 16,
              runSpacing: 6,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isGraph ? Icons.account_tree_outlined : Icons.quiz_outlined,
                      size: 18,
                      color: scheme.primary,
                    ),
                    const SizedBox(width: 6),
                    Text(isGraph ? '角色图谱' : '测试数据集',
                        style: Theme.of(context).textTheme.titleSmall),
                  ],
                ),
                Text('条目: ${data.entries.length}',
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                for (final s in data.stats)
                  Text(s, style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
              ],
            ),
          ),
        ),
        const SizedBox(height: 10),
        if (data.entries.isEmpty)
          const Card(
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Text('（无结构化条目，可展开下方原始 JSON 查看）',
                  style: TextStyle(color: AppColors.subtle)),
            ),
          ),
        for (final e in data.entries) _EntryCard(entry: e, isGraph: isGraph),
        const SizedBox(height: 8),
        // 原始 JSON（次要）
        Card(
          elevation: 0,
          color: scheme.surfaceContainerHigh,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: Theme(
            data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
            child: ExpansionTile(
              title: const Text('原始 JSON',
                  style: TextStyle(fontSize: 13, color: AppColors.subtle)),
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                  child: JsonTree(data: _raw?.data ?? const {}),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/// 结构化条目卡片：标题摘要 + 可展开（预览/详情/连接）。
class _EntryCard extends StatelessWidget {
  final InspectEntryItem entry;
  final bool isGraph;
  const _EntryCard({required this.entry, required this.isGraph});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final outgoing = entry.links.where((l) => l.isOutgoing).toList();
    final incoming = entry.links.where((l) => !l.isOutgoing).toList();
    final hasDetail = entry.detailLines.isNotEmpty || entry.links.isNotEmpty;

    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHigh,
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          leading: Icon(
            isGraph ? Icons.circle_outlined : Icons.quiz_outlined,
            size: 18,
            color: scheme.primary,
          ),
          title: Text(entry.summary,
              maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 13)),
          subtitle: entry.previewLines.isEmpty
              ? null
              : Text(entry.previewLines.first,
                  maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 11)),
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final p in entry.previewLines.skip(1))
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 1),
                      child: Text(p, style: const TextStyle(fontSize: 12)),
                    ),
                  if (entry.detailLines.isNotEmpty) ...[
                    const Divider(height: 16),
                    for (final dl in entry.detailLines)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 1),
                        child: Text(dl,
                            style: const TextStyle(
                                fontFamily: 'monospace', fontSize: 11, height: 1.4)),
                      ),
                  ],
                  if (hasDetail && (outgoing.isNotEmpty || incoming.isNotEmpty)) ...[
                    const Divider(height: 16),
                    if (outgoing.isNotEmpty) ...[
                      Text('出边 (→)',
                          style: TextStyle(fontSize: 11, color: scheme.primary)),
                      for (final l in outgoing) _LinkRow(link: l),
                    ],
                    if (incoming.isNotEmpty) ...[
                      const SizedBox(height: 4),
                      Text('入边 (←)',
                          style: TextStyle(fontSize: 11, color: scheme.primary)),
                      for (final l in incoming) _LinkRow(link: l),
                    ],
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LinkRow extends StatelessWidget {
  final InspectLink link;
  const _LinkRow({required this.link});

  @override
  Widget build(BuildContext context) {
    final target = link.isOutgoing ? link.toId : link.fromId;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Icon(link.isOutgoing ? Icons.arrow_forward : Icons.arrow_back,
              size: 13, color: AppColors.subtle),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              '$target  ${link.linkTypeDesc}  [${link.intensity.toStringAsFixed(2)}]',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}
