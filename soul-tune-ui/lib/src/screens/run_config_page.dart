import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import 'run_page.dart';

/// 新建测试配置页：算法与模式 → 数据集 → 参数 → 开始测试。
class RunConfigPage extends StatefulWidget {
  const RunConfigPage({super.key});

  @override
  State<RunConfigPage> createState() => _RunConfigPageState();
}

class _RunConfigPageState extends State<RunConfigPage> {
  String _mode = 'full';
  final _pathCtrl = TextEditingController();
  final _topK = TextEditingController(text: '10');
  final _threshold = TextEditingController(text: '0.7');
  DatasetMeta? _meta;
  String? _metaError;
  bool _loadingMeta = false;
  static final List<String> _recent = [];

  @override
  void dispose() {
    _pathCtrl.dispose();
    _topK.dispose();
    _threshold.dispose();
    super.dispose();
  }

  String get _algoName => switch (_mode) {
        'embedding' => 'retrieve/embedding',
        'association' => 'retrieve/association',
        _ => 'retrieve/full',
      };

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['json'],
      dialogTitle: '选择测试数据集 (question.json)',
    );
    if (result != null && result.files.single.path != null) {
      _pathCtrl.text = result.files.single.path!;
      await _loadMeta();
    }
  }

  Future<void> _loadMeta() async {
    final p = _pathCtrl.text.trim();
    if (p.isEmpty) return;
    setState(() {
      _loadingMeta = true;
      _metaError = null;
    });
    try {
      final meta = await datasetMeta(p);
      if (!mounted) return;
      setState(() {
        _meta = meta;
        _metaError = meta.error;
        _loadingMeta = false;
      });
      if (meta.error == null && !_recent.contains(p)) {
        _recent.insert(0, p);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _metaError = '读取失败: $e';
        _loadingMeta = false;
      });
    }
  }

  void _resetParams() {
    setState(() {
      _topK.text = '10';
      _threshold.text = '0.7';
    });
  }

  bool get _canStart => _pathCtrl.text.trim().isNotEmpty && _meta?.error == null;

  void _start() {
    if (!_canStart) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => RunPage(
          algo: _algoName,
          dataset: _pathCtrl.text.trim(),
          params: {'top_k': _topK.text.trim(), 'threshold': _threshold.text.trim()},
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('新建测试')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              // ① 算法与模式
              Text('算法与模式', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'embedding', label: Text('embedding'), tooltip: '仅相似度检索'),
                  ButtonSegment(value: 'association', label: Text('association'), tooltip: '相似度 + PPR 关联'),
                  ButtonSegment(value: 'full', label: Text('full'), tooltip: '相似度 + 关联 + 动作全管线'),
                ],
                selected: {_mode},
                onSelectionChanged: (s) => setState(() => _mode = s.first),
              ),
              const SizedBox(height: 24),

              // ② 数据集
              Text('数据集', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _pathCtrl,
                      decoration: const InputDecoration(
                        labelText: '数据集路径',
                        hintText: 'question.json 的完整路径',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: (_) => _loadMeta(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _pickFile,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('选择文件…'),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (_recent.isNotEmpty) ...[
                Text('最近使用', style: Theme.of(context).textTheme.bodySmall),
                Wrap(
                  spacing: 8,
                  children: [
                    for (final r in _recent.take(5))
                      ActionChip(
                        label: Text(r.split(RegExp(r'[\\/]')).last),
                        onPressed: () {
                          _pathCtrl.text = r;
                          _loadMeta();
                        },
                      ),
                  ],
                ),
                const SizedBox(height: 8),
              ],
              // 预览卡
              if (_loadingMeta)
                const LinearProgressIndicator()
              else if (_meta != null && _metaError == null)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(_meta!.name.isEmpty ? '(未命名)' : _meta!.name,
                            style: Theme.of(context).textTheme.titleSmall),
                        const SizedBox(height: 4),
                        Text(_meta!.description,
                            style: const TextStyle(color: Colors.grey)),
                        const SizedBox(height: 8),
                        Text('用例数: ${_meta!.caseCount}   图谱: ${_meta!.graphPath}',
                            style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                      ],
                    ),
                  ),
                )
              else if (_metaError != null)
                Text(_metaError!, style: const TextStyle(color: Colors.redAccent)),
              const SizedBox(height: 24),

              // ③ 参数
              Text('参数', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _topK,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        labelText: 'top_k',
                        helperText: '检索返回数量上限',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: TextField(
                      controller: _threshold,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        labelText: 'threshold',
                        helperText: '相似度阈值',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  TextButton(onPressed: _resetParams, child: const Text('恢复默认')),
                ],
              ),
              const SizedBox(height: 32),

              // 开始测试
              FilledButton.icon(
                onPressed: _canStart ? _start : null,
                icon: const Icon(Icons.play_arrow),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('开始测试', style: TextStyle(fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
