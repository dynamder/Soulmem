import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import 'compare_results_page.dart';

/// 对比测试配置页：选择数据集 + 参数 → 运行 embedding vs full 对比。
class CompareConfigPage extends StatefulWidget {
  const CompareConfigPage({super.key});

  @override
  State<CompareConfigPage> createState() => _CompareConfigPageState();
}

class _CompareConfigPageState extends State<CompareConfigPage> {
  final _pathCtrl = TextEditingController();
  final _topK = TextEditingController(text: '10');
  final _threshold = TextEditingController(text: '0.7');
  DatasetMeta? _meta;
  String? _metaError;
  bool _loadingMeta = false;

  @override
  void dispose() {
    _pathCtrl.dispose();
    _topK.dispose();
    _threshold.dispose();
    super.dispose();
  }

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
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _metaError = '读取失败: $e';
        _loadingMeta = false;
      });
    }
  }

  bool get _canStart => _pathCtrl.text.trim().isNotEmpty && _meta?.error == null;

  void _start() {
    if (!_canStart) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CompareRunPage(
          dataset: _pathCtrl.text.trim(),
          params: {'top_k': _topK.text.trim(), 'threshold': _threshold.text.trim()},
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('对比测试 (embedding vs full)')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
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
                        Text('用例数: ${_meta!.caseCount}   图谱: ${_meta!.graphPath}',
                            style: const TextStyle(
                                fontFamily: 'monospace', fontSize: 12, color: AppColors.subtle)),
                      ],
                    ),
                  ),
                )
              else if (_metaError != null)
                Text(_metaError!, style: const TextStyle(color: AppColors.fail)),
              const SizedBox(height: 24),
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
                ],
              ),
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: _canStart ? _start : null,
                icon: const Icon(Icons.compare_arrows),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('开始对比', style: TextStyle(fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 对比运行页：两阶段进度（embedding → full），完成后进入结果页。
class CompareRunPage extends StatefulWidget {
  final String dataset;
  final Map<String, String> params;
  const CompareRunPage({super.key, required this.dataset, required this.params});

  @override
  State<CompareRunPage> createState() => _CompareRunPageState();
}

class _CompareRunPageState extends State<CompareRunPage> {
  StreamSubscription<CompareEvent>? _sub;
  String _phase = '';
  String _loadingMsg = '正在加载…';
  int _done = 0, _total = 0, _passed = 0, _failed = 0;
  int _elapsedMs = 0;
  String _caseName = '';
  bool _loading = true;
  bool _finished = false;

  @override
  void initState() {
    super.initState();
    _start();
  }

  void _start() {
    resetCancel();
    _sub = runCompare(dataset: widget.dataset, params: widget.params).listen((e) {
      if (!mounted) return;
      setState(() {
        switch (e) {
          case CompareLoading(:final message):
            _loading = true;
            _loadingMsg = message;
          case CompareProgress(:final phase, :final done, :final total, :final passed,
              :final failed, :final elapsedMs, :final caseName):
            _loading = false;
            _phase = phase;
            _done = done;
            _total = total;
            _passed = passed;
            _failed = failed;
            _elapsedMs = elapsedMs;
            _caseName = caseName;
          case CompareDone(:final report):
            _finished = true;
            Navigator.of(context).pushReplacement(
              MaterialPageRoute(builder: (_) => CompareResultsPage(report: report)),
            );
          case CompareError(:final message):
            _finished = true;
            _showError(message);
          case CompareCancelled():
            _finished = true;
            Navigator.of(context).pop();
        }
      });
    }, onError: (Object e) {
      if (!mounted) return;
      _finished = true;
      _showError('桥接错误: $e');
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: AppColors.fail),
    );
    Navigator.of(context).pop();
  }

  void _cancel() {
    resetCancel();
    Navigator.of(context).pop();
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ratio = _total == 0 ? 0.0 : _done / _total;
    return Scaffold(
      appBar: AppBar(title: Text('对比中 · ${widget.dataset.split(RegExp(r'[\\/]')).last}')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (_loading)
                  Column(
                    children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(_loadingMsg, style: const TextStyle(color: AppColors.subtle)),
                    ],
                  )
                else ...[
                  Row(
                    children: [
                      _PhaseBadge(label: 'embedding', active: _phase == 'embedding'),
                      const SizedBox(width: 8),
                      _PhaseBadge(label: 'full', active: _phase == 'full'),
                    ],
                  ),
                  const SizedBox(height: 16),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: ratio,
                      minHeight: 12,
                      backgroundColor: Colors.grey.shade800,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '$_done / $_total  (${(ratio * 100).toStringAsFixed(0)}%)',
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 16),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    '通过 $_passed · 失败 $_failed · 耗时 ${(_elapsedMs / 1000).toStringAsFixed(1)}s',
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                        fontFamily: 'monospace', fontSize: 13, color: AppColors.subtle),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    '当前: $_caseName',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: AppColors.subtle),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 24),
                  if (!_finished)
                    OutlinedButton.icon(
                      onPressed: _cancel,
                      icon: const Icon(Icons.stop_circle_outlined),
                      style: OutlinedButton.styleFrom(foregroundColor: AppColors.fail),
                      label: const Text('取消'),
                    ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _PhaseBadge extends StatelessWidget {
  final String label;
  final bool active;
  const _PhaseBadge({required this.label, required this.active});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: active ? AppColors.runningBg : Colors.transparent,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: active ? AppColors.running : Colors.grey.shade700),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: active ? AppColors.running : AppColors.subtle,
          fontSize: 12,
          fontWeight: active ? FontWeight.w700 : FontWeight.w400,
        ),
      ),
    );
  }
}
