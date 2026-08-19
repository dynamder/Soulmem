import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import 'results_page.dart';

/// 批量配置页：目录 + 模式 + 参数 → 开始批量。
class BatchConfigPage extends StatefulWidget {
  const BatchConfigPage({super.key});

  @override
  State<BatchConfigPage> createState() => _BatchConfigPageState();
}

class _BatchConfigPageState extends State<BatchConfigPage> {
  String _mode = 'full';
  final _dirCtrl = TextEditingController();
  final _topK = TextEditingController(text: '10');
  final _threshold = TextEditingController(text: '0.7');

  @override
  void dispose() {
    _dirCtrl.dispose();
    _topK.dispose();
    _threshold.dispose();
    super.dispose();
  }

  String get _modeName => switch (_mode) {
        'embedding' => 'embedding',
        'association' => 'association',
        _ => 'full',
      };

  Future<void> _pickDir() async {
    final dir = await FilePicker.platform.getDirectoryPath(
      dialogTitle: '选择包含 question.json 的目录',
    );
    if (dir != null) {
      setState(() => _dirCtrl.text = dir);
    }
  }

  bool get _canStart => _dirCtrl.text.trim().isNotEmpty;

  void _start() {
    if (!_canStart) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => BatchPage(
          dir: _dirCtrl.text.trim(),
          mode: _modeName,
          params: {'top_k': _topK.text.trim(), 'threshold': _threshold.text.trim()},
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('批量测试')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              Text('检索模式', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'embedding', label: Text('embedding')),
                  ButtonSegment(value: 'association', label: Text('association')),
                  ButtonSegment(value: 'full', label: Text('full')),
                ],
                selected: {_mode},
                onSelectionChanged: (s) => setState(() => _mode = s.first),
              ),
              const SizedBox(height: 24),
              Text('数据集目录', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _dirCtrl,
                      decoration: const InputDecoration(
                        labelText: '目录路径',
                        hintText: '递归扫描该目录下的 question.json',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _pickDir,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('选择目录…'),
                  ),
                ],
              ),
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
                icon: const Icon(Icons.playlist_play),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('开始批量测试', style: TextStyle(fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 批量运行页：订阅 run_batch 流，实时更新数据集表格，完成后可钻取。
class BatchPage extends StatefulWidget {
  final String dir;
  final String mode;
  final Map<String, String> params;

  const BatchPage({
    super.key,
    required this.dir,
    required this.mode,
    required this.params,
  });

  @override
  State<BatchPage> createState() => _BatchPageState();
}

class _BatchPageState extends State<BatchPage> {
  StreamSubscription<BatchEvent>? _sub;
  String _status = '扫描中…';
  int _done = 0, _total = 0;
  final List<DatasetResultJson> _rows = [];
  BatchReport? _finalReport;
  String? _error;
  bool _finished = false;

  @override
  void initState() {
    super.initState();
    _start();
  }

  void _start() {
    resetCancel();
    _sub = runBatch(dir: widget.dir, mode: widget.mode, params: widget.params).listen((e) {
      if (!mounted) return;
      setState(() {
        switch (e) {
          case BatchScanning(:final dir):
            _status = '扫描目录: $dir';
          case BatchProgress(:final done, :final total):
            _done = done;
            _total = total;
            _status = '运行中';
          case BatchDatasetDone(:final name, :final total, :final passed, :final failed,
              :final passRate, :final elapsedMs, :final error):
            _rows.add(DatasetResultJson(
              name: name,
              path: '',
              total: total,
              passed: passed,
              failed: failed,
              passRate: passRate,
              elapsedMs: elapsedMs,
              error: error,
              outcomes: const [],
            ));
          case BatchDone(:final result):
            _finalReport = result;
            _rows
              ..clear()
              ..addAll(result.datasets);
            _finished = true;
          case BatchError(:final message):
            _error = message;
            _finished = true;
          case BatchCancelled():
            _finished = true;
            Navigator.of(context).pop();
        }
      });
    }, onError: (Object e) {
      if (!mounted) return;
      setState(() {
        _error = '桥接错误: $e';
        _finished = true;
      });
    });
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

  void _openDataset(DatasetResultJson ds) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ResultsPage(
          report: Report(
            algo: 'retrieve/${widget.mode}',
            datasetName: ds.name,
            datasetPath: ds.path,
            total: ds.total,
            passed: ds.passed,
            failed: ds.failed,
            elapsedSecs: ds.elapsedMs / 1000,
            metrics: const [],
            detailHeader: '',
            detailRows: const [],
            outcomes: ds.outcomes,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final totalPassed = _rows.fold<int>(0, (a, r) => a + r.passed);
    final totalFailed = _rows.fold<int>(0, (a, r) => a + r.failed);
    final sorted = [..._rows]..sort((a, b) => b.passRate.compareTo(a.passRate));

    return Scaffold(
      appBar: AppBar(title: Text('批量 · ${widget.mode}')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                LinearProgressIndicator(
                  value: _total == 0 ? null : _done / _total,
                  minHeight: 10,
                  borderRadius: BorderRadius.circular(4),
                ),
                const SizedBox(height: 8),
                Text(
                  _finalReport != null
                      ? '完成: ${_finalReport!.totalDatasets} 个数据集 · 通过 ${_finalReport!.totalPassed} / 失败 ${_finalReport!.totalFailed} · 耗时 ${_finalReport!.elapsedSecs.toStringAsFixed(1)}s'
                      : (_error ?? '$_status · $_done/$_total 数据集 · 累计通过 $totalPassed / 失败 $totalFailed'),
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 13),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: _rows.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : ListView.builder(
                    itemCount: sorted.length,
                    itemBuilder: (context, i) {
                      final r = sorted[i];
                      final statusColor = r.error != null
                          ? AppColors.warn
                          : AppColors.passRate(r.passRate);
                      return ListTile(
                        dense: true,
                        leading: Icon(
                          r.error != null ? Icons.error_outline : Icons.data_object,
                          color: r.error != null ? AppColors.warn : null,
                        ),
                        title: Text(r.name, maxLines: 1, overflow: TextOverflow.ellipsis),
                        subtitle: Text(
                          '${r.total} 用例 · 通过 ${r.passed} / 失败 ${r.failed} · ${(r.elapsedMs / 1000).toStringAsFixed(1)}s',
                          style: const TextStyle(fontSize: 12),
                        ),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              '${(r.passRate * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                fontFamily: 'monospace',
                                fontWeight: FontWeight.w700,
                                color: statusColor,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Icon(Icons.chevron_right, color: Colors.grey),
                          ],
                        ),
                        onTap: _finalReport != null || r.outcomes.isNotEmpty
                            ? () => _openDataset(r)
                            : null,
                      );
                    },
                  ),
          ),
          if (!_finished) ...[
            const Divider(height: 1),
            Padding(
              padding: const EdgeInsets.all(8),
              child: OutlinedButton.icon(
                onPressed: _cancel,
                icon: const Icon(Icons.stop_circle_outlined),
                style: OutlinedButton.styleFrom(foregroundColor: AppColors.fail),
                label: const Text('取消'),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
