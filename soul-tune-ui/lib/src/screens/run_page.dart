import 'dart:async';

import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/stat_card.dart';
import 'results_page.dart';

/// 运行页：订阅 run_suite 事件流，实时展示进度；完成后跳转结果页。
class RunPage extends StatefulWidget {
  final String algo;
  final String dataset;
  final Map<String, String> params;

  const RunPage({
    super.key,
    required this.algo,
    required this.dataset,
    required this.params,
  });

  @override
  State<RunPage> createState() => _RunPageState();
}

class _RunPageState extends State<RunPage> {
  StreamSubscription<RunEvent>? _sub;
  String? _loadingMsg;
  int _done = 0, _total = 0, _passed = 0, _failed = 0;
  int _elapsedMs = 0;
  String _caseName = '';
  bool _finished = false;

  @override
  void initState() {
    super.initState();
    _start();
  }

  void _start() {
    resetCancel();
    _sub = runSuite(algo: widget.algo, dataset: widget.dataset, params: widget.params)
        .listen((e) {
      if (!mounted) return;
      setState(() {
        switch (e) {
          case RunLoading(:final message):
            _loadingMsg = message;
          case RunProgress(
              :final done,
              :final total,
              :final passed,
              :final failed,
              :final elapsedMs,
              :final caseName
            ):
            _done = done;
            _total = total;
            _passed = passed;
            _failed = failed;
            _elapsedMs = elapsedMs;
            _caseName = caseName;
          case RunDone(:final report):
            _finished = true;
            _navigateToResults(report);
          case RunError(:final message):
            _finished = true;
            _showError(message);
          case RunCancelled():
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

  void _navigateToResults(Report report) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => ResultsPage(report: report)),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.redAccent),
    );
    Navigator.of(context).pop();
  }

  void _cancel() {
    resetCancel();
    // Rust 侧用例间检查标志后推 cancelled 事件；此处立即回退
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
    final loading = _total == 0 && !_finished;
    return Scaffold(
      appBar: AppBar(title: Text('运行中 · ${widget.algo}')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (loading)
                  Column(
                    children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(_loadingMsg ?? '正在加载…',
                          style: const TextStyle(color: Colors.grey)),
                    ],
                  )
                else ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: ratio,
                      minHeight: 14,
                      backgroundColor: Colors.grey.shade800,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '$_done / $_total  (${(ratio * 100).toStringAsFixed(0)}%)',
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 18),
                  ),
                  const SizedBox(height: 24),
                  Wrap(
                    alignment: WrapAlignment.center,
                    spacing: 12,
                    runSpacing: 12,
                    children: [
                      StatCard(label: '通过', value: '$_passed', valueColor: AppColors.pass),
                      StatCard(label: '失败', value: '$_failed', valueColor: AppColors.fail),
                      StatCard(label: '耗时', value: '${(_elapsedMs / 1000).toStringAsFixed(1)}s'),
                    ],
                  ),
                  const SizedBox(height: 24),
                  Text(
                    '当前: $_caseName',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.grey),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 32),
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
