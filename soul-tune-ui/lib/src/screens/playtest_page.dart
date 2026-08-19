import 'dart:async';
import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';

/// Playtest 配置页：选择角色图目录（须含 graph.json）→ 启动会话。
class PlaytestConfigPage extends StatefulWidget {
  const PlaytestConfigPage({super.key});

  @override
  State<PlaytestConfigPage> createState() => _PlaytestConfigPageState();
}

class _PlaytestConfigPageState extends State<PlaytestConfigPage> {
  final _pathCtrl = TextEditingController();
  String? _error;
  bool _starting = false;

  @override
  void dispose() {
    _pathCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    // 选 graph.json 文件（比目录选择器在 Windows 上更可靠），Rust 侧自动取其父目录
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['json'],
      dialogTitle: '选择角色图（graph.json）',
    );
    if (result != null && result.files.single.path != null) {
      setState(() {
        _pathCtrl.text = result.files.single.path!;
        _error = null;
      });
    }
  }

  Future<void> _start() async {
    final path = _pathCtrl.text.trim();
    if (path.isEmpty || _starting) return;
    setState(() {
      _starting = true;
      _error = null;
    });
    final result = await playtestStart(path);
    if (!mounted) return;
    setState(() => _starting = false);
    if (result.ok) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => PlaytestChatPage(characterName: result.characterName),
        ),
      );
    } else {
      setState(() => _error = result.error ?? '启动失败');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('角色扮演测试')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              Text('角色图', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _pathCtrl,
                      decoration: const InputDecoration(
                        labelText: 'graph.json 路径',
                        hintText: '选择文件或直接粘贴路径',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _starting ? null : _pickFile,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('选择…'),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                '选择角色图目录下的 graph.json（Rust 侧自动定位其父目录）；'
                '环境变量 SOUL_TUNE_CANDLE_MODEL_PATH 指向本地 LLM 模型，'
                '或 SOUL_TUNE_LLAMA_URL 直连已运行的 llama-server。',
                style: TextStyle(color: AppColors.subtle, fontSize: 12),
              ),
              if (_error != null) ...[
                const SizedBox(height: 8),
                Text(_error!, style: const TextStyle(color: AppColors.fail, fontSize: 12)),
              ],
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _pathCtrl.text.trim().isEmpty || _starting ? null : _start,
                icon: _starting
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.chat_bubble_outline),
                label: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  child: Text(_starting ? '加载中…' : '开始对话',
                      style: const TextStyle(fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Playtest 对话页：逐轮发送消息，展示 embedding/full 双回复 + 可展开检索轨迹。
class PlaytestChatPage extends StatefulWidget {
  final String characterName;
  const PlaytestChatPage({super.key, required this.characterName});

  @override
  State<PlaytestChatPage> createState() => _PlaytestChatPageState();
}

class _PlaytestChatPageState extends State<PlaytestChatPage> {
  final _inputCtrl = TextEditingController();
  final List<PlayTurn> _turns = [];
  bool _processing = false;
  final ScrollController _scroll = ScrollController();
  String? _fatalError;

  @override
  void dispose() {
    _inputCtrl.dispose();
    _scroll.dispose();
    playtestFinish();
    super.dispose();
  }

  Future<void> _send() async {
    final msg = _inputCtrl.text.trim();
    if (msg.isEmpty || _processing) return;
    setState(() {
      _inputCtrl.clear();
      _processing = true;
    });
    // 本地先显示一条占位轮次（用户消息），结果到达后替换
    final pending = PlayTurn(
      index: _turns.length,
      userMessage: msg,
      generatedQueriesJson: '',
      embedding: null,
      full: null,
    );
    setState(() => _turns.add(pending));
    _scrollToBottom();

    try {
      await for (final turn in playtestTurn(msg)) {
        if (!mounted) return;
        setState(() {
          _turns[_turns.length - 1] = turn;
          _processing = false;
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _processing = false;
        _fatalError = '桥接错误: $e';
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('角色扮演 · ${widget.characterName}'),
        actions: [
          TextButton(
            onPressed: () {
              playtestFinish();
              Navigator.of(context).pop();
            },
            child: const Text('结束'),
          ),
        ],
      ),
      body: Column(
        children: [
          if (_fatalError != null)
            Container(
              width: double.infinity,
              color: AppColors.failBg,
              padding: const EdgeInsets.all(8),
              child: Text(_fatalError!, style: const TextStyle(color: AppColors.fail)),
            ),
          Expanded(
            child: _turns.isEmpty
                ? Center(
                    child: Text(
                      '对 ${widget.characterName} 说点什么…\n（每轮同时跑 embedding 与 full pipeline 检索）',
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: AppColors.subtle),
                    ),
                  )
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(16),
                    itemCount: _turns.length,
                    itemBuilder: (context, i) => _TurnCard(turn: _turns[i]),
                  ),
          ),
          if (_processing)
            const Padding(
              padding: EdgeInsets.all(8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
                  SizedBox(width: 8),
                  Text('检索与回复生成中…', style: TextStyle(color: AppColors.subtle, fontSize: 12)),
                ],
              ),
            ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 10),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _inputCtrl,
                      decoration: const InputDecoration(
                        hintText: '输入消息…',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      onSubmitted: (_) => _send(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton.filled(
                    onPressed: _processing ? null : _send,
                    icon: const Icon(Icons.send),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// 单轮对话卡片：用户消息 + 双回复 A/B + 可展开检索轨迹 + 偏好标记。
class _TurnCard extends StatelessWidget {
  final PlayTurn turn;
  const _TurnCard({required this.turn});

  @override
  Widget build(BuildContext context) {
    final isPending = turn.embedding == null && turn.full == null && turn.error == null;
    return Card(
      elevation: 0,
      color: Theme.of(context).colorScheme.surfaceContainerHigh,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.person, size: 18, color: AppColors.running),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(turn.userMessage, style: const TextStyle(fontSize: 14, height: 1.4)),
                ),
              ],
            ),
            const Divider(height: 20),
            if (turn.error != null)
              Text('错误: ${turn.error}', style: const TextStyle(color: AppColors.fail, fontSize: 12))
            else if (isPending)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(8),
                  child: Text('…', style: TextStyle(color: AppColors.subtle)),
                ),
              )
            else ...[
              Row(
                children: [
                  const Expanded(
                    child: Text('A · embedding',
                        style: TextStyle(
                            color: AppColors.running, fontWeight: FontWeight.w600, fontSize: 12)),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text('B · full pipeline',
                        style: TextStyle(
                            color: Theme.of(context).colorScheme.primary,
                            fontWeight: FontWeight.w600,
                            fontSize: 12)),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _ResponseCard(run: turn.embedding, label: 'A')),
                  const SizedBox(width: 8),
                  Expanded(child: _ResponseCard(run: turn.full, label: 'B')),
                ],
              ),
              if (turn.generatedQueriesJson.isNotEmpty) ...[
                const SizedBox(height: 8),
                ExpansionTile(
                  dense: true,
                  tilePadding: EdgeInsets.zero,
                  title: const Text('生成的查询', style: TextStyle(fontSize: 12)),
                  children: [
                    SelectableText(
                      _prettyJson(turn.generatedQueriesJson),
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
                    ),
                  ],
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }

  String _prettyJson(String raw) {
    try {
      final v = jsonDecode(raw);
      return const JsonEncoder.withIndent('  ').convert(v);
    } catch (_) {
      return raw;
    }
  }
}

class _ResponseCard extends StatelessWidget {
  final PlayRun? run;
  final String label;
  const _ResponseCard({required this.run, required this.label});

  @override
  Widget build(BuildContext context) {
    final response = run?.response;
    final trace = run?.trace;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (response == null)
            const Text('（无回复）', style: TextStyle(color: AppColors.subtle, fontSize: 12))
          else
            SelectableText(response, style: const TextStyle(fontSize: 13, height: 1.5)),
          if (trace != null) ...[
            const SizedBox(height: 8),
            ExpansionTile(
              dense: true,
              tilePadding: EdgeInsets.zero,
              title: Text('检索轨迹 · ${trace.mode} · ${(trace.totalElapsedMs / 1000).toStringAsFixed(2)}s',
                  style: const TextStyle(fontSize: 11)),
              children: [
                if (trace.merged.isNotEmpty) ...[
                  _traceHeader('检索结果'),
                  for (final n in trace.merged)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Text(
                        '${n.name} [${n.stage}] ${n.score.toStringAsFixed(3)}',
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                ],
                if (trace.perQuery.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  _traceHeader('子查询'),
                  for (final q in trace.perQuery)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Text(
                        '${q.dropped ? '[丢弃] ' : ''}${q.preview.isEmpty ? '（空）' : q.preview}'
                        '  sim:${q.sim} ppr:${q.ppr} act:${q.action}',
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 10),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                ],
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _traceHeader(String title) => Padding(
        padding: const EdgeInsets.only(top: 4, bottom: 2),
        child: Text(title, style: const TextStyle(color: AppColors.subtle, fontSize: 10)),
      );
}
