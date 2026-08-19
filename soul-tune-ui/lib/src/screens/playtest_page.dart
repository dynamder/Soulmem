import 'dart:async';
import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/model_status_banner.dart';

/// Playtest 配置页：选择角色图（graph.json）+ 用户角色 → 启动会话。
class PlaytestConfigPage extends StatefulWidget {
  const PlaytestConfigPage({super.key});

  @override
  State<PlaytestConfigPage> createState() => _PlaytestConfigPageState();
}

class _PlaytestConfigPageState extends State<PlaytestConfigPage> {
  final _pathCtrl = TextEditingController();
  final _roleCtrl = TextEditingController();
  String? _error;
  bool _starting = false;

  @override
  void dispose() {
    _pathCtrl.dispose();
    _roleCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
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
    final result = await playtestStart(path, userRole: _roleCtrl.text.trim());
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
              const SizedBox(height: 16),
              Text('用户角色（可选）', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              TextField(
                controller: _roleCtrl,
                decoration: const InputDecoration(
                  labelText: '我扮演的是…',
                  hintText: '例如：玩家 / 帕秋莉',
                  border: OutlineInputBorder(),
                  isDense: true,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                '选择角色图目录下的 graph.json（Rust 侧自动定位其父目录）；'
                'LLM 统一解析：先复用运行中的 llama-server，没有则自动拉起本地缓存模型。',
                style: TextStyle(color: AppColors.subtle, fontSize: 12),
              ),
              if (_error != null) ...[
                const SizedBox(height: 8),
                Text(_error!, style: const TextStyle(color: AppColors.fail, fontSize: 12)),
              ],
              const SizedBox(height: 16),
              const ModelStatusBanner(),
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

/// Playtest 对话页：逐轮发送消息，A/B 双回复 + 结构化查询/轨迹 + 人工投票。
class PlaytestChatPage extends StatefulWidget {
  final String characterName;
  const PlaytestChatPage({super.key, required this.characterName});

  @override
  State<PlaytestChatPage> createState() => _PlaytestChatPageState();
}

class _PlaytestChatPageState extends State<PlaytestChatPage> {
  final _inputCtrl = TextEditingController();
  final List<PlayTurn> _turns = [];
  final Map<int, int> _votes = {}; // turnIndex → 0=A 1=B 2=持平
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

  Future<void> _vote(int turnIndex, int pick) async {
    setState(() => _votes[turnIndex] = pick);
    await playtestVote(turnIndex, pick);
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
                    itemBuilder: (context, i) => _TurnCard(
                      turn: _turns[i],
                      vote: _votes[_turns[i].index],
                      onVote: (pick) => _vote(_turns[i].index, pick),
                    ),
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

/// 单轮对话卡片：用户消息 + A/B 双回复 + 查询/轨迹 + 投票。
class _TurnCard extends StatelessWidget {
  final PlayTurn turn;
  final int? vote;
  final ValueChanged<int> onVote;
  const _TurnCard({required this.turn, this.vote, required this.onVote});

  @override
  Widget build(BuildContext context) {
    final isPending = turn.embedding == null && turn.full == null && turn.error == null;
    final tracesIdentical = _tracesIdentical(turn.embedding?.trace, turn.full?.trace);
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
              if (tracesIdentical)
                Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Text('提示：两种模式的检索轨迹相同，差异仅体现在回复生成上',
                      style: const TextStyle(color: AppColors.subtle, fontSize: 11)),
                ),
              // 双回复
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
              // 投票（TUI 评判机制迁移）
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _VoteChip(label: 'A 更好', selected: vote == 0, onTap: () => onVote(0)),
                  const SizedBox(width: 6),
                  _VoteChip(label: '持平', selected: vote == 2, onTap: () => onVote(2)),
                  const SizedBox(width: 6),
                  _VoteChip(label: 'B 更好', selected: vote == 1, onTap: () => onVote(1)),
                ],
              ),
              // 生成的查询（结构化）
              if (turn.generatedQueriesJson.isNotEmpty) ...[
                const SizedBox(height: 10),
                _QuerySection(raw: turn.generatedQueriesJson),
              ],
            ],
          ],
        ),
      ),
    );
  }

  bool _tracesIdentical(PlayTrace? a, PlayTrace? b) {
    if (a == null || b == null) return false;
    if (a.merged.length != b.merged.length) return false;
    for (var i = 0; i < a.merged.length; i++) {
      if (a.merged[i].name != b.merged[i].name ||
          a.merged[i].score.toStringAsFixed(4) != b.merged[i].score.toStringAsFixed(4)) {
        return false;
      }
    }
    if (a.perQuery.length != b.perQuery.length) return false;
    for (var i = 0; i < a.perQuery.length; i++) {
      if (a.perQuery[i].sim != b.perQuery[i].sim ||
          a.perQuery[i].ppr != b.perQuery[i].ppr ||
          a.perQuery[i].action != b.perQuery[i].action) {
        return false;
      }
    }
    return true;
  }
}

class _VoteChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _VoteChip({required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
        decoration: BoxDecoration(
          color: selected ? Theme.of(context).colorScheme.primary.withValues(alpha: 0.18) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: selected ? Theme.of(context).colorScheme.primary : Colors.grey.shade700,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: selected ? Theme.of(context).colorScheme.primary : AppColors.subtle,
            fontWeight: selected ? FontWeight.w700 : FontWeight.w400,
          ),
        ),
      ),
    );
  }
}

/// 生成的查询（结构化卡片，不再贴原始 JSON）。
class _QuerySection extends StatelessWidget {
  final String raw;
  const _QuerySection({required this.raw});

  @override
  Widget build(BuildContext context) {
    final queries = _parseQueries(raw);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('生成的查询 (${queries.length})',
            style: const TextStyle(fontSize: 12, color: AppColors.subtle)),
        const SizedBox(height: 6),
        for (var i = 0; i < queries.length; i++)
          _QueryCard(index: i, query: queries[i]),
      ],
    );
  }

  List<Map<String, dynamic>> _parseQueries(String raw) {
    try {
      final decoded = jsonDecode(raw);
      if (decoded is List) {
        return decoded.whereType<Map>().cast<Map<String, dynamic>>().toList();
      }
      if (decoded is Map) {
        final arr = decoded['queries'];
        if (arr is List) return arr.whereType<Map>().cast<Map<String, dynamic>>().toList();
      }
    } catch (_) {}
    return const [];
  }
}

class _QueryCard extends StatelessWidget {
  final int index;
  final Map<String, dynamic> query;
  const _QueryCard({required this.index, required this.query});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final priority = query['priority'];
    final tags = _tagsOf(query);
    final lines = _variantLines(query['variant']);
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerHighest,
      margin: const EdgeInsets.only(bottom: 6),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('Q${index + 1}',
                    style: TextStyle(
                        color: scheme.primary, fontWeight: FontWeight.w700, fontSize: 12)),
                if (priority != null) ...[
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: AppColors.runningBg,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text('优先级 $priority',
                        style: const TextStyle(fontSize: 10, color: AppColors.running)),
                  ),
                ],
                const Spacer(),
                if (query['dropped'] == true)
                  const Text('已丢弃', style: TextStyle(fontSize: 10, color: AppColors.fail)),
              ],
            ),
            if (tags.isNotEmpty) ...[
              const SizedBox(height: 6),
              Wrap(
                spacing: 4,
                runSpacing: 4,
                children: [
                  for (final t in tags)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: scheme.surfaceContainerLow,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(t, style: const TextStyle(fontSize: 10)),
                    ),
                ],
              ),
            ],
            if (lines.isNotEmpty) ...[
              const SizedBox(height: 6),
              for (final l in lines.take(3))
                Text(l,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
              if (lines.length > 3)
                Text('… 共 ${lines.length} 条',
                    style: const TextStyle(fontSize: 10, color: AppColors.subtle)),
            ],
          ],
        ),
      ),
    );
  }

  List<String> _tagsOf(Map<String, dynamic> q) {
    final direct = q['tags'];
    if (direct is List) return direct.whereType<String>().toList();
    final variant = q['variant'];
    if (variant is Map) {
      final out = <String>[];
      for (final v in variant.values) {
        if (v is List) {
          for (final unit in v) {
            if (unit is Map && unit['tags'] is List) {
              out.addAll((unit['tags'] as List).whereType<String>());
            }
          }
        }
      }
      return out;
    }
    return const [];
  }

  List<String> _variantLines(dynamic variant) {
    final out = <String>[];
    if (variant is! Map) return out;
    for (final v in variant.values) {
      if (v is List) {
        for (final unit in v) {
          if (unit is Map) {
            final parts = <String>[];
            for (final uv in unit.values) {
              if (uv is String && uv.trim().isNotEmpty) {
                parts.add(uv);
              } else if (uv is List) {
                parts.addAll(uv.whereType<String>());
              }
            }
            if (parts.isNotEmpty) out.add(parts.join(' · '));
          }
        }
      }
    }
    return out;
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
                  const _TraceHeader('检索结果'),
                  for (final n in trace.merged)
                    _NodeRow(name: n.name, stage: n.stage, score: n.score, content: n.content),
                ],
                if (trace.actions.isNotEmpty) ...[
                  const _TraceHeader('行为倾向'),
                  for (final n in trace.actions)
                    _NodeRow(name: n.name, stage: n.stage, score: n.score, content: n.content),
                ],
                if (trace.speech.isNotEmpty) ...[
                  const _TraceHeader('说话风格'),
                  for (final n in trace.speech)
                    _NodeRow(name: n.name, stage: n.stage, score: n.score, content: n.content),
                ],
                if (trace.think.isNotEmpty) ...[
                  const _TraceHeader('思维习惯'),
                  for (final n in trace.think)
                    _NodeRow(name: n.name, stage: n.stage, score: n.score, content: n.content),
                ],
                if (trace.perQuery.isNotEmpty) ...[
                  const _TraceHeader('子查询'),
                  for (final q in trace.perQuery)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              '${q.dropped ? '[丢弃] ' : ''}${q.preview.isEmpty ? '（空）' : q.preview}',
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(fontFamily: 'monospace', fontSize: 10),
                            ),
                          ),
                          _StageBadge(label: '相似 ${q.sim}'),
                          if (q.ppr > 0) ...[
                            const SizedBox(width: 4),
                            _StageBadge(label: 'PPR ${q.ppr}'),
                          ],
                          if (q.action > 0) ...[
                            const SizedBox(width: 4),
                            _StageBadge(label: '动作 ${q.action}'),
                          ],
                        ],
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
}

class _TraceHeader extends StatelessWidget {
  final String title;
  const _TraceHeader(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 4, bottom: 2),
      child: Text(title, style: const TextStyle(color: AppColors.subtle, fontSize: 10)),
    );
  }
}

class _NodeRow extends StatelessWidget {
  final String name;
  final String stage;
  final double score;
  final String content;
  const _NodeRow({
    required this.name,
    required this.stage,
    required this.score,
    required this.content,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Expanded(
            child: Text(
              content.isEmpty ? name : '$name · ${content.length > 24 ? '${content.substring(0, 24)}…' : content}',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
            ),
          ),
          _StageBadge(label: stage),
          const SizedBox(width: 6),
          Text(score.toStringAsFixed(3),
              style: const TextStyle(fontFamily: 'monospace', fontSize: 10, color: AppColors.subtle)),
        ],
      ),
    );
  }
}

/// 命中阶段徽章：similarity 蓝 / ppr 紫 / action 琥珀 / both 青。
class _StageBadge extends StatelessWidget {
  final String label;
  const _StageBadge({required this.label});

  @override
  Widget build(BuildContext context) {
    final color = switch (label) {
      'similarity' => const Color(0xFF4FC3F7),
      'ppr' => const Color(0xFFBA68C8),
      'action' => const Color(0xFFFFB74D),
      'both' => const Color(0xFF4DB6AC),
      _ => AppColors.subtle,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(label,
          style: TextStyle(fontSize: 9, color: color, fontFamily: 'monospace')),
    );
  }
}
