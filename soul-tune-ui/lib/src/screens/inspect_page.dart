import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../bridge.dart';
import '../models.dart';
import '../theme.dart';
import '../widgets/json_tree.dart';

/// 检视数据集：信息架构与交互逻辑复刻 TUI `states::inspect.rs`——
///
/// 布局（自顶向下）：
/// 1. 头部条：`检视数据集 · {文件名} [{图|查询}] · {N}条 | {首条统计}`；
/// 2. 左栏：图统计面板（仅图）+ 可滚动、可选的条目列表（节点/用例）；
/// 3. 右栏：选中条目的 **预览**（preview_lines）或 **详情**
///    （基本信息 detail_lines 固定 + 连接列表可滚动，点击连接可跳转邻居）；
/// 4. 底部状态栏：当前选中与操作提示。
///
/// 交互逻辑：点击条目 → 预览；双击 / 详情按钮 → 详情；点击连接 → 跳转邻居
/// （导航栈记录，可逐层返回）；键盘 ↑↓/Enter/Backspace/Esc 与 TUI 一致。
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

  // ── 导航状态（对应 TUI InspectState）──
  int _cursor = 0; // 当前选中条目
  bool _detail = false; // 详情模式（false = 预览）
  final List<(int, int)> _navStack = []; // (上一条目, 链接索引)
  final ScrollController _listCtrl = ScrollController();
  final ScrollController _detailCtrl = ScrollController();

  @override
  void dispose() {
    _listCtrl.dispose();
    _detailCtrl.dispose();
    super.dispose();
  }

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
      _cursor = 0;
      _detail = false;
      _navStack.clear();
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
        // 加载后默认直接展开第一个节点的详情（滚动回顶延迟到 build 后）
        if ((results[0] as InspectEntries).entries.isNotEmpty) {
          _detail = true;
          _cursor = 0;
        }
      });
      _jumpDetailTop();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '读取失败: $e';
        _loading = false;
      });
    }
  }

  // ── 导航逻辑（对应 TUI handle_key）──

  void _selectEntry(int i) {
    setState(() {
      _cursor = i.clamp(0, (_data?.entries.length ?? 1) - 1);
      _scrollListToCursor();
    });
  }

  void _scrollListToCursor() {
    if (!_listCtrl.hasClients) return;
    const extent = 48.0;
    final target = _cursor * extent;
    final pos = _listCtrl.position;
    if (target < pos.pixels) {
      _listCtrl.animateTo(target, duration: const Duration(milliseconds: 120), curve: Curves.easeOut);
    } else if (target + extent > pos.pixels + pos.viewportDimension) {
      _listCtrl.animateTo(
          (target + extent - pos.viewportDimension).clamp(0.0, pos.maxScrollExtent),
          duration: const Duration(milliseconds: 120),
          curve: Curves.easeOut);
    }
  }

  void _openDetail() {
    setState(() {
      _detail = true;
    });
    _jumpDetailTop();
  }

  void _closeDetail() {
    setState(() {
      _detail = false;
      _navStack.clear();
    });
    _jumpDetailTop();
  }

  /// 详情滚动回顶部：延迟到 build 之后执行，且仅在 controller 已 attach 时生效
  /// （避免 "ScrollController not attached to any scroll views" 断言）。
  void _jumpDetailTop() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_detailCtrl.hasClients) _detailCtrl.jumpTo(0);
    });
  }

  /// 点击连接：跳转邻居（压入导航栈），对应 TUI Enter on link。
  void _jumpTo(int linkIdx) {
    final data = _data;
    if (data == null || _cursor >= data.entries.length) return;
    final links = data.entries[_cursor].links;
    if (linkIdx < 0 || linkIdx >= links.length) return;
    final link = links[linkIdx];
    final target = link.targetIdx;
    if (target >= data.entries.length || target == _cursor) return;
    setState(() {
      _navStack.add((_cursor, linkIdx));
      _cursor = target;
      _detail = true;
      _scrollListToCursor();
    });
    _jumpDetailTop();
    // 跳转反馈：明确告知用户已跟随连接跳转（可返回）
    final targetName = link.isOutgoing ? link.toId : link.fromId;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(
        content: Text('已跳转 → $targetName  （可按 Backspace/Esc 返回）',
            style: const TextStyle(fontSize: 12)),
        behavior: SnackBarBehavior.floating,
        duration: const Duration(milliseconds: 1500),
        margin: const EdgeInsets.only(left: 340, right: 16, bottom: 40),
      ));
  }

  /// 返回一层（导航栈）或退出详情，对应 TUI Backspace/Esc。
  void _back() {
    if (_navStack.isNotEmpty) {
      final (prevIdx, prevLink) = _navStack.removeLast();
      setState(() {
        _cursor = prevIdx;
        _detail = true;
        _scrollListToCursor();
      });
      _jumpDetailTop();
    } else if (_detail) {
      _closeDetail();
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = _data;
    final error = _error;
    return Scaffold(
      appBar: AppBar(title: const Text('检视数据集')),
      body: Focus(
        autofocus: true,
        onKeyEvent: (node, event) {
          if (event is! KeyDownEvent) return KeyEventResult.ignored;
          final key = event.logicalKey;
          if (key == LogicalKeyboardKey.arrowUp) {
            if (_cursor > 0) _selectEntry(_cursor - 1);
            return KeyEventResult.handled;
          } else if (key == LogicalKeyboardKey.arrowDown) {
            final n = _data?.entries.length ?? 0;
            if (n > 0 && _cursor + 1 < n) _selectEntry(_cursor + 1);
            return KeyEventResult.handled;
          } else if (key == LogicalKeyboardKey.enter) {
            if (_data != null && _data!.entries.isNotEmpty) _openDetail();
            return KeyEventResult.handled;
          } else if (key == LogicalKeyboardKey.backspace || key == LogicalKeyboardKey.escape) {
            _back();
            return KeyEventResult.handled;
          }
          return KeyEventResult.ignored;
        },
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1180),
            child: Column(
              children: [
                // 打开文件行
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
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
                              style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                Expanded(
                  child: _loading
                      ? const Center(child: CircularProgressIndicator())
                      : error != null
                          ? Center(
                              child: Text(error,
                                  style: const TextStyle(color: AppColors.fail)))
                          : data != null
                              ? _buildContent(data, _raw)
                              : const Center(
                                  child: Text('选择一个 question.json 或 graph.json 开始检视',
                                      style: TextStyle(color: AppColors.subtle)),
                                ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContent(InspectEntries data, InspectFile? raw) {
    if (data.entries.isEmpty) {
      return ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _headerBar(data),
          const SizedBox(height: 10),
          const Card(
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Text('（无结构化条目，可展开下方原始 JSON 查看）',
                  style: TextStyle(color: AppColors.subtle)),
            ),
          ),
          const SizedBox(height: 8),
          _rawJsonCard(raw),
        ],
      );
    }
    // 越界保护（导航跳转目标可能被裁剪）
    if (_cursor >= data.entries.length) _cursor = data.entries.length - 1;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _headerBar(data),
        const SizedBox(height: 10),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ── 左栏：统计 + 条目列表 ──
              SizedBox(
                width: 320,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (data.stats.isNotEmpty) ...[
                      _statsPanel(data.stats),
                      const SizedBox(height: 8),
                    ],
                    Expanded(child: _entryList(data)),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              // ── 右栏：预览 / 详情 + 原始 JSON ──
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      child: _detail
                          ? _detailPanel(data, _cursor)
                          : _previewPanel(data, _cursor),
                    ),
                    const SizedBox(height: 8),
                    _rawJsonCard(raw),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        _statusBar(data),
      ],
    );
  }

  // ── 头部条（TUI Block::title）──
  Widget _headerBar(InspectEntries data) {
    final isGraph = data.fileType == 'graph';
    final scheme = Theme.of(context).colorScheme;
    final typeLabel = isGraph ? '图' : '查询';
    final firstStat = data.stats.isEmpty ? '' : ' | ${data.stats.first}';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant, width: 0.5),
      ),
      child: Row(
        children: [
          Icon(isGraph ? Icons.account_tree_outlined : Icons.quiz_outlined,
              size: 18, color: scheme.primary),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              '检视数据集 · ${_fileName(data.filePath)} [$typeLabel] · ${data.entries.length}条$firstStat',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }

  String _fileName(String path) {
    final parts = path.replaceAll('\\', '/').split('/');
    return parts.isEmpty ? path : parts.last;
  }

  // ── 图统计面板（TUI render_stats_panel）──
  // 可折叠：默认只显示前几项摘要（紧凑，不挤压节点列表），
  // 点击展开后以两列网格滚动查看全部统计项。
  Widget _statsPanel(List<String> stats) {
    return _StatsPanel(stats: stats);
  }

  // ── 条目列表（TUI render_entry_list）──
  Widget _entryList(InspectEntries data) {
    final isGraph = data.fileType == 'graph';
    final scheme = Theme.of(context).colorScheme;
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: scheme.outlineVariant, width: 0.5),
        borderRadius: BorderRadius.circular(10),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            color: scheme.surfaceContainerHigh,
            child: Text(isGraph ? '节点列表' : '用例列表',
                style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: scheme.primary)),
          ),
          Expanded(
            child: ListView.builder(
              controller: _listCtrl,
              itemExtent: 48,
              itemCount: data.entries.length,
              itemBuilder: (context, i) {
                final e = data.entries[i];
                final selected = i == _cursor;
                final hasLinks = e.links.isNotEmpty;
                return Material(
                  color: selected
                      ? scheme.primary.withValues(alpha: 0.14)
                      : Colors.transparent,
                  child: InkWell(
                    onTap: () {
                      _selectEntry(i);
                      _openDetail();
                    },
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      child: Row(
                        children: [
                          Icon(
                            isGraph
                                ? _nodeTypeIcon(e.summary)
                                : Icons.quiz_outlined,
                            size: 15,
                            color: selected ? scheme.primary : AppColors.subtle,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              e.summary,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontSize: 12,
                                fontFamily: 'monospace',
                                color: selected ? scheme.primary : null,
                              ),
                            ),
                          ),
                          if (hasLinks) ...[
                            const SizedBox(width: 6),
                            Icon(Icons.alt_route, size: 13, color: AppColors.subtle),
                            Text('${e.links.length}',
                                style: const TextStyle(
                                    fontFamily: 'monospace', fontSize: 10, color: AppColors.subtle)),
                          ],
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  IconData _nodeTypeIcon(String summary) {
    if (summary.startsWith('Semantic')) return Icons.memory_outlined;
    if (summary.startsWith('Situation')) return Icons.place_outlined;
    if (summary.startsWith('Procedure')) return Icons.build_outlined;
    if (summary.startsWith('AbstractSit')) return Icons.layers_outlined;
    return Icons.circle_outlined;
  }

  // ── 预览面板（TUI render_preview）──
  Widget _previewPanel(InspectEntries data, int idx) {
    final e = data.entries[idx];
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Text('预览 · ${e.id}',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: scheme.primary)),
              const Spacer(),
              if (e.links.isNotEmpty)
                TextButton.icon(
                  onPressed: _openDetail,
                  icon: const Icon(Icons.open_in_full, size: 15),
                  label: const Text('查看详情', style: TextStyle(fontSize: 12)),
                ),
            ],
          ),
          const Divider(height: 14),
          Expanded(
            child: e.previewLines.isEmpty
                ? const Text('（无预览）', style: TextStyle(color: AppColors.subtle))
                : ListView(
                    children: [
                      for (final l in e.previewLines)
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 2),
                          child: SelectableText(l,
                              style: const TextStyle(
                                  fontFamily: 'monospace', fontSize: 12, height: 1.5)),
                        ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  // ── 详情面板（TUI render_full_detail）──
  // 布局：标题+返回 → **连接区（固定可见）** → 基本信息（可滚动）。
  // 出边/入边必须一眼可见（用户核心诉求），不埋在长文本里。
  Widget _detailPanel(InspectEntries data, int idx) {
    final e = data.entries[idx];
    final scheme = Theme.of(context).colorScheme;
    final outgoing = e.links.where((l) => l.isOutgoing).toList();
    final incoming = e.links.where((l) => !l.isOutgoing).toList();

    return Container(
      padding: const EdgeInsets.fromLTRB(14, 8, 14, 10),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // 标题 + 返回
          Row(
            children: [
              if (_navStack.isNotEmpty)
                IconButton(
                  tooltip: '返回上一层（${_navStack.length}）',
                  onPressed: _back,
                  icon: const Icon(Icons.arrow_back, size: 18),
                )
              else
                IconButton(
                  tooltip: '返回预览',
                  onPressed: _closeDetail,
                  icon: const Icon(Icons.close, size: 18),
                ),
              Expanded(
                child: Text('详情 · ${e.id}',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w700, color: scheme.primary)),
              ),
              if (_navStack.isNotEmpty)
                Text('导航栈 ${_navStack.length} 层',
                    style: const TextStyle(fontSize: 11, color: AppColors.subtle)),
            ],
          ),
          const SizedBox(height: 2),
          // ── 连接区（占据详情主要空间、内部可滚动）──
          // 记忆节点的基本信息通常很短，不应抢占空间；连接（出边/入边）才是核心导航。
          Expanded(
            flex: 3,
            child: Container(
              decoration: BoxDecoration(
                color: scheme.surfaceContainerLow.withValues(alpha: 0.55),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(10, 6, 10, 2),
                    child: Row(
                      children: [
                        Text('── 连接 (Links) ${e.links.length}条 ──',
                            style: TextStyle(
                                fontSize: 11.5,
                                fontWeight: FontWeight.w700,
                                color: scheme.primary)),
                        const Spacer(),
                        const Text('点击跳转邻居',
                            style: TextStyle(fontSize: 10, color: AppColors.subtle)),
                      ],
                    ),
                  ),
                  Expanded(
                    child: e.links.isEmpty
                        ? const Padding(
                            padding: EdgeInsets.fromLTRB(10, 4, 10, 0),
                            child: Text('(无连接)',
                                style: TextStyle(color: AppColors.subtle, fontSize: 12)),
                          )
                        : ListView(
                            padding: const EdgeInsets.fromLTRB(4, 0, 4, 6),
                            children: [
                              if (outgoing.isNotEmpty) ...[
                                const Padding(
                                  padding: EdgeInsets.fromLTRB(6, 2, 6, 2),
                                  child: Text('出边 (→)',
                                      style: TextStyle(fontSize: 11, color: AppColors.subtle)),
                                ),
                                for (var i = 0; i < e.links.length; i++)
                                  if (e.links[i].isOutgoing)
                                    _LinkRow(link: e.links[i], onTap: () => _jumpTo(i)),
                              ],
                              if (incoming.isNotEmpty) ...[
                                const Padding(
                                  padding: EdgeInsets.fromLTRB(6, 3, 6, 2),
                                  child: Text('入边 (←)',
                                      style: TextStyle(fontSize: 11, color: AppColors.subtle)),
                                ),
                                for (var i = 0; i < e.links.length; i++)
                                  if (!e.links[i].isOutgoing)
                                    _LinkRow(link: e.links[i], onTap: () => _jumpTo(i)),
                              ],
                            ],
                          ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 6),
          const Divider(height: 8),
          // ── 基本信息（内容少则短，最多占 240px，内部可滚动）──
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: 240),
            child: SingleChildScrollView(
              controller: _detailCtrl,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text('── 基本信息 ──',
                      style: TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w700,
                          color: scheme.primary)),
                  const SizedBox(height: 6),
                  for (final l in e.detailLines)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 1),
                      child: SelectableText(l,
                          style: const TextStyle(
                              fontFamily: 'monospace', fontSize: 11.5, height: 1.45)),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── 底部状态栏（TUI status_bar）──
  Widget _statusBar(InspectEntries data) {
    final scheme = Theme.of(context).colorScheme;
    final hints = _detail
        ? ' [↑↓] 选择条目（自动进详情） · 点击连接跳转邻居 · [Back/Esc] 返回'
        : ' [↑↓] 选择条目 · 点击查看详情 · 点击连接跳转邻居';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: scheme.outlineVariant, width: 0.5),
      ),
      child: Row(
        children: [
          Icon(Icons.keyboard, size: 14, color: AppColors.subtle),
          const SizedBox(width: 6),
          Expanded(
            child: Text('${_cursor + 1} / ${data.entries.length}$hints',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontSize: 11, color: AppColors.subtle)),
          ),
        ],
      ),
    );
  }

  Widget _rawJsonCard(InspectFile? raw) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      margin: EdgeInsets.zero,
      color: scheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          dense: true,
          title: const Text('原始 JSON',
              style: TextStyle(fontSize: 12, color: AppColors.subtle)),
          children: [
            ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 300),
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                child: JsonTree(data: raw?.data ?? const {}),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 图统计面板：可折叠两列网格。
/// 折叠态（默认）只显示前 4 项摘要，高度紧凑；展开后滚动查看全部。
class _StatsPanel extends StatefulWidget {
  final List<String> stats;
  const _StatsPanel({required this.stats});

  @override
  State<_StatsPanel> createState() => _StatsPanelState();
}

class _StatsPanelState extends State<_StatsPanel> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final stats = widget.stats;
    final shown = _expanded ? stats : stats.take(4).toList();
    return Container(
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: scheme.outlineVariant, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          InkWell(
            onTap: () => setState(() => _expanded = !_expanded),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(9)),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              child: Row(
                children: [
                  Icon(Icons.query_stats, size: 14, color: scheme.primary),
                  const SizedBox(width: 6),
                  Text('图统计 · ${stats.length} 项',
                      style: TextStyle(
                          fontSize: 12, fontWeight: FontWeight.w700, color: scheme.primary)),
                  const Spacer(),
                  Icon(
                    _expanded ? Icons.expand_less : Icons.expand_more,
                    size: 16,
                    color: AppColors.subtle,
                  ),
                ],
              ),
            ),
          ),
          if (!_expanded)
            // 折叠摘要：两列网格显示前 4 项
            Padding(
              padding: const EdgeInsets.fromLTRB(10, 0, 10, 8),
              child: _StatsGrid(stats: shown, scrollable: false),
            )
          else
            // 展开：固定高度滚动显示全部（不能用 Flexible——左栏 Column 主轴无界会触发 RenderFlex 异常）
            Container(
              height: 220,
              padding: const EdgeInsets.fromLTRB(10, 0, 10, 8),
              child: _StatsGrid(stats: shown, scrollable: true),
            ),
        ],
      ),
    );
  }
}

/// 图统计两列网格（可滚动/不可滚动）。
class _StatsGrid extends StatelessWidget {
  final List<String> stats;
  final bool scrollable;
  const _StatsGrid({required this.stats, required this.scrollable});

  @override
  Widget build(BuildContext context) {
    final cells = [
      for (final s in stats)
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 1.5),
          child: Text(s,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style:
                  const TextStyle(fontFamily: 'monospace', fontSize: 12, height: 1.3)),
        ),
    ];
    if (!scrollable) {
      return GridView.count(
        crossAxisCount: 2,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        childAspectRatio: 4.2,
        crossAxisSpacing: 10,
        children: cells,
      );
    }
    return GridView.count(
      crossAxisCount: 2,
      childAspectRatio: 4.2,
      crossAxisSpacing: 10,
      children: cells,
    );
  }
}

/// 连接行：可点击跳转邻居（对应 TUI 连接光标行）。
/// 带 hover 高亮 + 按压缩放反馈 + tooltip，明确可点击。
class _LinkRow extends StatefulWidget {
  final InspectLink link;
  final VoidCallback onTap;
  const _LinkRow({required this.link, required this.onTap});

  @override
  State<_LinkRow> createState() => _LinkRowState();
}

class _LinkRowState extends State<_LinkRow> {
  bool _hover = false;
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final link = widget.link;
    final target = link.isOutgoing ? link.toId : link.fromId;
    return Tooltip(
      message: '跳转到 ${link.isOutgoing ? "目标" : "来源"}节点 $target',
      waitDuration: const Duration(milliseconds: 400),
      child: MouseRegion(
        onEnter: (_) => setState(() => _hover = true),
        onExit: (_) => setState(() {
          _hover = false;
          _pressed = false;
        }),
        child: GestureDetector(
          onTapDown: (_) => setState(() => _pressed = true),
          onTapUp: (_) => setState(() => _pressed = false),
          onTapCancel: () => setState(() => _pressed = false),
          onTap: widget.onTap,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 100),
            transform:
                _pressed ? Matrix4.translationValues(0, 1, 0) : Matrix4.identity(),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
            decoration: BoxDecoration(
              color: _hover
                  ? scheme.primary.withValues(alpha: 0.16)
                  : scheme.surfaceContainerLow.withValues(alpha: 0.5),
              borderRadius: BorderRadius.circular(6),
              border: _hover
                  ? Border.all(color: scheme.primary.withValues(alpha: 0.5), width: 1)
                  : null,
            ),
            margin: const EdgeInsets.symmetric(vertical: 2),
            child: Row(
              children: [
                Icon(link.isOutgoing ? Icons.arrow_forward : Icons.arrow_back,
                    size: 13, color: _hover ? scheme.primary : AppColors.subtle),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    '$target  ${link.linkTypeDesc}  [${link.intensity.toStringAsFixed(2)}]',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                        fontFamily: 'monospace', fontSize: 11.5, color: scheme.onSurface),
                  ),
                ),
                Icon(Icons.north_east,
                    size: 12, color: _hover ? scheme.primary : AppColors.subtle),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
