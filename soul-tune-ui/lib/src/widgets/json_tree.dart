import 'package:flutter/material.dart';

/// 可折叠 JSON 树（检视数据集用）：递归渲染，Map 键色、标量等宽。
class JsonTree extends StatelessWidget {
  final dynamic data;
  const JsonTree({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    return _Node(data: data, keyName: 'root');
  }
}

class _Node extends StatefulWidget {
  final dynamic data;
  final String keyName;
  const _Node({required this.data, required this.keyName});

  @override
  State<_Node> createState() => _NodeState();
}

class _NodeState extends State<_Node> {
  late bool _expanded = widget.data is Map || widget.data is List;

  String get _preview {
    final d = widget.data;
    if (d is Map) return '{ ${d.length} 项 }';
    if (d is List) return '[ ${d.length} 项 ]';
    return d.toString();
  }

  @override
  Widget build(BuildContext context) {
    final d = widget.data;
    if (d is Map) {
      final entries = d.entries.toList();
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _Toggle(keyName: widget.keyName, expanded: _expanded, preview: _preview,
              onTap: () => setState(() => _expanded = !_expanded)),
          if (_expanded)
            Padding(
              padding: const EdgeInsets.only(left: 18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final e in entries)
                    _Node(data: e.value, keyName: e.key.toString()),
                ],
              ),
            ),
        ],
      );
    }
    if (d is List) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _Toggle(keyName: widget.keyName, expanded: _expanded, preview: _preview,
              onTap: () => setState(() => _expanded = !_expanded)),
          if (_expanded)
            Padding(
              padding: const EdgeInsets.only(left: 18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (var i = 0; i < d.length; i++)
                    _Node(data: d[i], keyName: '[$i]'),
                ],
              ),
            ),
        ],
      );
    }
    // 标量
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(width: 8, child: Text('•', style: TextStyle(fontSize: 10, color: Colors.grey))),
          _KeyLabel(widget.keyName),
          const SizedBox(width: 8),
          Expanded(
            child: SelectableText(
              d.toString(),
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 13,
                color: _scalarColor(d),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _scalarColor(dynamic v) => switch (v) {
        final String _ => const Color(0xFF9CCC65),
        final num _ => const Color(0xFF4FC3F7),
        final bool _ => const Color(0xFFBA68C8),
        null => const Color(0xFF9E9E9E),
        _ => Colors.white,
      };
}

class _Toggle extends StatelessWidget {
  final String keyName;
  final bool expanded;
  final String preview;
  final VoidCallback onTap;
  const _Toggle({
    required this.keyName,
    required this.expanded,
    required this.preview,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(4),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(
          children: [
            Icon(
              expanded ? Icons.arrow_drop_down : Icons.arrow_right,
              size: 18,
              color: Theme.of(context).colorScheme.primary,
            ),
            _KeyLabel(keyName),
            const SizedBox(width: 8),
            Text(preview, style: const TextStyle(color: Color(0xFF9E9E9E), fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

class _KeyLabel extends StatelessWidget {
  final String keyName;
  const _KeyLabel(this.keyName);

  @override
  Widget build(BuildContext context) {
    return Text(
      keyName,
      style: TextStyle(
        color: Theme.of(context).colorScheme.primary,
        fontWeight: FontWeight.w600,
        fontSize: 13,
      ),
    );
  }
}
