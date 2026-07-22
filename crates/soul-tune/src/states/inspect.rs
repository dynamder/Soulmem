use std::collections::HashMap;
use std::path::PathBuf;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::widgets::list;
use crate::widgets::scroll::ScrollState;
use crate::widgets::status_bar;

#[derive(Clone, PartialEq)]
pub enum InspectFileType {
    Graph,
    Query,
}

pub struct LinkDisplay {
    pub from_id: String,
    pub to_id: String,
    pub target_idx: usize,
    pub link_type_desc: String,
    pub intensity: f64,
    pub is_outgoing: bool,
}

pub struct InspectEntry {
    pub id: String,
    pub summary: String,
    pub preview_lines: Vec<String>,
    pub detail_lines: Vec<String>,
    pub links: Vec<LinkDisplay>,
}

pub struct DetailState {
    pub link_cursor: Option<usize>,
    pub nav_stack: Vec<(usize, usize)>,
}

pub struct InspectState {
    pub file_path: PathBuf,
    pub file_type: InspectFileType,
    pub entries: Vec<InspectEntry>,
    pub stats: Option<Vec<String>>,
    pub list_scroll: ScrollState,
    pub detail: Option<DetailState>,
    pub detail_scroll: ScrollState,
}

impl InspectState {
    pub fn new(file_path: PathBuf) -> Self {
        let content = std::fs::read_to_string(&file_path).unwrap_or_default();
        let val: serde_json::Value =
            serde_json::from_str(&content).unwrap_or(serde_json::Value::Null);

        let (file_type, entries) = if val.is_array() {
            (InspectFileType::Graph, parse_graph_nodes(&val))
        } else if let Some(cases) = val.get("test_cases").and_then(|v| v.as_array()) {
            if !cases.is_empty() {
                (InspectFileType::Query, parse_query_cases(&val))
            } else {
                (InspectFileType::Query, parse_query_cases(&val))
            }
        } else if let Some(_nodes) = val.get("nodes").and_then(|v| v.as_array()) {
            (
                InspectFileType::Graph,
                parse_graph_nodes(&val.get("nodes").unwrap()),
            )
        } else {
            (InspectFileType::Query, parse_query_cases(&val))
        };

        // Load sibling graph_stats.json if available
        let stats = if file_type == InspectFileType::Graph {
            let parent = file_path.parent().unwrap_or(std::path::Path::new(""));
            let stats_path = parent.join("graph_stats.json");
            if stats_path.exists() {
                load_graph_stats(&stats_path)
            } else {
                None
            }
        } else {
            None
        };

        Self {
            file_path,
            file_type,
            entries,
            stats,
            list_scroll: ScrollState::new(),
            detail: None,
            detail_scroll: ScrollState::new(),
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .split(area);

        let type_label = match self.file_type {
            InspectFileType::Graph => "图",
            InspectFileType::Query => "查询",
        };
        let fname = self
            .file_path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        let title_suffix = if let Some(ref stat_lines) = self.stats {
            if let Some(first) = stat_lines.first() {
                format!(" | {}", first)
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        Block::bordered()
            .title(format!(
                " 检视数据集 · {} [{}] · {}条{}",
                fname,
                type_label,
                self.entries.len(),
                title_suffix
            ))
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let mid_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Fill(1), Constraint::Fill(1)])
            .split(layout[1]);

        // If stats exist, show them in a left panel above the entry list
        if self.stats.is_some() {
            let upper_left_size = if self.entries.len() > 10 { 8 } else { 6 };
            let left_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Length(upper_left_size),
                    Constraint::Fill(1),
                ])
                .split(mid_layout[0]);

            self.render_stats_panel(frame, left_layout[0]);
            self.render_entry_list(frame, left_layout[1]);
        } else {
            self.render_entry_list(frame, mid_layout[0]);
        }
        self.render_detail_panel(frame, mid_layout[1]);

        let nav_hints = if self.detail.is_some() {
            vec![
                ("[↑↓]".into(), "选择节点".into()),
                ("[Ctrl+↑↓]".into(), "选择链接".into()),
                ("[Enter]".into(), "跳转邻居".into()),
                ("[Back]".into(), "返回".into()),
                ("[Esc/Q]".into(), "退出检视".into()),
            ]
        } else {
            vec![
                ("[↑↓]".into(), "选择节点".into()),
                ("[Enter]".into(), "查看详情".into()),
                ("[Esc/Q]".into(), "返回".into()),
            ]
        };
        status_bar::render_status_bar(frame, layout[2], &nav_hints);
    }

    fn render_entry_list(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered().title(format!(
            " {} ",
            match self.file_type {
                InspectFileType::Graph => "节点列表",
                InspectFileType::Query => "用例列表",
            }
        ));
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let items: Vec<String> = self.entries.iter().map(|e| e.summary.clone()).collect();
        let (content_rect, bar_rect) = ScrollState::split_area(inner);
        let offset = ScrollState::offset(content_rect.height, items.len(), self.list_scroll.cursor);
        let s = ScrollState {
            cursor: self.list_scroll.cursor,
            offset,
        };
        list::render_simple_list(frame, content_rect, &items, &s);
        ScrollState::render_scrollbar(frame, bar_rect, items.len(), content_rect.height, offset);
    }

    fn render_detail_panel(&self, frame: &mut Frame, area: Rect) {
        if self.entries.is_empty() {
            let block = Block::bordered().title(" 详情 ");
            let inner = block.inner(area);
            block.render(area, frame.buffer_mut());
            frame.render_widget(Paragraph::new("(无数据)").fg(Color::DarkGray), inner);
            return;
        }

        let entry = &self.entries[self.list_scroll.cursor];

        if self.detail.is_some() {
            self.render_full_detail(frame, area, entry);
        } else {
            self.render_preview(frame, area, entry);
        }
    }

    fn render_preview(&self, frame: &mut Frame, area: Rect, entry: &InspectEntry) {
        let block = Block::bordered().title(format!(" 预览 · {} ", entry.id));
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let mut lines: Vec<Line> = Vec::new();
        for line in &entry.preview_lines {
            lines.push(Line::from(Span::raw(line)));
        }
        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn render_full_detail(&self, frame: &mut Frame, area: Rect, entry: &InspectEntry) {
        let block = Block::bordered().title(format!(" 详情 · {} ", entry.id));
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let green = Style::new().green();
        let yellow = Style::new().yellow().bold();
        let gray = Style::new().dark_gray();
        let _red = Style::new().red();

        // --- Fixed header: 基本信息 ---
        let mut header_lines: Vec<Line> = Vec::new();
        header_lines.push(Line::from(Span::styled(" ── 基本信息 ──", yellow)));
        for line in &entry.detail_lines {
            header_lines.push(Line::from(Span::raw(format!("  {}", line))));
        }
        header_lines.push(Line::from(""));

        // --- Scrollable content: 连接 + 导航信息 ---
        let mut link_lines: Vec<Line> = Vec::new();
        let mut link_cursor_line: Option<usize> = None;

        if !entry.links.is_empty() {
            link_lines.push(Line::from(Span::styled(" ── 连接 (Links) ──", yellow)));

            let outgoing: Vec<usize> = entry
                .links
                .iter()
                .enumerate()
                .filter(|(_, l)| l.is_outgoing)
                .map(|(i, _)| i)
                .collect();
            let incoming: Vec<usize> = entry
                .links
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.is_outgoing)
                .map(|(i, _)| i)
                .collect();

            let link_cursor = self.detail.as_ref().and_then(|d| d.link_cursor);

            if !outgoing.is_empty() {
                link_lines.push(Line::from(Span::raw("  出边 (→):")));
                for &idx in &outgoing {
                    let l = &entry.links[idx];
                    let is_active = link_cursor == Some(idx);
                    if is_active {
                        link_cursor_line = Some(link_lines.len());
                    }
                    let prefix = if is_active { "▶ " } else { "  " };
                    let style = if is_active {
                        Style::default().fg(Color::Black).bg(Color::Cyan)
                    } else {
                        Style::default()
                    };
                    link_lines.push(Line::from(vec![
                        Span::styled(format!("{}→ {}  ", prefix, l.to_id), style),
                        Span::styled(format!("{} ", l.link_type_desc), gray),
                        Span::styled(format!("{:.2}", l.intensity), green),
                    ]));
                }
            }

            if !incoming.is_empty() {
                link_lines.push(Line::from(Span::raw("  入边 (←):")));
                for &idx in &incoming {
                    let l = &entry.links[idx];
                    let is_active = link_cursor == Some(idx);
                    if is_active {
                        link_cursor_line = Some(link_lines.len());
                    }
                    let prefix = if is_active { "▶ " } else { "  " };
                    let style = if is_active {
                        Style::default().fg(Color::Black).bg(Color::Cyan)
                    } else {
                        Style::default()
                    };
                    link_lines.push(Line::from(vec![
                        Span::styled(format!("{}← {}  ", prefix, l.from_id), style),
                        Span::styled(format!("{} ", l.link_type_desc), gray),
                        Span::styled(format!("{:.2}", l.intensity), green),
                    ]));
                }
            }
        } else {
            link_lines.push(Line::from(Span::styled(" (无连接)", gray)));
        }

        // Show navigation info
        let stack_depth = self.detail.as_ref().map(|d| d.nav_stack.len()).unwrap_or(0);
        if stack_depth > 0 {
            link_lines.push(Line::from(""));
            link_lines.push(Line::from(Span::styled(
                format!(" (导航栈: {}层, Backspace可回退)", stack_depth),
                gray,
            )));
        }

        // --- Constraint layout: fixed header + scrollable links ---
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(header_lines.len().min(inner.height as usize) as u16),
                Constraint::Fill(1),
            ])
            .split(inner);

        // Render fixed header
        frame.render_widget(
            Paragraph::new(Text::from(header_lines)).wrap(Wrap { trim: false }),
            chunks[0],
        );

        // Render scrollable links with auto-scroll
        let (content_rect, bar_rect) = ScrollState::split_area(chunks[1]);
        let line_count = link_lines.len();
        let scroll_offset = if let Some(l) = link_cursor_line {
            ScrollState::offset(content_rect.height, line_count, l)
        } else {
            self.detail_scroll.offset
        };
        frame.render_widget(
            Paragraph::new(Text::from(link_lines))
                .wrap(Wrap { trim: false })
                .scroll((scroll_offset as u16, 0)),
            content_rect,
        );
        ScrollState::render_scrollbar(
            frame,
            bar_rect,
            line_count,
            content_rect.height,
            scroll_offset,
        );
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                if self.detail.is_some() {
                    if self
                        .detail
                        .as_ref()
                        .map(|d| !d.nav_stack.is_empty())
                        .unwrap_or(false)
                    {
                        self.pop_nav();
                    } else {
                        self.detail = None;
                        self.detail_scroll.reset();
                    }
                } else {
                    return Transition::ToMain;
                }
                Transition::None
            }
            KeyCode::Backspace => {
                if self.detail.is_some() {
                    if self
                        .detail
                        .as_ref()
                        .map(|d| !d.nav_stack.is_empty())
                        .unwrap_or(false)
                    {
                        self.pop_nav();
                    } else {
                        self.detail = None;
                        self.detail_scroll.reset();
                    }
                }
                Transition::None
            }
            KeyCode::Up => {
                if ctrl && self.detail.is_some() {
                    if let Some(ref mut detail) = self.detail {
                        let entry = &self.entries[self.list_scroll.cursor];
                        if entry.links.is_empty() {
                            return Transition::None;
                        }
                        match detail.link_cursor {
                            None | Some(0) => {
                                detail.link_cursor = Some(entry.links.len().saturating_sub(1));
                            }
                            Some(c) => {
                                detail.link_cursor = Some(c - 1);
                            }
                        }
                    }
                } else if self.list_scroll.cursor > 0 {
                    self.list_scroll.move_up();
                    let idx = self.first_outgoing_idx();
                    if let Some(ref mut detail) = self.detail {
                        detail.link_cursor = Some(idx);
                    }
                    self.detail_scroll.reset();
                }
                Transition::None
            }
            KeyCode::Down => {
                if ctrl && self.detail.is_some() {
                    if let Some(ref mut detail) = self.detail {
                        let entry = &self.entries[self.list_scroll.cursor];
                        if entry.links.is_empty() {
                            return Transition::None;
                        }
                        match detail.link_cursor {
                            None => {
                                detail.link_cursor = Some(0);
                            }
                            Some(c) => {
                                let next = c + 1;
                                if next >= entry.links.len() {
                                    detail.link_cursor = Some(0);
                                } else {
                                    detail.link_cursor = Some(next);
                                }
                            }
                        }
                    }
                } else if self.list_scroll.cursor + 1 < self.entries.len() {
                    self.list_scroll.move_down(self.entries.len());
                    let idx = self.first_outgoing_idx();
                    if let Some(ref mut detail) = self.detail {
                        detail.link_cursor = Some(idx);
                    }
                    self.detail_scroll.reset();
                }
                Transition::None
            }
            KeyCode::Enter => {
                if self.detail.is_none() {
                    // Open detail mode, cursor starts at first outgoing link
                    let has_links = !self.entries[self.list_scroll.cursor].links.is_empty();
                    self.detail = Some(DetailState {
                        link_cursor: if has_links {
                            Some(self.first_outgoing_idx())
                        } else {
                            None
                        },
                        nav_stack: Vec::new(),
                    });
                    self.detail_scroll.reset();
                } else if let Some(ref mut detail) = self.detail {
                    if let Some(cursor) = detail.link_cursor {
                        let entry = &self.entries[self.list_scroll.cursor];
                        if cursor < entry.links.len() {
                            let target = entry.links[cursor].target_idx;
                            if target < self.entries.len() && target != self.list_scroll.cursor {
                                let prev_cursor = self.list_scroll.cursor;
                                detail.nav_stack.push((prev_cursor, cursor));
                                self.list_scroll.move_to(target);
                                if let Some(ref mut d) = self.detail {
                                    d.link_cursor = None;
                                }
                                self.detail_scroll.reset();
                            }
                        }
                    }
                }
                Transition::None
            }
            _ => Transition::None,
        }
    }

    fn render_stats_panel(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered().title(" 图统计 ").fg(Color::Cyan);
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        if let Some(ref stat_lines) = self.stats {
            let lines: Vec<Line> = stat_lines
                .iter()
                .map(|l| Line::from(Span::raw(format!(" {}", l))))
                .collect();
            frame.render_widget(Paragraph::new(Text::from(lines)), inner);
        }
    }

    fn first_outgoing_idx(&self) -> usize {
        if let Some(entry) = self.entries.get(self.list_scroll.cursor) {
            entry.links.iter().position(|l| l.is_outgoing).unwrap_or(0)
        } else {
            0
        }
    }

    fn pop_nav(&mut self) {
        if let Some(ref mut detail) = self.detail {
            if let Some((prev_sel, prev_cursor)) = detail.nav_stack.pop() {
                self.list_scroll.move_to(prev_sel);
                detail.link_cursor = Some(prev_cursor);
                self.detail_scroll.reset();
            }
        }
    }
}

fn parse_graph_nodes(val: &serde_json::Value) -> Vec<InspectEntry> {
    let arr = match val.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };

    let mut entries: Vec<InspectEntry> = Vec::new();
    // First pass: collect all link references
    for node in arr {
        let id = node
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let tags: Vec<String> = node
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let mem_type = node.get("mem_type");
        let (type_label, preview_lines, detail_lines) = format_mem_type(mem_type, &tags);

        let entry = InspectEntry {
            id: id.clone(),
            summary: format!(
                "{} [{}]  {}",
                type_label,
                tags.join(","),
                preview_lines.first().unwrap_or(&String::new())
            ),
            preview_lines: preview_lines.clone(),
            detail_lines,
            links: Vec::new(),
        };
        entries.push(entry);
    }

    // Second pass: build link index from raw JSON to avoid borrowing entries
    let id_to_idx: HashMap<&str, usize> = arr
        .iter()
        .enumerate()
        .filter_map(|(i, node)| node.get("id").and_then(|v| v.as_str()).map(|id| (id, i)))
        .collect();

    for (i, node) in arr.iter().enumerate() {
        if let Some(links) = node.get("mem_links").and_then(|v| v.as_array()) {
            for link in links {
                let from = link.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let to = link.get("to").and_then(|v| v.as_str()).unwrap_or("");
                let intensity = link
                    .get("intensity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let link_type_desc = link
                    .get("link_type")
                    .map(format_link_type)
                    .unwrap_or_default();

                let from_idx = id_to_idx.get(from).copied().unwrap_or(i);
                let to_idx = id_to_idx.get(to).copied().unwrap_or(i);

                // Outgoing from `from`
                if from_idx < entries.len() {
                    entries[from_idx].links.push(LinkDisplay {
                        from_id: from.to_string(),
                        to_id: to.to_string(),
                        target_idx: to_idx,
                        link_type_desc: link_type_desc.clone(),
                        intensity,
                        is_outgoing: true,
                    });
                }
                // Incoming to `to`
                if to_idx < entries.len() && to_idx != from_idx {
                    entries[to_idx].links.push(LinkDisplay {
                        from_id: from.to_string(),
                        to_id: to.to_string(),
                        target_idx: from_idx,
                        link_type_desc,
                        intensity,
                        is_outgoing: false,
                    });
                }
            }
        }
    }

    // Sort links so outgoing come before incoming (matches render order)
    for entry in &mut entries {
        entry.links.sort_by_key(|l| !l.is_outgoing);
    }

    entries
}

fn parse_query_cases(val: &serde_json::Value) -> Vec<InspectEntry> {
    let cases = match val.get("test_cases").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };

    let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("?");
    let desc = val
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut entries: Vec<InspectEntry> = Vec::new();
    let config = val.get("config");

    for (_idx, tc) in cases.iter().enumerate() {
        let case_name = tc
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let case_desc = tc
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let sub_queries = tc
            .get("sub_queries")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        let expected: Vec<String> = tc
            .get("expected_combined_ranking")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let expected_str = if expected.is_empty() {
            "(空)".to_string()
        } else {
            format!("[{}]", expected.join(", "))
        };

        let bonus: Vec<String> = tc
            .get("bonus_combined_ranking")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let bonus_str = if bonus.is_empty() {
            "(空)".to_string()
        } else {
            format!("[{}]", bonus.join(", "))
        };

        // Preview lines
        let preview_lines = vec![
            format!("描述: {}", case_desc),
            format!("子查询数: {}", sub_queries),
            format!("期望结果: {}", expected_str),
            format!("奖励结果: {}", bonus_str),
        ];

        // Detail lines
        let mut detail_lines = Vec::new();
        detail_lines.push(format!("数据集: {}", name));
        detail_lines.push(format!("描述: {}", desc));
        detail_lines.push(format!(""));
        detail_lines.push(format!("用例名: {}", case_name));
        detail_lines.push(format!("用例描述: {}", case_desc));

        if let Some(cfg) = config {
            if let Some(thresh) = cfg.get("similarity_threshold").and_then(|v| v.as_f64()) {
                detail_lines.push(format!("相似度阈值: {:.2}", thresh));
            }
            if let Some(max_r) = cfg.get("max_results").and_then(|v| v.as_u64()) {
                detail_lines.push(format!("最大结果数: {}", max_r));
            }
            if let Some(k_vals) = cfg.get("test_k_values").and_then(|v| v.as_array()) {
                let ks: Vec<String> = k_vals
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n.to_string()))
                    .collect();
                detail_lines.push(format!("测试K值: [{}]", ks.join(", ")));
            }
        }

        detail_lines.push(format!(""));
        detail_lines.push(format!("子查询 ({}个):", sub_queries));
        if let Some(subs) = tc.get("sub_queries").and_then(|v| v.as_array()) {
            for (si, sq) in subs.iter().enumerate() {
                let prio = sq.get("priority").and_then(|v| v.as_u64()).unwrap_or(0);
                let tags: Vec<String> = sq
                    .get("tag")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                detail_lines.push(format!("  Q{} pri={} tags=[{}]", si, prio, tags.join(",")));
                // Show variant content
                if let Some(variant) = sq.get("variant") {
                    let vlines = format_variant_preview(variant, 4);
                    detail_lines.extend(vlines);
                }
            }
        }

        detail_lines.push(format!(""));
        detail_lines.push(format!("期望排序 (combined): {}", expected_str));
        detail_lines.push(format!("奖励排序 (bonus): {}", bonus_str));
        if let Some(per_q) = tc.get("expected_per_query").and_then(|v| v.as_array()) {
            for eq in per_q {
                let qidx = eq.get("q").and_then(|v| v.as_u64()).unwrap_or(0);
                let ranking: Vec<String> = eq
                    .get("ranking")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                detail_lines.push(format!("  Q{} 期望: [{}]", qidx, ranking.join(", ")));
                let bonus_ranking: Vec<String> = eq
                    .get("bonus_ranking")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if !bonus_ranking.is_empty() {
                    detail_lines.push(format!("     奖励: [{}]", bonus_ranking.join(", ")));
                }
            }
        }

        let summary = format!("{}  [{}子查询] {}", case_name, sub_queries, case_desc);

        entries.push(InspectEntry {
            id: case_name,
            summary,
            preview_lines,
            detail_lines,
            links: Vec::new(),
        });
    }

    entries
}

fn format_mem_type(
    mem_type: Option<&serde_json::Value>,
    tags: &[String],
) -> (String, Vec<String>, Vec<String>) {
    let type_label: String;
    let mut preview_lines = Vec::new();
    let mut detail_lines = Vec::new();

    match mem_type {
        None => {
            type_label = "?".to_string();
            preview_lines.push("(无类型)".to_string());
            detail_lines.push("(无类型)".to_string());
        }
        Some(val) if val.get("Semantic").is_some() => {
            type_label = "Semantic".to_string();
            let sem = &val["Semantic"];
            let content = sem.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let desc = sem
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let aliases: Vec<String> = sem
                .get("aliases")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let concept_type = sem
                .get("concept_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            preview_lines.push(format!("内容: {}", content));
            preview_lines.push(format!("描述: {}", desc));

            detail_lines.push(format!("类型: Semantic"));
            detail_lines.push(format!("标签: [{}]", tags.join(", ")));
            detail_lines.push(format!("内容: {}", content));
            detail_lines.push(format!("别名: [{}]", aliases.join(", ")));
            detail_lines.push(format!("概念类型: {}", concept_type));
            if !desc.is_empty() {
                detail_lines.push(format!("描述: {}", desc));
            }
        }
        Some(val) if val.get("Situation").is_some() => {
            let sit = &val["Situation"];

            if let Some(spec) = sit.get("SpecificSituation") {
                type_label = "Situation".to_string();
                let narrative = spec.get("narrative").and_then(|v| v.as_str()).unwrap_or("");
                let time_span = spec
                    .get("time_span")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");

                preview_lines.push(format!("叙事: {}", narrative));
                preview_lines.push(format!("时间: {}", time_span));

                detail_lines.push(format!("类型: Situation::SpecificSituation"));
                detail_lines.push(format!("标签: [{}]", tags.join(", ")));
                detail_lines.push(format!("叙事: {}", narrative));
                detail_lines.push(format!("时间: {}", time_span));

                if let Some(ctx) = spec.get("context") {
                    if let Some(loc) = ctx.get("location").and_then(|v| v.as_object()) {
                        let name = loc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let coords = loc
                            .get("coordinates")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        detail_lines.push(format!("地点: {} ({})", name, coords));
                    }
                    if let Some(parts) = ctx.get("participants").and_then(|v| v.as_array()) {
                        for p in parts {
                            let pname = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let role = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
                            detail_lines.push(format!("参与者: {} ({})", pname, role));
                        }
                    }
                    if let Some(env) = ctx.get("environment").and_then(|v| v.as_object()) {
                        let atm = env.get("atmosphere").and_then(|v| v.as_str()).unwrap_or("");
                        let tone = env.get("tone").and_then(|v| v.as_str()).unwrap_or("");
                        detail_lines.push(format!("环境: atm={} tone={}", atm, tone));
                    }
                    if let Some(events) = ctx.get("event").and_then(|v| v.as_array()) {
                        for ev in events {
                            let action = ev.get("action").and_then(|v| v.as_str()).unwrap_or("");
                            let init = ev.get("initiator").and_then(|v| v.as_str()).unwrap_or("");
                            let tgt = ev.get("target").and_then(|v| v.as_str()).unwrap_or("");
                            detail_lines.push(format!("事件: {} → {} ({}))", init, action, tgt));
                        }
                    }
                }
            } else if let Some(abs) = sit.get("AbstractSituation") {
                type_label = "AbstractSit".to_string();
                detail_lines.push(format!("类型: Situation::AbstractSituation"));
                detail_lines.push(format!("标签: [{}]", tags.join(", ")));

                if let Some(loc) = abs.get("Location") {
                    let name = loc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let coords = loc
                        .get("coordinates")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    preview_lines.push(format!("地点: {}", name));
                    detail_lines.push(format!("子类: Location"));
                    detail_lines.push(format!("名称: {}", name));
                    if !coords.is_empty() {
                        detail_lines.push(format!("坐标: {}", coords));
                    }
                } else if let Some(part) = abs.get("Participant") {
                    let name = part.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let role = part.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    preview_lines.push(format!("参与者: {}", name));
                    detail_lines.push(format!("子类: Participant"));
                    detail_lines.push(format!("名称: {}", name));
                    if !role.is_empty() {
                        detail_lines.push(format!("角色: {}", role));
                    }
                } else if let Some(env) = abs.get("Environment") {
                    let atm = env.get("atmosphere").and_then(|v| v.as_str()).unwrap_or("");
                    let tone = env.get("tone").and_then(|v| v.as_str()).unwrap_or("");
                    preview_lines.push(format!("环境: {} / {}", atm, tone));
                    detail_lines.push(format!("子类: Environment"));
                    detail_lines.push(format!("氛围: {}", atm));
                    detail_lines.push(format!("色调: {}", tone));
                } else if let Some(evt) = abs.get("Event") {
                    let action = evt.get("action").and_then(|v| v.as_str()).unwrap_or("");
                    let init = evt.get("initiator").and_then(|v| v.as_str()).unwrap_or("");
                    let tgt = evt.get("target").and_then(|v| v.as_str()).unwrap_or("");
                    preview_lines.push(format!("事件: {} → {}", init, action));
                    detail_lines.push(format!("子类: Event"));
                    detail_lines.push(format!("动作: {}", action));
                    if !init.is_empty() {
                        detail_lines.push(format!("发起者: {}", init));
                    }
                    if !tgt.is_empty() {
                        detail_lines.push(format!("目标: {}", tgt));
                    }
                } else {
                    preview_lines.push(format!("{:?}", abs));
                    detail_lines.push(format!("{:?}", abs));
                }
            } else {
                type_label = "Situation".to_string();
                preview_lines.push(format!("{:?}", sit));
                detail_lines.push(format!("{:?}", sit));
            }
        }
        Some(val) if val.get("Procedure").is_some() => {
            type_label = "Procedure".to_string();
            let action = &val["Procedure"]["action"];
            let content = action.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let action_type = action
                .get("action_type")
                .and_then(|v| v.as_str())
                .unwrap_or("?");

            preview_lines.push(format!("类型: Procedure"));
            preview_lines.push(format!("动作: {}", content));

            detail_lines.push(format!("类型: Procedure"));
            detail_lines.push(format!("标签: [{}]", tags.join(", ")));
            detail_lines.push(format!("动作: {}", content));
            detail_lines.push(format!("动作类型: {}", action_type));
        }
        Some(val) => {
            // Try to extract type from first key
            let first_key = val
                .as_object()
                .and_then(|o| o.keys().next())
                .map(|k| k.to_string())
                .unwrap_or_else(|| "?".to_string());
            type_label = first_key;
            preview_lines.push(format!("{:?}", val));
            detail_lines.push(format!("{:?}", val));
        }
    }

    (type_label, preview_lines, detail_lines)
}

fn format_link_type(val: &serde_json::Value) -> String {
    if let Some(obj) = val.as_object() {
        for (k, v) in obj {
            return match k.as_str() {
                "Sem" => {
                    let verb = v.get("verb").and_then(|x| x.as_str()).unwrap_or("?");
                    let conf = v.get("confidence").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    format!("Sem[{} conf={:.1}]", verb, conf)
                }
                "Proc" => {
                    if let Some(inner) = v.get("TrigToAction") {
                        let prob = inner.get("prob").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        format!("Proc::TrigToAction[prob={:.1}]", prob)
                    } else {
                        format!("Proc[{:?}]", v)
                    }
                }
                "Situation" => {
                    if v.get("AbstractToSpecific").is_some() {
                        "Sit::AbstractToSpecific".to_string()
                    } else {
                        "Situation[...]".to_string()
                    }
                }
                "Coref" => "Coref".to_string(),
                _ => format!("{}[...]", k),
            };
        }
    }
    val.to_string()
}

fn label_key(k: &str) -> &'static str {
    match k {
        "node_count" => "节点数",
        "edge_count" => "边数",
        "node_types" => "节点类型",
        "link_types" => "边类型",
        "connected_components" => "连通分量",
        "largest_component" => "最大分量",
        "isolated_nodes" => "孤立节点",
        "global_redundancy" => "全局冗余度",
        "avg_clustering" => "平均聚类系数",
        "community_modularity" => "社区模块度",
        "intra_community_ratio" => "社区内边比",
        "gini_coefficient" => "基尼系数",
        "has_self_node" => "有自身节点",
        "self_description_ok" => "自身描述有效",
        "is_clean" => "图结构清洁",
        "is_structurally_valid" => "结构有效",
        "proc_without_incoming_proc" => "孤立Procedure数",
        "abstract_sit_type_count" => "抽象情境类型数",
        "has_proc_none" => "有空Procedure",
        _ => "?",
    }
}

fn load_graph_stats(path: &std::path::Path) -> Option<Vec<String>> {
    let content = std::fs::read_to_string(path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    let obj = val.as_object()?;

    let mut lines = Vec::new();

    for (k, v) in obj {
        let lbl = match k.as_str() {
            "node_count"
            | "edge_count"
            | "connected_components"
            | "largest_component"
            | "isolated_nodes"
            | "abstract_sit_type_count" => v.as_u64().map(|n| format!("{}: {}", label_key(k), n)),
            "global_redundancy"
            | "avg_clustering"
            | "community_modularity"
            | "intra_community_ratio"
            | "gini_coefficient" => v.as_f64().map(|n| format!("{}: {:.3}", label_key(k), n)),
            "has_self_node"
            | "self_description_ok"
            | "is_clean"
            | "is_structurally_valid"
            | "has_proc_none" => v
                .as_bool()
                .map(|b| format!("{}: {}", label_key(k), if b { "✓" } else { "✗" })),
            "proc_without_incoming_proc" => v
                .as_array()
                .map(|a| format!("{}: {}个", label_key(k), a.len())),
            "node_types" | "link_types" => v.as_object().map(|o| {
                let inner: Vec<String> = o
                    .iter()
                    .map(|(sk, sv)| format!("{}={}", sk, sv.as_u64().unwrap_or(0)))
                    .collect();
                format!("{}: {}", label_key(k), inner.join(" "))
            }),
            _ => None,
        };
        if let Some(l) = lbl {
            lines.push(l);
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines)
    }
}

fn format_variant_preview(val: &serde_json::Value, indent: usize) -> Vec<String> {
    let pad = " ".repeat(indent);
    let mut out = Vec::new();
    if let Some(obj) = val.as_object() {
        for (k, inner) in obj {
            if let Some(arr) = inner.as_array() {
                if arr.is_empty() {
                    out.push(format!("{} {}: (空)", pad, k));
                } else {
                    out.push(format!("{} {}: {}条", pad, k, arr.len()));
                    for item in arr.iter().take(3) {
                        if let Some(item_obj) = item.as_object() {
                            for (fk, fv) in item_obj {
                                if let Some(s) = fv.as_str() {
                                    if s.chars().count() > 60 {
                                        let truncated: String = s.chars().take(60).collect();
                                        out.push(format!("{}   {}: {}...", pad, fk, truncated));
                                    } else {
                                        out.push(format!("{}   {}: {}", pad, fk, s));
                                    }
                                }
                            }
                        }
                    }
                    if arr.len() > 3 {
                        out.push(format!("{}   ... 还有{}条", pad, arr.len() - 3));
                    }
                }
            } else {
                out.push(format!("{} {}: {}", pad, k, inner));
            }
        }
    }
    out
}

impl Component for InspectState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => self.handle_key(key),
            _ => Transition::None,
        }
    }
    fn view(&self, frame: &mut Frame) {
        self.render(frame);
    }
}
