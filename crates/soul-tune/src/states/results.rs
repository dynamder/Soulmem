use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use crate::base::{TestReport, Transition};
use crate::component::{Component, ComponentEvent};
use crate::engine::retrieve::data::RetrieveCaseData;
use crate::engine::suite::{MetricFormat, ReportMetric};
use crate::widgets::expandable::ExpandableList;
use crate::widgets::scroll::ScrollState;
use crate::widgets::status_bar;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

#[derive(PartialEq, Eq)]
pub enum ResultTab {
    Summary,
    Detail,
}

pub struct ResultsState {
    pub report: TestReport,
    pub active_tab: ResultTab,
    pub kv_scroll: ScrollState,
    pub metric_group_idx: usize,
    pub log_filter: String,
    pub detail_selected: Option<usize>,
    pub drill_scroll: ScrollState,
    pub detail_scroll: ScrollState,
    pub compare_scroll: ScrollState,
    pub expanded: ExpandableList,
    case_details: Vec<RetrieveCaseData>,
    /// Line offsets (in `lines`) where each comparison row begins.
    /// Populated during `render_detail_drilldown` via interior mutability.
    comparison_lines: RefCell<Vec<usize>>,
    drill_viewport: Cell<usize>,
}

impl ResultsState {
    pub fn new(report: TestReport) -> Self {
        let mut by_weight: HashMap<(u32, u32), Vec<RetrieveCaseData>> = HashMap::new();
        for outcome in &report.suite_report.outcomes {
            if let Some(data) = outcome.data.downcast_ref::<RetrieveCaseData>() {
                let key = (
                    (data.tag_weight * 100.0).round() as u32,
                    (data.variant_weight * 100.0).round() as u32,
                );
                by_weight.entry(key).or_default().push(data.clone());
            }
        }
        let mut keys: Vec<_> = by_weight.keys().copied().collect();
        keys.sort();
        let mut case_details = Vec::new();
        for key in keys {
            if let Some(mut group) = by_weight.remove(&key) {
                case_details.append(&mut group);
            }
        }
        Self {
            report,
            active_tab: ResultTab::Summary,
            kv_scroll: ScrollState::new(),
            metric_group_idx: 0,
            log_filter: "ALL".into(),
            detail_selected: None,
            drill_scroll: ScrollState::new(),
            detail_scroll: ScrollState::new(),
            compare_scroll: ScrollState::new(),
            expanded: ExpandableList::new(0),
            case_details,
            comparison_lines: RefCell::new(Vec::new()),
            drill_viewport: Cell::new(0),
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .split(area);

        Block::bordered()
            .title(format!(
                " ✓ 完成 · {} · {} ",
                self.report.config.algo,
                self.report
                    .config
                    .dataset_path
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ))
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let tab_titles = vec![" Summary ", " Detail "];
        let tabs = Tabs::new(tab_titles)
            .select(match self.active_tab {
                ResultTab::Summary => 0,
                ResultTab::Detail => 1,
            })
            .highlight_style(Style::default().fg(Color::Yellow));
        frame.render_widget(tabs, layout[1]);

        match self.active_tab {
            ResultTab::Summary => self.render_summary(frame, layout[2]),
            ResultTab::Detail => self.render_detail(frame, layout[2]),
        }

        let summary_hints = vec![
            ("[←→]".into(), "切指标组".into()),
            ("[Tab]".into(), "详情".into()),
            ("[Q]".into(), "返回".into()),
        ];
        let detail_hints = vec![
            ("[↑↓]".into(), "滚动".into()),
            ("[Enter]".into(), "查看详情".into()),
            ("[N/P]".into(), "失/前失败".into()),
            ("[Q]".into(), "返回".into()),
        ];
        let drill_hints = vec![
            ("[↑↓]".into(), "滚动".into()),
            ("[Enter]".into(), "展开".into()),
            ("[Q]".into(), "返回列表".into()),
        ];
        status_bar::render_status_bar(
            frame,
            layout[3],
            match self.active_tab {
                ResultTab::Summary => &summary_hints,
                ResultTab::Detail if self.detail_selected.is_some() => &drill_hints,
                ResultTab::Detail => &detail_hints,
            },
        );
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let mut top_pad = 0u16;
        if let Some(ref error) = self.report.error {
            let err_rect = Rect::new(area.x, area.y, area.width, 4.min(area.height));
            let err_block = Block::bordered().title(" ⚠ 加载错误 ").fg(Color::Red);
            let err_inner = err_block.inner(err_rect);
            err_block.render(err_rect, frame.buffer_mut());
            frame.render_widget(
                Paragraph::new(error.as_str())
                    .wrap(Wrap { trim: false })
                    .fg(Color::Red),
                err_inner,
            );
            top_pad = 4.min(area.height);
        }

        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Fill(1), Constraint::Fill(2)])
            .split(Rect::new(
                area.x,
                area.y + top_pad,
                area.width,
                area.height.saturating_sub(top_pad),
            ));

        // ── Left: metric groups from SuiteReport ──
        let kv_block = Block::bordered().title(" 指标 ");
        let kv_inner = kv_block.inner(split[0]);
        kv_block.render(split[0], frame.buffer_mut());

        let mut y = kv_inner.y;
        let mut metric_groups: HashMap<String, Vec<&dyn ReportMetric>> = HashMap::new();
        for metric in &self.report.suite_report.metrics {
            metric_groups
                .entry(metric.group())
                .or_default()
                .push(metric.as_ref());
        }
        let mut group_keys: Vec<String> = metric_groups.keys().cloned().collect();
        group_keys.sort();
        for group_name in &group_keys {
            if let Some(metrics) = metric_groups.get(group_name) {
                frame.render_widget(
                    Paragraph::new(format!(" {} ", group_name))
                        .fg(Color::Yellow)
                        .bold(),
                    Rect::new(kv_inner.x, y, kv_inner.width, 1),
                );
                y += 1;
                for m in metrics {
                    if let MetricFormat::KeyValue { value } = m.format() {
                        let line = format!("  {}: {}", m.label(), value);
                        frame.render_widget(
                            Paragraph::new(line),
                            Rect::new(kv_inner.x + 1, y, kv_inner.width - 2, 1),
                        );
                        y += 1;
                    }
                }
                y += 1;
            }
        }

        // ── Right: per-case chart (passed=1, failed=0) ──
        let chart_area = split[1];
        let chart_points: Vec<(f64, f64)> = self
            .report
            .suite_report
            .detail_rows
            .iter()
            .enumerate()
            .map(|(i, row)| (i as f64, if row.has_error { 0.0 } else { 1.0 }))
            .collect();

        if chart_points.is_empty() {
            let chart_block = Block::bordered().title(" 逐用例状态 ");
            let chart_inner = chart_block.inner(chart_area);
            chart_block.render(chart_area, frame.buffer_mut());
            frame.render_widget(Paragraph::new("(无数据)").fg(Color::DarkGray), chart_inner);
        } else {
            let dataset = Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .data(&chart_points)
                .style(Style::default().fg(Color::Cyan));
            let n = (chart_points.len() as f64 - 1.0).max(0.0);
            let x_labels: Vec<String> = (0..chart_points.len())
                .step_by(1.max(chart_points.len() / 5))
                .map(|i| format!("{}", i))
                .collect();
            let chart = Chart::new(vec![dataset])
                .block(
                    Block::bordered()
                        .title(" 逐用例 通过/失败 ")
                        .fg(Color::Yellow),
                )
                .x_axis(
                    Axis::default()
                        .title("用例 #")
                        .labels(x_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                        .bounds([0.0, n]),
                )
                .y_axis(
                    Axis::default()
                        .title("状态")
                        .labels(vec!["失败", "通过"])
                        .bounds([-0.1, 1.1]),
                );
            frame.render_widget(chart, chart_area);
        }
    }

    fn render_detail(&self, frame: &mut Frame, area: Rect) {
        if let Some(idx) = self.detail_selected {
            self.render_detail_drilldown(frame, area, idx);
        } else {
            self.render_detail_list(frame, area);
        }
    }

    fn render_detail_list(&self, frame: &mut Frame, area: Rect) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(1), Constraint::Fill(1)])
            .split(area);

        let pass_rate = if self.report.total > 0 {
            self.report.passed as f64 / self.report.total as f64 * 100.0
        } else {
            0.0
        };
        let filter_text = format!(
            " 总 {} 用例 | 通过 {} | 失败 {} | {:.0}%",
            self.report.total, self.report.passed, self.report.failed, pass_rate
        );
        frame.render_widget(Paragraph::new(filter_text).fg(Color::DarkGray), layout[0]);

        let log_block = Block::bordered();
        let log_inner = log_block.inner(layout[1]);
        log_block.render(layout[1], frame.buffer_mut());

        // Header
        if !self.report.suite_report.detail_header.is_empty() {
            frame.render_widget(
                Paragraph::new(self.report.suite_report.detail_header.as_str())
                    .fg(Color::Cyan)
                    .bold(),
                Rect::new(log_inner.x, log_inner.y, log_inner.width, 1),
            );
        }

        // Rows with cursor highlight, auto-scroll to keep cursor visible
        let header_offset = 1;
        let _visible = (log_inner.height as usize).saturating_sub(header_offset);
        let (content_rect, bar_rect) = ScrollState::split_area(Rect::new(
            log_inner.x,
            log_inner.y + 1,
            log_inner.width,
            log_inner.height.saturating_sub(1),
        ));
        let offset = ScrollState::offset(
            content_rect.height,
            self.report.suite_report.detail_rows.len(),
            self.detail_scroll.cursor,
        );
        for (i, row) in self
            .report
            .suite_report
            .detail_rows
            .iter()
            .enumerate()
            .skip(offset)
        {
            let y = log_inner.y + 1u16 + (i - offset) as u16;
            if y >= log_inner.y + log_inner.height {
                break;
            }
            let is_active = i == self.detail_scroll.cursor;
            let (color, bg) = if is_active {
                (Color::Black, Color::Cyan)
            } else if row.has_error {
                (Color::Red, Color::Reset)
            } else {
                (Color::Reset, Color::Reset)
            };
            frame.render_widget(
                Paragraph::new(row.text.as_str()).fg(color).bg(bg),
                Rect::new(content_rect.x, y, content_rect.width, 1),
            );
        }
        ScrollState::render_scrollbar(
            frame,
            bar_rect,
            self.report.suite_report.detail_rows.len(),
            content_rect.height,
            offset,
        );
    }

    fn render_detail_drilldown(&self, frame: &mut Frame, area: Rect, index: usize) {
        let data = &self.case_details[index];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Fill(1)])
            .split(area);

        let mut lines: Vec<Line<'static>> = Vec::new();
        let hdr = Style::new().yellow().bold();
        let green = Style::new().green();
        let red = Style::new().red();
        let gray = Style::new().dark_gray();

        // Title
        lines.push(Line::from(Span::raw(format!(" 用例: {}", data.case_name))));

        // Status line with colored pass/fail
        let passed =
            data.combined_ranking_metrics.hit_rate > 0.0 || data.combined_ranking_metrics.mrr > 0.0;
        lines.push(Line::from(vec![
            Span::raw(format!(
                " Tag权重: {:.1}  Variant权重: {:.1}  状态: ",
                data.tag_weight, data.variant_weight,
            )),
            Span::styled(
                if passed { "✓ 通过" } else { "✗ 失败" },
                if passed { green } else { red },
            ),
        ]));
        lines.push(Line::from(""));

        // Query content
        lines.push(Line::from(Span::styled(" ── 查询内容 ──", hdr)));
        for (idx, sq) in data.sub_queries.iter().enumerate() {
            let tag_str = sq.tags.join(",");
            let variant_desc = format_variant_lines(&sq.variant);
            lines.push(Line::from(Span::raw(format!(
                "  Q{} pri={} [{}]",
                idx, sq.priority, tag_str,
            ))));
            for vline in &variant_desc {
                lines.push(Line::from(Span::raw(format!("    {}", vline))));
            }
        }
        lines.push(Line::from(""));

        // Combined ranking metrics
        lines.push(Line::from(Span::styled(" ── 综合排序指标 ──", hdr)));
        lines.push(Line::from(Span::raw("  K     Recall    Precision  NDCG")));
        for (k, r) in &data.combined_ranking_metrics.recall_at {
            let p = data
                .combined_ranking_metrics
                .precision_at
                .iter()
                .find(|(pk, _)| pk == k)
                .map(|(_, v)| v)
                .unwrap_or(&0.0);
            let n = data
                .combined_ranking_metrics
                .ndcg_at
                .iter()
                .find(|(nk, _)| nk == k)
                .map(|(_, v)| v)
                .unwrap_or(&0.0);
            lines.push(Line::from(Span::raw(format!(
                "  @{:<2}   {:.4}    {:.4}    {:.4}",
                k, r, p, n
            ))));
        }
        lines.push(Line::from(Span::styled(
            format!(
                "  MRR: {:.4}     Hit: {:.2}",
                data.combined_ranking_metrics.mrr, data.combined_ranking_metrics.hit_rate
            ),
            gray,
        )));
        lines.push(Line::from(""));

        // Per-sub-query metrics
        lines.push(Line::from(Span::styled(" ── 各子查询 ──", hdr)));
        for sq in &data.per_query_metrics {
            let sq_pass = sq.ranking_metrics.hit_rate > 0.0;
            lines.push(Line::from(vec![
                Span::raw(format!(
                    "  Q{}  MRR={:.4}  Hit={:.2}  Recall@3={:.4}  ",
                    sq.query_index,
                    sq.ranking_metrics.mrr,
                    sq.ranking_metrics.hit_rate,
                    sq.ranking_metrics
                        .recall_at
                        .iter()
                        .find(|(k, _)| *k == 3)
                        .map(|(_, v)| v)
                        .unwrap_or(&0.0),
                )),
                Span::styled(
                    if sq_pass { "✓" } else { "✗" },
                    if sq_pass { green } else { red },
                ),
            ]));
        }
        lines.push(Line::from(""));

        // Side-by-side comparison
        lines.push(Line::from(Span::styled(
            " ── 检索结果 vs 预期 (Enter展开) ──",
            hdr,
        )));
        let retrieved_set: std::collections::HashSet<&MemoryId> =
            data.combined_retrieved_ids.iter().take(10).collect();
        let n_max = data
            .combined_retrieved_ids
            .len()
            .min(10)
            .max(data.expected_combined_ranking.len().min(5));

        // resize happens in handle_key when opening drill-down or switching case.
        {
            self.comparison_lines.borrow_mut().clear();
        }
        for pos in 0..n_max {
            let is_cursor = pos == self.compare_scroll.cursor;
            let is_expanded = self.expanded.is_expanded(pos);
            let mut spans = Vec::new();
            spans.push(Span::styled(
                format!(" {}  #{}", if is_cursor { "▶" } else { " " }, pos + 1),
                if is_cursor { green } else { Style::new() },
            ));

            if let Some(id) = data.combined_retrieved_ids.get(pos) {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(id))
                    .cloned()
                    .unwrap_or_default();
                let is_hit = data.expected_combined_ranking.iter().any(|eid| eid == id);
                spans.push(Span::raw(format!(" {:<10} ", name)));
                spans.push(Span::styled(
                    if is_hit { "✓" } else { "-" },
                    if is_hit { green } else { gray },
                ));
            } else {
                spans.push(Span::raw(format!(" {:<10}  ", "—")));
            }
            spans.push(Span::raw("  "));

            if let Some(eid) = data.expected_combined_ranking.get(pos) {
                let ename = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(eid))
                    .cloned()
                    .unwrap_or_default();
                spans.push(Span::raw(format!("{:<10}", ename)));
                if !retrieved_set.contains(eid) {
                    spans.push(Span::styled(" ✗未命中", red));
                }
            } else {
                spans.push(Span::raw(format!(" {:<10}", "—")));
            }
            self.comparison_lines.borrow_mut().push(lines.len());
            lines.push(Line::from(spans));

            if is_expanded {
                let act = data.combined_retrieved_ids.get(pos);
                let exp = data.expected_combined_ranking.get(pos);
                for line in format_node_detail(act, exp, data).lines() {
                    lines.push(Line::from(Span::styled(format!("   {}", line), gray)));
                }
            }
        }
        lines.push(Line::from(""));

        // Missed
        let missed: Vec<&MemoryId> = data
            .expected_combined_ranking
            .iter()
            .filter(|eid| !retrieved_set.contains(eid))
            .collect();
        if !missed.is_empty() {
            let mut spans = vec![Span::styled("  未命中: ", red)];
            for (i, id) in missed.iter().enumerate() {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(*id))
                    .cloned()
                    .unwrap_or_default();
                if i > 0 {
                    spans.push(Span::raw(", "));
                }
                spans.push(Span::styled(name, red));
            }
            lines.push(Line::from(spans));
        }

        let block = Block::bordered().title(" 用例详情 ").fg(Color::Cyan);
        let inner = block.inner(layout[0]);
        block.render(layout[0], frame.buffer_mut());
        let (content_rect, bar_rect) = ScrollState::split_area(inner);
        self.drill_viewport.set(content_rect.height as usize);
        let line_count = lines.len();
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((self.drill_scroll.offset as u16, 0)),
            content_rect,
        );
        ScrollState::render_scrollbar(
            frame,
            bar_rect,
            line_count,
            content_rect.height,
            self.drill_scroll.offset,
        );
    }

    fn scroll_to_comparison_cursor(&mut self) {
        let vis = self.drill_viewport.get();
        if vis == 0 {
            return;
        }
        let lines = self.comparison_lines.borrow();
        if let Some(&line_off) = lines.get(self.compare_scroll.cursor) {
            if line_off < self.drill_scroll.offset {
                self.drill_scroll.offset = line_off;
            } else if line_off >= self.drill_scroll.offset + vis {
                self.drill_scroll.offset = line_off.saturating_sub(vis.saturating_sub(1));
            }
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                if self.detail_selected.is_some() {
                    self.detail_selected = None;
                    self.drill_scroll.reset();
                    self.expanded.clear_all();
                    Transition::None
                } else {
                    Transition::ToMain
                }
            }
            KeyCode::Tab => {
                self.detail_selected = None;
                self.drill_scroll.reset();
                self.expanded.clear_all();
                self.active_tab = match self.active_tab {
                    ResultTab::Summary => ResultTab::Detail,
                    ResultTab::Detail => ResultTab::Summary,
                };
                Transition::None
            }
            KeyCode::Enter => {
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_some() {
                    // In drill-down: toggle expand on cursor row
                    let row = self.compare_scroll.cursor;
                    self.expanded.toggle(row);
                    Transition::None
                } else if self.active_tab == ResultTab::Detail && self.detail_selected.is_none() {
                    // In list: open drill-down
                    if self.detail_scroll.cursor < self.case_details.len() {
                        self.detail_selected = Some(self.detail_scroll.cursor);
                        self.drill_scroll.reset();
                        self.compare_scroll.reset();
                        self.expanded.clear_all();
                        if let Some(detail) = self.detail_selected {
                            if let Some(data) = self.case_details.get(detail) {
                                let n_max = data
                                    .combined_retrieved_ids
                                    .len()
                                    .min(10)
                                    .max(data.expected_combined_ranking.len().min(5));
                                self.expanded.resize(n_max);
                            }
                        }
                    }
                    Transition::None
                } else {
                    Transition::None
                }
            }
            KeyCode::Char('n') | KeyCode::Char('N')
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_none() =>
            {
                // Jump to next failed case
                let rows = &self.report.suite_report.detail_rows;
                let start = self.detail_scroll.cursor + 1;
                let found = (start..rows.len()).find(|&i| rows[i].has_error);
                if let Some(idx) = found {
                    self.detail_scroll.move_to(idx);
                }
                Transition::None
            }
            KeyCode::Char('p') | KeyCode::Char('P')
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_none() =>
            {
                // Jump to previous failed case
                let rows = &self.report.suite_report.detail_rows;
                let found = (0..self.detail_scroll.cursor)
                    .rev()
                    .find(|&i| rows[i].has_error);
                if let Some(idx) = found {
                    self.detail_scroll.move_to(idx);
                }
                Transition::None
            }
            KeyCode::Char('x') | KeyCode::Char('X')
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_some() =>
            {
                self.expanded.clear_all();
                Transition::None
            }
            KeyCode::Left if self.active_tab == ResultTab::Summary && self.metric_group_idx > 0 => {
                self.metric_group_idx -= 1;
                Transition::None
            }
            KeyCode::Right if self.active_tab == ResultTab::Summary => {
                self.metric_group_idx += 1;
                Transition::None
            }
            KeyCode::Up => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_some() {
                    // Drill-down: move compare cursor
                    self.compare_scroll.move_up();
                    self.scroll_to_comparison_cursor();
                    if shift {
                        self.expanded.expand(self.compare_scroll.cursor);
                    }
                } else {
                    match self.active_tab {
                        ResultTab::Summary => self.kv_scroll.scroll_up(),
                        ResultTab::Detail => self.detail_scroll.move_up(),
                    }
                }
                Transition::None
            }
            KeyCode::Down => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_some() {
                    // Drill-down: move compare cursor down
                    let data = &self.case_details[self.detail_selected.unwrap()];
                    let n_max = data
                        .combined_retrieved_ids
                        .len()
                        .min(10)
                        .max(data.expected_combined_ranking.len().min(5));
                    if n_max > 0 {
                        self.compare_scroll.move_down(n_max);
                        self.scroll_to_comparison_cursor();
                        if shift {
                            self.expanded.expand(self.compare_scroll.cursor);
                        }
                    }
                } else {
                    match self.active_tab {
                        ResultTab::Summary => self.kv_scroll.scroll_down(),
                        ResultTab::Detail => {
                            let max = self.report.suite_report.detail_rows.len();
                            self.detail_scroll.move_down(max);
                        }
                    }
                }
                Transition::None
            }
            _ => Transition::None,
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if self.detail_selected.is_some() {
            match mouse.kind {
                MouseEventKind::ScrollDown => self.drill_scroll.scroll_down(),
                MouseEventKind::ScrollUp => self.drill_scroll.scroll_up(),
                _ => {}
            }
        } else {
            match mouse.kind {
                MouseEventKind::ScrollDown => {
                    let max = self.report.suite_report.detail_rows.len();
                    self.detail_scroll.move_down(max);
                }
                MouseEventKind::ScrollUp => {
                    self.detail_scroll.move_up();
                }
                _ => {}
            }
        }
    }
}

impl Component for ResultsState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => self.handle_key(key),
            ComponentEvent::Mouse(mouse) => {
                self.handle_mouse(mouse);
                Transition::None
            }
            ComponentEvent::Tick => Transition::None,
        }
    }
    fn view(&self, frame: &mut Frame) {
        self.render(frame);
    }
}

/// Build a multi-line detailed description of a retrieved node and its expected counterpart.
fn format_node_detail(
    actual: Option<&MemoryId>,
    expected: Option<&MemoryId>,
    data: &RetrieveCaseData,
) -> String {
    let mut lines = Vec::new();

    if let Some(id) = actual {
        let name = data
            .graph_names
            .as_ref()
            .and_then(|m| m.get(id))
            .cloned()
            .unwrap_or_default();
        if let Some(summary) = data.id_names.as_ref().and_then(|m| m.get(id)) {
            lines.push(format!(
                "实际: {} [{}]  {}",
                name, summary.type_label, summary.primary
            ));
            if !summary.secondary.is_empty() {
                lines.push(format!("      {}", summary.secondary));
            }
            if !summary.tags.is_empty() {
                lines.push(format!("     标签: [{}]", summary.tags.join(", ")));
            }
        } else {
            lines.push(format!("实际: {}", name));
        }
    }

    if let Some(eid) = expected {
        let ename = data
            .graph_names
            .as_ref()
            .and_then(|m| m.get(eid))
            .cloned()
            .unwrap_or_default();
        if let Some(esummary) = data.id_names.as_ref().and_then(|m| m.get(eid)) {
            lines.push(format!(
                "期望: {} [{}]  {}",
                ename, esummary.type_label, esummary.primary
            ));
            if !esummary.secondary.is_empty() {
                lines.push(format!("      {}", esummary.secondary));
            }
            if !esummary.tags.is_empty() {
                lines.push(format!("     标签: [{}]", esummary.tags.join(", ")));
            }
        } else {
            lines.push(format!("期望: {}", ename));
        }
    }

    lines.join("\n")
}

/// Format a query variant as an indented tree of field lines (Rust field names).
fn format_variant_lines(v: &MemoryRetrieveQueryVariant) -> Vec<String> {
    let mut out = Vec::new();
    match v {
        MemoryRetrieveQueryVariant::Semantic(units) => {
            out.push("Semantic".into());
            for u in units {
                if let Some(ci) = u.concept_identifier() {
                    out.push(format!("  concept_identifier: \"{}\"", ci));
                }
                if let Some(desc) = u.description() {
                    out.push(format!("  description: \"{}\"", desc));
                }
            }
        }
        MemoryRetrieveQueryVariant::Situation(units) => {
            out.push("Situation".into());
            for u in units {
                if let Some(n) = u.narrative() {
                    out.push(format!("  narrative: \"{}\"", n));
                }
                if let Some(locs) = u.location() {
                    for loc in locs {
                        let mut parts = vec![format!("name: \"{}\"", loc.name())];
                        if let Some(c) = loc.coordinates() {
                            parts.push(format!("coordinates: \"{}\"", c));
                        }
                        out.push(format!("  location: [{}]", parts.join(", ")));
                    }
                }
                if let Some(parts) = u.participants() {
                    for p in parts {
                        let mut pv = Vec::new();
                        if let Some(n) = p.name() {
                            pv.push(format!("name: \"{}\"", n));
                        }
                        if let Some(r) = p.role() {
                            pv.push(format!("role: \"{}\"", r));
                        }
                        out.push(format!("  participant: [{}]", pv.join(", ")));
                    }
                }
                if let Some(env) = u.environment() {
                    let mut ev = Vec::new();
                    if let Some(a) = env.atmosphere() {
                        ev.push(format!("atmosphere: \"{}\"", a));
                    }
                    if let Some(t) = env.tone() {
                        ev.push(format!("tone: \"{}\"", t));
                    }
                    out.push(format!("  environment: {{{}}}", ev.join(", ")));
                }
                if let Some(events) = u.event() {
                    for e in events {
                        let mut ev = vec![format!("action: \"{}\"", e.action())];
                        if let Some(i) = e.initiator() {
                            ev.push(format!("initiator: \"{}\"", i));
                        }
                        if let Some(t) = e.target() {
                            ev.push(format!("target: \"{}\"", t));
                        }
                        out.push(format!("  event: [{}]", ev.join(", ")));
                    }
                }
            }
        }
    }
    out
}
