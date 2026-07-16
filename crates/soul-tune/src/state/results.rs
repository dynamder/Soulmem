use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use crate::base::{TestReport, Transition};
use crate::eval::retrieve_suite::RetrieveCaseData;
use crate::tui::components::status_bar;
use soul_mem_core::memory_note::MemoryId;

#[derive(PartialEq, Eq)]
pub enum ResultTab {
    Summary,
    Detail,
}

pub struct ResultsState {
    pub report: TestReport,
    pub active_tab: ResultTab,
    pub kv_scroll: usize,
    pub metric_group_idx: usize,
    #[allow(dead_code)]
    pub chart_scroll: usize,
    pub log_scroll: usize,
    pub log_filter: String,
    #[allow(dead_code)]
    pub log_search: String,
    pub detail_selected: Option<usize>,
    pub drill_scroll: usize,
    pub detail_cursor: usize,
    case_details: Vec<RetrieveCaseData>,
}

impl ResultsState {
    pub fn new(report: TestReport) -> Self {
        let case_details = report
            .suite_report
            .outcomes
            .iter()
            .filter_map(|o| o.data.downcast_ref::<RetrieveCaseData>().cloned())
            .collect();
        Self {
            report,
            active_tab: ResultTab::Summary,
            kv_scroll: 0,
            metric_group_idx: 0,
            chart_scroll: 0,
            log_scroll: 0,
            log_filter: "ALL".into(),
            log_search: String::new(),
            detail_selected: None,
            drill_scroll: 0,
            detail_cursor: 0,
            case_details,
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
        for group in &self.report.suite_report.summary_groups {
            let title = format!(" {} ", group.label);
            frame.render_widget(
                Paragraph::new(title).fg(Color::Yellow).bold(),
                Rect::new(kv_inner.x, y, kv_inner.width, 1),
            );
            y += 1;
            for (k, v) in &group.items {
                let line = format!("  {}: {}", k, v);
                frame.render_widget(
                    Paragraph::new(line),
                    Rect::new(kv_inner.x + 1, y, kv_inner.width - 2, 1),
                );
                y += 1;
            }
            y += 1;
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

        // Rows with cursor highlight
        let header_offset = 1;
        for (i, row) in self
            .report
            .suite_report
            .detail_rows
            .iter()
            .enumerate()
            .skip(self.log_scroll)
        {
            let y = log_inner.y + header_offset + (i - self.log_scroll) as u16;
            if y >= log_inner.y + log_inner.height {
                break;
            }
            let is_active = i == self.detail_cursor;
            let (color, bg) = if is_active {
                (Color::Black, Color::Cyan)
            } else if row.has_error {
                (Color::Red, Color::Reset)
            } else {
                (Color::Reset, Color::Reset)
            };
            frame.render_widget(
                Paragraph::new(row.text.as_str()).fg(color).bg(bg),
                Rect::new(log_inner.x, y, log_inner.width, 1),
            );
        }
    }

    fn render_detail_drilldown(&self, frame: &mut Frame, area: Rect, index: usize) {
        let data = &self.case_details[index];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Fill(1)])
            .split(area);

        let mut lines = Vec::new();
        lines.push(format!(" 用例: {}", data.case_name));
        lines.push(format!(
            " Tag权重: {:.1}  Variant权重: {:.1}  状态: {}",
            data.tag_weight,
            data.variant_weight,
            if data.combined_ranking_metrics.hit_rate > 0.0
                || data.combined_ranking_metrics.mrr > 0.0
            {
                "✓ 通过"
            } else {
                "✗ 失败"
            }
        ));
        lines.push(String::new());

        // Combined ranking metrics
        lines.push(" ── 综合排序指标 ──".into());
        lines.push(format!("  K     Recall    Precision  NDCG"));
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
            lines.push(format!("  @{:<2}   {:.4}    {:.4}    {:.4}", k, r, p, n));
        }
        lines.push(format!(
            "  MRR: {:.4}     Hit: {:.2}",
            data.combined_ranking_metrics.mrr, data.combined_ranking_metrics.hit_rate
        ));
        lines.push(String::new());

        // Per-sub-query metrics
        lines.push(" ── 各子查询 ──".into());
        for sq in &data.per_query_metrics {
            lines.push(format!(
                "  Q{}  MRR={:.4}  Hit={:.2}  Recall@3={:.4}  {}",
                sq.query_index,
                sq.ranking_metrics.mrr,
                sq.ranking_metrics.hit_rate,
                sq.ranking_metrics
                    .recall_at
                    .iter()
                    .find(|(k, _)| *k == 3)
                    .map(|(_, v)| v)
                    .unwrap_or(&0.0),
                if sq.ranking_metrics.hit_rate > 0.0 {
                    "✓"
                } else {
                    "✗"
                },
            ));
        }
        lines.push(String::new());

        // ── Side-by-side: actual vs expected results ──
        lines.push(" ── 检索结果 vs 预期 ──".into());
        // Build hit set: which expected IDs were found in retrieved
        let retrieved_set: std::collections::HashSet<&MemoryId> =
            data.combined_retrieved_ids.iter().take(10).collect();
        let expected_hit: Vec<bool> = data
            .expected_combined_ranking
            .iter()
            .map(|eid| retrieved_set.contains(eid))
            .collect();

        let n_max = data
            .combined_retrieved_ids
            .len()
            .min(10)
            .max(data.expected_combined_ranking.len().min(5));
        lines.push(format!(
            "  {:<8} {:<13}  {:<6}  {:<13}",
            "实际检出", "", "期望排序", ""
        ));
        for pos in 0..n_max {
            let mut row = String::new();
            // Left: actual
            if let Some(id) = data.combined_retrieved_ids.get(pos) {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(id))
                    .cloned()
                    .unwrap_or_default();
                let summary = data.id_names.as_ref().and_then(|m| m.get(id));
                let (type_lbl, primary) = match summary {
                    Some(s) => (
                        s.type_label.chars().take(6).collect::<String>(),
                        s.primary.chars().take(12).collect::<String>(),
                    ),
                    None => (String::new(), String::new()),
                };
                let is_hit = data.expected_combined_ranking.iter().any(|eid| eid == id);
                let hit = if is_hit { " ✓" } else { " -" };
                row.push_str(&format!(
                    "  #{:<2} {:<10} [{:6}] {:<12} {}",
                    pos + 1,
                    name,
                    type_lbl,
                    primary,
                    hit
                ));
            } else {
                row.push_str(&format!("  #{:<2} {:24}  {}", pos + 1, "—", "  "));
            }
            // Right: expected
            if let Some(eid) = data.expected_combined_ranking.get(pos) {
                let ename = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(eid))
                    .cloned()
                    .unwrap_or_default();
                let esummary = data.id_names.as_ref().and_then(|m| m.get(eid));
                let (etype, eprimary) = match esummary {
                    Some(s) => (
                        s.type_label.chars().take(6).collect::<String>(),
                        s.primary.chars().take(12).collect::<String>(),
                    ),
                    None => (String::new(), String::new()),
                };
                let missed = if !retrieved_set.contains(eid) {
                    " ✗未命中"
                } else {
                    ""
                };
                row.push_str(&format!(
                    "  #{:<2} {:<10} [{:6}] {:<12}{}",
                    pos + 1,
                    ename,
                    etype,
                    eprimary,
                    missed
                ));
            } else {
                row.push_str(&format!("  #{:<2} {:>24}", pos + 1, "—"));
            }
            lines.push(row);
        }
        lines.push(String::new());

        // Missed expectations
        let missed: Vec<&MemoryId> = data
            .expected_combined_ranking
            .iter()
            .filter(|eid| !retrieved_set.contains(eid))
            .collect();
        if !missed.is_empty() {
            let mut missed_line = "  未命中期望: ".to_string();
            for (i, id) in missed.iter().enumerate() {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(*id))
                    .cloned()
                    .unwrap_or_default();
                let summary = data.id_names.as_ref().and_then(|m| m.get(*id));
                let extra = match summary {
                    Some(s) => format!(
                        "{} [{}]{}",
                        name,
                        s.type_label,
                        s.primary.chars().take(12).collect::<String>()
                    ),
                    None => name,
                };
                if i > 0 {
                    missed_line.push_str(", ");
                }
                missed_line.push_str(&extra);
            }
            lines.push(missed_line);
        }

        let content = lines[self.drill_scroll..]
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let block = Block::bordered().title(" 用例详情 ").fg(Color::Yellow);
        let inner = block.inner(layout[0]);
        block.render(layout[0], frame.buffer_mut());
        frame.render_widget(Paragraph::new(content).wrap(Wrap { trim: false }), inner);
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                if self.detail_selected.is_some() {
                    self.detail_selected = None;
                    self.drill_scroll = 0;
                    Transition::None
                } else {
                    Transition::ToMain
                }
            }
            KeyCode::Tab => {
                self.detail_selected = None;
                self.drill_scroll = 0;
                self.active_tab = match self.active_tab {
                    ResultTab::Summary => ResultTab::Detail,
                    ResultTab::Detail => ResultTab::Summary,
                };
                Transition::None
            }
            KeyCode::Enter
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_none() =>
            {
                if self.detail_cursor < self.case_details.len() {
                    self.detail_selected = Some(self.detail_cursor);
                    self.drill_scroll = 0;
                }
                Transition::None
            }
            KeyCode::Char('n') | KeyCode::Char('N')
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_none() =>
            {
                // Jump to next failed case
                let rows = &self.report.suite_report.detail_rows;
                let start = self.detail_cursor + 1;
                let found = (start..rows.len()).find(|&i| rows[i].has_error);
                if let Some(idx) = found {
                    self.detail_cursor = idx;
                    self.log_scroll = idx.saturating_sub(4);
                }
                Transition::None
            }
            KeyCode::Char('p') | KeyCode::Char('P')
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_none() =>
            {
                // Jump to previous failed case
                let rows = &self.report.suite_report.detail_rows;
                let found = (0..self.detail_cursor).rev().find(|&i| rows[i].has_error);
                if let Some(idx) = found {
                    self.detail_cursor = idx;
                    self.log_scroll = idx.saturating_sub(4);
                }
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
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_some() {
                    if self.drill_scroll > 0 {
                        self.drill_scroll -= 1;
                    }
                } else {
                    match self.active_tab {
                        ResultTab::Summary if self.kv_scroll > 0 => self.kv_scroll -= 1,
                        ResultTab::Detail if self.detail_cursor > 0 => {
                            self.detail_cursor -= 1;
                            // Auto-scroll: if cursor passes top of visible area
                            if self.detail_cursor < self.log_scroll && self.log_scroll > 0 {
                                self.log_scroll -= 1;
                            }
                        }
                        _ => {}
                    }
                }
                Transition::None
            }
            KeyCode::Down => {
                if self.active_tab == ResultTab::Detail && self.detail_selected.is_some() {
                    self.drill_scroll += 1;
                } else {
                    match self.active_tab {
                        ResultTab::Summary => self.kv_scroll += 1,
                        ResultTab::Detail => {
                            let max = self.report.suite_report.detail_rows.len().saturating_sub(1);
                            if self.detail_cursor < max {
                                self.detail_cursor += 1;
                                // Auto-scroll: if cursor passes bottom of visible area
                                if self.detail_cursor >= self.log_scroll + 10 {
                                    self.log_scroll += 1;
                                }
                            }
                        }
                    }
                }
                Transition::None
            }
            _ => Transition::None,
        }
    }
}
