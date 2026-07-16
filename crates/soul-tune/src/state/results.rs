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

        // Rows with highlight on visible row closest to center
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
            let is_active = i == self.log_scroll + (log_inner.height as usize / 2 - 1);
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

        // Retrieved ranking
        lines.push(" ── 检索结果 (top 10) ──".into());
        for (pos, id) in data.combined_retrieved_ids.iter().take(10).enumerate() {
            lines.push(format!("  #{:<2} {}", pos + 1, id));
        }
        lines.push(String::new());

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
                let center = self.log_scroll + 4;
                if center < self.case_details.len() {
                    self.detail_selected = Some(center);
                    self.drill_scroll = 0;
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
                        ResultTab::Detail if self.log_scroll > 0 => self.log_scroll -= 1,
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
                        ResultTab::Detail => self.log_scroll += 1,
                    }
                }
                Transition::None
            }
            _ => Transition::None,
        }
    }
}
