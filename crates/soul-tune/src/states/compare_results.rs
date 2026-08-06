use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::engine::compare::{CompareCaseData, CompareReport};
use crate::widgets::expandable::ExpandableList;
use crate::widgets::scroll::ScrollState;
use crate::widgets::status_bar;
use std::cell::{Cell, RefCell};

use soul_mem_core::memory_note::MemoryId;

pub struct CompareResultsState {
    report: CompareReport,
    detail_selected: Option<usize>,
    table_scroll: ScrollState,
    drill_scroll: ScrollState,
    compare_cursor: usize,
    expanded: ExpandableList,
    case_details: Vec<CompareCaseData>,
    comparison_lines: RefCell<Vec<usize>>,
    drill_viewport: Cell<usize>,
}

impl CompareResultsState {
    pub fn new(report: CompareReport) -> Self {
        let n_max = report
            .cases
            .iter()
            .map(|c| {
                c.embedding_retrieved
                    .len()
                    .min(10)
                    .max(c.expected_combined_ranking.len().min(5))
            })
            .max()
            .unwrap_or(5);
        Self {
            detail_selected: None,
            table_scroll: ScrollState::new(),
            drill_scroll: ScrollState::new(),
            compare_cursor: 0,
            expanded: ExpandableList::new(n_max),
            case_details: report.cases.clone(),
            report,
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
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .split(area);

        Block::bordered()
            .title(" 比对结果 · Embedding vs FullPipeline ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        if self.detail_selected.is_some() {
            self.render_drilldown(frame, layout[1]);
        } else {
            self.render_table(frame, layout[1]);
        }

        if self.detail_selected.is_some() {
            status_bar::render_status_bar(
                frame,
                layout[2],
                &[
                    ("[↑↓]".into(), "滚动".into()),
                    ("[Enter]".into(), "展开".into()),
                    ("[Q/Esc]".into(), "返回列表".into()),
                ],
            );
        } else {
            status_bar::render_status_bar(
                frame,
                layout[2],
                &[
                    ("[↑↓]".into(), "选择".into()),
                    ("[Enter]".into(), "查看详情".into()),
                    ("[Q/Esc]".into(), "返回".into()),
                ],
            );
        }
    }

    fn render_table(&self, frame: &mut Frame, area: Rect) {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(5), Constraint::Fill(1)])
            .split(area);

        // ── Aggregate summary ──
        let agg_block = Block::bordered().title(" 汇总 ").fg(Color::Yellow);
        let agg_inner = agg_block.inner(split[0]);
        agg_block.render(split[0], frame.buffer_mut());

        let agg = &self.report.aggregate;
        let hit_improve = agg.avg_fullpipeline_hit - agg.avg_embedding_hit;
        let mrr_improve = agg.avg_fullpipeline_mrr - agg.avg_embedding_mrr;

        let agg_lines = vec![
            Line::from(vec![
                Span::raw(format!(
                    "  平均 Hit   Emb={:.2}  Full={:.2}  ",
                    agg.avg_embedding_hit, agg.avg_fullpipeline_hit,
                )),
                diff_span(hit_improve),
            ]),
            Line::from(vec![
                Span::raw(format!(
                    "  平均 MRR   Emb={:.4}  Full={:.4}  ",
                    agg.avg_embedding_mrr, agg.avg_fullpipeline_mrr,
                )),
                diff_span(mrr_improve),
            ]),
            Line::from(Span::raw(format!(
                "  Hit 提升: {}/{} ({:.0}%)   MRR 提升: {}/{} ({:.0}%)",
                agg.hit_improvement_count,
                agg.case_count,
                if agg.case_count > 0 {
                    agg.hit_improvement_count as f64 / agg.case_count as f64 * 100.0
                } else {
                    0.0
                },
                agg.mrr_improvement_count,
                agg.case_count,
                if agg.case_count > 0 {
                    agg.mrr_improvement_count as f64 / agg.case_count as f64 * 100.0
                } else {
                    0.0
                },
            ))),
        ];
        frame.render_widget(Paragraph::new(Text::from(agg_lines)), agg_inner);

        // ── Per-case table ──
        let table_block = Block::bordered().title(" 每用例对比 ").fg(Color::Cyan);
        let table_inner = table_block.inner(split[1]);
        table_block.render(split[1], frame.buffer_mut());

        let (content_rect, bar_rect) = ScrollState::split_area(table_inner);

        // Header
        let header = Line::from(Span::styled(
            "  用例                     Emb_Hit  Full_Hit  ΔHit    Emb_MRR     Full_MRR    ΔMRR",
            Style::new().cyan().bold(),
        ));
        frame.render_widget(
            Paragraph::new(header),
            Rect::new(content_rect.x, content_rect.y, content_rect.width, 1),
        );

        let offset = ScrollState::offset(
            content_rect.height.saturating_sub(1),
            self.report.cases.len(),
            self.table_scroll.cursor,
        );

        for (i, case) in self.report.cases.iter().enumerate().skip(offset) {
            let y = content_rect.y + 1 + (i - offset) as u16;
            if y >= content_rect.y + content_rect.height {
                break;
            }
            let is_active = i == self.table_scroll.cursor;
            let name = if case.case_name.chars().count() > 22 {
                let trimmed: String = case.case_name.chars().take(20).collect();
                format!("{}..", trimmed)
            } else {
                format!("{:24}", case.case_name)
            };
            let hit_diff = case.fullpipeline_hit - case.embedding_hit;
            let mrr_diff = case.fullpipeline_mrr - case.embedding_mrr;

            let hit_str = if hit_diff > 0.0 {
                format!("+{:.2}  ", hit_diff)
            } else if hit_diff < 0.0 {
                format!("{:.2}  ", hit_diff)
            } else {
                "  -  ".to_string()
            };
            let mrr_str = if mrr_diff > 0.0 {
                format!("+{:.4}", mrr_diff)
            } else if mrr_diff < 0.0 {
                format!("{:.4}", mrr_diff)
            } else {
                "    -".to_string()
            };

            let (bg, fg) = if is_active {
                (Color::Cyan, Color::Black)
            } else {
                (Color::Reset, Color::Reset)
            };

            let line = Line::from(vec![
                Span::raw(format!(" {:24} ", name)),
                Span::styled(
                    format!(" {:.2}     ", case.embedding_hit),
                    Style::default().fg(fg).bg(bg),
                ),
                Span::styled(
                    format!("{:.2}     ", case.fullpipeline_hit),
                    Style::default().fg(fg).bg(bg),
                ),
                diff_span_styled(hit_diff, hit_str, bg),
                Span::styled(
                    format!(" {:.4}   ", case.embedding_mrr),
                    Style::default().fg(fg).bg(bg),
                ),
                Span::styled(
                    format!("{:.4}   ", case.fullpipeline_mrr),
                    Style::default().fg(fg).bg(bg),
                ),
                diff_span_styled(mrr_diff, mrr_str, bg),
            ]);
            frame.render_widget(
                Paragraph::new(line).bg(bg),
                Rect::new(content_rect.x, y, content_rect.width, 1),
            );
        }

        ScrollState::render_scrollbar(
            frame,
            bar_rect,
            self.report.cases.len(),
            content_rect.height.saturating_sub(1),
            offset,
        );
    }

    fn render_drilldown(&self, frame: &mut Frame, area: Rect) {
        let idx = self.detail_selected.unwrap();
        let data = &self.case_details[idx];
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Fill(1)])
            .split(area);

        let mut lines: Vec<Line> = Vec::new();
        let hdr = Style::new().yellow().bold();
        let green = Style::new().green();
        let red = Style::new().red();
        let gray = Style::new().dark_gray();

        lines.push(Line::from(Span::raw(format!(
            " 用例: {} (tag={:.1}, variant={:.1})",
            data.case_name, data.tag_weight, data.variant_weight,
        ))));

        let emb_pass = data.embedding_hit > 0.0;
        let full_pass = data.fullpipeline_hit > 0.0;
        lines.push(Line::from(vec![
            Span::raw(" Embedding: "),
            Span::styled(
                if emb_pass { "✓ Pass" } else { "✗ Fail" },
                if emb_pass { green } else { red },
            ),
            Span::raw("  FullPipeline: "),
            Span::styled(
                if full_pass { "✓ Pass" } else { "✗ Fail" },
                if full_pass { green } else { red },
            ),
        ]));
        lines.push(Line::from(""));

        // Metrics
        lines.push(Line::from(Span::styled(" ── 指标对比 ──", hdr)));
        lines.push(Line::from(Span::raw(format!(
            "  Hit:   Emb={:.2}  Full={:.2}  {}",
            data.embedding_hit,
            data.fullpipeline_hit,
            diff_text(data.fullpipeline_hit - data.embedding_hit),
        ))));
        lines.push(Line::from(Span::raw(format!(
            "  MRR:   Emb={:.4}  Full={:.4}  {}",
            data.embedding_mrr,
            data.fullpipeline_mrr,
            diff_text(data.fullpipeline_mrr - data.embedding_mrr),
        ))));
        lines.push(Line::from(""));

        // Side-by-side
        lines.push(Line::from(Span::styled(
            " ── 检索结果对比 (Enter展开) ──",
            hdr,
        )));
        let n_max = data
            .embedding_retrieved
            .len()
            .min(10)
            .max(data.expected_combined_ranking.len().min(5));
        let expected_set: std::collections::HashSet<&MemoryId> =
            data.expected_combined_ranking.iter().collect();

        for pos in 0..n_max {
            if pos == 0 {
                self.comparison_lines.borrow_mut().clear();
            }
            let is_cursor = pos == self.compare_cursor;
            let is_expanded = self.expanded.is_expanded(pos);
            let mut spans = Vec::new();
            spans.push(Span::styled(
                format!(" {}  #{}", if is_cursor { "▶" } else { " " }, pos + 1),
                if is_cursor { green } else { Style::new() },
            ));

            if let Some(id) = data.embedding_retrieved.get(pos) {
                let is_hit = expected_set.contains(id);
                spans.push(Span::raw(" Emb:"));
                spans.push(Span::styled(
                    if is_hit { "✓" } else { "-" },
                    if is_hit { green } else { gray },
                ));
            } else {
                spans.push(Span::raw(" Emb:—"));
            }
            spans.push(Span::raw(" "));

            if let Some(id) = data.fullpipeline_retrieved.get(pos) {
                let is_hit = expected_set.contains(id);
                spans.push(Span::raw(" Full:"));
                spans.push(Span::styled(
                    if is_hit { "✓" } else { "-" },
                    if is_hit { green } else { gray },
                ));
            } else {
                spans.push(Span::raw(" Full:—"));
            }
            spans.push(Span::raw(" "));

            if let Some(eid) = data.expected_combined_ranking.get(pos) {
                if !expected_set.contains(eid) {
                    spans.push(Span::styled("✗未命中", red));
                }
            }
            self.comparison_lines.borrow_mut().push(lines.len());
            lines.push(Line::from(spans));

            if is_expanded {
                if let Some(id) = data.embedding_retrieved.get(pos) {
                    lines.push(Line::from(Span::styled(format!("   Emb: {:?}", id), gray)));
                }
                if let Some(id) = data.fullpipeline_retrieved.get(pos) {
                    lines.push(Line::from(Span::styled(format!("   Full: {:?}", id), gray)));
                }
                if let Some(id) = data.expected_combined_ranking.get(pos) {
                    lines.push(Line::from(Span::styled(format!("   期望: {:?}", id), gray)));
                }
            }
        }

        let block = Block::bordered().title(" 用例详情 ").fg(Color::Cyan);
        let inner = block.inner(layout[0]);
        self.drill_viewport.set(inner.height as usize);
        block.render(layout[0], frame.buffer_mut());
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((self.drill_scroll.offset as u16, 0)),
            inner,
        );
    }

    fn scroll_to_comparison_cursor(&mut self) {
        let vis = self.drill_viewport.get();
        if vis == 0 {
            return;
        }
        let lines = self.comparison_lines.borrow();
        if let Some(&line_off) = lines.get(self.compare_cursor) {
            if line_off < self.drill_scroll.offset {
                self.drill_scroll.offset = line_off;
            } else if line_off >= self.drill_scroll.offset + vis {
                self.drill_scroll.offset = line_off.saturating_sub(vis.saturating_sub(1));
            }
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                if self.detail_selected.is_some() {
                    self.detail_selected = None;
                    self.drill_scroll.reset();
                    self.expanded.clear_all();
                    self.compare_cursor = 0;
                } else {
                    return Transition::ToMain;
                }
                Transition::None
            }
            KeyCode::Enter => {
                if self.detail_selected.is_some() {
                    self.expanded.toggle(self.compare_cursor);
                } else if self.table_scroll.cursor < self.case_details.len() {
                    self.detail_selected = Some(self.table_scroll.cursor);
                    self.drill_scroll.reset();
                    self.expanded.clear_all();
                    self.compare_cursor = 0;
                    if let Some(data) = self.case_details.get(self.table_scroll.cursor) {
                        let n_max = data
                            .embedding_retrieved
                            .len()
                            .min(10)
                            .max(data.expected_combined_ranking.len().min(5));
                        self.expanded.resize(n_max);
                    }
                }
                Transition::None
            }
            KeyCode::Up => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                if self.detail_selected.is_some() {
                    if self.compare_cursor > 0 {
                        self.compare_cursor -= 1;
                        self.scroll_to_comparison_cursor();
                        if shift {
                            self.expanded.expand(self.compare_cursor);
                        }
                    }
                } else {
                    self.table_scroll.move_up();
                }
                Transition::None
            }
            KeyCode::Down => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                if self.detail_selected.is_some() {
                    if let Some(data) = self.case_details.get(self.detail_selected.unwrap()) {
                        let n_max = data
                            .embedding_retrieved
                            .len()
                            .min(10)
                            .max(data.expected_combined_ranking.len().min(5));
                        if self.compare_cursor + 1 < n_max {
                            self.compare_cursor += 1;
                            self.scroll_to_comparison_cursor();
                            if shift {
                                self.expanded.expand(self.compare_cursor);
                            }
                        }
                    }
                } else {
                    self.table_scroll.move_down(self.case_details.len());
                }
                Transition::None
            }
            KeyCode::Char('x') | KeyCode::Char('X') if self.detail_selected.is_some() => {
                self.expanded.clear_all();
                Transition::None
            }
            _ => Transition::None,
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        if self.detail_selected.is_some() {
            match mouse.kind {
                MouseEventKind::ScrollDown => self.drill_scroll.scroll_down(),
                MouseEventKind::ScrollUp => self.drill_scroll.scroll_up(),
                _ => {}
            }
        } else {
            match mouse.kind {
                MouseEventKind::ScrollDown => {
                    self.table_scroll.move_down(self.case_details.len());
                }
                MouseEventKind::ScrollUp => {
                    self.table_scroll.move_up();
                }
                _ => {}
            }
        }
    }
}

impl Component for CompareResultsState {
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

fn diff_span(diff: f64) -> Span<'static> {
    if diff > 0.0 {
        Span::styled(format!("Δ+{:.4}", diff), Style::new().green())
    } else if diff < 0.0 {
        Span::styled(format!("Δ{:.4}", diff), Style::new().red())
    } else {
        Span::styled("Δ    -".to_string(), Style::default())
    }
}

fn diff_span_styled(diff: f64, text: String, bg: Color) -> Span<'static> {
    if diff > 0.0 {
        Span::styled(text, Style::new().green().bg(bg))
    } else if diff < 0.0 {
        Span::styled(text, Style::new().red().bg(bg))
    } else {
        Span::styled(text, Style::default().bg(bg))
    }
}

fn diff_text(diff: f64) -> String {
    if diff > 0.0 {
        format!("Δ+{:.4}", diff)
    } else if diff < 0.0 {
        format!("Δ{:.4}", diff)
    } else {
        "Δ    -".to_string()
    }
}
