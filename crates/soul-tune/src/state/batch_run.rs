use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Gauge, Paragraph, Wrap};
use ratatui::Frame;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use soul_mem_core::memory_note::MemoryId;

use crate::base::{RetrieveMode, Transition};
use crate::component::{Component, ComponentEvent};
use crate::eval::batch::{process_one_dataset, scan_question_jsons, BatchResult, DatasetResult};
use crate::eval::retrieve_suite::RetrieveCaseData;
use crate::tui::components::expandable_list::ExpandableList;
use crate::tui::components::scroll_container::ScrollContainer;
use crate::tui::components::status_bar;

struct WorkerSlot {
    active: bool,
    current_name: String,
    status_line: String,
    progress: f64,
    elapsed: f64,
}

struct BatchProgress {
    done: usize,
    total: usize,
    workers: Vec<WorkerSlot>,
    results: Vec<Option<DatasetResult>>,
}

pub struct BatchRunState {
    dir: PathBuf,
    mode: RetrieveMode,
    phase: BatchPhase,
    result: Option<BatchResult>,
    progress: Option<Arc<Mutex<BatchProgress>>>,
    result_rx: Option<mpsc::Receiver<(usize, DatasetResult)>>,
    table_scroll: ScrollContainer,
    drill_dataset: Option<usize>,
    drill_case: usize,
    compare_cursor: usize,
    drill_scroll: ScrollContainer,
    expanded: ExpandableList,
}

#[derive(PartialEq)]
enum BatchPhase {
    Scanning,
    Running,
    Done,
}

impl BatchRunState {
    pub fn new(dir: PathBuf, mode: RetrieveMode) -> Self {
        Self {
            dir,
            mode,
            phase: BatchPhase::Scanning,
            result: None,
            progress: None,
            result_rx: None,
            table_scroll: ScrollContainer::new(),
            drill_dataset: None,
            drill_case: 0,
            compare_cursor: 0,
            drill_scroll: ScrollContainer::new(),
            expanded: ExpandableList::new(0),
        }
    }

    pub fn tick(&mut self) -> Option<Transition> {
        match self.phase {
            BatchPhase::Scanning => {
                let datasets = scan_question_jsons(&self.dir);
                let total = datasets.len();
                if total == 0 {
                    self.phase = BatchPhase::Done;
                    self.result = Some(BatchResult {
                        datasets: vec![],
                        total_cases: 0,
                        total_passed: 0,
                        total_failed: 0,
                        elapsed: std::time::Duration::ZERO,
                    });
                    return Some(Transition::None);
                }

                let n_workers = 4.min(total).max(1);
                let mut workers = Vec::with_capacity(n_workers);
                for _ in 0..n_workers {
                    workers.push(WorkerSlot {
                        active: false,
                        current_name: String::new(),
                        status_line: "空闲".into(),
                        progress: 0.0,
                        elapsed: 0.0,
                    });
                }
                let mut results = Vec::with_capacity(total);
                for _ in 0..total {
                    results.push(None);
                }
                let progress = Arc::new(Mutex::new(BatchProgress {
                    done: 0,
                    total,
                    workers,
                    results,
                }));
                self.progress = Some(progress.clone());
                self.phase = BatchPhase::Running;

                let (tx, rx) = mpsc::channel::<(usize, DatasetResult)>();
                self.result_rx = Some(rx);

                let datasets = datasets.clone();
                let mode = self.mode;
                let counter = Arc::new(AtomicUsize::new(0));

                for slot_idx in 0..n_workers {
                    let datasets = datasets.clone();
                    let mode = mode;
                    let counter = Arc::clone(&counter);
                    let tx = tx.clone();
                    let progress = progress.clone();
                    std::thread::Builder::new()
                        .name(format!("batch-w{}", slot_idx))
                        .spawn(move || {
                            let update_slot =
                                |p: &Arc<Mutex<BatchProgress>>,
                                 idx: usize,
                                 name: &str,
                                 status: &str,
                                 prog: f64,
                                 start: Instant| {
                                    if let Ok(mut g) = p.lock() {
                                        if idx < g.workers.len() {
                                            g.workers[idx].active = !name.is_empty();
                                            g.workers[idx].current_name = name.to_string();
                                            g.workers[idx].status_line = status.to_string();
                                            g.workers[idx].progress = prog;
                                            g.workers[idx].elapsed = start.elapsed().as_secs_f64();
                                        }
                                    }
                                };

                            loop {
                                let i = counter.fetch_add(1, Ordering::Relaxed);
                                if i >= datasets.len() {
                                    update_slot(
                                        &progress,
                                        slot_idx,
                                        "",
                                        "空闲",
                                        0.0,
                                        Instant::now(),
                                    );
                                    break;
                                }
                                let name = datasets[i]
                                    .parent()
                                    .and_then(|p| p.file_name())
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| "?".into());
                                let ds_start = Instant::now();
                                update_slot(
                                    &progress,
                                    slot_idx,
                                    &name,
                                    "加载图数据...",
                                    0.1,
                                    ds_start,
                                );

                                let ds = process_one_dataset(
                                    &datasets[i],
                                    mode,
                                    ds_start,
                                    |pct, status| {
                                        update_slot(
                                            &progress, slot_idx, &name, status, pct, ds_start,
                                        );
                                    },
                                    |_msg| {},
                                );

                                update_slot(&progress, slot_idx, &name, "完成", 1.0, ds_start);
                                let _ = tx.send((i, ds));
                            }
                        })
                        .ok();
                }
                drop(tx);

                Some(Transition::None)
            }
            BatchPhase::Running => {
                if let Some(ref rx) = self.result_rx {
                    let mut received = 0;
                    while let Ok((idx, ds)) = rx.try_recv() {
                        if let Ok(mut p) = self.progress.as_ref().unwrap().lock() {
                            p.results[idx] = Some(ds);
                            p.done += 1;
                        }
                        received += 1;
                        if received > 10 {
                            break;
                        }
                    }
                }
                let total = self
                    .progress
                    .as_ref()
                    .map(|p| p.lock().map(|g| g.total).unwrap_or(1))
                    .unwrap_or(1);
                let done = self
                    .progress
                    .as_ref()
                    .map(|p| p.lock().map(|g| g.done).unwrap_or(0))
                    .unwrap_or(0);
                if done >= total {
                    let mut all_results = Vec::new();
                    let mut total_cases = 0;
                    let mut total_passed = 0;
                    let mut total_failed = 0;
                    if let Ok(mut p) = self.progress.as_ref().unwrap().lock() {
                        for opt in p.results.drain(..) {
                            if let Some(ds) = opt {
                                total_cases += ds.total;
                                total_passed += ds.passed;
                                total_failed += ds.failed;
                                all_results.push(ds);
                            }
                        }
                    }
                    self.result = Some(BatchResult {
                        datasets: all_results,
                        total_cases,
                        total_passed,
                        total_failed,
                        elapsed: std::time::Duration::ZERO,
                    });
                    self.phase = BatchPhase::Done;
                }
                Some(Transition::None)
            }
            BatchPhase::Done => Some(Transition::None),
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        let constraints = if self.phase == BatchPhase::Running {
            let n = self
                .progress
                .as_ref()
                .map(|p| p.lock().map(|g| g.workers.len()).unwrap_or(0))
                .unwrap_or(0);
            if n > 0 && self.phase == BatchPhase::Running {
                // Title + each worker takes 3 lines (gauge + status + spacer) + status bar
                let mut cs = vec![Constraint::Length(3)];
                for _ in 0..n {
                    cs.push(Constraint::Length(3));
                }
                cs.push(Constraint::Fill(1));
                cs.push(Constraint::Length(1));
                cs
            } else {
                vec![
                    Constraint::Length(3),
                    Constraint::Fill(1),
                    Constraint::Length(1),
                ]
            }
        } else {
            vec![
                Constraint::Length(3),
                Constraint::Fill(1),
                Constraint::Length(1),
            ]
        };
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        let n_workers = self
            .progress
            .as_ref()
            .and_then(|p| p.lock().ok().map(|g| g.workers.len()))
            .unwrap_or(0);

        Block::bordered()
            .title(" 批量运行 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        match self.phase {
            BatchPhase::Scanning => {
                frame.render_widget(Paragraph::new("正在扫描目录..."), layout[1]);
            }
            BatchPhase::Running => {
                for (i, slot_layout) in layout[1..1 + n_workers].iter().enumerate() {
                    let slot = self
                        .progress
                        .as_ref()
                        .and_then(|p| p.lock().ok())
                        .map(|g| {
                            g.workers
                                .get(i)
                                .map(|w| {
                                    (
                                        w.current_name.clone(),
                                        w.status_line.clone(),
                                        w.progress,
                                        w.elapsed,
                                    )
                                })
                                .unwrap_or_default()
                        })
                        .unwrap_or_default();
                    let (name, status, progress, elapsed) = slot;

                    let gauge = Gauge::default()
                        .ratio(progress)
                        .fg(Color::Cyan)
                        .label(format!(
                            "  {:<12} {:>5.1}s  ",
                            name.chars().take(10).collect::<String>(),
                            elapsed,
                        ));
                    frame.render_widget(
                        gauge,
                        Rect::new(slot_layout.x, slot_layout.y, slot_layout.width, 1),
                    );

                    frame.render_widget(
                        Paragraph::new(status).fg(Color::DarkGray),
                        Rect::new(
                            slot_layout.x + 2,
                            slot_layout.y + 1,
                            slot_layout.width.saturating_sub(2),
                            1,
                        ),
                    );
                }
            }
            BatchPhase::Done => {
                if let Some(idx) = self.drill_dataset {
                    self.render_drilldown(frame, layout[1], idx);
                } else if let Some(ref result) = self.result {
                    self.render_dataset_table(frame, layout[1], result);
                } else {
                    frame.render_widget(
                        Paragraph::new("(无结果)").alignment(Alignment::Center),
                        layout[1],
                    );
                }
            }
        }

        let status_idx = layout.len().saturating_sub(1);
        if status_idx < layout.len() {
            let is_done = self.phase == BatchPhase::Done;
            if is_done && self.drill_dataset.is_some() {
                let case_hint = if let Some(ref result) = self.result {
                    let ds_idx = self.drill_dataset.unwrap();
                    result
                        .datasets
                        .get(ds_idx)
                        .map(|ds| format!("用例 {}/{}", self.drill_case + 1, ds.outcomes.len()))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                status_bar::render_status_bar(
                    frame,
                    layout[status_idx],
                    &[
                        ("[←→]".into(), case_hint.into()),
                        ("[↑↓]".into(), "滚屏".into()),
                        ("[Enter]".into(), "展开".into()),
                        ("[Q]".into(), "返回".into()),
                    ],
                );
            } else if is_done {
                status_bar::render_status_bar(
                    frame,
                    layout[status_idx],
                    &[
                        ("[↑↓]".into(), "选择".into()),
                        ("[Enter]".into(), "查看详情".into()),
                        ("[Esc]".into(), "返回".into()),
                    ],
                );
            } else {
                status_bar::render_status_bar(
                    frame,
                    layout[status_idx],
                    &[("[Esc]".into(), "返回".into())],
                );
            }
        }
    }

    fn render_dataset_table(&self, frame: &mut Frame, area: Rect, result: &BatchResult) {
        let block = Block::bordered().title(" 批量结果 ");
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());

        let mut sorted: Vec<usize> = (0..result.datasets.len()).collect();
        sorted.sort_by(|&a, &b| {
            let da = &result.datasets[a];
            let db = &result.datasets[b];
            db.pass_rate
                .partial_cmp(&da.pass_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (content_rect, bar_rect) = ScrollContainer::split_area(inner);
        let scroll_offset = ScrollContainer::offset(
            inner.height.saturating_sub(1),
            result.datasets.len(),
            self.table_scroll.cursor,
        );

        let col_widths: [Constraint; 7] = [
            Constraint::Length(1),
            Constraint::Length(22),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Fill(1),
        ];

        // Header row
        let h_row = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(col_widths)
            .split(Rect::new(inner.x, inner.y, inner.width, 1));
        let hs = Style::new().fg(Color::Cyan).bold();
        frame.render_widget(Paragraph::new("数据集").style(hs), h_row[1]);
        frame.render_widget(Paragraph::new("用例").style(hs), h_row[2]);
        frame.render_widget(Paragraph::new("通过").style(hs), h_row[3]);
        frame.render_widget(Paragraph::new("失败").style(hs), h_row[4]);
        frame.render_widget(Paragraph::new("通过率").style(hs), h_row[5]);
        frame.render_widget(Paragraph::new("耗时").style(hs), h_row[6]);

        // Data rows
        for (disp_i, &orig_idx) in sorted.iter().enumerate().skip(scroll_offset) {
            let y = inner.y + 1 + (disp_i - scroll_offset) as u16;
            if y >= inner.y + inner.height {
                break;
            }
            let ds = &result.datasets[orig_idx];
            let active = disp_i == self.table_scroll.cursor;
            let fg = if ds.error.is_some() {
                Color::Red
            } else if ds.pass_rate >= 80.0 {
                Color::Green
            } else {
                Color::Yellow
            };
            let style = if active {
                Style::default().fg(Color::Black).bg(Color::Cyan)
            } else {
                Style::default().fg(fg)
            };

            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(col_widths)
                .split(Rect::new(content_rect.x, y, content_rect.width, 1));

            frame.render_widget(
                Paragraph::new(if active { "▶" } else { " " }).style(style),
                cols[0],
            );
            frame.render_widget(Paragraph::new(ds.name.as_str()).style(style), cols[1]);
            frame.render_widget(
                Paragraph::new(format!("{}", ds.total)).style(style),
                cols[2],
            );
            frame.render_widget(
                Paragraph::new(format!("{}", ds.passed)).style(style),
                cols[3],
            );
            frame.render_widget(
                Paragraph::new(format!("{}", ds.failed)).style(style),
                cols[4],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.1}%", ds.pass_rate)).style(style),
                cols[5],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.1}s", ds.elapsed.as_secs_f64())).style(style),
                cols[6],
            );
        }
        ScrollContainer::render_scrollbar(
            frame,
            bar_rect,
            result.datasets.len(),
            inner.height.saturating_sub(1),
            scroll_offset,
        );
    }

    fn render_drilldown(&self, frame: &mut Frame, area: Rect, ds_idx: usize) {
        if let Some(ref result) = self.result {
            let ds = &result.datasets[ds_idx];
            let data = &ds.outcomes;
            if self.drill_case >= data.len() {
                return;
            }

            let case = &data[self.drill_case];
            let lines: Vec<Line> = if let Some(rc) = case.data.downcast_ref::<RetrieveCaseData>() {
                let mut l = Vec::new();
                let hdr = Style::new().yellow().bold();
                let green = Style::new().green();
                let red = Style::new().red();
                let gray = Style::new().dark_gray();

                l.push(Line::from(Span::raw(format!(
                    " {} · {}",
                    ds.name, rc.case_name
                ))));
                let passed = rc.combined_ranking_metrics.hit_rate > 0.0
                    || rc.combined_ranking_metrics.mrr > 0.0;
                l.push(Line::from(vec![
                    Span::raw(" 状态: "),
                    Span::styled(
                        if passed { "✓ 通过" } else { "✗ 失败" },
                        if passed { green } else { red },
                    ),
                ]));
                l.push(Line::from(""));

                l.push(Line::from(Span::styled(" ── 综合排序指标 ──", hdr)));
                l.push(Line::from(Span::raw("  K     Recall    Precision  NDCG")));
                for (k, r) in &rc.combined_ranking_metrics.recall_at {
                    let p = rc
                        .combined_ranking_metrics
                        .precision_at
                        .iter()
                        .find(|(pk, _)| pk == k)
                        .map(|(_, v)| v)
                        .unwrap_or(&0.0);
                    let n = rc
                        .combined_ranking_metrics
                        .ndcg_at
                        .iter()
                        .find(|(nk, _)| nk == k)
                        .map(|(_, v)| v)
                        .unwrap_or(&0.0);
                    l.push(Line::from(Span::raw(format!(
                        "  @{:<2}   {:.4}    {:.4}    {:.4}",
                        k, r, p, n
                    ))));
                }
                l.push(Line::from(Span::styled(
                    format!(
                        "  MRR: {:.4}     Hit: {:.2}",
                        rc.combined_ranking_metrics.mrr, rc.combined_ranking_metrics.hit_rate
                    ),
                    gray,
                )));

                l.push(Line::from(""));
                l.push(Line::from(Span::styled(" ── 检索结果 vs 预期 ──", hdr)));
                let retrieved_set: std::collections::HashSet<&MemoryId> =
                    rc.combined_retrieved_ids.iter().take(10).collect();
                let n_max = rc
                    .combined_retrieved_ids
                    .len()
                    .min(10)
                    .max(rc.expected_combined_ranking.len().min(5));
                for pos in 0..n_max {
                    let is_cursor = pos == self.compare_cursor;
                    let mut spans = Vec::new();
                    let prefix = if is_cursor { "▶" } else { " " };
                    spans.push(Span::styled(
                        format!(" {}  #{}", prefix, pos + 1),
                        if is_cursor {
                            Style::new().yellow().bold()
                        } else {
                            Style::new()
                        },
                    ));
                    if let Some(id) = rc.combined_retrieved_ids.get(pos) {
                        let name = rc
                            .graph_names
                            .as_ref()
                            .and_then(|m| m.get(id))
                            .cloned()
                            .unwrap_or_default();
                        let is_hit = rc.expected_combined_ranking.iter().any(|eid| eid == id);
                        spans.push(Span::raw(format!(" {:<10} ", name)));
                        spans.push(Span::styled(
                            if is_hit { "✓" } else { "-" },
                            if is_hit { green } else { gray },
                        ));
                    } else {
                        spans.push(Span::raw(format!(" {:<10}  ", "—")));
                    }
                    spans.push(Span::raw("  "));
                    if let Some(eid) = rc.expected_combined_ranking.get(pos) {
                        let ename = rc
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
                    let line_style = if is_cursor {
                        Style::default().bg(Color::DarkGray)
                    } else {
                        Style::default()
                    };
                    l.push(Line::from(spans).style(line_style));

                    // Expanded detail for this row
                    if self.expanded.is_expanded(pos) {
                        let retrieved_id = rc.combined_retrieved_ids.get(pos);
                        let expected_id = rc.expected_combined_ranking.get(pos);
                        let gray = Style::new().dark_gray();

                        if let Some(id) = retrieved_id {
                            let name = rc
                                .graph_names
                                .as_ref()
                                .and_then(|m| m.get(id))
                                .cloned()
                                .unwrap_or_default();
                            if let Some(s) = rc.id_names.as_ref().and_then(|m| m.get(id)) {
                                l.push(Line::from(Span::styled(
                                    format!("   实际: {} [{}]  {}", name, s.type_label, s.primary),
                                    gray,
                                )));
                                if !s.secondary.is_empty() {
                                    l.push(Line::from(Span::styled(
                                        format!("         {}", s.secondary),
                                        gray,
                                    )));
                                }
                                if !s.tags.is_empty() {
                                    l.push(Line::from(Span::styled(
                                        format!("        标签: [{}]", s.tags.join(", ")),
                                        gray,
                                    )));
                                }
                            } else {
                                l.push(Line::from(Span::styled(
                                    format!("   实际: {}", name),
                                    gray,
                                )));
                            }
                        }
                        if let Some(eid) = expected_id {
                            let ename = rc
                                .graph_names
                                .as_ref()
                                .and_then(|m| m.get(eid))
                                .cloned()
                                .unwrap_or_default();
                            if let Some(s) = rc.id_names.as_ref().and_then(|m| m.get(eid)) {
                                l.push(Line::from(Span::styled(
                                    format!("   期望: {} [{}]  {}", ename, s.type_label, s.primary),
                                    gray,
                                )));
                                if !s.secondary.is_empty() {
                                    l.push(Line::from(Span::styled(
                                        format!("         {}", s.secondary),
                                        gray,
                                    )));
                                }
                                if !s.tags.is_empty() {
                                    l.push(Line::from(Span::styled(
                                        format!("        标签: [{}]", s.tags.join(", ")),
                                        gray,
                                    )));
                                }
                            } else {
                                l.push(Line::from(Span::styled(
                                    format!("   期望: {}", ename),
                                    gray,
                                )));
                            }
                        }
                    }
                }
                l
            } else {
                vec![Line::from(Span::raw(format!(
                    " 无详情数据 (outcome type: {})",
                    case.case_name
                )))]
            };

            let prev_sym = if self.drill_case > 0 { "◀" } else { " " };
            let next_sym = if self.drill_case + 1 < data.len() {
                "▶"
            } else {
                " "
            };
            let block = Block::bordered()
                .title(format!(
                    " {} {}/{} {} ",
                    prev_sym,
                    self.drill_case + 1,
                    data.len(),
                    next_sym
                ))
                .fg(Color::Cyan);
            let inner = block.inner(area);
            block.render(area, frame.buffer_mut());
            frame.render_widget(
                Paragraph::new(Text::from(lines))
                    .wrap(Wrap { trim: false })
                    .scroll((self.drill_scroll.offset as u16, 0)),
                inner,
            );
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => {
                if self.drill_dataset.is_some() {
                    self.drill_dataset = None;
                    self.drill_scroll.reset();
                    self.compare_cursor = 0;
                    self.expanded.clear_all();
                } else {
                    return Transition::ToMain;
                }
                Transition::None
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                if self.drill_dataset.is_some() {
                    self.drill_dataset = None;
                    self.drill_scroll.reset();
                    self.compare_cursor = 0;
                    self.expanded.clear_all();
                } else {
                    return Transition::ToMain;
                }
                Transition::None
            }
            // ←/→ switch between test cases in drill-down
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => {
                if self.drill_dataset.is_some() {
                    if self.drill_case > 0 {
                        self.drill_case -= 1;
                        self.compare_cursor = 0;
                        self.drill_scroll.reset();
                        self.expanded.clear_all();
                        self.sync_expanded_size();
                    }
                }
                Transition::None
            }
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') => {
                if self.drill_dataset.is_some() {
                    if let Some(ref result) = self.result {
                        let ds_idx = self.drill_dataset.unwrap();
                        if let Some(ds) = result.datasets.get(ds_idx) {
                            if self.drill_case + 1 < ds.outcomes.len() {
                                self.drill_case += 1;
                                self.compare_cursor = 0;
                                self.drill_scroll.reset();
                                self.expanded.clear_all();
                                self.sync_expanded_size();
                            }
                        }
                    }
                }
                Transition::None
            }
            // In drill-down: ↑↓ navigate comparison rows (retrieved vs expected);
            // outside drill-down: navigate dataset table
            KeyCode::Up => {
                let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                if self.drill_dataset.is_some() {
                    if self.compare_cursor > 0 {
                        self.compare_cursor -= 1;
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
                if self.drill_dataset.is_some() {
                    if let Some(ref result) = self.result {
                        let ds_idx = self.drill_dataset.unwrap();
                        if let Some(ds) = result.datasets.get(ds_idx) {
                            if let Some(data) = ds.outcomes.get(self.drill_case) {
                                if let Some(rc) = data.data.downcast_ref::<RetrieveCaseData>() {
                                    let n_max = rc
                                        .combined_retrieved_ids
                                        .len()
                                        .min(10)
                                        .max(rc.expected_combined_ranking.len().min(5));
                                    if self.compare_cursor + 1 < n_max {
                                        self.compare_cursor += 1;
                                        if shift {
                                            self.expanded.expand(self.compare_cursor);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(ref result) = self.result {
                    self.table_scroll.move_down(result.datasets.len());
                }
                Transition::None
            }
            // x: collapse all expanded rows in drill-down
            KeyCode::Char('x') | KeyCode::Char('X') if self.drill_dataset.is_some() => {
                self.expanded.clear_all();
                Transition::None
            }
            // Enter: enter drill-down from table, or toggle expand in drill-down
            KeyCode::Enter => {
                if let Some(ref result) = self.result {
                    if self.drill_dataset.is_some() {
                        // Toggle expanded row
                        self.expanded.toggle(self.compare_cursor);
                    } else {
                        let mut sorted: Vec<usize> = (0..result.datasets.len()).collect();
                        sorted.sort_by(|&a, &b| {
                            let da = &result.datasets[a];
                            let db = &result.datasets[b];
                            db.pass_rate
                                .partial_cmp(&da.pass_rate)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if let Some(&orig_idx) = sorted.get(self.table_scroll.cursor) {
                            let ds = &result.datasets[orig_idx];
                            if !ds.outcomes.is_empty() {
                                self.drill_dataset = Some(orig_idx);
                                self.drill_case = 0;
                                self.compare_cursor = 0;
                                self.drill_scroll.reset();
                                self.expanded.clear_all();
                                self.sync_expanded_size();
                            }
                        }
                    }
                }
                Transition::None
            }
            _ => Transition::None,
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if self.drill_dataset.is_some() {
            match mouse.kind {
                MouseEventKind::ScrollDown => {
                    self.drill_scroll.scroll_down();
                }
                MouseEventKind::ScrollUp => {
                    self.drill_scroll.scroll_up();
                }
                _ => {}
            }
        }
    }

    fn sync_expanded_size(&mut self) {
        if let Some(ref result) = self.result {
            if let Some(ds_idx) = self.drill_dataset {
                if let Some(ds) = result.datasets.get(ds_idx) {
                    if let Some(data) = ds.outcomes.get(self.drill_case) {
                        if let Some(rc) = data.data.downcast_ref::<RetrieveCaseData>() {
                            let n_max = rc
                                .combined_retrieved_ids
                                .len()
                                .min(10)
                                .max(rc.expected_combined_ranking.len().min(5));
                            self.expanded.resize(n_max);
                        }
                    }
                }
            }
        }
    }
}

impl Component for BatchRunState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => self.handle_key(key),
            ComponentEvent::Mouse(mouse) => {
                self.handle_mouse(mouse);
                Transition::None
            }
            ComponentEvent::Tick => self.tick().unwrap_or(Transition::None),
        }
    }
    fn view(&self, frame: &mut Frame) {
        self.render(frame);
    }
}
