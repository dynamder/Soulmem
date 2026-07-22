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

use std::collections::HashMap;

use soul_mem_core::memory_note::MemoryId;

use crate::base::{RetrieveMode, Transition};
use crate::component::{Component, ComponentEvent};
use crate::engine::batch::{scan_question_jsons, BatchResult, DatasetResult};
use crate::engine::retrieve::batch::{
    process_one_compare_dataset, process_one_dataset, BatchCompareResult, CompareDatasetResult,
};
use crate::engine::retrieve::data::RetrieveCaseData;
use crate::widgets::expandable::ExpandableList;
use crate::widgets::scroll::ScrollState;
use crate::widgets::status_bar;

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

struct CmpWorkerSlot {
    active: bool,
    current_name: String,
    status_line: String,
    progress: f64,
    elapsed: f64,
}

struct CmpBatchProgress {
    done: usize,
    total: usize,
    workers: Vec<CmpWorkerSlot>,
    results: Vec<Option<CompareDatasetResult>>,
}

enum RunMode {
    Normal {
        dir: PathBuf,
        mode: RetrieveMode,
        params: HashMap<String, String>,
    },
    Compare {
        dir: PathBuf,
        params: HashMap<String, String>,
    },
}

#[derive(PartialEq)]
enum BatchPhase {
    Scanning,
    Running,
    Done,
}

pub struct BatchRunState {
    mode: RunMode,
    phase: BatchPhase,
    result: Option<BatchResult>,
    compare_result: Option<BatchCompareResult>,
    // Normal mode progress
    normal_progress: Option<Arc<Mutex<BatchProgress>>>,
    normal_rx: Option<mpsc::Receiver<(usize, DatasetResult)>>,
    // Compare mode progress
    cmp_progress: Option<Arc<Mutex<CmpBatchProgress>>>,
    cmp_rx: Option<mpsc::Receiver<(usize, CompareDatasetResult)>>,
    // UI state
    table_scroll: ScrollState,
    drill_dataset: Option<usize>,
    drill_case: usize,
    compare_cursor: usize,
    drill_scroll: ScrollState,
    expanded: ExpandableList,
}

impl BatchRunState {
    pub fn new(dir: PathBuf, mode: RetrieveMode, params: HashMap<String, String>) -> Self {
        Self {
            mode: RunMode::Normal { dir, mode, params },
            phase: BatchPhase::Scanning,
            result: None,
            compare_result: None,
            normal_progress: None,
            normal_rx: None,
            cmp_progress: None,
            cmp_rx: None,
            table_scroll: ScrollState::new(),
            drill_dataset: None,
            drill_case: 0,
            compare_cursor: 0,
            drill_scroll: ScrollState::new(),
            expanded: ExpandableList::new(0),
        }
    }

    pub fn new_compare(dir: PathBuf, params: HashMap<String, String>) -> Self {
        Self {
            mode: RunMode::Compare { dir, params },
            phase: BatchPhase::Scanning,
            result: None,
            compare_result: None,
            normal_progress: None,
            normal_rx: None,
            cmp_progress: None,
            cmp_rx: None,
            table_scroll: ScrollState::new(),
            drill_dataset: None,
            drill_case: 0,
            compare_cursor: 0,
            drill_scroll: ScrollState::new(),
            expanded: ExpandableList::new(0),
        }
    }

    fn is_compare(&self) -> bool {
        matches!(self.mode, RunMode::Compare { .. })
    }

    fn dir(&self) -> &PathBuf {
        match &self.mode {
            RunMode::Normal { dir, .. } | RunMode::Compare { dir, .. } => dir,
        }
    }

    fn title(&self) -> &str {
        if self.is_compare() {
            " 批量比对 "
        } else {
            " 批量运行 "
        }
    }

    pub fn tick(&mut self) -> Option<Transition> {
        match self.phase {
            BatchPhase::Scanning => self.tick_scanning(),
            BatchPhase::Running => self.tick_running(),
            BatchPhase::Done => None,
        }
    }

    fn tick_scanning(&mut self) -> Option<Transition> {
        let datasets = scan_question_jsons(self.dir());
        let total = datasets.len();
        if total == 0 {
            self.phase = BatchPhase::Done;
            return None;
        }
        let n_workers = 4.min(total).max(1);
        let counter = Arc::new(AtomicUsize::new(0));
        let datasets = datasets.clone();

        if self.is_compare() {
            let params = match &self.mode {
                RunMode::Compare { params, .. } => params.clone(),
                _ => HashMap::default(),
            };
            let params_ref = Arc::new(params);
            let workers = (0..n_workers)
                .map(|_| CmpWorkerSlot {
                    active: false,
                    current_name: String::new(),
                    status_line: "空闲".into(),
                    progress: 0.0,
                    elapsed: 0.0,
                })
                .collect();
            let progress = Arc::new(Mutex::new(CmpBatchProgress {
                done: 0,
                total,
                workers,
                results: (0..total).map(|_| None).collect(),
            }));
            self.cmp_progress = Some(progress.clone());
            let (tx, rx) = mpsc::channel();
            self.cmp_rx = Some(rx);

            for slot_idx in 0..n_workers {
                let datasets = datasets.clone();
                let counter = Arc::clone(&counter);
                let tx = tx.clone();
                let progress = progress.clone();
                let params_ref = Arc::clone(&params_ref);
                std::thread::Builder::new()
                    .name(format!("cmp-w{}", slot_idx))
                    .spawn(move || loop {
                        let i = counter.fetch_add(1, Ordering::Relaxed);
                        if i >= datasets.len() {
                            break;
                        }
                        let name = datasets[i]
                            .parent()
                            .and_then(|p| p.file_name())
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "?".into());
                        let ds_start = Instant::now();
                        let p2 = progress.clone();
                        let n2 = name.clone();
                        let ds = process_one_compare_dataset(
                            &datasets[i],
                            Some(&params_ref),
                            ds_start,
                            |pct, status| {
                                if let Ok(mut g) = p2.lock() {
                                    if slot_idx < g.workers.len() {
                                        g.workers[slot_idx].current_name = n2.clone();
                                        g.workers[slot_idx].status_line = status.to_string();
                                        g.workers[slot_idx].progress = pct;
                                        g.workers[slot_idx].elapsed =
                                            ds_start.elapsed().as_secs_f64();
                                    }
                                }
                            },
                            |_| {},
                        );
                        let _ = tx.send((i, ds));
                    })
                    .ok();
            }
            drop(tx);
        } else {
            let mode = match &self.mode {
                RunMode::Normal { mode, .. } => *mode,
                _ => unreachable!(),
            };
            let params = match &self.mode {
                RunMode::Normal { params, .. } => params.clone(),
                _ => HashMap::default(),
            };
            let params_arc = Arc::new(params);
            let workers = (0..n_workers)
                .map(|_| WorkerSlot {
                    active: false,
                    current_name: String::new(),
                    status_line: "空闲".into(),
                    progress: 0.0,
                    elapsed: 0.0,
                })
                .collect();
            let progress = Arc::new(Mutex::new(BatchProgress {
                done: 0,
                total,
                workers,
                results: (0..total).map(|_| None).collect(),
            }));
            self.normal_progress = Some(progress.clone());
            let (tx, rx) = mpsc::channel();
            self.normal_rx = Some(rx);

            for slot_idx in 0..n_workers {
                let datasets = datasets.clone();
                let mode = mode;
                let counter = Arc::clone(&counter);
                let tx = tx.clone();
                let progress = progress.clone();
                let params_arc = Arc::clone(&params_arc);
                std::thread::Builder::new()
                    .name(format!("batch-w{}", slot_idx))
                    .spawn(move || loop {
                        let i = counter.fetch_add(1, Ordering::Relaxed);
                        if i >= datasets.len() {
                            break;
                        }
                        let name = datasets[i]
                            .parent()
                            .and_then(|p| p.file_name())
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "?".into());
                        let ds_start = Instant::now();
                        let p2 = progress.clone();
                        let n2 = name.clone();
                        let ds = process_one_dataset(
                            &datasets[i],
                            mode,
                            Some(&params_arc),
                            ds_start,
                            |pct, status| {
                                if let Ok(mut g) = p2.lock() {
                                    if slot_idx < g.workers.len() {
                                        g.workers[slot_idx].current_name = n2.clone();
                                        g.workers[slot_idx].status_line = status.to_string();
                                        g.workers[slot_idx].progress = pct;
                                        g.workers[slot_idx].elapsed =
                                            ds_start.elapsed().as_secs_f64();
                                    }
                                }
                            },
                            |_| {},
                        );
                        let _ = tx.send((i, ds));
                    })
                    .ok();
            }
            drop(tx);
        }

        self.phase = BatchPhase::Running;
        None
    }

    fn tick_running(&mut self) -> Option<Transition> {
        if self.is_compare() {
            if let Some(ref rx) = self.cmp_rx {
                let mut received = 0;
                while let Ok((idx, ds)) = rx.try_recv() {
                    if let Ok(mut p) = self.cmp_progress.as_ref().unwrap().lock() {
                        p.results[idx] = Some(ds);
                        p.done += 1;
                    }
                    received += 1;
                    if received > 10 {
                        break;
                    }
                }
            }
            let done = self
                .cmp_progress
                .as_ref()
                .map(|p| p.lock().map(|g| g.done).unwrap_or(0))
                .unwrap_or(0);
            let total = self
                .cmp_progress
                .as_ref()
                .map(|p| p.lock().map(|g| g.total).unwrap_or(1))
                .unwrap_or(1);
            if done >= total {
                let mut all_results = Vec::new();
                if let Ok(mut p) = self.cmp_progress.as_ref().unwrap().lock() {
                    for opt in p.results.drain(..) {
                        if let Some(ds) = opt {
                            all_results.push(ds);
                        }
                    }
                }
                let n = all_results.len();
                let avg_e = if n > 0 {
                    all_results.iter().map(|d| d.avg_emb_hit).sum::<f64>() / n as f64
                } else {
                    0.0
                };
                let avg_f = if n > 0 {
                    all_results.iter().map(|d| d.avg_full_hit).sum::<f64>() / n as f64
                } else {
                    0.0
                };
                let avg_em = if n > 0 {
                    all_results.iter().map(|d| d.avg_emb_mrr).sum::<f64>() / n as f64
                } else {
                    0.0
                };
                let avg_fm = if n > 0 {
                    all_results.iter().map(|d| d.avg_full_mrr).sum::<f64>() / n as f64
                } else {
                    0.0
                };
                self.compare_result = Some(BatchCompareResult {
                    datasets: all_results,
                    total_datasets: n,
                    avg_emb_hit: avg_e,
                    avg_full_hit: avg_f,
                    hit_delta: avg_f - avg_e,
                    avg_emb_mrr: avg_em,
                    avg_full_mrr: avg_fm,
                    mrr_delta: avg_fm - avg_em,
                    elapsed: std::time::Duration::ZERO,
                });
                self.expanded.resize(n);
                self.phase = BatchPhase::Done;
            }
        } else {
            if let Some(ref rx) = self.normal_rx {
                let mut received = 0;
                while let Ok((idx, ds)) = rx.try_recv() {
                    if let Ok(mut p) = self.normal_progress.as_ref().unwrap().lock() {
                        p.results[idx] = Some(ds);
                        p.done += 1;
                    }
                    received += 1;
                    if received > 10 {
                        break;
                    }
                }
            }
            let done = self
                .normal_progress
                .as_ref()
                .map(|p| p.lock().map(|g| g.done).unwrap_or(0))
                .unwrap_or(0);
            let total = self
                .normal_progress
                .as_ref()
                .map(|p| p.lock().map(|g| g.total).unwrap_or(1))
                .unwrap_or(1);
            if done >= total {
                let mut all_results = Vec::new();
                let mut tc = 0;
                let mut tp = 0;
                let mut tf = 0;
                if let Ok(mut p) = self.normal_progress.as_ref().unwrap().lock() {
                    for opt in p.results.drain(..) {
                        if let Some(ds) = opt {
                            tc += ds.total;
                            tp += ds.passed;
                            tf += ds.failed;
                            all_results.push(ds);
                        }
                    }
                }
                self.result = Some(BatchResult {
                    datasets: all_results,
                    total_cases: tc,
                    total_passed: tp,
                    total_failed: tf,
                    elapsed: std::time::Duration::ZERO,
                });
                self.phase = BatchPhase::Done;
            }
        }
        None
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let area = if self.phase == BatchPhase::Running {
            let n_workers = if self.is_compare() {
                self.cmp_progress
                    .as_ref()
                    .map(|p| p.lock().map(|g| g.workers.len()).unwrap_or(0))
                    .unwrap_or(0)
            } else {
                self.normal_progress
                    .as_ref()
                    .map(|p| p.lock().map(|g| g.workers.len()).unwrap_or(0))
                    .unwrap_or(0)
            };
            if n_workers > 0 {
                let mut cs = vec![Constraint::Length(3)];
                for _ in 0..n_workers {
                    cs.push(Constraint::Length(3));
                }
                cs.push(Constraint::Fill(1));
                cs.push(Constraint::Length(1));
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(cs)
                    .split(area)
            } else {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(vec![
                        Constraint::Length(3),
                        Constraint::Fill(1),
                        Constraint::Length(1),
                    ])
                    .split(area)
            }
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Length(3),
                    Constraint::Fill(1),
                    Constraint::Length(1),
                ])
                .split(area)
        };

        Block::bordered()
            .title(self.title())
            .fg(Color::Cyan)
            .render(area[0], frame.buffer_mut());

        match self.phase {
            BatchPhase::Scanning => {
                frame.render_widget(Paragraph::new("正在扫描目录..."), area[1]);
            }
            BatchPhase::Running => {
                let n_workers = if self.is_compare() {
                    self.cmp_progress
                        .as_ref()
                        .map(|p| p.lock().map(|g| g.workers.len()).unwrap_or(0))
                        .unwrap_or(0)
                } else {
                    self.normal_progress
                        .as_ref()
                        .map(|p| p.lock().map(|g| g.workers.len()).unwrap_or(0))
                        .unwrap_or(0)
                };
                for (i, slot_layout) in area[1..1 + n_workers].iter().enumerate() {
                    let (name, status, progress, elapsed) = if self.is_compare() {
                        self.cmp_progress
                            .as_ref()
                            .and_then(|p| p.lock().ok())
                            .and_then(|g| {
                                g.workers.get(i).map(|w| {
                                    (
                                        w.current_name.clone(),
                                        w.status_line.clone(),
                                        w.progress,
                                        w.elapsed,
                                    )
                                })
                            })
                            .unwrap_or_default()
                    } else {
                        self.normal_progress
                            .as_ref()
                            .and_then(|p| p.lock().ok())
                            .and_then(|g| {
                                g.workers.get(i).map(|w| {
                                    (
                                        w.current_name.clone(),
                                        w.status_line.clone(),
                                        w.progress,
                                        w.elapsed,
                                    )
                                })
                            })
                            .unwrap_or_default()
                    };
                    let gauge = Gauge::default()
                        .ratio(progress)
                        .fg(Color::Cyan)
                        .label(format!(
                            "  {:<12} {:>5.1}s  ",
                            name.chars().take(10).collect::<String>(),
                            elapsed
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
                if self.is_compare() {
                    if let Some(ref result) = self.compare_result {
                        self.render_compare_table(frame, area[1], result);
                    } else {
                        frame.render_widget(
                            Paragraph::new("(无结果)").alignment(Alignment::Center),
                            area[1],
                        );
                    }
                } else if let Some(idx) = self.drill_dataset {
                    self.render_drilldown(frame, area[1], idx);
                } else if let Some(ref result) = self.result {
                    self.render_dataset_table(frame, area[1], result);
                } else {
                    frame.render_widget(
                        Paragraph::new("(无结果)").alignment(Alignment::Center),
                        area[1],
                    );
                }
            }
        }

        let si = area.len().saturating_sub(1);
        let is_done = self.phase == BatchPhase::Done;
        if is_done && self.is_compare() {
            status_bar::render_status_bar(
                frame,
                area[si],
                &[
                    ("[↑↓]".into(), "选择".into()),
                    ("[Q/Esc]".into(), "返回".into()),
                ],
            );
        } else if is_done && self.drill_dataset.is_some() {
            let hint = self
                .result
                .as_ref()
                .and_then(|r| {
                    let idx = self.drill_dataset?;
                    r.datasets
                        .get(idx)
                        .map(|ds| format!("用例 {}/{}", self.drill_case + 1, ds.outcomes.len()))
                })
                .unwrap_or_default();
            status_bar::render_status_bar(
                frame,
                area[si],
                &[
                    ("[←→]".into(), hint.into()),
                    ("[↑↓]".into(), "滚屏".into()),
                    ("[Enter]".into(), "展开".into()),
                    ("[Q]".into(), "返回".into()),
                ],
            );
        } else if is_done {
            status_bar::render_status_bar(
                frame,
                area[si],
                &[
                    ("[↑↓]".into(), "选择".into()),
                    ("[Enter]".into(), "查看详情".into()),
                    ("[Esc]".into(), "返回".into()),
                ],
            );
        } else {
            status_bar::render_status_bar(frame, area[si], &[("[Esc]".into(), "返回".into())]);
        }
    }

    fn render_compare_table(&self, frame: &mut Frame, area: Rect, result: &BatchCompareResult) {
        let cw: [Constraint; 9] = [
            Constraint::Length(1),  // selector
            Constraint::Length(22), // dataset name
            Constraint::Length(5),  // case count
            Constraint::Length(8),  // Emb_Hit
            Constraint::Length(8),  // Full_Hit
            Constraint::Length(7),  // delta_Hit
            Constraint::Length(10), // Emb_MRR
            Constraint::Length(10), // Full_MRR
            Constraint::Fill(1),    // delta_MRR
        ];
        let hs = Style::new().fg(Color::Cyan).bold();
        let gray = Style::new().dark_gray();
        let yellow = Style::new().yellow();

        let block = Block::bordered().title(" 批量比对结果 ");
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());
        let (cr, bar_rect) = ScrollState::split_area(inner);

        // ── Summary row ──
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(
                    " 综合: Hit {:.2}→{:.2} (Δ{:.2})  MRR {:.4}→{:.4} (Δ{:.4})  数据集{}个",
                    result.avg_emb_hit,
                    result.avg_full_hit,
                    result.hit_delta,
                    result.avg_emb_mrr,
                    result.avg_full_mrr,
                    result.mrr_delta,
                    result.total_datasets,
                ),
                yellow,
            ))),
            Rect::new(cr.x, cr.y, cr.width, 1),
        );

        // ── Header row ──
        let hr = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(cw)
            .split(Rect::new(cr.x, cr.y + 1, cr.width, 1));
        frame.render_widget(Paragraph::new("").style(hs), hr[0]);
        frame.render_widget(Paragraph::new("数据集").style(hs), hr[1]);
        frame.render_widget(Paragraph::new("用例").style(hs), hr[2]);
        frame.render_widget(Paragraph::new("EmbHit").style(hs), hr[3]);
        frame.render_widget(Paragraph::new("FullHit").style(hs), hr[4]);
        frame.render_widget(Paragraph::new("ΔHit").style(hs), hr[5]);
        frame.render_widget(Paragraph::new("EmbMRR").style(hs), hr[6]);
        frame.render_widget(Paragraph::new("FullMRR").style(hs), hr[7]);
        frame.render_widget(Paragraph::new("ΔMRR").style(hs), hr[8]);

        // ── Data rows ──
        let data_top = cr.y + 2;
        let visible = cr.height.saturating_sub(2);
        let so = ScrollState::offset(visible, result.datasets.len(), self.table_scroll.cursor);

        let mut y = data_top;
        let mut skipped = 0usize;
        for (i, ds) in result.datasets.iter().enumerate() {
            let is_expanded = self.expanded.is_expanded(i);

            if skipped < so {
                skipped += 1;
                if is_expanded {
                    skipped += 3;
                }
                continue;
            }

            if y >= cr.y + cr.height {
                break;
            }

            let is_cursor = i == self.table_scroll.cursor;
            let (bg, fg) = if is_cursor {
                (Color::Cyan, Color::Black)
            } else {
                (Color::Reset, Color::Reset)
            };

            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(cw)
                .split(Rect::new(cr.x, y, cr.width, 1));

            let sym = if is_cursor { "▶" } else { " " };
            let name = truncate_to(&ds.name, 22);
            let base = Style::default().fg(fg).bg(bg);
            frame.render_widget(Paragraph::new(sym).style(base), cols[0]);
            frame.render_widget(Paragraph::new(name).style(base), cols[1]);
            frame.render_widget(
                Paragraph::new(format!("{}", ds.case_count)).style(base),
                cols[2],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.2}", ds.avg_emb_hit)).style(base),
                cols[3],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.2}", ds.avg_full_hit)).style(base),
                cols[4],
            );
            frame.render_widget(
                Paragraph::new(delta_str(ds.hit_delta)).style(if ds.hit_delta > 0.0 {
                    Style::new().green().bg(bg)
                } else if ds.hit_delta < 0.0 {
                    Style::new().red().bg(bg)
                } else {
                    base
                }),
                cols[5],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.4}", ds.avg_emb_mrr)).style(base),
                cols[6],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.4}", ds.avg_full_mrr)).style(base),
                cols[7],
            );
            frame.render_widget(
                Paragraph::new(delta_str(ds.mrr_delta)).style(if ds.mrr_delta > 0.0 {
                    Style::new().green().bg(bg)
                } else if ds.mrr_delta < 0.0 {
                    Style::new().red().bg(bg)
                } else {
                    base
                }),
                cols[8],
            );

            y += 1;

            if is_expanded {
                let detail_lines = [
                    format!(
                        "Emb通过: {}/{}  Full通过: {}/{}",
                        ds.emb_passed, ds.case_count, ds.full_passed, ds.case_count
                    ),
                    format!(
                        "Hit: {:.2} → {:.2} (Δ{})",
                        ds.avg_emb_hit,
                        ds.avg_full_hit,
                        delta_str(ds.hit_delta)
                    ),
                    format!(
                        "MRR: {:.4} → {:.4} (Δ{})",
                        ds.avg_emb_mrr,
                        ds.avg_full_mrr,
                        delta_str(ds.mrr_delta)
                    ),
                ];
                for line in &detail_lines {
                    if y >= cr.y + cr.height {
                        break;
                    }
                    frame.render_widget(
                        Paragraph::new(line.clone()).style(gray),
                        Rect::new(cr.x + 2, y, cr.width - 2, 1),
                    );
                    y += 1;
                }
            }
        }

        let total_expanded = (0..result.datasets.len())
            .filter(|i| self.expanded.is_expanded(*i))
            .count();
        let total_lines = result.datasets.len() + total_expanded * 3;
        let expanded_before = (0..so).filter(|i| self.expanded.is_expanded(*i)).count();
        let scroll_ylines = so + expanded_before * 3;
        ScrollState::render_scrollbar(frame, bar_rect, total_lines, visible, scroll_ylines);
    }

    fn render_dataset_table(&self, frame: &mut Frame, area: Rect, result: &BatchResult) {
        let block = Block::bordered().title(" 批量结果 ");
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());
        let mut sorted: Vec<usize> = (0..result.datasets.len()).collect();
        sorted.sort_by(|&a, &b| {
            result.datasets[b]
                .pass_rate
                .partial_cmp(&result.datasets[a].pass_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (cr, br) = ScrollState::split_area(inner);
        let so = ScrollState::offset(
            cr.height.saturating_sub(1),
            result.datasets.len(),
            self.table_scroll.cursor,
        );
        let cw: [Constraint; 7] = [
            Constraint::Length(1),
            Constraint::Length(22),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Fill(1),
        ];
        let hr = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(cw)
            .split(Rect::new(inner.x, inner.y, inner.width, 1));
        let hs = Style::new().fg(Color::Cyan).bold();
        frame.render_widget(Paragraph::new("数据集").style(hs), hr[1]);
        frame.render_widget(Paragraph::new("用例").style(hs), hr[2]);
        frame.render_widget(Paragraph::new("通过").style(hs), hr[3]);
        frame.render_widget(Paragraph::new("失败").style(hs), hr[4]);
        frame.render_widget(Paragraph::new("通过率").style(hs), hr[5]);
        frame.render_widget(Paragraph::new("耗时").style(hs), hr[6]);
        for (di, &oi) in sorted.iter().enumerate().skip(so) {
            let y = inner.y + 1 + (di - so) as u16;
            if y >= inner.y + inner.height {
                break;
            }
            let ds = &result.datasets[oi];
            let act = di == self.table_scroll.cursor;
            let fg = if ds.error.is_some() {
                Color::Red
            } else if ds.pass_rate >= 80.0 {
                Color::Green
            } else {
                Color::Yellow
            };
            let st = if act {
                Style::default().fg(Color::Black).bg(Color::Cyan)
            } else {
                Style::default().fg(fg)
            };
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(cw)
                .split(Rect::new(cr.x, y, cr.width, 1));
            frame.render_widget(
                Paragraph::new(if act { "▶" } else { " " }).style(st),
                cols[0],
            );
            frame.render_widget(Paragraph::new(ds.name.as_str()).style(st), cols[1]);
            frame.render_widget(Paragraph::new(format!("{}", ds.total)).style(st), cols[2]);
            frame.render_widget(Paragraph::new(format!("{}", ds.passed)).style(st), cols[3]);
            frame.render_widget(Paragraph::new(format!("{}", ds.failed)).style(st), cols[4]);
            frame.render_widget(
                Paragraph::new(format!("{:.1}%", ds.pass_rate)).style(st),
                cols[5],
            );
            frame.render_widget(
                Paragraph::new(format!("{:.1}s", ds.elapsed.as_secs_f64())).style(st),
                cols[6],
            );
        }
        ScrollState::render_scrollbar(
            frame,
            br,
            result.datasets.len(),
            cr.height.saturating_sub(1),
            so,
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
                let (mut l, hdr, green, red, gray) = (
                    Vec::new(),
                    Style::new().yellow().bold(),
                    Style::new().green(),
                    Style::new().red(),
                    Style::new().dark_gray(),
                );
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
                let rs: std::collections::HashSet<&MemoryId> =
                    rc.combined_retrieved_ids.iter().take(10).collect();
                let nm = rc
                    .combined_retrieved_ids
                    .len()
                    .min(10)
                    .max(rc.expected_combined_ranking.len().min(5));
                for pos in 0..nm {
                    let is_cursor = pos == self.compare_cursor;
                    let (mut spans, prefix) = (Vec::new(), if is_cursor { "▶" } else { " " });
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
                        if !rs.contains(eid) {
                            spans.push(Span::styled(" ✗未命中", red));
                        }
                    } else {
                        spans.push(Span::raw(format!(" {:<10}", "—")));
                    }
                    l.push(Line::from(spans).style(if is_cursor {
                        Style::default().bg(Color::DarkGray)
                    } else {
                        Style::default()
                    }));
                    if self.expanded.is_expanded(pos) {
                        if let Some(id) = rc.combined_retrieved_ids.get(pos) {
                            l.push(Line::from(Span::styled(format!("   实际: {:?}", id), gray)));
                        }
                        if let Some(eid) = rc.expected_combined_ranking.get(pos) {
                            l.push(Line::from(Span::styled(
                                format!("   期望: {:?}", eid),
                                gray,
                            )));
                        }
                    }
                }
                l
            } else {
                vec![Line::from(Span::raw(format!(
                    " 无详情数据: {}",
                    case.case_name
                )))]
            };
            let (ps, ns) = (
                if self.drill_case > 0 { "◀" } else { " " },
                if self.drill_case + 1 < data.len() {
                    "▶"
                } else {
                    " "
                },
            );
            let block = Block::bordered()
                .title(format!(
                    " {} {}/{} {} ",
                    ps,
                    self.drill_case + 1,
                    data.len(),
                    ns
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
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
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
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => {
                if self.drill_dataset.is_some() && self.drill_case > 0 {
                    self.drill_case -= 1;
                    self.compare_cursor = 0;
                    self.drill_scroll.reset();
                    self.expanded.clear_all();
                    self.sync_expanded_size();
                }
                Transition::None
            }
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') => {
                if self.drill_dataset.is_some() {
                    if let Some(ref result) = self.result {
                        let idx = self.drill_dataset.unwrap();
                        if let Some(ds) = result.datasets.get(idx) {
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
                        if let Some(ds) = result.datasets.get(self.drill_dataset.unwrap()) {
                            if let Some(data) = ds.outcomes.get(self.drill_case) {
                                if let Some(rc) = data.data.downcast_ref::<RetrieveCaseData>() {
                                    let nm = rc
                                        .combined_retrieved_ids
                                        .len()
                                        .min(10)
                                        .max(rc.expected_combined_ranking.len().min(5));
                                    if self.compare_cursor + 1 < nm {
                                        self.compare_cursor += 1;
                                        if shift {
                                            self.expanded.expand(self.compare_cursor);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if self.is_compare() {
                    if let Some(ref result) = self.compare_result {
                        self.table_scroll.move_down(result.datasets.len() + 3);
                    }
                } else if let Some(ref result) = self.result {
                    self.table_scroll.move_down(result.datasets.len());
                }
                Transition::None
            }
            KeyCode::Char('x') | KeyCode::Char('X') if self.drill_dataset.is_some() => {
                self.expanded.clear_all();
                Transition::None
            }
            KeyCode::Enter => {
                if self.drill_dataset.is_some() {
                    self.expanded.toggle(self.compare_cursor);
                } else if self.is_compare() {
                    self.expanded.toggle(self.table_scroll.cursor);
                } else {
                    if let Some(ref result) = self.result {
                        let mut sorted: Vec<usize> = (0..result.datasets.len()).collect();
                        sorted.sort_by(|&a, &b| {
                            result.datasets[b]
                                .pass_rate
                                .partial_cmp(&result.datasets[a].pass_rate)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if let Some(&oi) = sorted.get(self.table_scroll.cursor) {
                            if !result.datasets[oi].outcomes.is_empty() {
                                self.drill_dataset = Some(oi);
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
                MouseEventKind::ScrollDown => self.drill_scroll.scroll_down(),
                MouseEventKind::ScrollUp => self.drill_scroll.scroll_up(),
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
                            self.expanded.resize(
                                rc.combined_retrieved_ids
                                    .len()
                                    .min(10)
                                    .max(rc.expected_combined_ranking.len().min(5)),
                            );
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

/// 安全截断字符串到指定字符宽度（处理多字节 Unicode）
fn truncate_to(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > max_chars {
        format!(
            "{}..",
            chars[..max_chars.saturating_sub(2)]
                .iter()
                .collect::<String>()
        )
    } else {
        s.to_string()
    }
}

/// 左对齐填充到指定字符宽度（处理多字节 Unicode）
fn pad_right(s: &str, width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= width {
        chars.iter().take(width).collect()
    } else {
        let mut result = s.to_string();
        result.push_str(&" ".repeat(width - chars.len()));
        result
    }
}

/// 格式化 delta 差值字符串
fn delta_str(delta: f64) -> String {
    if delta > 0.0 {
        format!("+{:.2}", delta)
    } else if delta < 0.0 {
        format!("{:.2}", delta)
    } else {
        "-".to_string()
    }
}
