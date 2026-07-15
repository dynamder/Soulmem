use std::path::PathBuf;
use std::time::Instant;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge, Paragraph, Wrap};
use ratatui::Frame;

use crate::base::{AlgoType, RetrieveMode, Transition};
use crate::eval::batch::{print_batch_result, run_batch, scan_question_jsons, BatchResult};
use crate::tui::components::status_bar;

pub struct BatchRunState {
    dir: PathBuf,
    mode: RetrieveMode,
    phase: BatchPhase,
    result: Option<BatchResult>,
}

enum BatchPhase {
    Scanning,
    Running(usize, usize), // current, total
    Done,
}

impl BatchRunState {
    pub fn new(dir: PathBuf, mode: RetrieveMode) -> Self {
        Self {
            dir,
            mode,
            phase: BatchPhase::Scanning,
            result: None,
        }
    }

    pub fn tick(&mut self) -> Option<Transition> {
        match &self.phase {
            BatchPhase::Scanning => {
                // Scan is fast, do it and start running
                let datasets = scan_question_jsons(&self.dir);
                if datasets.is_empty() {
                    self.phase = BatchPhase::Done;
                    self.result = Some(BatchResult {
                        datasets: vec![],
                        total_cases: 0,
                        total_passed: 0,
                        total_failed: 0,
                        elapsed: std::time::Duration::ZERO,
                    });
                } else {
                    let total = datasets.len();
                    let datasets = datasets.clone(); // clone once for the run
                    self.phase = BatchPhase::Running(0, total);

                    let mode = self.mode;
                    let result = run_batch(
                        &datasets,
                        mode,
                        Some(&|done, _| {
                            // Progress is handled via phase updates in the thread
                        }),
                    );
                    self.result = Some(result);
                    self.phase = BatchPhase::Done;
                }
                Some(Transition::None)
            }
            BatchPhase::Running(..) => {
                // Currently run_batch is synchronous, so this state is brief
                Some(Transition::None)
            }
            BatchPhase::Done => {
                // Stay until user presses a key to return to main
                Some(Transition::None)
            }
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
            .title(" 批量运行 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        match &self.phase {
            BatchPhase::Scanning => {
                frame.render_widget(
                    Paragraph::new("正在扫描目录...")
                        .alignment(Alignment::Center)
                        .fg(Color::Cyan),
                    layout[1],
                );
            }
            BatchPhase::Running(current, total) => {
                let ratio = if *total > 0 {
                    *current as f64 / *total as f64
                } else {
                    0.0
                };
                let gauge = Gauge::default().ratio(ratio).fg(Color::Cyan).label(format!(
                    "  {}/{} ({:.0}%)  ",
                    current,
                    total,
                    ratio * 100.0
                ));
                frame.render_widget(
                    gauge,
                    Layout::default()
                        .direction(Direction::Vertical)
                        .constraints(vec![Constraint::Fill(1)])
                        .split(layout[1])[0],
                );
            }
            BatchPhase::Done => {
                if let Some(ref result) = self.result {
                    let content = format_batch_summary(result);
                    frame.render_widget(
                        Paragraph::new(content).wrap(Wrap { trim: false }),
                        layout[1],
                    );
                } else {
                    frame.render_widget(
                        Paragraph::new("(无结果)").alignment(Alignment::Center),
                        layout[1],
                    );
                }
            }
        }

        status_bar::render_status_bar(frame, layout[2], &[("[Esc]".into(), "返回".into())]);
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => Transition::ToMain,
            _ => Transition::None,
        }
    }
}

fn format_batch_summary(result: &BatchResult) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "数据集数: {} | 总用例: {} | 通过: {} | 失败: {} | 总耗时: {:.2}s",
        result.datasets.len(),
        result.total_cases,
        result.total_passed,
        result.total_failed,
        result.elapsed.as_secs_f64(),
    ));
    lines.push(String::new());

    let mut sorted: Vec<_> = result.datasets.iter().collect();
    sorted.sort_by(|a, b| {
        b.pass_rate
            .partial_cmp(&a.pass_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    lines.push(format!(
        "{:<26} {:>5} {:>5} {:>5} {:>7}",
        "数据集", "用例", "通过", "失败", "通过率"
    ));
    lines.push("-".repeat(55));
    for ds in &sorted {
        let mark = if ds.error.is_some() {
            "!"
        } else if ds.pass_rate >= 80.0 {
            "+"
        } else {
            " "
        };
        lines.push(format!(
            "{:<26} {:>5} {:>5} {:>5} {:>6.1}%{}",
            ds.name.chars().take(24).collect::<String>(),
            ds.total,
            ds.passed,
            ds.failed,
            ds.pass_rate,
            mark,
        ));
        if let Some(ref err) = ds.error {
            lines.push(format!("   错误: {}", err));
        }
    }
    lines.join("\n")
}
