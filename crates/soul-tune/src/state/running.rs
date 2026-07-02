use std::time::Duration;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::Frame;

use crate::base::{TestConfig, TestReport, Transition};
use crate::tui::components::status_bar;

pub struct RunningState {
    pub config: TestConfig,
    pub total: usize,
    pub current: usize,
    pub passed: usize,
    pub failed: usize,
    pub elapsed: Duration,
    pub current_description: String,
}

impl RunningState {
    pub fn new(config: TestConfig) -> Self {
        Self {
            total: 100,
            current: 0,
            passed: 0,
            failed: 0,
            elapsed: Duration::ZERO,
            current_description: "准备中...".into(),
            config,
        }
    }

    pub fn tick(&mut self) -> Option<Transition> {
        if self.current >= self.total {
            return Some(Transition::ToTestResults(TestReport {
                config: self.config.clone(),
                total: self.total,
                passed: self.passed,
                failed: self.failed,
                elapsed: self.elapsed,
            }));
        }

        let step = 5;
        let new_current = (self.current + step).min(self.total);
        for i in self.current..new_current {
            if i % 7 == 0 {
                self.failed += 1;
            } else {
                self.passed += 1;
            }
        }
        self.current = new_current;
        self.elapsed += Duration::from_millis(50);
        self.current_description = format!("模拟运行中... 第 {}/{} 条", self.current, self.total);
        None
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
            .title(format!(
                " ▶ 运行中 · {} · {} ",
                self.config.algo,
                self.config
                    .dataset_path
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ))
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let content_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Fill(1),
            ])
            .split(layout[1]);

        let ratio = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        };
        let gauge = Gauge::default().ratio(ratio).fg(Color::Cyan).label(format!(
            "  {}/{} ({:.0}%)  ",
            self.current,
            self.total,
            ratio * 100.0
        ));
        frame.render_widget(gauge, content_area[0]);

        let info = vec![
            format!("当前: {}", self.current_description),
            format!(
                "通过: {}    失败: {}    耗时: {:.1}s",
                self.passed,
                self.failed,
                self.elapsed.as_secs_f64()
            ),
        ]
        .join("\n");
        frame.render_widget(
            Paragraph::new(info).alignment(Alignment::Center),
            content_area[1],
        );

        status_bar::render_status_bar(frame, layout[2], &[("[Ctrl+C]".into(), "中止".into())]);
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Transition::ToMain
            }
            _ => Transition::None,
        }
    }
}
