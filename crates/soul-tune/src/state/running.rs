use std::time::Duration;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::Frame;

use crate::base::{AlgoType, TestConfig, TestReport, Transition};
use crate::eval::retrieve_suite::RetrieveSuite;
use crate::eval::runner::{SuiteReport, TestCaseOutcome, TestSuite};
use crate::tui::components::status_bar;

pub struct RunningState {
    pub config: TestConfig,
    pub total: usize,
    pub current: usize,
    pub passed: usize,
    pub failed: usize,
    pub elapsed_secs: f64,
    pub current_description: String,
    outcomes: Vec<TestCaseOutcome>,
    suite: Box<dyn TestSuite>,
}

impl RunningState {
    pub fn new(config: TestConfig) -> Self {
        let (suite, total, desc) = match config.algo {
            AlgoType::Retrieve => match RetrieveSuite::load(&config.dataset_path) {
                Ok(s) => {
                    let n = s.case_count();
                    (
                        Box::new(s) as Box<dyn TestSuite>,
                        n,
                        format!("准备就绪，共 {} 个测试用例", n),
                    )
                }
                Err(e) => {
                    let msg = format!("加载失败: {}", e);
                    let suite: Box<dyn TestSuite> = Box::new(NoopSuite);
                    (suite, 0, msg)
                }
            },
            AlgoType::Consolidate | AlgoType::Forget => {
                let suite: Box<dyn TestSuite> = Box::new(NoopSuite);
                (suite, 0, format!("{} 尚未实现", config.algo))
            }
        };

        Self {
            config,
            total,
            current: 0,
            passed: 0,
            failed: 0,
            elapsed_secs: 0.0,
            current_description: desc,
            outcomes: Vec::new(),
            suite,
        }
    }

    pub fn tick(&mut self) -> Option<Transition> {
        if self.current >= self.total {
            let report = self.suite.build_report(
                std::mem::take(&mut self.outcomes),
                std::time::Duration::from_secs_f64(self.elapsed_secs),
                self.total,
                self.passed,
                self.failed,
            );
            return Some(Transition::ToTestResults(TestReport {
                config: self.config.clone(),
                total: self.total,
                passed: self.passed,
                failed: self.failed,
                elapsed: std::time::Duration::from_secs_f64(self.elapsed_secs),
                suite_report: report,
            }));
        }

        let start = std::time::Instant::now();
        let outcome = self.suite.run_case(self.current);
        self.current_description = format!("执行: {}", outcome.case_name);

        if outcome.passed {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
        self.outcomes.push(outcome);
        self.current += 1;
        self.elapsed_secs += start.elapsed().as_secs_f64();

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
                self.passed, self.failed, self.elapsed_secs
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

/// 退化的空套件，当算法尚未实现时使用
struct NoopSuite;

impl TestSuite for NoopSuite {
    fn case_count(&self) -> usize {
        0
    }
    fn run_case(&self, _: usize) -> TestCaseOutcome {
        TestCaseOutcome {
            case_name: "?".into(),
            description: String::new(),
            passed: false,
            data: Box::new(()),
        }
    }
    fn build_report(
        &self,
        _: Vec<TestCaseOutcome>,
        _: Duration,
        _: usize,
        _: usize,
        _: usize,
    ) -> SuiteReport {
        SuiteReport {
            summary_groups: vec![],
            detail_header: String::new(),
            detail_rows: vec![],
        }
    }
}
