use std::sync::{Arc, Mutex};
use std::time::Duration;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::Frame;

use std::collections::HashMap;

use crate::base::{AlgoType, TestConfig, TestReport, Transition};
use crate::component::{Component, ComponentEvent};
use crate::eval::retrieve_suite::RetrieveSuite;
use crate::eval::runner::{SuiteReport, TestCaseOutcome, TestSuite};
use crate::tui::components::status_bar;

type LoadResult = Result<(Box<dyn TestSuite>, usize, String), String>;

pub struct RunningState {
    pub config: TestConfig,
    pub total: usize,
    pub current: usize,
    pub passed: usize,
    pub failed: usize,
    pub elapsed_secs: f64,
    pub current_description: String,
    outcomes: Vec<TestCaseOutcome>,
    suite: Option<Box<dyn TestSuite>>,
    /// Shared loading state: None = still loading, Some(Ok(...)) = done, Some(Err(...)) = failed
    load_result: Arc<Mutex<Option<LoadResult>>>,
    #[allow(dead_code)]
    _load_thread: Option<std::thread::JoinHandle<()>>,
    spinner_frame: usize,
    loading_error: Option<String>,
}

impl RunningState {
    pub fn new(config: TestConfig) -> Self {
        let load_result: Arc<Mutex<Option<LoadResult>>> = Arc::new(Mutex::new(None));
        let load_result_clone = load_result.clone();
        let path = config.dataset_path.clone();
        let algo = config.algo;

        let params: HashMap<String, String> = config.params.clone();
        let _load_thread = std::thread::Builder::new()
            .name("suite-loader".into())
            .spawn(move || {
                let result: LoadResult =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> LoadResult {
                        match algo {
                            AlgoType::Retrieve(mode) => {
                                match RetrieveSuite::load_with_params(&path, mode, Some(&params)) {
                                    Ok(s) => {
                                        let n = s.case_count();
                                        let desc = format!("准备就绪，共 {} 个测试用例", n);
                                        Ok((Box::new(s) as Box<dyn TestSuite>, n, desc))
                                    }
                                    Err(e) => Err(format!(
                                        "加载 '{}' 失败: {}",
                                        path.file_name()
                                            .map(|n| n.to_string_lossy())
                                            .unwrap_or_default(),
                                        e
                                    )),
                                }
                            }
                            _ => Err(format!("{} 尚未实现", algo)),
                        }
                    }))
                    .unwrap_or_else(|_| Err("加载过程中发生内部错误 (panic)".to_string()));
                *load_result_clone.lock().unwrap() = Some(result);
            })
            .ok();

        Self {
            config,
            total: 0,
            current: 0,
            passed: 0,
            failed: 0,
            elapsed_secs: 0.0,
            current_description: "正在加载模型和数据...".to_string(),
            outcomes: Vec::new(),
            suite: None,
            load_result,
            _load_thread,
            spinner_frame: 0,
            loading_error: None,
        }
    }

    /// 从预加载的 suite 直接构造（跳过异步加载线程）
    pub fn new_loaded(
        config: TestConfig,
        suite: Box<dyn TestSuite>,
        total: usize,
        desc: String,
    ) -> Self {
        Self {
            config,
            total,
            current: 0,
            passed: 0,
            failed: 0,
            elapsed_secs: 0.0,
            current_description: desc,
            outcomes: Vec::new(),
            suite: Some(suite),
            load_result: Arc::new(Mutex::new(None)),
            _load_thread: None,
            spinner_frame: 0,
            loading_error: None,
        }
    }

    pub fn tick(&mut self) -> Option<Transition> {
        // Phase 1: waiting for async load to finish
        if self.suite.is_none() {
            self.spinner_frame = (self.spinner_frame + 1) % 10;
            if let Some(result) = self.load_result.lock().unwrap().take() {
                match result {
                    Ok((suite, total, desc)) => {
                        self.suite = Some(suite);
                        self.total = total;
                        self.current_description = desc;
                    }
                    Err(e) => {
                        self.current_description = e.clone();
                        self.loading_error = Some(e);
                        let suite: Box<dyn TestSuite> = Box::new(NoopSuite);
                        self.suite = Some(suite);
                        self.total = 0;
                    }
                }
            }
            return None;
        }

        // Phase 2: running test cases
        let suite = self.suite.as_ref().unwrap();
        if self.current >= self.total {
            let report = suite.build_report(
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
                error: self.loading_error.take(),
            }));
        }

        let start = std::time::Instant::now();
        let outcome = suite.run_case(self.current);
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

        if self.suite.is_none() {
            // Loading animation
            let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let s = spinner[self.spinner_frame % spinner.len()];
            frame.render_widget(
                Paragraph::new(format!("{} {} ", s, self.current_description))
                    .alignment(Alignment::Center)
                    .fg(Color::Cyan),
                content_area[0],
            );
            let note = Paragraph::new("正在加载 BGE 嵌入模型（首次运行需下载）...")
                .alignment(Alignment::Center)
                .fg(Color::DarkGray);
            frame.render_widget(note, content_area[1]);
        } else {
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
        }

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

    fn handle_tick(&mut self) -> Option<Transition> {
        self.tick()
    }
}

impl Component for RunningState {
    fn handle_event(&mut self, event: ComponentEvent) -> Transition {
        match event {
            ComponentEvent::Key(key) => self.handle_key(key),
            ComponentEvent::Tick => self.handle_tick().unwrap_or(Transition::None),
            _ => Transition::None,
        }
    }
    fn view(&self, frame: &mut Frame) {
        self.render(frame);
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
        outcomes: Vec<TestCaseOutcome>,
        _: Duration,
        _: usize,
        _: usize,
        _: usize,
    ) -> SuiteReport {
        SuiteReport {
            summary_groups: vec![],
            detail_header: String::new(),
            detail_rows: vec![],
            outcomes,
        }
    }
}
