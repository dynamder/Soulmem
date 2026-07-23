use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::Frame;
use ratatui_textarea::TextArea;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::engine::llm::LlamaServer;
use crate::engine::playtest::{DialogueFile, PlayTestResult, PlayTestRunner, PlayTurnResult};
use crate::widgets::status_bar;

enum WorkerMsg {
    Loading(String),
    LoadError(String),
    TurnComplete {
        turn: PlayTurnResult,
        total_turns: usize,
    },
    AllDone {
        llm: LlamaServer,
        runner: Arc<PlayTestRunner>,
    },
}

enum RunPhase {
    RoleInput,
    Loading,
    Processing {
        total: usize,
        current: usize,
        results: Vec<PlayTurnResult>,
    },
    Done(PlayTestResult, Arc<PlayTestRunner>),
}

pub struct PlayTestRunState {
    runner: Option<Arc<PlayTestRunner>>,
    llm: Option<LlamaServer>,
    dialogue: Arc<DialogueFile>,
    phase: RunPhase,
    load_error: Option<String>,
    spinner_frame: usize,
    current_description: String,
    worker_rx: Option<mpsc::Receiver<WorkerMsg>>,
    worker_thread: Option<std::thread::JoinHandle<()>>,
    human_role_input: TextArea<'static>,
    graph_dir: PathBuf,
    _path: PathBuf,
}

impl PlayTestRunState {
    pub fn new(dialogue: DialogueFile, _path: PathBuf) -> Self {
        let graph_dir = {
            let gp = Path::new(&dialogue.graph_path);
            if gp.is_absolute() {
                gp.parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| gp.to_path_buf())
            } else {
                _path
                    .parent()
                    .map(|p| {
                        p.join(&dialogue.graph_path)
                            .parent()
                            .unwrap_or(p)
                            .to_path_buf()
                    })
                    .unwrap_or_else(|| {
                        let mut base = _path.parent().unwrap_or(Path::new(".")).to_path_buf();
                        base.push(&dialogue.graph_path);
                        base
                    })
            }
        };

        let mut textarea = TextArea::default();
        textarea.set_placeholder_text("我是同事小李");

        Self {
            runner: None,
            llm: None,
            dialogue: Arc::new(dialogue),
            phase: RunPhase::RoleInput,
            load_error: None,
            spinner_frame: 0,
            current_description: String::new(),
            worker_rx: None,
            worker_thread: None,
            human_role_input: textarea,
            graph_dir,
            _path,
        }
    }

    fn spawn_load_thread(&mut self, human_role: Option<String>) {
        let dialogue = self.dialogue.clone();
        let graph_dir = self.graph_dir.clone();
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);

        let thread = std::thread::Builder::new()
            .name("playtest-worker".into())
            .spawn({
                let tx = tx.clone();
                move || {
                    let _ = tx.send(WorkerMsg::Loading("加载角色图...".into()));

                    let runner = match PlayTestRunner::load(&graph_dir) {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = tx.send(WorkerMsg::LoadError(format!("加载图失败: {}", e)));
                            return;
                        }
                    };
                    let runner = if let Some(ref config) = dialogue.config {
                        runner.with_config(config.clone())
                    } else {
                        runner
                    };
                    let runner = runner.with_human_role(human_role);
                    let runner = Arc::new(runner);

                    let _ = tx.send(WorkerMsg::Loading("启动 LLM 模型...".into()));

                    let model_path = match std::env::var("SOUL_TUNE_CANDLE_MODEL_PATH") {
                        Ok(p) => p,
                        Err(_) => {
                            let _ = tx.send(WorkerMsg::LoadError(
                                "环境变量 SOUL_TUNE_CANDLE_MODEL_PATH 未设置".into(),
                            ));
                            return;
                        }
                    };
                    let mut llm = match LlamaServer::load(&model_path) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ =
                                tx.send(WorkerMsg::LoadError(format!("启动 LLM 服务失败: {}", e)));
                            return;
                        }
                    };

                    let total = dialogue.conversations.len();
                    for (i, entry) in dialogue.conversations.iter().enumerate() {
                        let turn = runner.process_turn(entry, i, &mut llm);
                        let _ = tx.send(WorkerMsg::TurnComplete {
                            turn,
                            total_turns: total,
                        });
                    }
                    let _ = tx.send(WorkerMsg::AllDone { llm, runner });
                }
            })
            .ok();

        self.worker_thread = thread;
        self.phase = RunPhase::Loading;
        self.current_description = "正在加载角色图和 LLM 模型...".to_string();
    }

    fn tick(&mut self) -> Option<Transition> {
        let phase = std::mem::replace(
            &mut self.phase,
            RunPhase::Processing {
                total: 0,
                current: 0,
                results: vec![],
            },
        );

        let next_phase = match phase {
            RunPhase::RoleInput => phase,
            RunPhase::Loading => self.tick_loading(),
            RunPhase::Processing {
                total,
                current,
                results,
            } => self.tick_processing(total, current, results),
            RunPhase::Done(..) => phase,
        };

        self.phase = next_phase;
        None
    }

    fn tick_loading(&mut self) -> RunPhase {
        self.spinner_frame = (self.spinner_frame + 1) % 10;

        if let Some(ref rx) = self.worker_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::Loading(desc) => {
                        self.current_description = desc;
                    }
                    WorkerMsg::LoadError(msg) => {
                        self.load_error = Some(msg.clone());
                        self.current_description = msg;
                    }
                    WorkerMsg::TurnComplete { turn, total_turns } => {
                        self.current_description = format!("轮次 1/{}: 处理中...", total_turns);
                        return RunPhase::Processing {
                            total: total_turns,
                            current: 1,
                            results: vec![turn],
                        };
                    }
                    WorkerMsg::AllDone { llm, runner } => {
                        let result = PlayTestResult {
                            character_name: runner.system_prompt.clone(),
                            config: runner.config.clone(),
                            turns: vec![],
                            human_role: runner.human_role.clone(),
                        };
                        self.llm = Some(llm);
                        self.runner = Some(runner.clone());
                        return RunPhase::Done(result, runner);
                    }
                }
            }
        }

        RunPhase::Loading
    }

    fn tick_processing(
        &mut self,
        total: usize,
        _current: usize,
        mut results: Vec<PlayTurnResult>,
    ) -> RunPhase {
        if let Some(ref rx) = self.worker_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::TurnComplete { turn, .. } => {
                        results.push(turn);
                        let n = results.len();
                        self.current_description = format!("轮次 {}/{}: 处理中...", n, total);
                    }
                    WorkerMsg::AllDone { llm, runner } => {
                        self.llm = Some(llm);
                        self.runner = Some(runner.clone());
                        let turns = std::mem::take(&mut results);
                        let result = PlayTestResult {
                            character_name: runner.system_prompt.clone(),
                            config: runner.config.clone(),
                            turns,
                            human_role: runner.human_role.clone(),
                        };
                        return RunPhase::Done(result, runner);
                    }
                    _ => {}
                }
            }
        }

        let current = results.len();
        RunPhase::Processing {
            total,
            current,
            results,
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
            .title(" ▶ 角色扮演测试 ")
            .fg(Color::Cyan)
            .render(layout[0], frame.buffer_mut());

        let content_area = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Length(3), Constraint::Fill(1)])
            .split(layout[1]);

        match &self.phase {
            RunPhase::RoleInput => {
                let hints = Paragraph::new("请输入你的角色（告诉 AI 你是谁，留空则跳过）:")
                    .alignment(Alignment::Center)
                    .fg(Color::Yellow);
                frame.render_widget(hints, content_area[0]);
                let input_rect = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(20),
                        Constraint::Percentage(60),
                        Constraint::Percentage(20),
                    ])
                    .split(content_area[1])[1];
                frame.render_widget(&self.human_role_input, input_rect);
            }
            RunPhase::Loading => {
                let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let s = spinner[self.spinner_frame % spinner.len()];
                frame.render_widget(
                    Paragraph::new(format!("{} {} ", s, self.current_description))
                        .alignment(Alignment::Center)
                        .fg(Color::Cyan),
                    content_area[0],
                );
                let note = if self.load_error.is_some() {
                    Paragraph::new(self.load_error.as_deref().unwrap_or(""))
                        .alignment(Alignment::Center)
                        .fg(Color::Red)
                } else {
                    Paragraph::new("正在加载 BGE 嵌入模型和 Qwen3 GGUF 模型...")
                        .alignment(Alignment::Center)
                        .fg(Color::DarkGray)
                };
                frame.render_widget(note, content_area[1]);
            }
            RunPhase::Processing {
                total,
                current,
                results: _,
            } => {
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
                frame.render_widget(gauge, content_area[0]);
                frame.render_widget(
                    Paragraph::new(self.current_description.clone()).alignment(Alignment::Center),
                    content_area[1],
                );
            }
            RunPhase::Done(result, _runner) => {
                let summary = format!("测试完成！共 {} 轮对话\n请进行人工评分", result.turns.len());
                frame.render_widget(
                    Paragraph::new(summary)
                        .alignment(Alignment::Center)
                        .fg(Color::Green),
                    content_area[0],
                );
            }
        }

        let hints = match &self.phase {
            RunPhase::RoleInput => {
                vec![
                    ("[Enter]".into(), "确认".into()),
                    ("[Esc]".into(), "返回".into()),
                ]
            }
            _ => {
                vec![
                    ("[Enter]".into(), "开始评分".into()),
                    ("[Ctrl+C]".into(), "退出".into()),
                ]
            }
        };
        status_bar::render_status_bar(frame, layout[2], &hints);
    }

    fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Transition::ToMain
            }
            KeyCode::Enter => match &self.phase {
                RunPhase::RoleInput => {
                    let role = self.human_role_input.lines().join("\n");
                    let role = if role.trim().is_empty() {
                        None
                    } else {
                        Some(role.trim().to_string())
                    };
                    self.spawn_load_thread(role);
                    Transition::None
                }
                RunPhase::Done(result, _) => Transition::ToPlayTestJudge(result.clone()),
                _ => Transition::None,
            },
            _ => {
                if matches!(self.phase, RunPhase::RoleInput) {
                    self.human_role_input.input(key);
                }
                Transition::None
            }
        }
    }

    fn handle_tick(&mut self) -> Option<Transition> {
        self.tick()
    }
}

impl Component for PlayTestRunState {
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
