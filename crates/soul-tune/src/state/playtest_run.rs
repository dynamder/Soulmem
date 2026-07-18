use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::prelude::Widget;
use ratatui::style::{Color, Stylize};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::Frame;

use crate::base::Transition;
use crate::component::{Component, ComponentEvent};
use crate::eval::playtest::{DialogueFile, PlayTestResult, PlayTestRunner, PlayTurnResult};
use crate::tui::components::status_bar;
use soul_mem_runtime::working_memory::llm::client::LlmClient;
use soul_mem_runtime::working_memory::llm::config::LLMConfig;

enum RunPhase {
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
    dialogue: Arc<DialogueFile>,
    graph_dir: PathBuf,
    phase: RunPhase,
    runtime: Option<Arc<tokio::runtime::Runtime>>,
    load_error: Option<String>,
    spinner_frame: usize,
    current_description: String,
    load_result: Arc<Mutex<Option<Result<Arc<PlayTestRunner>, String>>>>,
    _load_thread: Option<std::thread::JoinHandle<()>>,
}

impl PlayTestRunState {
    pub fn new(dialogue: DialogueFile, path: PathBuf) -> Self {
        let graph_dir = path
            .parent()
            .map(|p| {
                p.join(&dialogue.graph_path)
                    .parent()
                    .unwrap_or(p)
                    .to_path_buf()
            })
            .unwrap_or_else(|| {
                let p = Path::new(&dialogue.graph_path);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    let mut base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
                    base.push(&dialogue.graph_path);
                    base
                }
            });

        let dialogue = Arc::new(dialogue);
        let dialogue_spawn = dialogue.clone();
        let graph_dir_clone = graph_dir.clone();
        let load_result: Arc<Mutex<Option<Result<Arc<PlayTestRunner>, String>>>> =
            Arc::new(Mutex::new(None));
        let load_result_clone = load_result.clone();

        let _load_thread = std::thread::Builder::new()
            .name("playtest-loader".into())
            .spawn(move || {
                let result = match PlayTestRunner::load(&graph_dir_clone) {
                    Ok(runner) => {
                        let runner = if let Some(ref config) = dialogue_spawn.config {
                            runner.with_config(config.clone())
                        } else {
                            runner
                        };
                        Ok(Arc::new(runner))
                    }
                    Err(e) => Err(format!("加载失败: {}", e)),
                };
                *load_result_clone.lock().unwrap() = Some(result);
            })
            .ok();

        Self {
            runner: None,
            dialogue,
            graph_dir,
            phase: RunPhase::Loading,
            runtime: None,
            load_error: None,
            spinner_frame: 0,
            current_description: "正在加载角色图和模型...".to_string(),
            load_result,
            _load_thread,
        }
    }

    fn tick(&mut self) -> Option<Transition> {
        match &self.phase {
            RunPhase::Loading => {
                self.spinner_frame = (self.spinner_frame + 1) % 10;
                if let Some(result) = self.load_result.lock().unwrap().take() {
                    match result {
                        Ok(runner) => {
                            let n = self.dialogue.conversations.len();
                            let rt = tokio::runtime::Runtime::new().ok();
                            self.runtime = rt.map(Arc::new);
                            self.runner = Some(runner);
                            self.current_description = format!("准备就绪，共 {} 轮对话", n);
                            self.phase = RunPhase::Processing {
                                total: n,
                                current: 0,
                                results: Vec::with_capacity(n),
                            };
                        }
                        Err(e) => {
                            self.load_error = Some(e.clone());
                            self.current_description = e;
                        }
                    }
                }
                None
            }
            RunPhase::Processing {
                total,
                current,
                results,
            } => {
                if *current >= *total {
                    let runner = self.runner.clone().unwrap();
                    let result = PlayTestResult {
                        character_name: runner.system_prompt.clone(),
                        config: runner.config.clone(),
                        turns: results.clone(),
                    };
                    self.phase = RunPhase::Done(result, runner);
                    None
                } else {
                    let entry = &self.dialogue.conversations[*current];
                    let runner = self.runner.clone().unwrap();
                    let runtime = self.runtime.clone().unwrap();

                    let llm = Self::create_llm();

                    let turn = runner.process_turn(entry, *current, &llm, &runtime);

                    let idx = *current;
                    let mut res = results.clone();
                    res.push(turn);

                    self.current_description = format!(
                        "轮次 {}/{}: {}",
                        idx + 1,
                        total,
                        entry.user_message.chars().take(40).collect::<String>()
                    );

                    self.phase = RunPhase::Processing {
                        total: *total,
                        current: idx + 1,
                        results: res,
                    };
                    None
                }
            }
            RunPhase::Done(_, _) => None,
        }
    }

    fn create_llm() -> LlmClient {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "not-needed".to_string());
        let api_base = std::env::var("OPENAI_API_BASE")
            .unwrap_or_else(|_| "http://localhost:8080/v1".to_string());
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "qwen3.5-4b".to_string());
        LlmClient::new(LLMConfig::new(&api_key, &api_base, &model))
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
            RunPhase::Loading => {
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
            }
            RunPhase::Processing {
                total,
                current,
                results,
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

                let info = vec![
                    format!("当前: {}", self.current_description),
                    format!("完成: {} / {} 轮", current, total),
                ]
                .join("\n");
                frame.render_widget(
                    Paragraph::new(info).alignment(Alignment::Center),
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

        status_bar::render_status_bar(
            frame,
            layout[2],
            &[
                ("[Enter]".into(), "开始评分".into()),
                ("[Ctrl+C]".into(), "退出".into()),
            ],
        );
    }

    fn handle_key(&mut self, key: KeyEvent) -> Transition {
        match key.code {
            KeyCode::Esc => Transition::ToMain,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Transition::ToMain
            }
            KeyCode::Enter => {
                if let RunPhase::Done(result, _) = &self.phase {
                    Transition::ToPlayTestJudge(result.clone())
                } else {
                    Transition::None
                }
            }
            _ => Transition::None,
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
