use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyEvent, KeyEventKind, MouseEvent};
use ratatui::crossterm::execute;
use ratatui::{DefaultTerminal, Frame};

use crate::base::{AlgoType, RetrieveMode, Transition};
use crate::cmd::CmdRegistry;
use crate::component::{Component, ComponentEvent};
use crate::metric::MetricRegistry;
use crate::reporter::ReporterRegistry;
use crate::state::batch_mode::BatchModeState;
use crate::state::batch_run::BatchRunState;
use crate::state::command::CommandState;
use crate::state::dataset::DatasetState;
use crate::state::inspect::InspectState;
use crate::state::main::MainState;
use crate::state::params::ParamState;
use crate::state::results::ResultsState;
use crate::state::running::RunningState;

pub enum AppState {
    Main(MainState),
    CommandMode(CommandState),
    SelectDataset(DatasetState),
    ConfigParams(ParamState),
    TestRunning(RunningState),
    TestResults(ResultsState),
    SelectBatchDir(DatasetState),
    BatchModeSelect(BatchModeState),
    BatchRunning(BatchRunState),
    InspectData(InspectState),
}

pub struct App {
    terminal: DefaultTerminal,
    app_state: AppState,
    saved_state: Option<AppState>,
    cmd_registry: CmdRegistry,
    #[allow(dead_code)]
    metric_registry: MetricRegistry,
    #[allow(dead_code)]
    reporter_registry: ReporterRegistry,
}

impl App {
    pub fn new() -> color_eyre::Result<Self> {
        let terminal = ratatui::init();
        let _ = execute!(
            std::io::stdout(),
            ratatui::crossterm::event::EnableMouseCapture
        );
        let mut cmd_registry = CmdRegistry::new();

        // Register built-in commands
        use crate::base::SoulTuneEvent;
        use crate::cmd::UserCmdBuilder;
        cmd_registry.register(
            UserCmdBuilder::new("test")
                .aliases(["t"])
                .description("运行算法测试: test <retrieve|consolidate|forget>")
                .usage("test <algo>")
                .handler(|args| {
                    if args.is_empty() {
                        return None;
                    }
                    let algo = match args[0].as_str() {
                        "retrieve" | "r" => AlgoType::Retrieve(RetrieveMode::FullPipeline),
                        "retrieve/embedding" | "re" => AlgoType::Retrieve(RetrieveMode::Embedding),
                        "retrieve/association" | "ra" => {
                            AlgoType::Retrieve(RetrieveMode::Association)
                        }
                        "retrieve/full" | "rf" => AlgoType::Retrieve(RetrieveMode::FullPipeline),
                        "consolidate" | "c" => AlgoType::Consolidate,
                        "forget" | "f" => AlgoType::Forget,
                        _ => return None,
                    };
                    Some(SoulTuneEvent::StartTest(algo, None))
                })
                .build(),
        );
        cmd_registry.register(
            UserCmdBuilder::new("inspect")
                .aliases(["i"])
                .description("直接检视测试数据集: inspect <path>")
                .usage("inspect <path>")
                .handler(|_| None)
                .build(),
        );
        cmd_registry.register(
            UserCmdBuilder::new("help")
                .aliases(["h"])
                .description("显示帮助信息")
                .usage("help")
                .handler(|_| None)
                .build(),
        );
        cmd_registry.register(
            UserCmdBuilder::new("quit")
                .aliases(["q"])
                .description("退出程序")
                .usage("quit")
                .handler(|_| Some(SoulTuneEvent::Quit))
                .build(),
        );

        Ok(Self {
            terminal,
            app_state: AppState::Main(MainState),
            saved_state: None,
            cmd_registry,
            metric_registry: MetricRegistry::new(),
            reporter_registry: ReporterRegistry::new(),
        })
    }

    pub fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            let state = &self.app_state;
            self.terminal
                .draw(|frame| App::render_state(frame, state))?;

            self.tick_running();

            if event::poll(Duration::from_millis(40))? {
                match event::read()? {
                    Event::Key(key) => {
                        if self.handle_key(key) {
                            break;
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse(mouse);
                    }
                    _ => {}
                }
            }
        }
        let _ = execute!(
            std::io::stdout(),
            ratatui::crossterm::event::DisableMouseCapture
        );
        ratatui::restore();
        Ok(())
    }

    fn tick_running(&mut self) {
        let t = match &mut self.app_state {
            AppState::TestRunning(s) => s.handle_event(ComponentEvent::Tick),
            AppState::BatchRunning(s) => s.handle_event(ComponentEvent::Tick),
            _ => Transition::None,
        };
        self.apply(t);
    }

    fn render_state(frame: &mut Frame, state: &AppState) {
        match state {
            AppState::Main(s) => Component::view(s, frame),
            AppState::CommandMode(s) => s.render(frame),
            AppState::SelectDataset(s) => s.view(frame),
            AppState::SelectBatchDir(s) => s.view(frame),
            AppState::ConfigParams(s) => s.view(frame),
            AppState::TestRunning(s) => s.view(frame),
            AppState::TestResults(s) => s.view(frame),
            AppState::BatchModeSelect(s) => s.view(frame),
            AppState::BatchRunning(s) => s.view(frame),
            AppState::InspectData(s) => s.view(frame),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        let transition = match &mut self.app_state {
            AppState::Main(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::TestResults(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::BatchModeSelect(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::CommandMode(s) => s.handle_key(key, &self.cmd_registry),
            AppState::SelectDataset(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::SelectBatchDir(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::ConfigParams(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::TestRunning(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::BatchRunning(s) => s.handle_event(ComponentEvent::Key(key)),
            AppState::InspectData(s) => s.handle_event(ComponentEvent::Key(key)),
        };
        self.apply(transition)
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match &mut self.app_state {
            AppState::TestResults(s) => {
                s.handle_event(ComponentEvent::Mouse(mouse));
            }
            AppState::BatchRunning(s) => {
                s.handle_event(ComponentEvent::Mouse(mouse));
            }
            _ => {}
        }
    }

    fn apply(&mut self, transition: Transition) -> bool {
        match transition {
            Transition::None => false,
            Transition::ToMain => {
                if let Some(saved) = self.saved_state.take() {
                    self.app_state = saved;
                } else {
                    self.app_state = AppState::Main(MainState);
                }
                false
            }
            Transition::ToCommand(prefill) => {
                let mut cmd = CommandState::new();
                cmd.input.insert_str(&prefill);
                cmd.update_suggestions(&self.cmd_registry);
                self.app_state = AppState::CommandMode(cmd);
                false
            }
            Transition::ToSelectDataset(algo) => {
                self.app_state = AppState::SelectDataset(DatasetState::new(algo));
                false
            }
            Transition::ToSelectBatchDir => {
                self.app_state = AppState::SelectBatchDir(DatasetState::new_batch());
                false
            }
            Transition::ToBatchModeSelect(dir) => {
                self.app_state = AppState::BatchModeSelect(BatchModeState::new(dir));
                false
            }
            Transition::ToBatchRun(dir, mode) => {
                self.app_state = AppState::BatchRunning(BatchRunState::new(dir, mode));
                false
            }
            Transition::ToInspect(path) => {
                let prev = std::mem::replace(
                    &mut self.app_state,
                    AppState::InspectData(InspectState::new(path)),
                );
                self.saved_state = Some(prev);
                false
            }
            Transition::ToConfigParams(algo, path) => {
                self.app_state = AppState::ConfigParams(ParamState::new(algo, path));
                false
            }
            Transition::ToTestRunning(config) => {
                self.app_state = AppState::TestRunning(RunningState::new(config));
                false
            }
            Transition::ToTestResults(report) => {
                self.app_state = AppState::TestResults(ResultsState::new(report));
                false
            }
            Transition::Quit => true,
        }
    }
}
