use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyEvent, KeyEventKind};
use ratatui::{DefaultTerminal, Frame};

use crate::base::{AlgoType, RetrieveMode, Transition};
use crate::cmd::CmdRegistry;
use crate::metric::MetricRegistry;
use crate::reporter::ReporterRegistry;
use crate::state::batch_mode::BatchModeState;
use crate::state::batch_run::BatchRunState;
use crate::state::command::CommandState;
use crate::state::dataset::DatasetState;
use crate::state::main::MainState;
use crate::state::params::ParamState;
use crate::state::results::ResultsState;
use crate::state::running::RunningState;

pub enum AppState {
    Main,
    CommandMode(CommandState),
    SelectDataset(DatasetState),
    ConfigParams(ParamState),
    TestRunning(RunningState),
    TestResults(ResultsState),
    SelectBatchDir(DatasetState),
    BatchModeSelect(BatchModeState),
    BatchRunning(BatchRunState),
}

pub struct App {
    terminal: DefaultTerminal,
    app_state: AppState,
    cmd_registry: CmdRegistry,
    #[allow(dead_code)]
    metric_registry: MetricRegistry,
    #[allow(dead_code)]
    reporter_registry: ReporterRegistry,
}

impl App {
    pub fn new() -> color_eyre::Result<Self> {
        let terminal = ratatui::init();
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
                        "retrieve" | "r" => AlgoType::Retrieve(RetrieveMode::Embedding),
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
            app_state: AppState::Main,
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
                    _ => {}
                }
            }
        }
        ratatui::restore();
        Ok(())
    }

    fn tick_running(&mut self) {
        if let AppState::TestRunning(state) = &mut self.app_state {
            if let Some(transition) = state.tick() {
                self.apply(transition);
            }
        }
    }

    fn render_state(frame: &mut Frame, state: &AppState) {
        match state {
            AppState::Main => MainState::render(frame),
            AppState::CommandMode(s) => s.render(frame),
            AppState::SelectDataset(s) => s.render(frame),
            AppState::SelectBatchDir(s) => s.render(frame),
            AppState::ConfigParams(s) => s.render(frame),
            AppState::TestRunning(s) => s.render(frame),
            AppState::TestResults(s) => s.render(frame),
            AppState::BatchModeSelect(s) => s.render(frame),
            AppState::BatchRunning(s) => s.render(frame),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        let cmd_registry = &self.cmd_registry;
        let transition = match &mut self.app_state {
            AppState::Main => MainState::handle_key(key),
            AppState::CommandMode(s) => s.handle_key(key, cmd_registry),
            AppState::SelectDataset(s) => s.handle_key(key),
            AppState::SelectBatchDir(s) => s.handle_key(key),
            AppState::ConfigParams(s) => s.handle_key(key),
            AppState::TestRunning(s) => s.handle_key(key),
            AppState::TestResults(s) => s.handle_key(key),
            AppState::BatchModeSelect(s) => s.handle_key(key),
            AppState::BatchRunning(s) => s.handle_key(key),
        };
        self.apply(transition)
    }

    fn apply(&mut self, transition: Transition) -> bool {
        match transition {
            Transition::None => false,
            Transition::ToMain => {
                self.app_state = AppState::Main;
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
