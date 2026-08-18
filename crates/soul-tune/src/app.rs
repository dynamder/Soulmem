use std::path::PathBuf;

use ratatui::crossterm::execute;
use ratatui::DefaultTerminal;

use crate::base::{AlgoType, ForgetMode, RetrieveMode, Transition};
use crate::cmd::CmdRegistry;
use crate::engine::playtest::DialogueFile;
use crate::states::batch::running::BatchRunState;
use crate::states::batch::BatchModeState;
use crate::states::command_palette::CommandState;
use crate::states::compare_mode::SelectAlgoState;
use crate::states::compare_results::CompareResultsState;
use crate::states::dataset_browser::DatasetState;
use crate::states::inspect::InspectState;
use crate::states::main_menu::MainState;
use crate::states::params_config::ParamState;
use crate::states::playtest::input::PlayTestInputState;
use crate::states::playtest::judge::PlayTestJudgeState;
use crate::states::playtest::run_state::PlayTestRunState;
use crate::states::forget_observer::ForgetObserverState;
use crate::states::results::ResultsState;
use crate::states::retrieve_mode::RetrieveModeSelectState;
use crate::states::running::RunningState;

pub mod event_loop;

pub enum AppState {
    Main(MainState),
    CommandMode(CommandState),
    SelectDataset(DatasetState),
    RetrieveModeSelect(RetrieveModeSelectState),
    SelectAlgo(SelectAlgoState),
    ConfigParams(ParamState),
    TestRunning(RunningState),
    TestResults(ResultsState),
    /// 遗忘测试观测页（Forget 算法跑完后进入，逐节点观测遗忘结果）
    ForgetObserver(ForgetObserverState),
    CompareResults(CompareResultsState),
    PlayTestSelect(DatasetState),
    PlayTestInput(PlayTestInputState),
    PlayTestRun(PlayTestRunState),
    PlayTestJudge(PlayTestJudgeState),
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
}

impl App {
    pub fn new() -> color_eyre::Result<Self> {
        let terminal = ratatui::init();
        let _ = execute!(
            std::io::stdout(),
            ratatui::crossterm::event::EnableMouseCapture
        );
        let mut cmd_registry = CmdRegistry::new();

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
                        "compare" => AlgoType::Compare,
                        "consolidate" | "c" => AlgoType::Consolidate,
                        // TUI 命令面板默认跑全管线（三阶段独立测试走 headless CLI）
                        "forget" | "f" => AlgoType::Forget(ForgetMode::Pipeline),
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
        })
    }

    pub fn apply(&mut self, transition: Transition) -> bool {
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
            Transition::ToRetrieveModeSelect => {
                self.app_state = AppState::RetrieveModeSelect(RetrieveModeSelectState::new());
                false
            }
            Transition::ToSelectAlgo => {
                self.app_state = AppState::SelectAlgo(SelectAlgoState::new());
                false
            }
            Transition::ToSelectCompareDataset => {
                self.app_state = AppState::SelectDataset(DatasetState::new_compare());
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
            Transition::ToBatchConfigParams(algo, dir) => {
                self.app_state = AppState::ConfigParams(ParamState::new(algo, dir, true));
                false
            }
            Transition::ToBatchRun(dir, mode, params) => {
                self.app_state = AppState::BatchRunning(BatchRunState::new(dir, mode, params));
                false
            }
            Transition::ToBatchCompareRun(dir, params) => {
                self.app_state = AppState::BatchRunning(BatchRunState::new_compare(dir, params));
                false
            }
            Transition::ToPlayTestInput => {
                self.app_state = AppState::PlayTestInput(PlayTestInputState::new());
                false
            }
            Transition::ToGraphBrowse => {
                let prev = std::mem::replace(
                    &mut self.app_state,
                    AppState::PlayTestSelect({
                        let mut ds = DatasetState::new(AlgoType::PlayTest);
                        ds.graph_pick_mode = true;
                        ds
                    }),
                );
                self.saved_state = Some(prev);
                false
            }
            Transition::ToGraphSelected(dir) => {
                if let Some(saved) = self.saved_state.take() {
                    if let AppState::PlayTestInput(state) = saved {
                        self.app_state = AppState::PlayTestInput(state.with_graph_dir(dir));
                    } else {
                        self.app_state = saved;
                    }
                } else {
                    self.app_state = AppState::Main(MainState);
                }
                false
            }
            Transition::ToPlayTestManualRun(dialogue) => {
                self.app_state =
                    AppState::PlayTestRun(PlayTestRunState::new(dialogue, PathBuf::from(".")));
                false
            }
            Transition::ToPlayTestSelect => {
                self.app_state = AppState::PlayTestSelect(DatasetState::new(AlgoType::PlayTest));
                false
            }
            Transition::ToPlayTestJudge(result) => {
                self.app_state = AppState::PlayTestJudge(PlayTestJudgeState::new(result));
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
            Transition::ToConfigParams(algo, path) => match algo {
                AlgoType::PlayTest => {
                    match std::fs::read_to_string(&path).and_then(|s| {
                        serde_json::from_str::<DialogueFile>(&s).map_err(|e| e.into())
                    }) {
                        Ok(dialogue) => {
                            self.app_state =
                                AppState::PlayTestRun(PlayTestRunState::new(dialogue, path));
                        }
                        Err(_e) => {
                            self.app_state = AppState::Main(MainState);
                        }
                    }
                    false
                }
                _ => {
                    self.app_state = AppState::ConfigParams(ParamState::new(algo, path, false));
                    false
                }
            },
            Transition::ToTestRunning(config) => {
                self.app_state = AppState::TestRunning(RunningState::new(config));
                false
            }
            Transition::ToTestResults(report) => {
                // 遗忘算法跑完后进入专门的观测页（逐节点看原文/遮罩/LLM 回复），
                // 其余算法走通用结果页
                if matches!(report.config.algo, AlgoType::Forget(_)) {
                    self.app_state = AppState::ForgetObserver(ForgetObserverState::new(report));
                } else {
                    self.app_state = AppState::TestResults(ResultsState::new(report));
                }
                false
            }
            Transition::ToCompareResults(report) => {
                self.app_state = AppState::CompareResults(CompareResultsState::new(report));
                false
            }
            Transition::Quit => true,
        }
    }
}
