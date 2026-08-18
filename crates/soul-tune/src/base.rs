use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use ratatui::crossterm;

use crate::engine::compare::CompareReport;
use crate::engine::playtest::{DialogueFile, PlayTestResult};
use crate::engine::suite::SuiteReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrieveMode {
    Embedding,
    Association,
    FullPipeline,
}

impl std::fmt::Display for RetrieveMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrieveMode::Embedding => write!(f, "embedding"),
            RetrieveMode::Association => write!(f, "association"),
            RetrieveMode::FullPipeline => write!(f, "full"),
        }
    }
}

/// 遗忘算法测试模式：三阶段独立验证
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgetMode {
    /// 阶段 1：只验证遮罩（纯算法、无 LLM、确定性）
    Mask,
    /// 阶段 2：只验证遮罩补全（直接驱动 llama-server，贴 LLM 原始回复）
    Revise,
    /// 阶段 3：全管线（衰减 → 遮罩 → LLM 补全 → 边衰减）
    Pipeline,
}

impl std::fmt::Display for ForgetMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForgetMode::Mask => write!(f, "mask"),
            ForgetMode::Revise => write!(f, "revise"),
            ForgetMode::Pipeline => write!(f, "full"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AlgoType {
    Retrieve(RetrieveMode),
    Compare,
    PlayTest,
    Consolidate,
    Forget(ForgetMode),
}

impl std::fmt::Display for AlgoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgoType::Retrieve(mode) => write!(f, "retrieve/{}", mode),
            AlgoType::Compare => write!(f, "compare"),
            AlgoType::PlayTest => write!(f, "playtest"),
            AlgoType::Consolidate => write!(f, "consolidate"),
            AlgoType::Forget(mode) => write!(f, "forget/{}", mode),
        }
    }
}

#[derive(Clone)]
pub struct TestConfig {
    pub algo: AlgoType,
    pub dataset_path: PathBuf,
    pub params: HashMap<String, String>,
}

pub struct TestReport {
    pub config: TestConfig,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub elapsed: Duration,
    pub suite_report: SuiteReport,
    pub error: Option<String>,
}

pub enum Transition {
    None,
    ToMain,
    ToCommand(String),
    ToSelectDataset(AlgoType),
    ToRetrieveModeSelect,
    ToForgetModeSelect,
    ToSelectAlgo,
    ToSelectCompareDataset,
    ToConfigParams(AlgoType, PathBuf),
    ToTestRunning(TestConfig),
    ToTestResults(TestReport),
    ToCompareResults(CompareReport),
    ToPlayTestInput,
    ToGraphBrowse,
    ToGraphSelected(std::path::PathBuf),
    ToPlayTestManualRun(DialogueFile),
    ToPlayTestSelect,
    ToPlayTestJudge(PlayTestResult),
    ToSelectBatchDir,
    ToBatchModeSelect(PathBuf),
    ToBatchConfigParams(AlgoType, PathBuf),
    ToBatchCompareRun(PathBuf, HashMap<String, String>),
    ToBatchRun(PathBuf, RetrieveMode, HashMap<String, String>),
    ToInspect(PathBuf),
    Quit,
}

pub enum SoulTuneEvent {
    CrossTerm(crossterm::event::Event),
    StartTest(AlgoType, Option<PathBuf>),
    TestComplete,
    Quit,
}

impl From<crossterm::event::Event> for SoulTuneEvent {
    fn from(event: crossterm::event::Event) -> Self {
        Self::CrossTerm(event)
    }
}

#[allow(dead_code)]
pub trait EventHandler {
    fn handle_event(&mut self, event: SoulTuneEvent) -> Option<SoulTuneEvent>;
}
