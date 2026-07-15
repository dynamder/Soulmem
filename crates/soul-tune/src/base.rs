use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use ratatui::crossterm;

use crate::eval::runner::SuiteReport;

#[derive(Clone, Copy, PartialEq, Eq)]
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AlgoType {
    Retrieve(RetrieveMode),
    Consolidate,
    Forget,
}

impl std::fmt::Display for AlgoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgoType::Retrieve(mode) => write!(f, "retrieve/{}", mode),
            AlgoType::Consolidate => write!(f, "consolidate"),
            AlgoType::Forget => write!(f, "forget"),
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
    ToConfigParams(AlgoType, PathBuf),
    ToTestRunning(TestConfig),
    ToTestResults(TestReport),
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
