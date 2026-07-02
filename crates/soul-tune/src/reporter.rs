#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReportLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for ReportLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportLevel::Debug => write!(f, "DEBUG"),
            ReportLevel::Info => write!(f, "INFO"),
            ReportLevel::Warn => write!(f, "WARN"),
            ReportLevel::Error => write!(f, "ERROR"),
        }
    }
}

pub struct ReportEntry {
    pub time: String,
    pub level: ReportLevel,
    pub source: String,
    pub message: String,
}

pub trait TestReporter: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn entries(&self) -> Vec<ReportEntry>;
}

pub struct ReporterRegistry {
    reporters: Vec<Box<dyn TestReporter>>,
}

impl ReporterRegistry {
    pub fn new() -> Self {
        Self {
            reporters: Vec::new(),
        }
    }

    pub fn register(&mut self, reporter: Box<dyn TestReporter>) {
        self.reporters.push(reporter);
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn TestReporter> {
        self.reporters.iter().map(|r| r.as_ref())
    }
}
