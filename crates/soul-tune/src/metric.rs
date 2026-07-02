use std::time::Duration;

pub enum MetricDisplayKind {
    KeyValue,
    Chart {
        x_label: &'static str,
        y_label: &'static str,
    },
}

pub enum MetricValue {
    Int(i64),
    Float(f64),
    String(String),
    Duration(Duration),
    Percent(f64),
    Bool(bool),
}

pub enum MetricData {
    Single(MetricValue),
    ChartPoints(Vec<(f64, f64)>),
}

pub trait Metric: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> &str;
    fn category(&self) -> &str;
    fn display_kind(&self) -> MetricDisplayKind;
    fn value(&self) -> MetricData;
}

pub struct MetricRegistry {
    metrics: Vec<Box<dyn Metric>>,
}

impl MetricRegistry {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn register(&mut self, metric: Box<dyn Metric>) {
        self.metrics.push(metric);
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Metric> {
        self.metrics.iter().map(|m| m.as_ref())
    }
}
