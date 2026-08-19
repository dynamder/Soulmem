use std::any::Any;
use std::time::Duration;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MetricFormat {
    KeyValue {
        value: String,
    },
    Chart {
        x_label: String,
        y_label: String,
        datasets: Vec<Series>,
    },
}

/// 可序列化的指标条目（替代原 trait 对象 `Box<dyn ReportMetric>`）。
///
/// 通过 `#[serde(flatten)]` 与内部标签枚举组合，JSON 形状为：
/// `{"label":..,"group":..,"kind":"key_value"|"chart",...}`，
/// 供 FRB 桥接层（JSON-over-FRB）直接跨边界传递。
#[derive(Debug, Clone, Serialize)]
pub struct MetricEntry {
    pub label: String,
    pub group: String,
    #[serde(flatten)]
    pub format: MetricFormat,
}

impl MetricEntry {
    pub fn label(&self) -> String {
        self.label.clone()
    }
    pub fn group(&self) -> String {
        self.group.clone()
    }
    pub fn format(&self) -> MetricFormat {
        self.format.clone()
    }
}

pub fn key_value_metric(
    label: impl Into<String>,
    group: impl Into<String>,
    value: impl Into<String>,
) -> MetricEntry {
    MetricEntry {
        label: label.into(),
        group: group.into(),
        format: MetricFormat::KeyValue {
            value: value.into(),
        },
    }
}

pub fn chart_metric(
    label: impl Into<String>,
    group: impl Into<String>,
    x_label: impl Into<String>,
    y_label: impl Into<String>,
    datasets: Vec<Series>,
) -> MetricEntry {
    MetricEntry {
        label: label.into(),
        group: group.into(),
        format: MetricFormat::Chart {
            x_label: x_label.into(),
            y_label: y_label.into(),
            datasets,
        },
    }
}

pub trait TestSuite: Send {
    fn case_count(&self) -> usize;

    fn run_case(&self, index: usize) -> TestCaseOutcome;

    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        elapsed: Duration,
        total: usize,
        passed: usize,
        failed: usize,
    ) -> SuiteReport;
}

pub struct TestCaseOutcome {
    pub case_name: String,
    pub description: String,
    pub passed: bool,
    pub data: Box<dyn Any + Send>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetailRow {
    pub text: String,
    pub has_error: bool,
}

pub struct SuiteReport {
    pub metrics: Vec<MetricEntry>,
    pub detail_header: String,
    pub detail_rows: Vec<DetailRow>,
    pub outcomes: Vec<TestCaseOutcome>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_value_metric_format() {
        let m = key_value_metric("Hit Rate", "Accuracy", "0.85");
        assert_eq!(m.label(), "Hit Rate");
        assert_eq!(m.group(), "Accuracy");
        match m.format() {
            MetricFormat::KeyValue { value } => assert_eq!(value, "0.85"),
            _ => panic!("expected KeyValue"),
        }
    }

    #[test]
    fn test_chart_metric_format() {
        let datasets = vec![Series {
            label: "s1".into(),
            points: vec![(0.0, 1.0)],
        }];
        let m = chart_metric("Curve", "Perf", "time", "count", datasets);
        assert_eq!(m.label(), "Curve");
        assert_eq!(m.group(), "Perf");
        match m.format() {
            MetricFormat::Chart {
                x_label,
                y_label,
                datasets,
            } => {
                assert_eq!(x_label, "time");
                assert_eq!(y_label, "count");
                assert_eq!(datasets.len(), 1);
            }
            _ => panic!("expected Chart"),
        }
    }

    #[test]
    fn test_series_clone() {
        let s = Series {
            label: "a".into(),
            points: vec![(1.0, 2.0)],
        };
        let c = s.clone();
        assert_eq!(c.label, "a");
    }

    #[test]
    fn test_test_case_outcome_data_downcast() {
        let outcome = TestCaseOutcome {
            case_name: "test".into(),
            description: "desc".into(),
            passed: true,
            data: Box::new(42i32),
        };
        let v = outcome.data.downcast_ref::<i32>().unwrap();
        assert_eq!(*v, 42);
    }

    #[test]
    fn test_metric_entry_serialize_shape() {
        let m = key_value_metric("平均 MRR", "权重 tag=0.4, variant=0.6", "0.8123");
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json["label"], "平均 MRR");
        assert_eq!(json["group"], "权重 tag=0.4, variant=0.6");
        assert_eq!(json["kind"], "key_value");
        assert_eq!(json["value"], "0.8123");
    }
}
