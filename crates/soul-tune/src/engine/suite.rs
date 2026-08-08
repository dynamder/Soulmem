use std::any::Any;
use std::time::Duration;

#[derive(Clone)]
pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

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

pub trait ReportMetric {
    fn label(&self) -> String;
    fn group(&self) -> String;
    fn format(&self) -> MetricFormat;
}

pub fn key_value_metric(
    label: impl Into<String>,
    group: impl Into<String>,
    value: impl Into<String>,
) -> impl ReportMetric {
    KeyValueMetric {
        label: label.into(),
        group: group.into(),
        value: value.into(),
    }
}

pub fn chart_metric(
    label: impl Into<String>,
    group: impl Into<String>,
    x_label: impl Into<String>,
    y_label: impl Into<String>,
    datasets: Vec<Series>,
) -> impl ReportMetric {
    ChartMetric {
        label: label.into(),
        group: group.into(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        datasets,
    }
}

struct KeyValueMetric {
    label: String,
    group: String,
    value: String,
}

impl ReportMetric for KeyValueMetric {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn group(&self) -> String {
        self.group.clone()
    }
    fn format(&self) -> MetricFormat {
        MetricFormat::KeyValue {
            value: self.value.clone(),
        }
    }
}

struct ChartMetric {
    label: String,
    group: String,
    x_label: String,
    y_label: String,
    datasets: Vec<Series>,
}

impl ReportMetric for ChartMetric {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn group(&self) -> String {
        self.group.clone()
    }
    fn format(&self) -> MetricFormat {
        MetricFormat::Chart {
            x_label: self.x_label.clone(),
            y_label: self.y_label.clone(),
            datasets: self.datasets.clone(),
        }
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

pub struct DetailRow {
    pub text: String,
    pub has_error: bool,
}

pub struct SuiteReport {
    pub metrics: Vec<Box<dyn ReportMetric>>,
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
}
