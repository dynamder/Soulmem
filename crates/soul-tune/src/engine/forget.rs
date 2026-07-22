use std::time::Duration;

use crate::engine::suite::{SuiteReport, TestCaseOutcome, TestSuite};

pub struct ForgetSuite;

impl TestSuite for ForgetSuite {
    fn case_count(&self) -> usize {
        0
    }

    fn run_case(&self, _index: usize) -> TestCaseOutcome {
        TestCaseOutcome {
            case_name: String::from("forget-stub"),
            description: String::new(),
            passed: false,
            data: Box::new(()),
        }
    }

    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        _elapsed: Duration,
        _total: usize,
        _passed: usize,
        _failed: usize,
    ) -> SuiteReport {
        SuiteReport {
            metrics: vec![],
            detail_header: String::new(),
            detail_rows: vec![],
            outcomes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_zero_cases() {
        let suite = ForgetSuite;
        assert_eq!(suite.case_count(), 0);
    }

    #[test]
    fn test_stub_run_case_not_passed() {
        let suite = ForgetSuite;
        let outcome = suite.run_case(0);
        assert!(!outcome.passed);
        assert_eq!(outcome.case_name, "forget-stub");
    }

    #[test]
    fn test_stub_empty_report() {
        let suite = ForgetSuite;
        let report = suite.build_report(vec![], Duration::ZERO, 0, 0, 0);
        assert!(report.metrics.is_empty());
    }
}
