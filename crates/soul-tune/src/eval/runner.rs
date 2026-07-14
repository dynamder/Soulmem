use std::any::Any;
use std::time::Duration;

/// 算法测试套件的抽象接口。
/// 每个算法（retrieve / consolidate / forget）实现此 trait。
pub trait TestSuite: Send {
    /// 测试用例总数
    fn case_count(&self) -> usize;

    /// 执行第 index 个测试用例
    fn run_case(&self, index: usize) -> TestCaseOutcome;

    /// 所有用例执行完成后，构建供 UI 渲染的报告
    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        elapsed: Duration,
        total: usize,
        passed: usize,
        failed: usize,
    ) -> SuiteReport;
}

/// 单个测试用例的输出（框架可见部分）
pub struct TestCaseOutcome {
    pub case_name: String,
    pub description: String,
    pub passed: bool,
    /// 算法特定的附加数据，build_report 中 downcast 使用
    pub data: Box<dyn Any + Send>,
}

/// 一个指标分组，如 "准确率" / "性能"
pub struct MetricGroup {
    pub label: String,
    pub items: Vec<(String, String)>,
}

/// 详情页的一行
pub struct DetailRow {
    pub text: String,
    pub has_error: bool,
}

/// 框架渲染结果页所需的全部数据
pub struct SuiteReport {
    pub summary_groups: Vec<MetricGroup>,
    pub detail_header: String,
    pub detail_rows: Vec<DetailRow>,
}
