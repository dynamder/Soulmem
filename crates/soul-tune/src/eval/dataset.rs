use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

/// 一个可执行的检索测试用例，绑定图 + 一组查询
#[derive(Debug, Clone)]
pub struct RetrTestCase {
    pub scenario_id: String,
    pub graph_path: String,
    pub cases: Vec<TestCaseQuery>,
}

/// 测试配置元数据
#[derive(Debug, Clone)]
pub struct TestCaseConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub test_k_values: Vec<usize>,
}

/// 单个测试用例
#[derive(Debug, Clone)]
pub struct TestCaseQuery {
    pub name: String,
    pub description: String,
    pub sub_queries: Vec<SubQuery>,
    pub expected_per_query: Vec<PerQueryExpectation>,
    pub expected_combined_ranking: Vec<MemoryId>,
    pub expected_actions: Vec<MemoryId>,
}

/// 子查询
#[derive(Debug, Clone)]
pub struct SubQuery {
    pub priority: u32,
    pub tags: Vec<String>,
    pub variant: MemoryRetrieveQueryVariant,
}

/// 子查询的期望结果
#[derive(Debug, Clone)]
pub struct PerQueryExpectation {
    pub query_index: usize,
    pub ranking: Vec<MemoryId>,
}
