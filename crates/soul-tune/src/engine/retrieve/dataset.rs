use serde::Serialize;
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

#[derive(Debug, Clone)]
pub struct RetrTestCase {
    pub scenario_id: String,
    pub graph_path: String,
    pub cases: Vec<TestCaseQuery>,
}

#[derive(Debug, Clone)]
pub struct TestCaseQuery {
    pub name: String,
    pub description: String,
    pub sub_queries: Vec<SubQuery>,
    pub expected_per_query: Vec<PerQueryExpectation>,
    pub expected_combined_ranking: Vec<MemoryId>,
    pub bonus_combined_ranking: Vec<MemoryId>,
    pub expected_actions: Vec<MemoryId>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubQuery {
    pub priority: u32,
    pub tags: Vec<String>,
    pub variant: MemoryRetrieveQueryVariant,
}

#[derive(Debug, Clone)]
pub struct PerQueryExpectation {
    pub query_index: usize,
    pub ranking: Vec<MemoryId>,
    pub bonus_ranking: Vec<MemoryId>,
}
