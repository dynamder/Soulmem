#[derive(Debug, Clone)]
pub struct TestCaseConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub test_k_values: Vec<usize>,
    /// DB 模式下的每槽位 HNSW KNN 候选召回预算（None = 用默认启发式）。
    pub db_candidate_k: Option<usize>,
}
