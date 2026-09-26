#[derive(Debug, Clone)]
pub struct TestCaseConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub test_k_values: Vec<usize>,
    /// DB 模式下的每槽位 HNSW KNN 候选召回预算（None = 用默认启发式）。
    pub db_candidate_k: Option<usize>,
    /// DB 模式下的邻居扩展跳数（None = 默认 1 跳）。
    /// `0` 关闭扩展（只把相似命中写进子图），用于观测"邻居扩展到底带来了什么"。
    pub db_neighbor_depth: Option<usize>,
}

impl TestCaseConfig {
    /// DB 预取参数：候选预算缺省用启发式 `max(2 * max_results, 20)`，邻居跳数缺省 1。
    pub fn db_prefetch_config(&self) -> soul_mem_algo::algo::retrieve::DbPrefetchConfig {
        soul_mem_algo::algo::retrieve::DbPrefetchConfig::new(
            self.db_candidate_k
                .unwrap_or_else(|| default_db_candidate_k(self.max_results)),
            self.db_neighbor_depth.unwrap_or(1),
        )
    }
}

/// DB 模式默认候选预算启发式：DB 只做候选召回（精确重排与 top-k 截断在内存侧），
/// 预算需按槽位数留出余量，同时保持"召回子图小于全图"以真实反映 DB 路径。
///
/// **注意它与图规模无关**：只随 `max_results` 变化，因此在大图上是否仍然够用
/// 需要单独验证（见 `engine::retrieve::depth_audit`）。
pub fn default_db_candidate_k(max_results: usize) -> usize {
    (max_results * 2).max(20)
}
