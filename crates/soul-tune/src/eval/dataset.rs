use soul_mem_core::memory_note::MemoryId;

#[derive(Debug, Clone)]
pub struct GoldenTestCase {
    pub name: String,
    pub description: String,
    pub expected_ranking: Vec<MemoryId>,
    pub expected_actions: Vec<MemoryId>,
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub case_name: String,
    pub retrieved_ids: Vec<MemoryId>,
    pub retrieved_action_ids: Vec<MemoryId>,
    pub ranking_metrics: RankingMetrics,
    pub action_metrics: ActionMetrics,
}

#[derive(Debug, Clone)]
pub struct RankingMetrics {
    pub recall_at: Vec<(usize, f64)>,
    pub precision_at: Vec<(usize, f64)>,
    pub mrr: f64,
    pub ndcg_at: Vec<(usize, f64)>,
    pub hit_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ActionMetrics {
    pub action_hit_rate: f64,
    pub action_recall_at: Vec<(usize, f64)>,
}
