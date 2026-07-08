use std::path::PathBuf;

use soul_mem_core::memory_note::MemoryId;

/// 一个可执行的检索测试用例，绑定图 + 一组查询
#[derive(Debug, Clone)]
pub struct RetrTestCase {
    /// 场景标识，如 "sakuya_scarlet_mansion"
    pub scenario_id: String,
    /// 指向图数据文件的路径
    pub graph_path: PathBuf,
    /// 该场景下的所有查询用例
    pub cases: Vec<TestCaseQuery>,
}

/// 单个查询：输入 + 期望输出
#[derive(Debug, Clone)]
pub struct TestCaseQuery {
    pub name: String,
    pub priority: u32,
    /// 查询中的 tag 标签列表
    pub tags: Vec<String>,
    /// 期望检索到的记忆节点 node_id（有序，按重要程度排列）
    pub expected_ranking: Vec<MemoryId>,
    /// 期望命中的动作节点 node_id
    pub expected_actions: Vec<MemoryId>,
}

/// 金标准测试用例（保留作为内部结构）
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
