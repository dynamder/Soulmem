use std::time::Duration;

use soul_mem_core::memory_note::MemoryId;

use crate::base::RetrieveMode;

#[derive(Debug, Clone)]
pub struct TracedNode {
    pub id: MemoryId,
    pub name: String,
    pub content: String,
    pub score: f64,
    pub stage: HitStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitStage {
    Similarity,
    Ppr,
    Action,
    Both,
}

#[derive(Debug, Clone)]
pub struct QueryTrace {
    pub query: soul_mem_query::query::retrieve::MemoryRetrieveQuery,
    pub sim_nodes: Vec<TracedNode>,
    pub sim_elapsed: Duration,
    pub ppr_nodes: Vec<TracedNode>,
    pub ppr_elapsed: Duration,
    pub action_nodes: Vec<TracedNode>,
    pub action_elapsed: Duration,
    pub total_elapsed: Duration,
    /// 生成后校验中被丢弃（低于兜底分或嵌入失败）的查询标记；
    /// 被丢弃的查询不参与检索，仅在 trace 中保留以便观察。
    pub dropped: bool,
}

#[derive(Debug, Clone)]
pub struct RetrievalTrace {
    pub mode: RetrieveMode,
    pub total_elapsed: Duration,
    pub merged_nodes: Vec<TracedNode>,
    /// 独立于记忆分数的动作节点（procedure top-k）：按动作自身分数单独排名，
    /// 不参与 merged_nodes 的分数合并与截断，保证"瞬时行为倾向"始终进入最终结果。
    pub action_nodes: Vec<TracedNode>,
    pub per_query: Vec<QueryTrace>,
}
