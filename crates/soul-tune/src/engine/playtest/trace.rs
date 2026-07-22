use std::time::Duration;

use soul_mem_core::memory_note::MemoryId;

use crate::base::RetrieveMode;

#[derive(Debug, Clone)]
pub struct TracedNode {
    pub id: MemoryId,
    pub name: String,
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
}

#[derive(Debug, Clone)]
pub struct RetrievalTrace {
    pub mode: RetrieveMode,
    pub total_elapsed: Duration,
    pub merged_nodes: Vec<TracedNode>,
    pub per_query: Vec<QueryTrace>,
}
