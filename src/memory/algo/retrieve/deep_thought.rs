use std::sync::Arc;

use crate::memory::{
    algo::retrieve::RetrRequest, memory_cluster::MemorySubCluster, memory_note::MemoryId,
    working_memory::WorkingMemory,
};

use super::RetrStrategy;
// 采用 LLM进行的Plan-on-Graph
pub struct RetrDeepThought {
    pub max_iteration: usize,
}
pub struct DeepThoughtRequest {
    working_mem: Arc<WorkingMemory>,
    initial_mem_notes: Vec<MemoryId>,
}

impl RetrRequest for DeepThoughtRequest {}

impl RetrStrategy for RetrDeepThought {
    type Request = DeepThoughtRequest;
    fn retrieve(&self, request: Self::Request) -> Vec<MemoryId> {
        todo!()
    }
}

//Place holder for LLM call
fn llm_call() -> MemoryId {
    todo!()
}
