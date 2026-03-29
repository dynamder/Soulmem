use std::sync::Arc;

use crate::memory::{
    algo::retrieve::RetrRequest, memory_note::MemoryId, working_memory::WorkingMemory,
};

use super::RetrStrategy;

//用PPR变种算法进行联想
pub struct RetrAssociation {
    pub max_results: usize,
}
pub struct AssociationRequest {
    working_mem: Arc<WorkingMemory>,
}

impl RetrRequest for AssociationRequest {}

impl RetrStrategy for RetrAssociation {
    type Request = AssociationRequest;
    fn retrieve(&self, request: Self::Request) -> Vec<MemoryId> {
        todo!()
    }
}
