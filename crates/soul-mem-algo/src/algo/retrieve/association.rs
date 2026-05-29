use std::sync::Arc;

use crate::algo::retrieve::RetrRequest;
use soul_mem_core::memory_note::MemoryId;
use soul_mem_runtime::working_memory::WorkingMemory;

use super::RetrStrategy;

//用PPR变种算法进行联想
pub struct RetrAssociation {
    pub max_results: usize,
}

#[allow(dead_code)]
pub struct AssociationRequest {
    working_mem: Arc<WorkingMemory>,
}

impl RetrRequest for AssociationRequest {}

impl RetrStrategy for RetrAssociation {
    type Request = AssociationRequest;
    type Return<'a> = Vec<MemoryId>;
    fn retrieve(&self, _request: Self::Request) -> Self::Return<'_> {
        todo!()
    }
}
