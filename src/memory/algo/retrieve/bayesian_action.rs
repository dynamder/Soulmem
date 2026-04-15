use std::sync::Arc;

use crate::memory::{
    algo::retrieve::{RetrRequest, RetrStrategy},
    memory_note::MemoryId,
    working_memory::WorkingMemory,
};

pub struct RetrBayesAction;

pub struct BayesActionRequest {
    working_mem: Arc<WorkingMemory>,
    source: Vec<(MemoryId, f64)>,
    top_k: usize,
}

impl RetrRequest for BayesActionRequest {}

impl RetrStrategy for RetrBayesAction {
    type Request = BayesActionRequest;
    type Return<'a> = Vec<(MemoryId, f64)>;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        todo!()
    }
}
