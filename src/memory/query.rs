use crate::memory::memory_note::MemoryId;

pub mod retrieve;

pub trait QueryCompute {
    fn compute(&self) -> QueryComputeResult;
    fn anonymous_compute(&self) -> f32;
}

pub struct QueryComputeResult {
    pub id: MemoryId,
    pub score: f32,
}

impl QueryComputeResult {
    pub fn new(id: MemoryId, score: f32) -> Self {
        QueryComputeResult { id, score }
    }
}
