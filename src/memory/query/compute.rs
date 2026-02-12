use crate::memory::{
    embedding::sem::SemanticEmbedding,
    memory_note::MemoryNote,
    query::{self, retrieve::SemanticQueryUnit},
};

pub trait AnonymousQueryCompute {
    type Query;
    fn anonymous_compute(&self, query: &Self::Query) -> f32;
}

pub trait QueryCompute: AnonymousQueryCompute {
    fn compute(&self, query: &Self::Query) -> QueryComputeResult;
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
////////////////////////////////////////////////////////////
impl AnonymousQueryCompute for SemanticEmbedding {
    type Query = SemanticQueryUnit;
    fn anonymous_compute(&self, query: &Self::Query) -> f32 {
        todo!()
    }
}
