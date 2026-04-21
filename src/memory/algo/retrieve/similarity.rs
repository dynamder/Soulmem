use serde::Deserialize;

use super::RetrStrategy;
use crate::memory::{
    algo::retrieve::RetrRequest, embedding::query::note::MemoryRetrieveQueryEmbedding,
    memory_note::MemoryId, working_memory::WorkingMemory,
};
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
pub struct SimilarityConfig {
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_similarity_threshold() -> f64 {
    0.7
}
fn default_max_results() -> usize {
    10
}

impl SimilarityConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        query: Arc<MemoryRetrieveQueryEmbedding>,
    ) -> SimilarityRequest {
        SimilarityRequest {
            working_mem,
            query,
            similarity_threshold: self.similarity_threshold,
            max_results: self.max_results,
        }
    }
}

pub struct RetrSimilarity;

pub struct SimilarityRequest {
    working_mem: Arc<WorkingMemory>,
    query: Arc<MemoryRetrieveQueryEmbedding>,
    similarity_threshold: f64,
    max_results: usize,
}

impl RetrRequest for SimilarityRequest {}

impl RetrStrategy for RetrSimilarity {
    type Request = SimilarityRequest;
    type Return<'a> = Vec<(MemoryId, f64)>;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        todo!("This will only be a wrapper for database operation.")
    }
}
