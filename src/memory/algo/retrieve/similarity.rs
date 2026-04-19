use serde::Deserialize;

use super::RetrStrategy;
use crate::memory::{
    algo::retrieve::RetrRequest, memory_note::MemoryId, working_memory::WorkingMemory,
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
    pub fn into_request(self, working_mem: Arc<WorkingMemory>) -> SimilarityRequest {
        SimilarityRequest { working_mem }
    }
}

pub struct RetrSimilarity {
    pub similarity_threshold: f64,
    pub max_results: usize,
}

pub struct SimilarityRequest {
    working_mem: Arc<WorkingMemory>,
}

impl RetrRequest for SimilarityRequest {}

impl RetrStrategy for RetrSimilarity {
    type Request = SimilarityRequest;
    type Return<'a> = Vec<MemoryId>;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        todo!("This will only be a wrapper for database operation.")
    }
}
