//仅提取相似记忆策略，即仅提取相似度大于阈值的记忆片段
use super::RetrStrategy;
use crate::algo::retrieve::RetrRequest;
use soul_mem_core::memory_note::MemoryId;
use soul_mem_runtime::working_memory::WorkingMemory;
use std::sync::Arc;

pub struct RetrSimilarity {
    pub similarity_threshold: f64,
    pub max_results: usize,
}
#[allow(dead_code)]
pub struct SimilarityRequest {
    working_mem: Arc<WorkingMemory>,
}
impl RetrRequest for SimilarityRequest {}
impl RetrStrategy for RetrSimilarity {
    type Request = SimilarityRequest;
    type Return<'a> = Vec<MemoryId>;

    fn retrieve(&self, _request: Self::Request) -> Self::Return<'_> {
        todo!("This will only be a wrapper for database operation.")
    }
}
