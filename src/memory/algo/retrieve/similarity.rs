//仅提取相似记忆策略，即仅提取相似度大于阈值的记忆片段
use super::RetrStrategy;
use crate::memory::working_memory::WorkingMemory;
use std::sync::Arc;
pub struct RetrSimilarity {
    similarity_threshold: f64,
    max_results: usize,
}
pub struct SimilarityRequest {
    working_mem: Arc<WorkingMemory>,
}
impl RetrStrategy for RetrSimilarity {
    type RetrRequest = SimilarityRequest;
    fn retrieve(&self, request: Self::RetrRequest) -> Vec<String> {
        todo!()
    }
}
