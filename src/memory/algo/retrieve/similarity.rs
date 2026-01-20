//仅提取相似记忆策略，即仅提取相似度大于阈值的记忆片段
use super::RetrStrategy;
use crate::memory::{
    algo::retrieve::RetrRequest, memory_note::MemoryId, working_memory::WorkingMemory,
};
use std::sync::Arc;
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
    fn retrieve(&self, request: Self::Request) -> Vec<MemoryId> {
        //TODO: 减少计算量，以下只是一个最初步的实现
        let cos_similarities: Vec<(f64, MemoryId)> = vec![];
        cos_similarities
            .into_iter()
            .filter(|(similarity, _)| *similarity > self.similarity_threshold)
            .map(|(_, id)| id)
            .take(self.max_results)
            .collect()
    }
}
