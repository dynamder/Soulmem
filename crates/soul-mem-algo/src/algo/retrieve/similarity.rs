use serde::Deserialize;

use super::RetrStrategy;
use crate::algo::retrieve::RetrRequest;
use soul_mem_core::memory_note::MemoryId;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
use soul_mem_query::query::compute::QueryCompute;
use soul_mem_runtime::working_memory::WorkingMemory;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
pub struct SimilarityConfig {
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_similarity_threshold() -> f32 {
    0.7
}
fn default_max_results() -> usize {
    10
}

impl SimilarityConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        query: MemoryRetrieveQueryEmbedding,
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
    query: MemoryRetrieveQueryEmbedding,
    similarity_threshold: f32,
    max_results: usize,
}

impl RetrRequest for SimilarityRequest {}

impl RetrStrategy for RetrSimilarity {
    type Request = SimilarityRequest;
    type Return<'a> = Vec<(MemoryId, f32)>;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        //TODO: 添加从数据库的向量相似结果并混合
        let cluster = request.working_mem.memory_cluster();
        cluster.read_or_compute(|mem_cluster| {
            let node_weights = mem_cluster.graph().node_weights().collect::<Vec<_>>();

            let mut query_calc = node_weights
                .into_par_iter()
                .filter_map(|mem_note| {
                    let compute_res = mem_note.compute(&request.query).ok();
                    match compute_res {
                        Some(res) => {
                            if !res.score.is_finite() {
                                None
                            } else {
                                if res.score < request.similarity_threshold {
                                    None
                                } else {
                                    Some(res)
                                }
                            }
                        }
                        None => None,
                    }
                })
                .collect::<Vec<_>>();

            query_calc.sort_by(|a, b| b.score.total_cmp(&a.score));

            query_calc
                .into_iter()
                .take(request.max_results)
                .map(|r| (r.id, r.score))
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_note::{
        MemoryId, MemoryNoteBuilder, MemoryType,
        sem_mem::{ConceptType, SemMemory},
    };
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_runtime::working_memory::WorkingMemory;

    fn create_mock_working_memory_with_nodes() -> (WorkingMemory, Vec<MemoryId>) {
        let mut wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let ids: Vec<_> = (0..5).map(|_| MemoryId::new()).collect();

        cluster.write(|c| {
            for (i, id) in ids.iter().enumerate() {
                let mem_type = MemoryType::Semantic(SemMemory {
                    content: format!("Memory {}", i),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                });
                let note = MemoryNoteBuilder::new(mem_type).id(*id).build().unwrap();
                let embedding = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                    )),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding });
            }
        });

        (wm, ids)
    }

    #[test]
    fn test_retr_similarity_basic() {
        let (wm, _ids) = create_mock_working_memory_with_nodes();
        let config = SimilarityConfig {
            similarity_threshold: 0.5,
            max_results: 3,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(128));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert!(!result.is_empty());
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_retr_similarity_max_results() {
        let (wm, _ids) = create_mock_working_memory_with_nodes();
        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 2,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(128));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_retr_similarity_empty_cluster() {
        let wm = WorkingMemory::new(10);
        let config = SimilarityConfig {
            similarity_threshold: 0.5,
            max_results: 10,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(128));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert!(result.is_empty());
    }
}
