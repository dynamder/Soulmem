use serde::Deserialize;

use super::RetrStrategy;
use crate::memory::embedding::EmbeddingVec;
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
        #[cfg(not(test))]
        {
            todo!("This will only be a wrapper for database operation.")
        }
        #[cfg(test)]
        {
            let cluster = request.working_mem.memory_cluster();
            cluster.read_or_compute(|mem_cluster| {
                mem_cluster
                    .graph()
                    .node_indices()
                    .map(|idx| {
                        let id = mem_cluster.graph().node_weight(idx).unwrap().note().id();
                        (id, 0.5_f64)
                    })
                    .take(request.max_results)
                    .collect()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::cluster::memory_cluster::MemoryCluster;
    use crate::memory::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use crate::memory::embedding::sem::SemanticEmbedding;
    use crate::memory::embedding::{Embeddable, EmbeddingModel, EmbeddingVec};
    use crate::memory::memory_note::{
        sem_mem::{ConceptType, SemMemory},
        MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType,
    };
    use crate::memory::working_memory::WorkingMemory;

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
                let embedding = MemoryEmbedding::new_for_test(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new_for_test(
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
        let query_embedding = MemoryRetrieveQueryEmbedding::new_for_test(EmbeddingVec::zero(128));
        let request = config.into_request(Arc::new(wm), Arc::new(query_embedding));
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
        let query_embedding = MemoryRetrieveQueryEmbedding::new_for_test(EmbeddingVec::zero(128));
        let request = config.into_request(Arc::new(wm), Arc::new(query_embedding));
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
        let query_embedding = MemoryRetrieveQueryEmbedding::new_for_test(EmbeddingVec::zero(128));
        let request = config.into_request(Arc::new(wm), Arc::new(query_embedding));
        let result = RetrSimilarity {}.retrieve(request);

        assert!(result.is_empty());
    }
}
