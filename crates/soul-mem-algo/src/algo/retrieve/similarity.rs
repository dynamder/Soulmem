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
    use soul_mem_query::embedding::Embeddable;
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::query::retrieve::{
        MemoryRetrieveQuery, MemoryRetrieveQueryVariant, SemanticQueryUnit,
    };
    use soul_mem_runtime::working_memory::WorkingMemory;

    const EMB_DIM: usize = 4;

    fn unit_tag_vec(idx: usize) -> EmbeddingVec {
        let mut v = vec![0.0f32; EMB_DIM];
        v[idx] = 1.0;
        EmbeddingVec::new(v)
    }

    fn zero_sem_embedding() -> SemanticEmbedding {
        SemanticEmbedding::new(
            EmbeddingVec::zero(EMB_DIM),
            EmbeddingVec::zero(EMB_DIM),
            EmbeddingVec::zero(EMB_DIM),
        )
    }

    /// Expected score for a node whose tag matches query tag ([1,0,0,0] vs [1,0,0,0]):
    ///   tag_score = cosim = 1.0 → final = 0.4 × 1.0 + 0.6 × 0.0 = 0.4
    const MATCH_SCORE: f32 = 0.4;

    fn build_cluster(tags: &[EmbeddingVec]) -> (WorkingMemory, Vec<MemoryId>) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let ids: Vec<MemoryId> = (0..tags.len()).map(|_| MemoryId::new()).collect();
        cluster.write(|c| {
            for (i, tag) in tags.iter().enumerate() {
                let mem_type = MemoryType::Semantic(SemMemory {
                    content: "test".to_string(),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                });
                let note = MemoryNoteBuilder::new(mem_type).id(ids[i]).build().unwrap();
                let embedding = MemoryEmbedding::new(
                    tag.clone(),
                    MemoryEmbeddingVariant::Semantic(zero_sem_embedding()),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding });
            }
        });
        (wm, ids)
    }

    #[test]
    fn test_retr_similarity_basic() {
        let tags = vec![unit_tag_vec(0), EmbeddingVec::zero(EMB_DIM)];
        let (wm, _ids) = build_cluster(&tags);
        let config = SimilarityConfig {
            similarity_threshold: 0.2,
            max_results: 10,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(unit_tag_vec(0));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert_eq!(result.len(), 1);
        assert!((result[0].1 - MATCH_SCORE).abs() < f32::EPSILON);
    }

    #[test]
    fn test_retr_similarity_max_results() {
        let tags = vec![
            unit_tag_vec(0),
            unit_tag_vec(0),
            unit_tag_vec(1),
            unit_tag_vec(1),
            EmbeddingVec::zero(EMB_DIM),
        ];
        let (wm, _ids) = build_cluster(&tags);
        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 2,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(unit_tag_vec(0));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert_eq!(result.len(), 2);
        for entry in &result {
            assert!((entry.1 - MATCH_SCORE).abs() < f32::EPSILON);
        }
        assert!(result[0].1 >= result[1].1);
    }

    #[test]
    fn test_retr_similarity_empty_cluster() {
        let wm = WorkingMemory::new(10);
        let config = SimilarityConfig {
            similarity_threshold: 0.5,
            max_results: 10,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(unit_tag_vec(0));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert!(result.is_empty());
    }

    #[test]
    fn test_retr_similarity_all_below_threshold() {
        let tags = vec![
            unit_tag_vec(0), // 0.4
            unit_tag_vec(0), // 0.4
        ];
        let (wm, _ids) = build_cluster(&tags);
        let config = SimilarityConfig {
            similarity_threshold: 0.5,
            max_results: 10,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(unit_tag_vec(0));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert!(result.is_empty());
    }

    #[test]
    fn test_retr_similarity_nan_filtered() {
        let nan_vec = EmbeddingVec::new(vec![f32::NAN, 0.0, 0.0, 0.0]);
        let tags = vec![nan_vec];
        let (wm, _ids) = build_cluster(&tags);
        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 10,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(unit_tag_vec(0));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert!(result.is_empty());
    }

    #[test]
    fn test_retr_similarity_sort_stable() {
        let tags = vec![unit_tag_vec(0), unit_tag_vec(0), unit_tag_vec(0)];
        let (wm, ids) = build_cluster(&tags);
        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 2,
        };
        let query_embedding = MemoryRetrieveQueryEmbedding::new(unit_tag_vec(0));
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, ids[0]);
        assert_eq!(result[1].0, ids[1]);
    }

    #[test]
    fn test_retr_similarity_with_bge_embedding() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let mem_type = MemoryType::Semantic(SemMemory {
            content: "Rust编程语言".to_string(),
            aliases: vec!["Rust".to_string()],
            concept_type: ConceptType::Entity,
            description: "一种系统编程语言".to_string(),
        });
        let note = MemoryNoteBuilder::new(mem_type)
            .tags(vec!["Rust".to_string(), "编程".to_string()])
            .build()
            .unwrap();
        let note_id = note.id();
        let embedding = note.embed(&model).unwrap();

        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        cluster.write(|c| {
            c.add_single_node(EmbeddedMemoryNote { note, embedding });
        });

        let retrieve_query = MemoryRetrieveQuery::new(
            vec!["Rust".to_string(), "编程".to_string()],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("Rust编程语言".to_string()),
            ]),
        );
        let query_embedding = retrieve_query.embed(&model).unwrap();

        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 10,
        };
        let request = config.into_request(Arc::new(wm), query_embedding);
        let result = RetrSimilarity {}.retrieve(request);

        assert!(!result.is_empty());
        assert!(result.iter().any(|(id, _)| *id == note_id));
    }
}
