use serde::Deserialize;

use super::RetrStrategy;
use crate::algo::retrieve::RetrRequest;
use soul_mem_core::memory_note::MemoryId;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
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
    4
}

impl SimilarityConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        query: EmbeddedMemoryRetrieveQuery,
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
    query: EmbeddedMemoryRetrieveQuery,
    similarity_threshold: f32,
    max_results: usize,
}

impl RetrRequest for SimilarityRequest {}

impl RetrStrategy for RetrSimilarity {
    type Request = SimilarityRequest;
    type Return<'a> = Vec<(MemoryId, f32)>;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        //TODO: 添加从数据库的向量相似结果并混合
        let string_blend_alpha = request.query.embedding.string_blend_alpha;
        let cluster = request.working_mem.memory_cluster();
        cluster.read_or_compute(|mem_cluster| {
            let node_weights = mem_cluster.graph().node_weights().collect::<Vec<_>>();

            let mut query_calc = node_weights
                .into_par_iter()
                .filter_map(|mem_note| {
                    let compute_res = mem_note
                        .compute_fused(&request.query, string_blend_alpha)
                        .ok();
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
        sem_mem::{ConceptType, SemMemory},
        MemoryId, MemoryNoteBuilder, MemoryType,
    };
    use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::Embeddable;
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::query::compute::QueryCompute;
    use soul_mem_query::query::retrieve::{
        MemoryRetrieveQuery, MemoryRetrieveQueryVariant, SemanticQueryUnit,
    };
    use soul_mem_runtime::working_memory::WorkingMemory;

    fn empty_embedded_query(tag: EmbeddingVec) -> EmbeddedMemoryRetrieveQuery {
        EmbeddedMemoryRetrieveQuery {
            embedding: MemoryRetrieveQueryEmbedding::new(tag),
            query: MemoryRetrieveQuery::new(vec![], MemoryRetrieveQueryVariant::Semantic(vec![])),
        }
    }

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
    ///   embedding score = 0.4 × tag_cosim(1.0) + 0.6 × variant(0.0) = 0.4
    ///   query has no concept_identifier → string_score = 0.0
    ///   fused = string_blend_alpha(0.6) × 0.4 + (1-0.6) × 0.0 = 0.24
    const MATCH_SCORE: f32 = 0.24;

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
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
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
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
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
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
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
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
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
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
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
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
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
                SemanticQueryUnit::new().with_concept_identifier("Rust编程语言".to_string())
            ]),
        );
        let query_embedding = retrieve_query.embed(&model).unwrap();

        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 10,
        };
        let request = config.into_request(
            Arc::new(wm),
            EmbeddedMemoryRetrieveQuery {
                embedding: query_embedding,
                query: retrieve_query,
            },
        );
        let result = RetrSimilarity {}.retrieve(request);

        assert!(!result.is_empty());
        assert!(result.iter().any(|(id, _)| *id == note_id));
    }

    #[test]
    fn test_retr_similarity_string_boost_lifts_score() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let mem_type = MemoryType::Semantic(SemMemory {
            content: "小酒馆".to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "与酒馆相关的描述".to_string(),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let note_id = note.id();
        let embedding = note.embed(&model).unwrap();
        let embedded_note = EmbeddedMemoryNote { note, embedding };

        let retrieve_query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("酒馆".to_string())
            ]),
        );
        let query_embedding = retrieve_query.embed(&model).unwrap();

        // 纯 embedding 得分（字符串分=0 时的基线）
        let pure = embedded_note.compute(&query_embedding).unwrap().score;

        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        cluster.write(|c| {
            c.add_single_node(embedded_note);
        });

        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 10,
        };
        let request = config.into_request(
            Arc::new(wm),
            EmbeddedMemoryRetrieveQuery {
                embedding: query_embedding,
                query: retrieve_query,
            },
        );
        let result = RetrSimilarity {}.retrieve(request);

        assert_eq!(result.len(), 1);
        let (id, fused) = result[0];
        assert_eq!(id, note_id);
        assert!(fused.is_finite());
        assert!((0.0..=1.0).contains(&fused), "fused out of range: {fused}");
        // "酒馆" vs "小酒馆" 的字形接近被字符串分兜底，融合分应高于纯 embedding 分
        assert!(
            fused > pure,
            "string boost should lift fused ({fused}) above pure embedding ({pure})"
        );
    }
}
