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
    0.35
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
        // similarity_threshold 语义为"最低兜底分"：低于该分的节点直接过滤，
        // 达到该分的节点按分数取 top-k（max_results），绝对阈值不再饿死查询。
        let floor = request.similarity_threshold;
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
                                if res.score < floor {
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
    use soul_mem_query::query::string_distance::compute_note_string_score;
    use soul_mem_query::query::retrieve::{
        MemoryRetrieveQuery, MemoryRetrieveQueryVariant, SemanticQueryUnit,
    };
    use soul_mem_runtime::working_memory::WorkingMemory;

    #[test]
    fn test_default_constants() {
        assert_eq!(default_similarity_threshold(), 0.35);
        assert_eq!(default_max_results(), 4);
    }

    #[test]
    fn test_similarity_config_defaults() {
        let config = SimilarityConfig {
            similarity_threshold: default_similarity_threshold(),
            max_results: default_max_results(),
        };
        assert_eq!(config.similarity_threshold, 0.35);
        assert_eq!(config.max_results, 4);
    }

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
    ///   embedding score = 0.3 × tag_cosim(1.0) + 0.7 × variant(0.0) = 0.3
    ///   query has no concept_identifier → string_score = 0.0
    ///   string 分量缺失时 fused 退化为纯 embedding 分 = 0.3
    const MATCH_SCORE: f32 = 0.3;

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
                    ..Default::default()
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
    fn test_retr_similarity_default_floor_filters_below() {
        // 默认兜底分 0.35：tag-only 匹配（0.3）被过滤（纯 top-k 不生效）
        let tags = vec![unit_tag_vec(0)];
        let (wm, _ids) = build_cluster(&tags);
        let config = SimilarityConfig {
            similarity_threshold: default_similarity_threshold(),
            max_results: 10,
        };
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
        let result = RetrSimilarity {}.retrieve(request);
        // tag-only 匹配分 = 0.3 < 0.35 → 被兜底分过滤
        assert!(result.is_empty());
    }

    fn make_sem_embedding(content: f32, aliases: f32, desc: f32) -> SemanticEmbedding {
        SemanticEmbedding::new(
            EmbeddingVec::new(vec![content, 0.0, 0.0, 0.0]),
            EmbeddingVec::new(vec![aliases, 0.0, 0.0, 0.0]),
            EmbeddingVec::new(vec![desc, 0.0, 0.0, 0.0]),
        )
    }

    #[test]
    fn test_retr_similarity_variant_affects_score() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let id = MemoryId::new();
        cluster.write(|c| {
            let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "test".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(id)
            .build()
            .unwrap();
            let embedding = MemoryEmbedding::new(
                EmbeddingVec::zero(EMB_DIM),
                MemoryEmbeddingVariant::Semantic(make_sem_embedding(0.8, 0.0, 0.6)),
            );
            c.add_single_node(EmbeddedMemoryNote { note, embedding });
        });
        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 10,
        };
        let request = config.into_request(
            Arc::new(wm),
            empty_embedded_query(EmbeddingVec::zero(EMB_DIM)),
        );
        let result = RetrSimilarity {}.retrieve(request);

        assert!(!result.is_empty());
        let (_, score) = result[0];
        insta::assert_debug_snapshot!("variant_affects_score", score);
    }

    #[test]
    fn test_retr_similarity_ranking_by_tag_and_variant() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let best_id = MemoryId::new();
        let mid_id = MemoryId::new();
        let worst_id = MemoryId::new();

        cluster.write(|c| {
            for (i, (note_id, tag_val, content_val)) in [
                (best_id, 1.0f32, 0.9f32),
                (mid_id, 1.0f32, 0.5f32),
                (worst_id, 0.0f32, 0.0f32),
            ]
            .into_iter()
            .enumerate()
            {
                let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                    content: format!("mem_{i}"),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                }))
                .id(note_id)
                .build()
                .unwrap();
                let embedding = MemoryEmbedding::new(
                    unit_tag_vec(0) * tag_val,
                    MemoryEmbeddingVariant::Semantic(make_sem_embedding(content_val, 0.0, 0.0)),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding });
            }
        });

        let config = SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 10,
        };
        let request = config.into_request(Arc::new(wm), empty_embedded_query(unit_tag_vec(0)));
        let result = RetrSimilarity {}.retrieve(request);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, best_id);
        assert_eq!(result[1].0, mid_id);
        assert_eq!(result[2].0, worst_id);

        let scores: Vec<f32> = result.iter().map(|(_, s)| *s).collect();
        insta::assert_debug_snapshot!("ranking_by_tag_and_variant", scores);
    }

    #[test]
    fn test_retr_similarity_with_bge_embedding() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let mem_type = MemoryType::Semantic(SemMemory {
            content: "Rust编程语言".to_string(),
            aliases: vec!["Rust".to_string()],
            concept_type: ConceptType::Entity,
            description: "一种系统编程语言".to_string(),
            ..Default::default()
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
        let note_for_string = embedded_note.note().clone();

        let retrieve_query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("酒馆".to_string())
            ]),
        );
        let retrieve_query_for_string = retrieve_query.clone();
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
        // "酒馆" vs "小酒馆" 的字形接近，字符串分参与混合（str > 0）。
        // 字符串通道只加分：fused = max(emb, 0.6*emb + 0.4*str)，不会低于纯 embedding 分。
        let str_score = compute_note_string_score(&note_for_string, &retrieve_query_for_string);
        assert!(str_score > 0.0, "string score should be positive: {str_score}");
        let expected = pure.max(0.6 * pure + 0.4 * str_score);
        assert!(
            (fused - expected).abs() < 1e-5,
            "fused {fused} != expected {expected}"
        );
    }
}
