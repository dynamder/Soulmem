use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;

use crate::algo::retrieve::{
    complex::{AssociateWithActionConfig, RetrAssociateWithAction},
    short_only::{RetrShortOnly, ShortOnlyConfig},
    similarity::{RetrSimilarity, SimilarityConfig},
    RetrRequest, RetrStrategy,
};
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
use soul_mem_runtime::working_memory::{sliding_window::Information, WorkingMemory};

#[derive(Debug, Clone, Deserialize)]
pub struct DefaultPipelineConfig {
    short_mem_with_history: ShortOnlyConfig,
    similarity: SimilarityConfig,
    assoc_with_action: AssociateWithActionConfig,
}

impl DefaultPipelineConfig {
    pub fn into_request(
        self,
        working_memory: Arc<WorkingMemory>,
        query: EmbeddedMemoryRetrieveQuery,
        priority: u32,
    ) -> DefaultPipelineRequest {
        DefaultPipelineRequest {
            working_mem: working_memory,
            query,
            priority,
            pipeline_config: self,
        }
    }
}

pub struct RetrDefaultPipeline;

/// 默认pipeline返回的记忆note数量上限
const MAX_PIPELINE_NOTES: usize = 10;

/// 将相似性种子与PPR关联结果按MemoryId合并去重（同id取更高分），
/// 按分数降序后截断到 MAX_PIPELINE_NOTES。
/// 两种来源的分数均处于 [0,1] 量纲，可直接比较。
fn merge_note_scores(
    similarity: Vec<(MemoryId, f32)>,
    association: Vec<(MemoryId, f64)>,
) -> Vec<(MemoryId, f64)> {
    let mut score_map: HashMap<MemoryId, f64> = HashMap::new();
    for (id, score) in similarity {
        let score = score as f64;
        score_map
            .entry(id)
            .and_modify(|s| *s = s.max(score))
            .or_insert(score);
    }
    for (id, score) in association {
        score_map
            .entry(id)
            .and_modify(|s| *s = s.max(score))
            .or_insert(score);
    }
    let mut merged: Vec<(MemoryId, f64)> = score_map.into_iter().collect();
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    merged.truncate(MAX_PIPELINE_NOTES);
    merged
}

pub struct DefaultPipelineRequest {
    working_mem: Arc<WorkingMemory>,
    pipeline_config: DefaultPipelineConfig,
    query: EmbeddedMemoryRetrieveQuery,
    priority: u32,
}

pub struct DefaultPipelineResult {
    pub association: Vec<(MemoryId, f64)>,
    pub action: Vec<(MemoryId, f64)>,
    pub short_history: Arc<[Information]>,
    pub short_mem: Arc<str>,
    pub priority: u32,
}

impl RetrRequest for DefaultPipelineRequest {}

impl RetrStrategy for RetrDefaultPipeline {
    type Request = DefaultPipelineRequest;
    type Return<'a> = DefaultPipelineResult;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        //短期记忆：滑动窗口和摘要
        let short_mem_request = request
            .pipeline_config
            .short_mem_with_history
            .into_request(Arc::clone(&request.working_mem));
        let short_mem_res = RetrShortOnly {}.retrieve(short_mem_request);

        //向量相似性搜索，提取ppr源节点
        let similarity_request = request
            .pipeline_config
            .similarity
            .into_request(Arc::clone(&request.working_mem), request.query);
        let similarity_res = RetrSimilarity {}.retrieve(similarity_request);

        //PPR联想及动作概率推理
        let association_with_action_request = request
            .pipeline_config
            .assoc_with_action
            .into_request(Arc::clone(&request.working_mem), similarity_res.clone());
        let association_with_action_res =
            RetrAssociateWithAction {}.retrieve(association_with_action_request);

        //合并去重：直接命中（similarity）与PPR关联（association）统一按分排序，上限MAX_PIPELINE_NOTES
        let merged_memory = merge_note_scores(similarity_res, association_with_action_res.memory);

        DefaultPipelineResult {
            association: merged_memory,
            action: association_with_action_res.action,
            short_history: short_mem_res.0,
            short_mem: short_mem_res.1,
            priority: request.priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
    use soul_mem_core::memory_note::{
        proc_mem::{Action, ActionType, ProcMemory},
        sem_mem::{ConceptType, SemMemory},
        MemoryNoteBuilder, MemoryType,
    };
    use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::Embeddable;
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::query::retrieve::{
        MemoryRetrieveQuery, MemoryRetrieveQueryVariant, SemanticQueryUnit,
    };
    use soul_mem_runtime::working_memory::sliding_window::{AssistantInformation, UserInformation};

    fn create_mock_working_memory_full() -> WorkingMemory {
        let mut wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let action_id = MemoryId::new();

        let sem_link = SemMemLink::new("related".to_string(), 0.8);
        let link_type = MemoryLinkType::Sem(sem_link);
        let link1 = MemoryLink::new(id1, id2, link_type);

        let proc_link = ProcMemLink::TrigToAction(TrigToAction::new(0.5));
        let link_type2 = MemoryLinkType::Proc(proc_link);
        let link2 = MemoryLink::new(id2, action_id, link_type2);

        cluster.write(|c| {
            let note1 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "Memory 1".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(id1)
            .mem_links(vec![link1])
            .build()
            .unwrap();
            let embedding1 = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: note1,
                embedding: embedding1,
            });

            let note2 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "Memory 2".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(id2)
            .mem_links(vec![link2])
            .build()
            .unwrap();
            let embedding2 = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: note2,
                embedding: embedding2,
            });

            let action_mem_type = MemoryType::Procedure(ProcMemory::new(Action::new(
                "TestAction".to_string(),
                ActionType::new_speak(),
            )));
            let action_note = MemoryNoteBuilder::new(action_mem_type)
                .id(action_id)
                .build()
                .unwrap();
            let action_embedding =
                MemoryEmbedding::new(EmbeddingVec::zero(128), MemoryEmbeddingVariant::Procedure());
            c.add_single_node(EmbeddedMemoryNote {
                note: action_note,
                embedding: action_embedding,
            });
        });

        let sw = wm.sliding_window_mut();
        let window = sw.window();
        let mut guard = window.write();
        guard.push_back(Information::User(UserInformation::new("Hello")));
        guard.push_back(Information::Assistant(AssistantInformation::new(
            "Hi there!",
        )));
        drop(guard);

        wm
    }

    fn create_default_pipeline_config() -> DefaultPipelineConfig {
        DefaultPipelineConfig {
            short_mem_with_history: ShortOnlyConfig {
                clipping_length: None,
                include_summary: true,
            },
            similarity: SimilarityConfig {
                similarity_threshold: 0.5,
                max_results: 10,
            },
            assoc_with_action: AssociateWithActionConfig {
                association: Default::default(),
                action_top_k: 3,
            },
        }
    }

    #[test]
    fn test_retr_default_pipeline_basic() {
        let wm = create_mock_working_memory_full();
        let config = create_default_pipeline_config();
        let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(128));
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query: MemoryRetrieveQuery::new(
                vec!["test".to_string()],
                MemoryRetrieveQueryVariant::Semantic(vec![]),
            ),
        };
        let request = config.into_request(Arc::new(wm), embedded_query, 1);
        let result = RetrDefaultPipeline {}.retrieve(request);

        assert_eq!(result.priority, 1);
    }

    #[test]
    fn test_retr_default_pipeline_with_empty_cluster() {
        let wm = WorkingMemory::new(10);
        let config = create_default_pipeline_config();
        let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(128));
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query: MemoryRetrieveQuery::new(
                vec!["test".to_string()],
                MemoryRetrieveQueryVariant::Semantic(vec![]),
            ),
        };
        let request = config.into_request(Arc::new(wm), embedded_query, 1);
        let result = RetrDefaultPipeline {}.retrieve(request);

        assert_eq!(result.priority, 1);
        assert!(result.short_history.is_empty());
        assert!(result.short_mem.is_empty());
    }

    #[test]
    fn test_default_pipeline_config_into_request() {
        let wm = WorkingMemory::new(10);
        let config = create_default_pipeline_config();
        let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(128));
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query: MemoryRetrieveQuery::new(
                vec!["test".to_string()],
                MemoryRetrieveQueryVariant::Semantic(vec![]),
            ),
        };
        let request = config.into_request(Arc::new(wm), embedded_query, 42);

        assert_eq!(request.priority, 42);
    }

    #[test]
    fn test_merge_note_scores_dedupe_and_cap() {
        let id_a = MemoryId::new();
        let id_b = MemoryId::new();
        let id_c = MemoryId::new();
        let x1 = MemoryId::new();
        let x2 = MemoryId::new();

        // 相似性种子4个：与关联重叠的取更高分
        let similarity = vec![(id_a, 0.9f32), (id_b, 0.2f32), (x1, 0.7f32), (x2, 0.6f32)];
        // 关联节点12个：与种子重叠 + 大量新节点，去重后应超10
        let mut association = vec![(id_a, 0.5f64), (id_b, 0.95f64), (id_c, 0.8f64)];
        for i in 0..9 {
            association.push((MemoryId::new(), 0.1 + i as f64 * 0.05));
        }

        let merged = merge_note_scores(similarity, association);

        assert!(merged.len() <= MAX_PIPELINE_NOTES);
        // 排序严格降序，且所有分数落在 [0,1] 量纲
        for w in merged.windows(2) {
            assert!(w[0].1 >= w[1].1);
            assert!(w[0].1.is_finite());
            assert!((0.0..=1.0).contains(&w[0].1));
        }
        // 重叠节点取更高分：id_a 应保留 0.9，id_b 应保留 0.95（f32→f64 有精度差，用容差比较）
        let get = |id: MemoryId| merged.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
        assert!((get(id_a).unwrap() - 0.9).abs() < 1e-5);
        assert!((get(id_b).unwrap() - 0.95).abs() < 1e-5);
        assert!((get(id_c).unwrap() - 0.8).abs() < 1e-5);
        assert!(merged[0].1 >= 0.95, "top score should be >= 0.95");
    }

    #[test]
    fn test_pipeline_magnitude_and_ppr_association() {
        // 端到端验证：
        //   1. 相似度直接命中节点进入结果
        //   2. 字形/语义都无关、被相似度阈值排除的节点，通过PPR链接被关联检索到
        //   3. 结果数量级正确：分数有限、落在[0,1]、note总数不超过10
        let model = BgeSmallZh::default_cpu().unwrap();
        let mut wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();
        let sem_link1 = MemoryLink::new(
            id1,
            id2,
            MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 0.8)),
        );
        let sem_link2 = MemoryLink::new(
            id2,
            id3,
            MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 0.8)),
        );

        let note1 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: "酒馆".to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "人们喝酒聊天的地方".to_string(),
        }))
        .id(id1)
        .mem_links(vec![sem_link1])
        .build()
        .unwrap();
        let embedding1 = note1.embed(&model).unwrap();

        let note2 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: "酒吧".to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "供应酒水饮料的场所".to_string(),
        }))
        .id(id2)
        .mem_links(vec![sem_link2])
        .build()
        .unwrap();
        let embedding2 = note2.embed(&model).unwrap();

        // 与查询完全无关的节点：只通过 id2 的 Sem 链接可达
        let note3 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: "火车站".to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "乘坐火车的地方".to_string(),
        }))
        .id(id3)
        .build()
        .unwrap();
        let embedding3 = note3.embed(&model).unwrap();

        cluster.write(|c| {
            c.add_single_node(EmbeddedMemoryNote {
                note: note1,
                embedding: embedding1,
            });
            c.add_single_node(EmbeddedMemoryNote {
                note: note2,
                embedding: embedding2,
            });
            c.add_single_node(EmbeddedMemoryNote {
                note: note3,
                embedding: embedding3,
            });
        });

        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("酒馆".to_string())
            ]),
        );
        let query_embedding = query.embed(&model).unwrap();
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query,
        };

        // 阈值0.3：id3（火车站）因相似度不足被排除，只能由PPR通过链接发现
        let config = DefaultPipelineConfig {
            short_mem_with_history: ShortOnlyConfig {
                clipping_length: None,
                include_summary: true,
            },
            similarity: SimilarityConfig {
                similarity_threshold: 0.3,
                max_results: 4,
            },
            assoc_with_action: AssociateWithActionConfig {
                association: Default::default(),
                action_top_k: 3,
            },
        };
        let request = config.into_request(Arc::new(wm), embedded_query, 1);
        let result = RetrDefaultPipeline {}.retrieve(request);

        // 数量级验证：分数有限、在[0,1]、note数不超过上限
        assert!(!result.association.is_empty());
        assert!(
            result.association.len() <= MAX_PIPELINE_NOTES,
            "notes {} exceed cap {MAX_PIPELINE_NOTES}",
            result.association.len()
        );
        for (_, score) in &result.association {
            assert!(score.is_finite(), "non-finite score: {score}");
            assert!((0.0..=1.0).contains(score), "score out of [0,1]: {score}");
        }
        // 直接命中的种子节点在结果中
        assert!(
            result.association.iter().any(|(id, _)| *id == id1),
            "direct similarity hit id1 should be present"
        );
        // 相似度被排除的 id3 通过 PPR 关联出现，证明PPR实际检索到了额外节点
        assert!(
            result.association.iter().any(|(id, _)| *id == id3),
            "PPR should surface linked-but-unsimilar node id3"
        );
    }
}
