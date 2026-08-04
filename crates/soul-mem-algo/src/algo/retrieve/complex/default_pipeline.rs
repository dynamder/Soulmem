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
            .into_request(Arc::clone(&request.working_mem), similarity_res);
        let association_with_action_res =
            RetrAssociateWithAction {}.retrieve(association_with_action_request);

        DefaultPipelineResult {
            association: association_with_action_res.memory,
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
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
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
}
