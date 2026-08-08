use std::sync::Arc;

use serde::Deserialize;

use crate::algo::retrieve::{
    association::{AssociationConfig, AssociationRequest, RetrAssociation},
    bayes_action::{BayesActionRequest, RetrBayesAction},
    RetrRequest, RetrStrategy,
};
use soul_mem_core::memory_note::MemoryId;
use soul_mem_runtime::working_memory::WorkingMemory;

#[derive(Debug, Clone, Deserialize)]
pub struct AssociateWithActionConfig {
    #[serde(default)]
    pub association: AssociationConfig,
    #[serde(default = "default_action_top_k")]
    pub action_top_k: usize,
}

fn default_action_top_k() -> usize {
    3
}

impl AssociateWithActionConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        source: Vec<(MemoryId, f32)>,
    ) -> AssociateWithActionRequest {
        AssociateWithActionRequest {
            association: self
                .association
                .into_request(Arc::clone(&working_mem), source),
            action_top_k: self.action_top_k,
        }
    }
}

pub struct RetrAssociateWithAction;

pub struct AssociateWithActionRequest {
    association: AssociationRequest,
    action_top_k: usize,
}

impl AssociateWithActionRequest {
    pub fn new(association: AssociationRequest) -> Self {
        Self {
            association,
            action_top_k: 3,
        }
    }
    pub fn with_action_top_k(mut self, action_top_k: usize) -> Self {
        self.action_top_k = action_top_k;
        self
    }
}

impl RetrRequest for AssociateWithActionRequest {}

pub struct AssociateWithActionResult {
    pub memory: Vec<(MemoryId, f64)>,
    pub action: Vec<(MemoryId, f64)>,
}

impl RetrStrategy for RetrAssociateWithAction {
    type Request = AssociateWithActionRequest;
    type Return<'a> = AssociateWithActionResult;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let working_mem = Arc::clone(&request.association.working_mem);
        let association_res = RetrAssociation {}.retrieve(request.association);
        if association_res.is_empty() {
            return AssociateWithActionResult {
                memory: Vec::new(),
                action: Vec::new(),
            };
        }

        let normalized_association_res = softmax(&association_res);

        let action_request = BayesActionRequest::new(working_mem, normalized_association_res)
            .with_top_k(request.action_top_k);

        let action_res = RetrBayesAction {}.retrieve(action_request);

        AssociateWithActionResult {
            memory: association_res,
            action: action_res,
        }
    }
}

fn softmax(logits: &[(MemoryId, f64)]) -> Vec<(MemoryId, f64)> {
    let max_x = logits
        .iter()
        .max_by(|&(_, v1), &(_, v2)| v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(_, v)| v)
        .unwrap_or(0.0);

    let sum = logits
        .iter()
        .map(|&(_, x)| f64::exp(x - max_x))
        .sum::<f64>();

    logits
        .iter()
        .map(|&(id, x)| (id, f64::exp(x - max_x) / sum))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_links::MemoryLink;
    use soul_mem_core::memory_links::MemoryLinkType;
    use soul_mem_core::memory_note::{
        proc_mem::{Action, ActionType, ProcMemory},
        sem_mem::{ConceptType, SemMemory},
        MemoryNoteBuilder, MemoryType,
    };
    use soul_mem_query::embedding::note::EmbeddedMemoryNote;
    use soul_mem_query::embedding::note::MemoryEmbedding;
    use soul_mem_query::embedding::note::MemoryEmbeddingVariant;

    #[test]
    fn test_default_action_top_k() {
        assert_eq!(default_action_top_k(), 3);
    }

    #[test]
    fn test_associate_with_action_config_defaults() {
        let config = AssociateWithActionConfig {
            association: AssociationConfig::default(),
            action_top_k: default_action_top_k(),
        };
        assert_eq!(config.action_top_k, 3);
    }
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::EmbeddingVec;

    fn create_mock_working_memory_with_assoc_and_action(
    ) -> (WorkingMemory, MemoryId, MemoryId, MemoryId) {
        let wm = WorkingMemory::new(10);
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

        (wm, id1, id2, action_id)
    }

    #[test]
    fn test_retr_associate_with_action_basic() {
        let (wm, source_id, _, _) = create_mock_working_memory_with_assoc_and_action();
        let config = AssociateWithActionConfig {
            association: AssociationConfig::default(),
            action_top_k: 3,
        };
        let request = config.into_request(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert!(!result.memory.is_empty() || !result.action.is_empty());
    }

    #[test]
    fn test_retr_associate_with_action_empty_source() {
        let (wm, _, _, _) = create_mock_working_memory_with_assoc_and_action();
        let config = AssociateWithActionConfig {
            association: AssociationConfig::default(),
            action_top_k: 3,
        };
        let request = config.into_request(Arc::new(wm), vec![]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert!(result.memory.is_empty());
        assert!(result.action.is_empty());
    }

    #[test]
    fn test_retr_associate_with_action_action_top_k() {
        let (wm, source_id, _, _) = create_mock_working_memory_with_assoc_and_action();
        let config = AssociateWithActionConfig {
            association: AssociationConfig::default(),
            action_top_k: 1,
        };
        let request = config.into_request(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert!(result.action.len() <= 1);
    }

    #[test]
    fn test_softmax_function() {
        let input = vec![
            (MemoryId::new(), 1.0),
            (MemoryId::new(), 2.0),
            (MemoryId::new(), 3.0),
        ];
        let result = softmax(&input);
        let sum: f64 = result.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|(_, p)| *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn test_softmax_empty() {
        let input: Vec<(MemoryId, f64)> = vec![];
        let result = softmax(&input);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_extreme_values() {
        let huge = vec![(MemoryId::new(), 1e10), (MemoryId::new(), 1e10 + 1.0)];
        let result = softmax(&huge);
        let sum: f64 = result.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|(_, p)| *p >= 0.0 && *p <= 1.0));

        let tiny = vec![(MemoryId::new(), -1e10), (MemoryId::new(), -1e10)];
        let result2 = softmax(&tiny);
        let sum2: f64 = result2.iter().map(|(_, p)| p).sum();
        assert!((sum2 - 1.0).abs() < 1e-5);

        let uniform = vec![
            (MemoryId::new(), 5.0),
            (MemoryId::new(), 5.0),
            (MemoryId::new(), 5.0),
        ];
        let result3 = softmax(&uniform);
        let sum3: f64 = result3.iter().map(|(_, p)| p).sum();
        assert!((sum3 - 1.0).abs() < 1e-5);
        assert!(result3.iter().all(|(_, p)| (*p - 1.0 / 3.0).abs() < 1e-5));
    }
}
