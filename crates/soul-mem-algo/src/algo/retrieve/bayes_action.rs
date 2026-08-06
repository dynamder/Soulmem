use std::{collections::HashMap, sync::Arc};

use petgraph::{Direction::Outgoing, visit::EdgeRef};

use serde::Deserialize;

use crate::algo::retrieve::{RetrRequest, RetrStrategy};

use soul_mem_core::memory_links::{
    MemoryLinkType,
    proc_mem::{ProcMemLink, TrigToAction},
};
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_runtime::cluster::memory_cluster::MemoryCluster;
use soul_mem_runtime::working_memory::WorkingMemory;

#[derive(Debug, Clone, Deserialize)]
pub struct BayesActionConfig {
    #[serde(default = "default_action_top_k")]
    pub top_k: usize,
}

fn default_action_top_k() -> usize {
    5
}

impl BayesActionConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        source: Vec<(MemoryId, f64)>,
    ) -> BayesActionRequest {
        BayesActionRequest {
            working_mem,
            source,
            top_k: self.top_k,
        }
    }
}

pub struct BayesActionRequest {
    pub working_mem: Arc<WorkingMemory>,
    pub source: Vec<(MemoryId, f64)>,
    pub top_k: usize,
}

impl BayesActionRequest {
    pub fn new(working_mem: Arc<WorkingMemory>, source: Vec<(MemoryId, f64)>) -> Self {
        Self {
            working_mem,
            source,
            top_k: 5,
        }
    }
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

pub struct RetrBayesAction;

impl RetrRequest for BayesActionRequest {}

impl RetrStrategy for RetrBayesAction {
    type Request = BayesActionRequest;
    type Return<'a> = Vec<(MemoryId, f64)>;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let cluster = request.working_mem.memory_cluster();

        cluster.read_or_compute(|mem_cluster| {
            let mut possible_actions = get_possible_actions(mem_cluster, &request.source);

            request.source.iter().for_each(|&(id, weight)| {
                let idx = mem_cluster.get_mem_index(id);

                if let Some(idx) = idx {
                    let links = mem_cluster.graph().edges_directed(idx, Outgoing);

                    for link in links {
                        let neighbor_idx = link.target();

                        mem_cluster
                            .graph()
                            .node_weight(neighbor_idx)
                            .map(|embed_note| {
                                let note_id = embed_note.note().id();
                                let link_type = link.weight().link_type();

                                if let MemoryType::Procedure(_) = embed_note.note().mem_type()
                                    && let MemoryLinkType::Proc(link_weight) = link_type
                                {
                                    match link_weight {
                                        ProcMemLink::TrigToAction(TrigToAction {
                                            prob, ..
                                        }) => {
                                            possible_actions
                                                .get_mut(&note_id)
                                                .map(|v| *v += prob * weight);
                                        }
                                    }
                                }
                            });
                    }
                }
            });

            let mut actions_vec = possible_actions.into_iter().collect::<Vec<_>>();
            actions_vec.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

            actions_vec.into_iter().take(request.top_k).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::MemoryLink;
    use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
    use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};
    use soul_mem_core::memory_note::{
        MemoryNoteBuilder, MemoryType,
        sem_mem::{ConceptType, SemMemory},
    };
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;

    fn create_mock_working_memory_with_actions() -> (WorkingMemory, MemoryId, MemoryId) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let action_id = MemoryId::new();

        let proc_link = ProcMemLink::TrigToAction(TrigToAction::new(0.5));
        let link_type = MemoryLinkType::Proc(proc_link);
        let source_link = MemoryLink::new(source_id, action_id, link_type);

        cluster.write(|c| {
            let source_mem_type = MemoryType::Semantic(SemMemory {
                content: "Source Memory".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            });
            let source_note = MemoryNoteBuilder::new(source_mem_type)
                .id(source_id)
                .mem_links(vec![source_link])
                .build()
                .unwrap();
            let source_embedding = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_embedding,
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

        (wm, source_id, action_id)
    }

    #[test]
    fn test_retr_bayes_action_basic() {
        let (wm, source_id, action_id) = create_mock_working_memory_with_actions();
        let request = BayesActionRequest::new(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrBayesAction {}.retrieve(request);

        assert!(!result.is_empty());
        let (id, score) = &result[0];
        assert_eq!(id, &action_id);
        assert!(*score > 0.0);
    }

    #[test]
    fn test_retr_bayes_action_empty_source() {
        let (wm, _, _) = create_mock_working_memory_with_actions();
        let request = BayesActionRequest::new(Arc::new(wm), vec![]);
        let result = RetrBayesAction {}.retrieve(request);

        assert!(result.is_empty());
    }

    #[test]
    fn test_retr_bayes_action_top_k() {
        let (wm, source_id, _) = create_mock_working_memory_with_actions();
        let request = BayesActionRequest::new(Arc::new(wm), vec![(source_id, 1.0)]).with_top_k(1);
        let result = RetrBayesAction {}.retrieve(request);

        assert!(result.len() <= 1);
    }

    #[test]
    fn test_get_possible_actions() {
        let (wm, source_id, action_id) = create_mock_working_memory_with_actions();
        let cluster = wm.memory_cluster();
        let result = cluster.read_or_compute(|c| get_possible_actions(c, &[(source_id, 1.0)]));

        assert!(result.contains_key(&action_id));
    }

    #[test]
    fn test_multi_action_inference() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let action_a = MemoryId::new();
        let action_b = MemoryId::new();

        cluster.write(|c| {
            let source_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(vec![
                MemoryLink::new(
                    source_id,
                    action_a,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.7))),
                ),
                MemoryLink::new(
                    source_id,
                    action_b,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.3))),
                ),
            ])
            .build()
            .unwrap();
            let source_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_emb,
            });

            for (aid, name) in [(action_a, "ActionA"), (action_b, "ActionB")] {
                let note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new(name.into(), ActionType::new_speak()),
                )))
                .id(aid)
                .build()
                .unwrap();
                let emb = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Procedure(),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding: emb });
            }
        });

        let request = BayesActionRequest::new(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrBayesAction {}.retrieve(request);

        assert_eq!(result.len(), 2);
        let scores: Vec<f64> = result.iter().map(|(_, s)| *s).collect();
        insta::assert_debug_snapshot!(scores);
    }

    #[test]
    fn test_multi_source_action_aggregation() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let src_a = MemoryId::new();
        let src_b = MemoryId::new();
        let action_id = MemoryId::new();

        cluster.write(|c| {
            for (sid, prob) in [(src_a, 0.6f64), (src_b, 0.4f64)] {
                let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                    content: "source".into(),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                }))
                .id(sid)
                .mem_links(vec![MemoryLink::new(
                    sid,
                    action_id,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(prob))),
                )])
                .build()
                .unwrap();
                let emb = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                    )),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding: emb });
            }

            let action_note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                Action::new("AggregateAction".into(), ActionType::new_speak()),
            )))
            .id(action_id)
            .build()
            .unwrap();
            let action_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Procedure(),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: action_note,
                embedding: action_emb,
            });
        });

        let sources = vec![(src_a, 0.8), (src_b, 0.5)];
        let request = BayesActionRequest::new(Arc::new(wm), sources);
        let result = RetrBayesAction {}.retrieve(request);

        assert_eq!(result.len(), 1);
        let (id, score) = result[0];
        assert_eq!(id, action_id);
        insta::assert_debug_snapshot!(score);
    }

    #[test]
    fn test_bayes_action_prob_accuracy() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let action_id = MemoryId::new();

        cluster.write(|c| {
            let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(vec![MemoryLink::new(
                source_id,
                action_id,
                MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.5))),
            )])
            .build()
            .unwrap();
            let emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote { note, embedding: emb });

            let a_note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                Action::new("Test".into(), ActionType::new_speak()),
            )))
            .id(action_id)
            .build()
            .unwrap();
            let a_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Procedure(),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: a_note,
                embedding: a_emb,
            });
        });

        let request = BayesActionRequest::new(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrBayesAction {}.retrieve(request);

        assert!(!result.is_empty());
        let score = result[0].1;
        let expected = 0.5 * 1.0;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected {expected}, got {score}"
        );
    }
}

fn get_possible_actions(
    cluster: &MemoryCluster,
    source: &[(MemoryId, f64)],
) -> HashMap<MemoryId, f64> {
    source
        .iter()
        .filter_map(|&(id, _weight)| {
            let idx = cluster.get_mem_index(id)?;
            //同时检查邻居节点类型(Procedure)与链接类型(Proc)，
            //只有经Proc链接可达的动作节点才会指导行为，避免Sem/Situation链接
            //可达的Procedure节点以0.0分挤占top_k
            let action_neighbors = cluster
                .graph()
                .edges_directed(idx, Outgoing)
                .filter_map(|edge| {
                    let node_idx = edge.target();
                    if !matches!(edge.weight().link_type(), MemoryLinkType::Proc(_)) {
                        return None;
                    }
                    let note = cluster.graph().node_weight(node_idx)?;
                    match note.note().mem_type() {
                        MemoryType::Procedure(_) => Some((note.note().id(), 0.0)),
                        _ => None,
                    }
                });
            Some(action_neighbors)
        })
        .flatten()
        .collect()
}
