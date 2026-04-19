use std::{collections::HashMap, sync::Arc};

use petgraph::{
    Direction::Outgoing,
    visit::{EdgeRef, NodeRef},
};

use serde::Deserialize;

use crate::memory::{
    algo::retrieve::{RetrRequest, RetrStrategy},
    cluster::memory_cluster::MemoryCluster,
    memory_links::{
        MemoryLinkType,
        proc_mem::{ProcMemLink, TrigToAction},
    },
    memory_note::{MemoryId, MemoryType},
    working_memory::WorkingMemory,
};

#[derive(Debug, Clone, Deserialize)]
pub struct BayesActionConfig {
    #[serde(default = "default_action_top_k")]
    pub top_k: usize,
}

fn default_action_top_k() -> usize { 5 }

impl BayesActionConfig {
    pub fn into_request(self, working_mem: Arc<WorkingMemory>, source: Vec<(MemoryId, f64)>) -> BayesActionRequest {
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
            //收集在当前source下所有可能被触发的action
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
                                                //P(act_i) = P(act_i | situation) * P(situation)
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

fn get_possible_actions(
    cluster: &MemoryCluster,
    source: &[(MemoryId, f64)],
) -> HashMap<MemoryId, f64> {
    source
        .iter()
        .filter_map(|&(id, weight)| {
            let idx = cluster.get_mem_index(id)?;
            let action_neighbors = cluster
                .graph()
                .neighbors_directed(idx, Outgoing)
                .filter_map(|node_idx| {
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
