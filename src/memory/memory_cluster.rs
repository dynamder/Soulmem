use chrono::{DateTime, Utc};
use fastembed::{Embedding, TextEmbedding};
use petgraph::prelude::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use petgraph::{Direction, Undirected};
use rayon::prelude::IntoParallelIterator;
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::sync::Arc;
use surrealdb::RecordId;
use thiserror::Error;
use uuid::Uuid;

use crate::memory::embedding::{Embeddable, EmbeddingModel};
use crate::memory::memory_links::{LinkId, MemoryLinkType};
use crate::memory::memory_note::EmbedMemoryNote;

use super::memory_note::MemoryId;

use super::embedding::MemoryEmbedding;
use super::memory_links::MemoryLink;
use super::memory_note::MemoryNote;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphMemoryLink {
    id: LinkId,
    link_type: MemoryLinkType,
}
impl GraphMemoryLink {
    pub fn id(&self) -> LinkId {
        self.id
    }
    pub fn link_type(&self) -> &MemoryLinkType {
        &self.link_type
    }
}
impl From<MemoryLink> for GraphMemoryLink {
    fn from(link: MemoryLink) -> Self {
        GraphMemoryLink {
            id: link.id(),
            link_type: link.into_link_type(), // extract the link type
        }
    }
}

impl From<(LinkId, MemoryLinkType)> for GraphMemoryLink {
    fn from(link: (LinkId, MemoryLinkType)) -> Self {
        GraphMemoryLink {
            id: link.0,
            link_type: link.1,
        }
    }
}
#[derive(Clone)]
//TODO: test it, the embedding injection and link store has changed
pub struct MemoryCluster {
    graph: StableDiGraph<MemoryNote, GraphMemoryLink>,
    mem_id_to_index: HashMap<MemoryId, NodeIndex>,
    link_id_to_index: HashMap<LinkId, EdgeIndex>,
    incompletely_linked_note: HashMap<MemoryId, Vec<(NodeIndex, MemoryLink)>>, //目标节点的uuid，Vec<(源节点的index，关系)>
    embedding_store: HashMap<MemoryId, MemoryEmbedding>, //由于link储存在source节点，source节点不在图中，link则不可知，因此source节点通常总是有效
}
impl MemoryCluster {
    pub fn new(embedding_model: TextEmbedding) -> Self {
        Self {
            graph: StableDiGraph::new(),
            mem_id_to_index: HashMap::new(),
            link_id_to_index: HashMap::new(),
            incompletely_linked_note: HashMap::new(),
            embedding_store: HashMap::new(),
        }
    }
    // 获取内部图的不可变引用
    pub fn graph(&self) -> &StableDiGraph<MemoryNote, GraphMemoryLink> {
        &self.graph
    }

    // 获取内部图的可变引用
    pub fn graph_mut(&mut self) -> &mut StableDiGraph<MemoryNote, GraphMemoryLink> {
        //Be careful when using this
        &mut self.graph
    }
    pub fn has_edge(&self, link_id: LinkId) -> bool {
        self.link_id_to_index.contains_key(&link_id)
    }
    fn add_embeddings(&mut self, node_id: MemoryId, embeddings: MemoryEmbedding) {
        self.embedding_store.insert(node_id, embeddings);
    }
    pub fn add_single_node(&mut self, embed_node: EmbedMemoryNote) {
        let (id, links) = (embed_node.note().id(), embed_node.note().links().to_owned());
        self.merge_node(embed_node);
        if let Some(&node_index) = self.mem_id_to_index.get(&id) {
            self.merge_edges(node_index, links)
        }
    }
    /// 在直接修改节点的连接后，必须调用此方法
    pub fn refresh_node(&mut self, node: &MemoryId) {
        if let Some(node_index) = self.mem_id_to_index.get(node) {
            if let Some(node) = self.graph.node_weight(*node_index) {
                self.merge_edges(*node_index, node.links().to_owned());
            }
        }
    }
    /// 删除单个节点，返回被删除的节点，并清理冗余项目
    pub fn remove_single_node(&mut self, node_id: MemoryId) -> Option<MemoryNote> {
        //TODO: test it
        if let Some(idx) = self.mem_id_to_index.remove(&node_id) {
            self.embedding_store.remove(&node_id);
            self.incompletely_linked_note
                .values_mut()
                .for_each(|v| v.retain(|(origin_idx, _)| *origin_idx != idx));
            self.graph.remove_node(idx)
        } else {
            None
        }
    }
    pub fn get_node(&self, node_id: MemoryId) -> Option<&MemoryNote> {
        self.mem_id_to_index
            .get(&node_id)
            .and_then(|&index| self.graph.node_weight(index))
    }
    pub fn get_embedding(&self, node_id: MemoryId) -> Option<&MemoryEmbedding> {
        self.embedding_store.get(&node_id)
    }
    pub fn get_node_mut(&mut self, node_id: MemoryId) -> Option<&mut MemoryNote> {
        self.mem_id_to_index
            .get(&node_id)
            .and_then(|&index| self.graph.node_weight_mut(index))
    }
    pub fn contains_node(&self, node_id: MemoryId) -> bool {
        if let Some(&index) = self.mem_id_to_index.get(&node_id) {
            self.graph.contains_node(index) //TODO: clean dirty index
        } else {
            false
        }
    }
    pub fn get_directed_linked_edges(
        &self,
        node_id: MemoryId,
        direction: Direction,
    ) -> Option<impl Iterator<Item = LinkId>> {
        if let Some(&index) = self.mem_id_to_index.get(&node_id) {
            Some(
                self.graph()
                    .edges_directed(index, direction)
                    .map(|edge| edge.weight().id()),
            )
        } else {
            None
        }
    }
    pub fn get_all_linked_edges(&self, node_id: MemoryId) -> Option<impl Iterator<Item = LinkId>> {
        if let Some(&index) = self.mem_id_to_index.get(&node_id) {
            Some(
                self.graph()
                    .edges_directed(index, Direction::Incoming)
                    .chain(self.graph().edges_directed(index, Direction::Outgoing))
                    .map(|edge| edge.weight().id()),
            )
        } else {
            None
        }
    }
    pub fn merge(&mut self, other: Vec<EmbedMemoryNote>) {
        let to_merged_edge = other
            .iter()
            .map(|x| (x.note().id(), x.note().links().to_owned()))
            .collect::<Vec<_>>();

        self.merge_nodes(other);
        let to_merged_edge = to_merged_edge
            .into_iter()
            .filter_map(|(id, links)| {
                if let Some(&node_index) = self.mem_id_to_index.get(&id) {
                    Some((node_index, links))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        self.merge_batch_edges(to_merged_edge);
    }
    pub fn merge_cluster(&mut self, _other: MemoryCluster) {
        todo!()
    }
    pub fn sub_cluster(
        &self,
        node_ids: impl Into<HashSet<MemoryId>>,
        edge_ids: impl Into<HashSet<LinkId>>,
    ) -> MemorySubCluster<'_> {
        MemorySubCluster {
            node_ids: node_ids.into(),
            edge_ids: edge_ids.into(),
            super_cluster: &self,
        }
    }
    fn merge_node(&mut self, embed_node: EmbedMemoryNote) -> NodeIndex {
        let node_id = embed_node.note().id();

        match self.mem_id_to_index.get(&node_id) {
            Some(&index) if self.graph.contains_node(index) => {
                // 节点存在且有效
                if let Some(existing_node) = self.graph.node_weight_mut(index) {
                    existing_node.retrieval_increment();
                }
                index
            }
            _ => {
                // 节点不存在或索引无效
                self.add_new_node(embed_node.into_tuple())
            }
        }
    }
    fn add_new_node(&mut self, embed_node: (MemoryNote, MemoryEmbedding)) -> NodeIndex {
        let node_id = embed_node.0.id();

        let index = self.graph.add_node(embed_node.0);

        // 清理可能存在的无效索引
        //self.id_to_index.remove(&node_id);
        self.mem_id_to_index.insert(node_id.clone(), index);

        // 处理悬挂边
        self.process_pending_edges(&node_id);

        index
    }
    fn process_pending_edges(&mut self, node_id: &MemoryId) {
        if let Some(pending_edges) = self.incompletely_linked_note.remove(node_id) {
            for (source_index, edge) in pending_edges {
                if !self.graph.contains_node(source_index) {
                    log::warn!("Attempted to add edge from invalid source node {node_id}");
                    // 处理源节点丢失的情况
                    continue;
                }
                self.merge_edge(source_index, edge);
            }
        }
    }
    fn merge_nodes(&mut self, nodes: Vec<EmbedMemoryNote>) -> Vec<NodeIndex> {
        nodes
            .into_iter()
            .map(|x| self.merge_node(x))
            .collect::<Vec<_>>()
    }
    fn merge_edges(&mut self, source: NodeIndex, edges: Vec<MemoryLink>) {
        for edge in edges {
            self.merge_edge(source, edge);
        }
    }
    fn merge_batch_edges(&mut self, edges: Vec<(NodeIndex, Vec<MemoryLink>)>) {
        for (source, edges) in edges {
            self.merge_edges(source, edges);
        }
    }
    fn merge_edge(&mut self, source: NodeIndex, edge: MemoryLink) {
        if !self.graph.contains_node(source) {
            log::warn!("Attempted to add edge from invalid source node");
            return;
        }

        let target_id = edge.to();
        let edge_id = edge.id();
        if let Some(&target_index) = self.mem_id_to_index.get(&target_id) {
            if !self.graph.contains_node(target_index) {
                self.mem_id_to_index.remove(&target_id);
                self.add_pending_edge(target_id, (source, edge));
                return;
            }
            if !self.has_edge(edge.id()) {
                let edge_index =
                    self.graph
                        .add_edge(source, target_index, GraphMemoryLink::from(edge));
                self.link_id_to_index.insert(edge_id, edge_index);
            }
        } else {
            self.add_pending_edge(target_id, (source, edge))
        }
    }
    fn add_pending_edge(&mut self, target_id: MemoryId, edge: (NodeIndex, MemoryLink)) {
        self.incompletely_linked_note
            .entry(target_id)
            .or_default()
            .push(edge);
    }
}
impl Debug for MemoryCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryCluster")
            .field("graph", &self.graph)
            .field("mem_id_to_index", &self.mem_id_to_index)
            .field("link_id_to_index", &self.link_id_to_index)
            .field("incompletely_linked_note", &self.incompletely_linked_note)
            .finish()
    }
}

pub enum LTQueryType {
    Text(String),
    Id(MemoryId),
    Embedding(Embedding),
}
pub enum BatchLTQueryType {
    Text(Vec<String>),
    Id(Vec<MemoryId>),
    Embedding(Vec<Embedding>), // TODO: 待实现
}
impl BatchLTQueryType {
    pub fn as_text(&self) -> Option<&Vec<String>> {
        match self {
            BatchLTQueryType::Text(text) => Some(text),
            _ => None,
        }
    }
    pub fn as_id(&self) -> Option<&Vec<MemoryId>> {
        match self {
            BatchLTQueryType::Id(id) => Some(id),
            _ => None,
        }
    }
}

//TODO: test it
#[derive(Debug, Clone)]
pub struct MemorySubCluster<'a> {
    node_ids: HashSet<MemoryId>,
    edge_ids: HashSet<LinkId>,
    super_cluster: &'a MemoryCluster,
}
impl<'a> MemorySubCluster<'a> {
    pub fn add_node(&mut self, mem_id: MemoryId) -> Result<(), ClusterError> {
        if !self.super_cluster.contains_node(mem_id) {
            return Err(ClusterError::NodeNotContained(mem_id));
        }
        self.node_ids.insert(mem_id);
        if let Some(edges) = self.super_cluster.get_all_linked_edges(mem_id) {
            self.edge_ids.extend(edges);
        }
        Ok(())
    }
    pub fn add_nodes(&mut self, mem_ids: &[MemoryId]) -> Result<(), Vec<ClusterError>> {
        let mut errors = Vec::with_capacity(mem_ids.len() / 2); // Initialize with half the capacity
        for mem_id in mem_ids {
            let res = self.add_node(*mem_id);
            if let Err(err) = res {
                errors.push(err);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    pub fn super_cluster(&self) -> &'a MemoryCluster {
        self.super_cluster
    }
}

#[derive(Debug, Error)]
pub enum ClusterError {
    #[error("node {0} not contained in Super.")]
    NodeNotContained(MemoryId),
    #[error("edge {0} not contained in Super.")]
    EdgeNotContained(LinkId),
    //PlaceHolder for now
}
//WARNING: Legacy Code below, maybe useful for later reuse

// pub struct LTMemoryQuery {
//     pub query_type: LTQueryType,
//     pub depth: Option<usize>,
//     pub filter: Option<qdrant_client::qdrant::Filter>,
//     pub relation: Option<Vec<String>>, //TODO: 未实现
//     pub vs_k: Option<usize>,           //vector_search_k
// }
// impl LTMemoryQuery {
//     pub fn new(query_type: LTQueryType) -> Self {
//         Self {
//             query_type,
//             depth: None,
//             filter: None,
//             relation: None,
//             vs_k: None,
//         }
//     }
//     pub fn with_depth(mut self, depth: usize) -> Self {
//         self.depth = Some(depth);
//         self
//     }
//     pub fn with_filter(mut self, filter: Filter) -> Self {
//         self.filter = Some(filter);
//         self
//     }

//     pub fn with_relation(mut self, relation: impl Into<Vec<String>>) -> Self {
//         self.relation = Some(relation.into());
//         self
//     }
//     pub fn with_vs_k(mut self, vs_k: usize) -> Self {
//         self.vs_k = Some(vs_k);
//         self
//     }
// }
// pub struct BatchLTMemoryQuery {
//     pub query_type: BatchLTQueryType,
//     pub depth: Option<usize>,
//     pub filter: Option<qdrant_client::qdrant::Filter>,
//     pub relation: Option<Vec<String>>,
//     pub vs_k: Option<usize>,
// }
// impl BatchLTMemoryQuery {
//     pub fn new(query_type: BatchLTQueryType) -> Self {
//         Self {
//             query_type,
//             depth: None,
//             filter: None,
//             relation: None,
//             vs_k: None,
//         }
//     }
//     pub fn with_depth(mut self, depth: usize) -> Self {
//         self.depth = Some(depth);
//         self
//     }
//     pub fn with_filter(mut self, filter: Filter) -> Self {
//         self.filter = Some(filter);
//         self
//     }

//     pub fn with_relation(mut self, relation: impl Into<Vec<String>>) -> Self {
//         self.relation = Some(relation.into());
//         self
//     }
//     pub fn with_vs_k(mut self, vs_k: usize) -> Self {
//         self.vs_k = Some(vs_k);
//         self
//     }
// }

mod test {
    // #[allow(unused_imports)]
    // use super::*;
    // use fastembed::InitOptions;
    // fn prepare_vec_graph() -> Vec<MemoryNote> {
    //     vec![
    //         MemoryNoteBuilder::new("test1")
    //             .id("test1")
    //             .links(vec![
    //                 MemoryLink::new("test2", None::<String>, "test".to_string(), 100.0),
    //                 MemoryLink::new("test3", None::<String>, "test".to_string(), 100.0),
    //             ])
    //             .build(),
    //         MemoryNoteBuilder::new("test2")
    //             .id("test2")
    //             .links(vec![
    //                 MemoryLink::new("test1", None::<String>, "test".to_string(), 100.0),
    //                 MemoryLink::new("test3", None::<String>, "test".to_string(), 100.0),
    //             ])
    //             .build(),
    //         MemoryNoteBuilder::new("test3")
    //             .id("test3")
    //             .links(vec![
    //                 MemoryLink::new("test1", None::<String>, "test".to_string(), 100.0),
    //                 MemoryLink::new("test2", None::<String>, "test".to_string(), 100.0),
    //             ])
    //             .build(),
    //     ]
    // }
    // fn prepare_merge_graph() -> Vec<MemoryNote> {
    //     vec![
    //         MemoryNoteBuilder::new("test4")
    //             .id("test4")
    //             .links(vec![
    //                 MemoryLink::new("test8", None::<String>, "test".to_string(), 100.0),
    //                 MemoryLink::new("test9", None::<String>, "test".to_string(), 100.0),
    //             ])
    //             .build(),
    //         MemoryNoteBuilder::new("test5")
    //             .id("test5")
    //             .links(vec![
    //                 MemoryLink::new("test4", None::<String>, "test".to_string(), 100.0),
    //                 MemoryLink::new("test6", None::<String>, "test".to_string(), 100.0),
    //             ])
    //             .build(),
    //         MemoryNoteBuilder::new("test6")
    //             .id("test6")
    //             .links(vec![
    //                 MemoryLink::new("test10", None::<String>, "test".to_string(), 100.0),
    //                 MemoryLink::new("test11", None::<String>, "test".to_string(), 100.0),
    //             ])
    //             .build(),
    //     ]
    // }
    // fn prepare_merge_graph_2() -> Vec<MemoryNote> {
    //     vec![
    //         MemoryNoteBuilder::new("test8").id("test8").build(),
    //         MemoryNoteBuilder::new("test9").id("test9").build(),
    //         MemoryNoteBuilder::new("test10").id("test10").build(),
    //         MemoryNoteBuilder::new("test11").id("test11").build(),
    //     ]
    // }
    // #[test]
    // fn test_memory_note_create_simple() {
    //     let memory_note = MemoryNote::new("test");
    //     assert_eq!(memory_note.content, "test");
    // }
    // #[test]
    // fn test_memory_note_create_builder() {
    //     let memory_note = MemoryNoteBuilder::new("test")
    //         .id("test_id")
    //         .keywords(vec!["test".to_string()])
    //         .links(vec![MemoryLink::new(
    //             "test",
    //             None::<String>,
    //             "test".to_string(),
    //             100.0,
    //         )])
    //         .retrieval_count(6u32)
    //         .timestamp(2025u64)
    //         .last_accessed(2025u64)
    //         .context("test")
    //         .evolution_history(vec!["test".to_string()])
    //         .category("test")
    //         .tags(vec!["test".to_string()])
    //         .build();

    //     assert_eq!(memory_note.content, "test");
    //     assert_eq!(memory_note.mem_id.as_str(), "test_id");
    //     assert_eq!(memory_note.keywords, vec!["test".to_string()]);
    //     assert_eq!(
    //         memory_note.links,
    //         vec![MemoryLink::new(
    //             "test",
    //             None::<String>,
    //             "test".to_string(),
    //             100.0
    //         )]
    //     );
    //     assert_eq!(memory_note.retrieval_count, 6u32);
    //     assert_eq!(memory_note.timestamp, 2025u64);
    //     assert_eq!(memory_note.last_accessed, 2025u64);
    //     assert_eq!(memory_note.context, "test");
    //     assert_eq!(memory_note.evolution_history, vec!["test".to_string()]);
    //     assert_eq!(memory_note.category, "test");
    //     assert_eq!(memory_note.tags, vec!["test".to_string()]);
    // }
    // #[test]
    // fn test_memory_cluster_create() {
    //     let mem_vec = prepare_vec_graph();
    //     let mut cluster = MemoryCluster::new(
    //         TextEmbedding::try_new(
    //             InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    //         )
    //         .unwrap(),
    //     );
    //     cluster.merge(mem_vec);
    //     assert_eq!(cluster.graph.node_count(), 3);
    //     assert_eq!(cluster.graph.edge_count(), 6);
    //     println!("{:?}", cluster);
    // }
    // #[test]
    // fn test_memory_merge() {
    //     let mem_vec = prepare_vec_graph();
    //     let mem_merge = prepare_merge_graph();
    //     let mut cluster = MemoryCluster::new(
    //         TextEmbedding::try_new(
    //             InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    //         )
    //         .unwrap(),
    //     );
    //     cluster.merge(mem_vec);
    //     cluster.merge(mem_merge);
    //     assert_eq!(cluster.graph.node_count(), 6);
    //     assert_eq!(cluster.graph.edge_count(), 8);
    //     assert_eq!(cluster.incompletely_linked_note.len(), 4);
    //     assert_eq!(cluster.id_to_index.len(), 6);
    //     assert_eq!(cluster.relation_map.len(), 8);
    //     let mem_merge_2 = prepare_merge_graph_2();
    //     cluster.merge(mem_merge_2);
    //     assert_eq!(cluster.graph.node_count(), 10);
    //     assert_eq!(cluster.graph.edge_count(), 12);
    //     assert_eq!(cluster.incompletely_linked_note.len(), 0);
    //     assert_eq!(cluster.id_to_index.len(), 10);
    //     assert_eq!(cluster.relation_map.len(), 12);
    //     println!("{:?}", cluster);
    // }
}
