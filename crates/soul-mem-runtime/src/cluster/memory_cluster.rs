use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use petgraph::Direction;
use petgraph::prelude::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;

use crate::cluster::cluster_handle::MemoryClusterHandle;
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding};

use soul_mem_core::memory_links::{LinkId, MemoryLinkType};

use soul_mem_core::memory_note::MemoryId;

use soul_mem_core::memory_links::{MemoryLink, MemoryLinkBuilder};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphMemoryLink {
    id: LinkId,
    link_type: MemoryLinkType,
    intensity: f64,
    /// 遗忘缺失度（0.0 新鲜 ~ 1.0 完全遗忘），边独立衰减
    #[serde(default = "default_missing_degree")]
    missing_degree: f32,
    /// 缺失度最近一次计算的时间，用于增量更新
    #[serde(default = "default_last_forget_time")]
    last_forget_time: DateTime<Utc>,
}

/// serde 默认：缺失度初始为 0
fn default_missing_degree() -> f32 {
    0.0
}

/// serde 默认：缺失度计算时间初始为当前
fn default_last_forget_time() -> DateTime<Utc> {
    Utc::now()
}

impl GraphMemoryLink {
    pub fn id(&self) -> LinkId {
        self.id
    }
    pub fn link_type(&self) -> &MemoryLinkType {
        &self.link_type
    }
    pub fn intensity(&self) -> f64 {
        self.intensity
    }
    pub fn missing_degree(&self) -> f32 {
        self.missing_degree
    }
    pub fn set_missing_degree(&mut self, missing_degree: f32) {
        self.missing_degree = missing_degree.clamp(0.0, 1.0);
    }
    pub fn last_forget_time(&self) -> DateTime<Utc> {
        self.last_forget_time
    }
    pub fn set_last_forget_time(&mut self, time: DateTime<Utc>) {
        self.last_forget_time = time;
    }
}
impl From<MemoryLink> for GraphMemoryLink {
    fn from(link: MemoryLink) -> Self {
        let missing_degree = link.missing_degree();
        let last_forget_time = link.last_forget_time();
        GraphMemoryLink {
            id: link.id(),
            intensity: link.intensity,
            link_type: link.into_link_type(), // extract the link type
            missing_degree,
            last_forget_time,
        }
    }
}

#[derive(Clone)]
//TODO: test it, the embedding injection and link store has changed
pub struct MemoryCluster {
    graph: StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink>,
    mem_id_to_index: HashMap<MemoryId, NodeIndex>,
    link_id_to_index: HashMap<LinkId, EdgeIndex>,
    incompletely_linked_note: HashMap<MemoryId, Vec<(MemoryId, MemoryLink)>>, //目标节点的uuid，Vec<(源节点的uuid，关系)>，存uuid而非NodeIndex，避免petgraph索引复用导致连错节点
                                                                              //embedding_store: HashMap<MemoryId, MemoryEmbedding>, //由于link储存在source节点，source节点不在图中，link则不可知，因此source节点通常总是有效
}
impl Default for MemoryCluster {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCluster {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            mem_id_to_index: HashMap::new(),
            link_id_to_index: HashMap::new(),
            incompletely_linked_note: HashMap::new(),
            //embedding_store: HashMap::new(),
        }
    }
    // 获取内部图的不可变引用
    pub fn graph(&self) -> &StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink> {
        &self.graph
    }

    // 获取内部图的可变引用
    pub fn graph_mut(&mut self) -> &mut StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink> {
        //Be careful when using this
        &mut self.graph
    }

    pub fn get_mem_index(&self, id: MemoryId) -> Option<NodeIndex> {
        self.mem_id_to_index.get(&id).copied()
    }
    pub fn get_link_index(&self, link_id: LinkId) -> Option<EdgeIndex> {
        self.link_id_to_index.get(&link_id).copied()
    }

    pub fn into_handle(self) -> MemoryClusterHandle {
        MemoryClusterHandle {
            cluster: Arc::new(RwLock::new(self)),
        }
    }

    pub fn has_edge(&self, link_id: LinkId) -> bool {
        self.link_id_to_index.contains_key(&link_id)
    }
    // fn add_embeddings(&mut self, node_id: MemoryId, embeddings: MemoryEmbedding) {
    //     self.embedding_store.insert(node_id, embeddings);
    // }
    pub fn add_single_node(&mut self, embed_node: EmbeddedMemoryNote) {
        let (id, links) = (embed_node.note().id(), embed_node.note().links().to_owned());
        self.merge_node(embed_node);
        if let Some(&node_index) = self.mem_id_to_index.get(&id) {
            self.merge_edges(node_index, links)
        }
    }
    /// 在直接修改节点的连接后，必须调用此方法
    pub fn refresh_node(&mut self, node: &MemoryId) {
        if let Some(node_index) = self.mem_id_to_index.get(node)
            && let Some(node) = self.graph.node_weight(*node_index)
        {
            self.merge_edges(*node_index, node.note.links().to_owned());
        }
    }
    /// 删除单个节点，返回被删除的节点，并清理冗余项目，添加pending边
    pub fn remove_single_node(&mut self, node_id: MemoryId) -> Option<EmbeddedMemoryNote> {
        //TODO: test it
        if let Some(idx) = self.mem_id_to_index.remove(&node_id) {
            //self.embedding_store.remove(&node_id);
            //清理所有pending的边中，源节点是node_id的项
            self.incompletely_linked_note
                .values_mut()
                .for_each(|v| v.retain(|(origin_id, _)| *origin_id != node_id));

            //因为删除了node_id节点，原来已经建立的链接，可能会丢失，将Incoming的链接加入pending边
            // 这里似乎性能看起来不是很好，不过先这样了，后续再说,remove操作本身不会非常频繁
            let incoming_neighbors = self
                .graph
                .edges_directed(idx, Direction::Incoming)
                .map(|edge_ref| {
                    //SAFEUNWRAP: 以下的unwrap是安全的，因为edge_ref中的source和target在这个时间点总存在
                    let source_id = self
                        .graph
                        .node_weight(edge_ref.source())
                        .unwrap()
                        .note()
                        .id();
                    let target_id = self
                        .graph
                        .node_weight(edge_ref.target())
                        .unwrap()
                        .note()
                        .id();
                    // 必须保留原边的 id 与全部状态：`MemoryLink::new` 会生成新的 LinkId
                    // 并把 intensity / missing_degree / last_forget_time 重置为默认值。
                    // 那样重放出来的边在 link_id_to_index 里是另一个身份，
                    // 既无法按原 id 找回（has_edge/get_link_index 失配），
                    // 也会把这条边独立累积的遗忘状态清零。
                    let mem_link = MemoryLinkBuilder::new(
                        source_id,
                        target_id,
                        edge_ref.weight().to_owned().link_type,
                    )
                    .id(edge_ref.weight().id())
                    .intensity(edge_ref.weight().intensity())
                    .missing_degree(edge_ref.weight().missing_degree())
                    .last_forget_time(edge_ref.weight().last_forget_time())
                    .build();
                    (source_id, mem_link)
                })
                .collect::<Vec<_>>();

            self.incompletely_linked_note
                .insert(node_id, incoming_neighbors);
            // graph.remove_node 会一并删除所有关联边（petgraph 文档保证：删除节点即删除其入射边），
            // 但 link_id_to_index 不会自动同步。残留条目会让 has_edge() 永久返回 true，
            // 于是 merge_edge 的 `if !self.has_edge(edge.id())` 守卫会永久跳过这条边——
            // 该链接再也无法重建（图里已无此边，note.links() 里却仍列着它）。
            // 另外 petgraph 会复用被释放的索引槽位，陈旧 EdgeIndex 可能指向后来新建的边。
            // 因此在 remove_node 之前，必须先把关联边的 id 从索引表里摘除。
            let incident_link_ids: Vec<LinkId> = self
                .graph
                .edges_directed(idx, Direction::Incoming)
                .chain(self.graph.edges_directed(idx, Direction::Outgoing))
                .map(|edge_ref| edge_ref.weight().id())
                .collect();
            for link_id in incident_link_ids {
                self.link_id_to_index.remove(&link_id);
            }
            self.graph.remove_node(idx)
        } else {
            None
        }
    }
    pub fn get_node(&self, node_id: MemoryId) -> Option<&EmbeddedMemoryNote> {
        self.mem_id_to_index
            .get(&node_id)
            .and_then(|&index| self.graph.node_weight(index))
    }
    pub fn get_embedding(&self, node_id: MemoryId) -> Option<&MemoryEmbedding> {
        let idx = self.mem_id_to_index.get(&node_id)?;
        self.graph.node_weight(*idx).map(|node| &node.embedding)
    }
    pub fn get_node_mut(&mut self, node_id: MemoryId) -> Option<&mut EmbeddedMemoryNote> {
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
    pub fn merge(&mut self, other: Vec<EmbeddedMemoryNote>) {
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
    pub fn merge_cluster(&mut self, _other: MemoryCluster) -> Result<(), ClusterError> {
        Err(ClusterError::NotImplemented("merge_cluster".to_string()))
    }
    pub fn sub_cluster(
        &self,
        node_ids: impl Into<HashSet<MemoryId>>,
        edge_ids: impl Into<HashSet<LinkId>>,
    ) -> MemorySubCluster<'_> {
        MemorySubCluster {
            node_ids: node_ids.into(),
            edge_ids: edge_ids.into(),
            super_cluster: self,
        }
    }
    fn merge_node(&mut self, embed_node: EmbeddedMemoryNote) -> NodeIndex {
        let node_id = embed_node.note().id();

        match self.mem_id_to_index.get(&node_id) {
            Some(&index) if self.graph.contains_node(index) => {
                // 节点存在且有效。注意：节点重加不算检索，检索计数统一由
                // WorkingMemory::record_retrieval维护Record.retrieval_count，
                // 这里不再递增note的retrieval_count，避免双计数漂移
                index
            }
            _ => {
                // 节点不存在或索引无效
                self.add_new_node(embed_node)
            }
        }
    }
    fn add_new_node(&mut self, embed_node: EmbeddedMemoryNote) -> NodeIndex {
        let node_id = embed_node.note().id();

        let index = self.graph.add_node(embed_node);

        // 清理可能存在的无效索引
        //self.id_to_index.remove(&node_id);
        self.mem_id_to_index.insert(node_id, index);

        // 处理悬挂边
        self.process_pending_edges(&node_id);

        index
    }
    fn process_pending_edges(&mut self, node_id: &MemoryId) {
        if let Some(pending_edges) = self.incompletely_linked_note.remove(node_id) {
            for (source_id, edge) in pending_edges {
                //重新解析源节点索引，并校验索引上的节点id与预期一致，防止petgraph索引复用导致连错节点
                let Some(&source_index) = self.mem_id_to_index.get(&source_id) else {
                    log::warn!("Attempted to add edge from invalid source id {source_id}");
                    continue;
                };
                if !self.graph.contains_node(source_index) {
                    log::warn!("Attempted to add edge from invalid source node {source_id}");
                    continue;
                }
                let valid = self
                    .graph
                    .node_weight(source_index)
                    .map(|n| n.note().id() == source_id)
                    .unwrap_or(false);
                if !valid {
                    log::warn!("Source node index reused for a different id {source_id}");
                    continue;
                }
                self.merge_edge(source_index, edge);
            }
        }
    }
    fn merge_nodes(&mut self, nodes: Vec<EmbeddedMemoryNote>) -> Vec<NodeIndex> {
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
        //pending边存源节点uuid，避免NodeIndex被复用后连错节点
        let source_id = self
            .graph
            .node_weight(source)
            .map(|n| n.note().id())
            .unwrap_or(edge.from());
        if let Some(&target_index) = self.mem_id_to_index.get(&target_id) {
            if !self.graph.contains_node(target_index) {
                self.mem_id_to_index.remove(&target_id);
                self.add_pending_edge(target_id, (source_id, edge));
                return;
            }
            if !self.has_edge(edge.id()) {
                let edge_index =
                    self.graph
                        .add_edge(source, target_index, GraphMemoryLink::from(edge));
                self.link_id_to_index.insert(edge_id, edge_index);
            }
        } else {
            self.add_pending_edge(target_id, (source_id, edge))
        }
    }
    fn add_pending_edge(&mut self, target_id: MemoryId, edge: (MemoryId, MemoryLink)) {
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
    #[error("operation {0} is not implemented yet.")]
    NotImplemented(String),
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

// 测试分两个模块：`tests.rs` 是两个针对索引复用/待补边不变量的回归测试，
// `tests2.rs` 是通用的图变更套件（增删、合并、子簇、索引查询）。
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests2;
