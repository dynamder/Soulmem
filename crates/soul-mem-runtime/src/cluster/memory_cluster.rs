use parking_lot::RwLock;
use chrono::{DateTime, Utc};
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

use soul_mem_core::memory_links::MemoryLink;

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
    incompletely_linked_note: HashMap<MemoryId, Vec<(MemoryId, MemoryLink)>>, //鐩爣鑺傜偣鐨剈uid锛孷ec<(婧愯妭鐐圭殑uuid锛屽叧绯?>锛屽瓨uuid鑰岄潪NodeIndex锛岄伩鍏峱etgraph绱㈠紩澶嶇敤瀵艰嚧杩為敊鑺傜偣
                                                                              //embedding_store: HashMap<MemoryId, MemoryEmbedding>, //鐢变簬link鍌ㄥ瓨鍦╯ource鑺傜偣锛宻ource鑺傜偣涓嶅湪鍥句腑锛宭ink鍒欎笉鍙煡锛屽洜姝ource鑺傜偣閫氬父鎬绘槸鏈夋晥
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
    // 鑾峰彇鍐呴儴鍥剧殑涓嶅彲鍙樺紩鐢?
    pub fn graph(&self) -> &StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink> {
        &self.graph
    }

    // 鑾峰彇鍐呴儴鍥剧殑鍙彉寮曠敤
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
    /// 鍦ㄧ洿鎺ヤ慨鏀硅妭鐐圭殑杩炴帴鍚庯紝蹇呴』璋冪敤姝ゆ柟娉?
    pub fn refresh_node(&mut self, node: &MemoryId) {
        if let Some(node_index) = self.mem_id_to_index.get(node)
            && let Some(node) = self.graph.node_weight(*node_index)
        {
            self.merge_edges(*node_index, node.note.links().to_owned());
        }
    }
    /// 鍒犻櫎鍗曚釜鑺傜偣锛岃繑鍥炶鍒犻櫎鐨勮妭鐐癸紝骞舵竻鐞嗗啑浣欓」鐩紝娣诲姞pending杈?
    pub fn remove_single_node(&mut self, node_id: MemoryId) -> Option<EmbeddedMemoryNote> {
        //TODO: test it
        if let Some(idx) = self.mem_id_to_index.remove(&node_id) {
            //self.embedding_store.remove(&node_id);
            //娓呯悊鎵€鏈塸ending鐨勮竟涓紝婧愯妭鐐规槸node_id鐨勯」
            self.incompletely_linked_note
                .values_mut()
                .for_each(|v| v.retain(|(origin_id, _)| *origin_id != node_id));

            //鍥犱负鍒犻櫎浜唍ode_id鑺傜偣锛屽師鏉ュ凡缁忓缓绔嬬殑閾炬帴锛屽彲鑳戒細涓㈠け锛屽皢Incoming鐨勯摼鎺ュ姞鍏ending杈?
            // 杩欓噷浼间箮鎬ц兘鐪嬭捣鏉ヤ笉鏄緢濂斤紝涓嶈繃鍏堣繖鏍蜂簡锛屽悗缁啀璇?remove鎿嶄綔鏈韩涓嶄細闈炲父棰戠箒
            let incoming_neighbors = self
                .graph
                .edges_directed(idx, Direction::Incoming)
                .map(|edge_ref| {
                    //SAFEUNWRAP: 浠ヤ笅鐨剈nwrap鏄畨鍏ㄧ殑锛屽洜涓篹dge_ref涓殑source鍜宼arget鍦ㄨ繖涓椂闂寸偣鎬诲瓨鍦?
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
                    let mem_link = MemoryLink::new(
                        source_id,
                        target_id,
                        edge_ref.weight().to_owned().link_type,
                    );
                    (source_id, mem_link)
                })
                .collect::<Vec<_>>();

            self.incompletely_linked_note
                .insert(node_id, incoming_neighbors);
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
                // 鑺傜偣瀛樺湪涓旀湁鏁堛€傛敞鎰忥細鑺傜偣閲嶅姞涓嶇畻妫€绱紝妫€绱㈣鏁扮粺涓€鐢?
                // WorkingMemory::record_retrieval缁存姢Record.retrieval_count锛?
                // 杩欓噷涓嶅啀閫掑note鐨剅etrieval_count锛岄伩鍏嶅弻璁℃暟婕傜Щ
                index
            }
            _ => {
                // 鑺傜偣涓嶅瓨鍦ㄦ垨绱㈠紩鏃犳晥
                self.add_new_node(embed_node)
            }
        }
    }
    fn add_new_node(&mut self, embed_node: EmbeddedMemoryNote) -> NodeIndex {
        let node_id = embed_node.note().id();

        let index = self.graph.add_node(embed_node);

        // 娓呯悊鍙兘瀛樺湪鐨勬棤鏁堢储寮?
        //self.id_to_index.remove(&node_id);
        self.mem_id_to_index.insert(node_id, index);

        // 澶勭悊鎮寕杈?
        self.process_pending_edges(&node_id);

        index
    }
    fn process_pending_edges(&mut self, node_id: &MemoryId) {
        if let Some(pending_edges) = self.incompletely_linked_note.remove(node_id) {
            for (source_id, edge) in pending_edges {
                //閲嶆柊瑙ｆ瀽婧愯妭鐐圭储寮曪紝骞舵牎楠岀储寮曚笂鐨勮妭鐐筰d涓庨鏈熶竴鑷达紝闃叉petgraph绱㈠紩澶嶇敤瀵艰嚧杩為敊鑺傜偣
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
        //pending杈瑰瓨婧愯妭鐐箄uid锛岄伩鍏峃odeIndex琚鐢ㄥ悗杩為敊鑺傜偣
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

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{MemoryEmbedding, MemoryEmbeddingVariant};
    use soul_mem_query::embedding::sem::SemanticEmbedding;

    fn mock_node(id: MemoryId) -> EmbeddedMemoryNote {
        let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: "node".to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: String::new(),
        }))
        .id(id)
        .build()
        .unwrap();
        let embedding = MemoryEmbedding::new(
            EmbeddingVec::zero(4),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    #[test]
    fn test_refresh_node_merges_new_links() {
        let handle = MemoryCluster::new().into_handle();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let link_ab = MemoryLink::new(
            a,
            b,
            MemoryLinkType::Sem(SemMemLink::new("relates".to_string(), 1.0)),
        );
        let link_ac = MemoryLink::new(
            a,
            c,
            MemoryLinkType::Sem(SemMemLink::new("relates".to_string(), 1.0)),
        );

        handle.write(|cluster| {
            let mut node_a = mock_node(a);
            node_a.note.links_mut().push(link_ab.clone());
            cluster.add_single_node(node_a);
            cluster.add_single_node(mock_node(b));
            cluster.add_single_node(mock_node(c));
        });
        assert!(handle.read_or_compute(|cluster| cluster.has_edge(link_ab.id())));

        // 修改 A 的链接后必须调用 refresh_node 才会合并新边
        handle.write(|cluster| {
            cluster
                .get_node_mut(a)
                .unwrap()
                .note
                .links_mut()
                .push(link_ac.clone());
            cluster.refresh_node(&a);
        });
        assert!(handle.read_or_compute(|cluster| cluster.has_edge(link_ac.id())));
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
//     pub relation: Option<Vec<String>>, //TODO: 鏈疄鐜?
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

#[cfg(test)]
mod tests2 {
    use super::*;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_query::embedding::note::MemoryEmbeddingVariant;
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::EmbeddingVec;

    fn sem_note(content: &str) -> EmbeddedMemoryNote {
        let mem_type = MemoryType::Semantic(SemMemory {
            content: content.to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: String::new(),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding = MemoryEmbedding::new(
            EmbeddingVec::zero(4),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    fn sem_link(from: MemoryId, to: MemoryId) -> MemoryLink {
        MemoryLink::from_tuple(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 0.9)),
            0.9,
        )
    }

    fn note_with_links(id: MemoryId, content: &str, links: Vec<MemoryLink>) -> EmbeddedMemoryNote {
        let mem_type = MemoryType::Semantic(SemMemory {
            content: content.to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: String::new(),
        });
        let note = MemoryNoteBuilder::new(mem_type)
            .id(id)
            .mem_links(links)
            .build()
            .unwrap();
        let embedding = MemoryEmbedding::new(
            EmbeddingVec::zero(4),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    #[test]
    fn test_add_single_node_and_lookup() {
        let mut cluster = MemoryCluster::new();
        let node = sem_note("A");
        let id = node.note().id();
        cluster.add_single_node(node);
        assert!(cluster.contains_node(id));
        assert!(cluster.get_node(id).is_some());
        assert!(cluster.get_embedding(id).is_some());
        assert_eq!(cluster.graph().node_count(), 1);
    }

    #[test]
    fn test_add_single_node_deduplicates() {
        let mut cluster = MemoryCluster::new();
        let node1 = sem_note("A");
        let id = node1.note().id();
        cluster.add_single_node(node1);
        let duplicate = note_with_links(id, "A", vec![]);
        cluster.add_single_node(duplicate);
        assert_eq!(cluster.graph().node_count(), 1);
        assert!(cluster.contains_node(id));
    }

    #[test]
    fn test_merge_creates_edges() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        let node_a_with_link = note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]);
        cluster.add_single_node(node_a_with_link);
        cluster.add_single_node(node_b);
        assert_eq!(cluster.graph().node_count(), 2);
        // A->B 一条边
        assert_eq!(cluster.graph().edge_count(), 1);
        let linked_edges = cluster
            .get_all_linked_edges(id_a)
            .expect("linked edges")
            .collect::<Vec<_>>();
        assert_eq!(linked_edges.len(), 1);
        // 返回的必须是真实存在的 link id
        assert!(cluster.has_edge(linked_edges[0]));
    }

    #[test]
    fn test_merge_handles_pending_edges() {
        // B 尚未加入时，A->B 的边进入 pending；随后加入 B 应补建边
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();

        let node_a_with_link = note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]);
        cluster.add_single_node(node_a_with_link);
        // B 未加入 → 边进入 incompletely_linked_note
        assert_eq!(cluster.graph().edge_count(), 0);

        cluster.add_single_node(node_b);
        assert_eq!(cluster.graph().edge_count(), 1);
    }

    #[test]
    fn test_merge_does_not_duplicate_edges() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        let link = sem_link(id_a, id_b);
        let link_id = link.id();

        let a1 = note_with_links(id_a, "A", vec![link.clone()]);
        cluster.add_single_node(a1);
        // 同一 link_id 再次 merge 不应产生第二条边
        cluster.add_single_node(node_b);
        assert_eq!(cluster.graph().edge_count(), 1);
        cluster.merge_edge(cluster.get_mem_index(id_a).unwrap(), link.clone());
        assert_eq!(cluster.graph().edge_count(), 1);
        assert!(cluster.has_edge(link_id));
    }

    #[test]
    fn test_merge_batch() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        let node_c = sem_note("C");
        let id_c = node_c.note().id();

        let a = note_with_links(
            id_a,
            "A",
            vec![sem_link(id_a, id_b), sem_link(id_a, id_c)],
        );
        let b = note_with_links(id_b, "B", vec![sem_link(id_b, id_c)]);
        cluster.merge(vec![a, b, node_c]);
        assert_eq!(cluster.graph().node_count(), 3);
        assert_eq!(cluster.graph().edge_count(), 3);
    }

    #[test]
    fn test_remove_single_node() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
        cluster.add_single_node(node_b);

        let removed = cluster.remove_single_node(id_b);
        assert!(removed.is_some());
        assert!(!cluster.contains_node(id_b));
        assert_eq!(cluster.graph().node_count(), 1);
        // 删除后入边应转为 pending，不再存在于图中
        assert_eq!(cluster.graph().edge_count(), 0);
        assert!(cluster.incompletely_linked_note.contains_key(&id_b));
    }

    #[test]
    fn test_remove_nonexistent_node() {
        let mut cluster = MemoryCluster::new();
        let result = cluster.remove_single_node(MemoryId::new());
        assert!(result.is_none());
    }

    #[test]
    fn test_has_edge_and_link_index() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        let link = sem_link(id_a, id_b);
        let link_id = link.id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![link]));
        cluster.add_single_node(node_b);
        assert!(cluster.has_edge(link_id));
        assert!(cluster.get_link_index(link_id).is_some());
    }

    #[test]
    fn test_sub_cluster_add_node() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
        cluster.add_single_node(node_b);

        let mut sub = cluster.sub_cluster(
            HashSet::from([id_a]),
            HashSet::new(),
        );
        assert!(sub.add_node(id_a).is_ok());
        assert!(sub.add_node(MemoryId::new()).is_err());
    }

    #[test]
    fn test_sub_cluster_add_nodes() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        cluster.add_single_node(node_a);

        let mut sub = cluster.sub_cluster(
            HashSet::new(),
            HashSet::new(),
        );
        assert!(sub.add_nodes(&[id_a]).is_ok());
        let missing = MemoryId::new();
        assert!(sub.add_nodes(&[id_a, missing]).is_err());
        assert_eq!(sub.super_cluster().graph().node_count(), 1);
    }

    #[test]
    fn test_refresh_node() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
        cluster.add_single_node(node_b);
        cluster.refresh_node(&id_a);
        assert_eq!(cluster.graph().edge_count(), 1);
    }

    #[test]
    fn test_refresh_node_adds_missing_edges() {
        // 直接修改图节点上的 links，再调用 refresh_node 应补建缺失的边
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        let node_c = sem_note("C");
        let id_c = node_c.note().id();
        // A 初始无 links
        cluster.add_single_node(note_with_links(id_a, "A", vec![]));
        cluster.add_single_node(node_b);
        cluster.add_single_node(node_c);
        assert_eq!(cluster.graph().edge_count(), 0);

        // 直接通过 get_node_mut 在图中给 A 增加一条 link（绕过 add 接口）
        let link_b = sem_link(id_a, id_b);
        let link_c = sem_link(id_a, id_c);
        if let Some(node) = cluster.get_node_mut(id_a) {
            // 构造新的 MemoryNote（含 links）替换节点
            *node = note_with_links(id_a, "A", vec![link_b, link_c]);
        }

        cluster.refresh_node(&id_a);
        assert_eq!(cluster.graph().edge_count(), 2);
    }

    #[test]
    fn test_get_node_mut() {
        let mut cluster = MemoryCluster::new();
        let node = sem_note("A");
        let id = node.note().id();
        cluster.add_single_node(node);
        let node_mut = cluster.get_node_mut(id).expect("node exists");
        assert_eq!(node_mut.note().id(), id);
        assert!(cluster.get_node_mut(MemoryId::new()).is_none());
    }

    #[test]
    fn test_remove_single_node_cleans_pending_edges_from_source() {
        // A 指向 B，但 B 尚未加入 → (A→B) 进入 pending（origin=A）
        // 删除 A 后，该 pending 边应被清除
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
        assert!(cluster.incompletely_linked_note.contains_key(&id_b));

        cluster.remove_single_node(id_a);
        // pending 列表中 origin 为 A 的边已被 retain 清除
        let pending = cluster
            .incompletely_linked_note
            .get(&id_b)
            .map(|v| v.clone())
            .unwrap_or_default();
        assert!(
            pending.iter().all(|(origin, _)| *origin != id_a),
            "pending edges from removed source should be cleaned: {pending:?}"
        );
    }

    #[test]
    fn test_get_directed_linked_edges() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
        cluster.add_single_node(node_b);

        let outgoing = cluster.get_directed_linked_edges(id_a, petgraph::Direction::Outgoing);
        assert!(outgoing.is_some());
        assert_eq!(outgoing.unwrap().count(), 1);
        assert!(cluster.get_directed_linked_edges(MemoryId::new(), petgraph::Direction::Outgoing).is_none());
    }

    #[test]
    fn test_graph_mut_roundtrip() {
        let mut cluster = MemoryCluster::new();
        let node = sem_note("A");
        let id = node.note().id();
        cluster.add_single_node(node);
        {
            let graph = cluster.graph_mut();
            assert_eq!(graph.node_count(), 1);
        }
        assert_eq!(cluster.graph().node_count(), 1);
        assert!(cluster.get_mem_index(id).is_some());
    }

    #[test]
    fn test_graph_memory_link_intensity_roundtrip() {
        let mut cluster = MemoryCluster::new();
        let node_a = sem_note("A");
        let id_a = node_a.note().id();
        let node_b = sem_note("B");
        let id_b = node_b.note().id();
        let link = sem_link(id_a, id_b);
        let link_id = link.id();
        cluster.add_single_node(note_with_links(id_a, "A", vec![link]));
        cluster.add_single_node(node_b);

        let edge_index = cluster.get_link_index(link_id).expect("edge index exists");
        let graph_link = cluster.graph().edge_weight(edge_index).expect("edge weight");
        assert_eq!(graph_link.intensity(), 0.9);
        assert_eq!(graph_link.id(), link_id);
        assert!(matches!(graph_link.link_type(), MemoryLinkType::Sem(_)));
    }

    #[test]
    fn test_get_indexes_none_for_missing() {
        let cluster = MemoryCluster::new();
        assert!(cluster.get_mem_index(MemoryId::new()).is_none());
        assert!(cluster.get_link_index(LinkId::new()).is_none());
        assert!(!cluster.has_edge(LinkId::new()));
    }

    #[test]
    fn test_cluster_error_not_implemented() {
        let mut cluster = MemoryCluster::new();
        let other = MemoryCluster::new();
        let result = cluster.merge_cluster(other);
        assert!(matches!(
            result,
            Err(ClusterError::NotImplemented(_))
        ));
    }

    #[test]
    fn test_cluster_debug_format() {
        let mut cluster = MemoryCluster::new();
        let node = sem_note("A");
        let id = node.note().id();
        cluster.add_single_node(node);
        let debug = format!("{:?}", cluster);
        assert!(debug.contains("MemoryCluster"), "debug was: {debug}");
        assert!(!debug.is_empty());
    }
}

