mod long_term;
mod working;
mod default_prompts;
mod probability;
mod temporary;
mod share;

use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use qdrant_client::Payload;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map};
use uuid::Uuid;
use anyhow::Result;
use fastembed::{Embedding, EmbeddingModel, TextEmbedding};
use petgraph::prelude::{EdgeIndex, NodeIndex, StableDiGraph};
use qdrant_client::qdrant::condition::ConditionOneOf::HasId;
use qdrant_client::qdrant::PointId;
use rayon::prelude::IntoParallelIterator;
use surrealdb::RecordId;
use crate::soul_embedding::CalcEmbedding;

///Stand for a link to another MemoryNote,the intensity mimics the strength of the link in human brains
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct MemoryLink {
    pub id : NodeRefId,
    pub relation: String,
    pub link_table: String,
    pub intensity: f32
}
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct GraphMemoryLink {
    pub relation: String,
    pub intensity: f32
}
impl From<MemoryLink> for GraphMemoryLink {
    fn from(link: MemoryLink) -> Self {
        GraphMemoryLink {
            relation: link.relation,
            intensity: link.intensity,
        }
    }
}



impl MemoryLink {
    pub fn new(id: impl Into<NodeRefId>, relation: Option<impl Into<String>>, link_table: impl Into<String>, intensity: f32) -> Self {
        MemoryLink {
            id: id.into(),
            relation: relation.map(|x| x.into()).unwrap_or("Simple".to_string()),
            link_table: link_table.into(),
            intensity,
        }
    }
    pub fn as_surreal_id(&self) -> RecordId {
        RecordId::from((self.link_table.as_str(),self.id.as_str()))
    }
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct MemoryEmbeddings {
    pub concept_embedding: Vec<f32>, //关键概念嵌入
    pub emotion_embedding: Vec<f32>, //情感嵌入
    pub context_embedding: Vec<f32>, //上下文情境嵌入
    pub last_updated: DateTime<Utc>, //最后更新时间
}
impl MemoryEmbeddings {
    pub fn new(
        concept_embedding: Vec<f32>,
        emotion_embedding: Vec<f32>,
        context_embedding: Vec<f32>,
    ) -> Self {
        Self {
            concept_embedding,
            emotion_embedding,
            context_embedding,
            last_updated: Utc::now(),
        }
    }
}


///The basic unit of memory
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct MemoryNote {
    pub content : String, //内容，自然文本
    mem_id: NodeRefId, //uuid
    pub keywords : Vec<String>,//关键词
    links : Vec<MemoryLink>,//链接记忆
    retrieval_count : u32,//被检索次数
    timestamp : u64,//创建时间
    last_accessed : u64,//最后访问时间
    pub context : String,//记忆情景
    evolution_history : Vec<String>,//进化历史
    pub category : String,//分类,用作Surreal db 的表名
    pub tags : Vec<String>,//标签，（认知，行为）
    pub base_emotion: String, //情感基调
}
impl MemoryNote {
    pub fn new(content: impl Into<String>) -> Self {
        let now = Utc::now().format("%Y%m%d%H%M%S").to_string().parse().unwrap_or(0);
        MemoryNote {
            content: content.into(),
            mem_id: Uuid::new_v4().into(),
            keywords: vec![],
            links: vec![],
            retrieval_count: 0,
            timestamp: now,
            last_accessed: now,
            context: String::default(),
            evolution_history: vec![],
            category: String::default(),
            tags: vec![],
            base_emotion: "无明显情感".to_string(),
        }
    }
    pub fn id(&self) -> &NodeRefId {
        &self.mem_id
    }
    pub fn surreal_id(&self) -> RecordId {
        RecordId::from((self.category.as_str(),self.mem_id.as_str()))
    }
    pub fn links(&self) -> &[MemoryLink] {
        &self.links
    }
    pub fn retrieval_count(&self) -> u32 {
        self.retrieval_count
    }
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
    pub fn last_accessed(&self) -> u64 {
        self.last_accessed
    }
    pub fn evolution_history(&self) -> &[String] {
        &self.evolution_history
    }
}
impl MemoryNote {
    pub fn set_links(&mut self, links: impl Into<Vec<MemoryLink>> ) {
        self.links = links.into();
    }
    pub fn add_link(&mut self, link: impl Into<MemoryLink>) {
        self.links.push(link.into());
    }
    pub fn append_links(&mut self, links: impl Into<Vec<MemoryLink>>) {
        self.links.append(&mut links.into());
    }
    pub fn retrieval_increment(&mut self) {
        self.retrieval_count += 1;
    }
    pub fn mark_accessed(&mut self) -> Result<()> {
        self.last_accessed = Utc::now().format("%Y%m%d%H%M%S").to_string().parse()?;
        Ok(())
    }
    pub fn append_evolution(&mut self, evolution: impl Into<String>) {
        self.evolution_history.push(evolution.into());
    }
}
impl MemoryNote {
    pub fn as_payload(&self) -> Result<Payload> {
        Ok(Payload::try_from(json!({
            "id": self.id(),
            "category": &self.category
        }))?)
    }
    pub fn get_embedding(&self, embedding_model: &TextEmbedding) -> Result<MemoryEmbeddings> {
        let embeddings = embedding_model.embed(
            vec![
                &self.keywords.join(" "),
                &self.base_emotion,
                &self.context
            ],
            None
        )?;
        let [concept, emotion, context]: [Embedding; 3] = embeddings.try_into()
            .map_err(|_| anyhow::anyhow!("Embedding error: the length doesn't match"))?;
        Ok(MemoryEmbeddings::new(concept, emotion, context))
    }
}
impl CalcEmbedding for MemoryNote {  //TODO: modify the calc embedding trait function to suit the latest embedding calc
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(vec![self.content.as_str()],None)
    }
}
impl CalcEmbedding for Vec<MemoryNote> {
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(self.iter().map(|x| x.content.as_str()).collect::<Vec<_>>(),None)
    }
}
impl CalcEmbedding for &[String] {
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(Vec::from(*self),None)
    }
}
impl CalcEmbedding for String {
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(vec![self],None)
    }
}
impl CalcEmbedding for &[&str] {
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(Vec::from(*self),None)
    }
}
impl CalcEmbedding for &str {
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(vec![self],None)
    }
}
impl CalcEmbedding for Vec<&str> {
    fn calc_embedding(&self,embedding_model: &TextEmbedding) -> Result<Vec<Embedding>> {
        embedding_model.embed(self.to_vec(),None)
    }
}

impl TryInto<Payload> for MemoryNote {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<Payload, Self::Error> {
        self.as_payload()
    }
}
impl TryInto<Payload> for &MemoryNote {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Payload, Self::Error> {
        self.as_payload()
    }
}
impl TryInto<Payload> for &mut MemoryNote {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Payload, Self::Error> {
        self.as_payload()
    }
}

impl TryFrom<Payload> for MemoryNote {
    type Error = anyhow::Error;
    fn try_from(payload: Payload) -> Result<Self, Self::Error> {
        Ok(
            serde_json::from_value(serde_json::Value::Object(
                Map::from(payload)
            ))?
        )
    }
}
impl TryFrom<HashMap<String, qdrant_client::qdrant::Value>> for MemoryNote {
    type Error = anyhow::Error;
    fn try_from(payload: HashMap<String, qdrant_client::qdrant::Value>) -> Result<Self, Self::Error> {
        Ok(
            serde_json::from_value(serde_json::Value::Object(
                Map::from(Payload::from(payload))
            ))?
        )
    }
}
#[derive(Debug, Clone,Serialize,Deserialize,Default,PartialOrd,PartialEq,Eq,Hash,Ord)]
pub struct NodeRefId(Arc<str>);
impl NodeRefId {
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl AsRef<str> for NodeRefId {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}
impl From<String> for NodeRefId {
    fn from(s: String) -> Self {
        Self(s.into())
    }
}
impl From<&str> for NodeRefId {
    fn from(id: &str) -> Self {
        Self(id.into())
    }
}
impl From<Uuid> for NodeRefId {
    fn from(id: Uuid) -> Self {
        Self(id.to_string().into())
    }
}
impl Into<String> for NodeRefId {
    fn into(self) -> String {
        self.0.to_string()
    }
}
impl Into<PointId> for NodeRefId {
    fn into(self) -> PointId {
        PointId::from(self.0.as_ref())
    }
}
impl Display for NodeRefId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

//TODO: test it
pub struct MemoryCluster {
    graph: StableDiGraph<MemoryNote, GraphMemoryLink>,
    id_to_index: HashMap<NodeRefId, NodeIndex>,
    relation_map: HashMap<(NodeIndex,String,NodeIndex),EdgeIndex>,
    incompletely_linked_note: HashMap<NodeRefId,Vec<(NodeIndex,MemoryLink)>>, //目标节点的uuid，Vec<(源节点的index，关系)>

    embedding_store: HashMap<NodeRefId, MemoryEmbeddings>,
    embedding_model: TextEmbedding

    //由于link储存在source节点，source节点不在图中，link则不可知，因此source节点通常总是有效
}
impl MemoryCluster {
    pub fn new(embedding_model: TextEmbedding) -> Self {
        Self {
            graph: StableDiGraph::new(),
            id_to_index: HashMap::new(),
            relation_map: HashMap::new(),
            incompletely_linked_note: HashMap::new(),
            embedding_store: HashMap::new(),
            embedding_model
        }
    }
    // 获取内部图的不可变引用
    pub fn graph(&self) -> &StableDiGraph<MemoryNote, GraphMemoryLink> {
        &self.graph
    }

    // 获取内部图的可变引用
    pub fn graph_mut(&mut self) -> &mut StableDiGraph<MemoryNote, GraphMemoryLink> { //Be careful when using this
        &mut self.graph
    }
    fn add_embeddings(&mut self, node_id: NodeRefId, embeddings: MemoryEmbeddings) {
        self.embedding_store.insert(node_id, embeddings);
    }
    pub fn add_single_node(&mut self, node: MemoryNote) {
        let (id,links) = (node.id().to_owned(),node.links().to_owned());
        self.merge_node(node);
        if let Some(&node_index) = self.id_to_index.get(&id) {
            self.merge_edges(node_index,links)
        }
    }
    /// 删除单个节点，返回被删除的节点，并清理冗余项目
    pub fn remove_single_node(&mut self, node_id: NodeRefId) -> Option<MemoryNote> { //TODO: test it
        if let Some(idx) = self.id_to_index.remove(&node_id) {
            self.embedding_store.remove(&node_id);
            self.relation_map.retain(|k,_| k.0 != idx && k.2 != idx);
            self.incompletely_linked_note.values_mut().for_each(|v| {
                v.retain(|(origin_idx,_)| *origin_idx != idx)
            });
            self.graph.remove_node(idx)
        }else {
            None
        }
    }
    pub fn get_node(&self, node_id: &NodeRefId) -> Option<&MemoryNote> {
        self.id_to_index.get(node_id).and_then(|&index| self.graph.node_weight(index))
    }
    pub fn get_embedding(&self, node_id: &NodeRefId) -> Option<&MemoryEmbeddings> {
        self.embedding_store.get(node_id)
    }
    pub fn get_node_mut(&mut self, node_id: &NodeRefId) -> Option<&mut MemoryNote> {
        self.id_to_index.get(node_id).and_then(|&index| self.graph.node_weight_mut(index))
    }
    pub fn contains_node(&self, node_id: &NodeRefId) -> bool {
        if let Some(&index) = self.id_to_index.get(node_id) {
            self.graph.contains_node(index) //TODO: clean dirty index
        } else {
            false
        }
    }
    pub fn merge(&mut self, other: Vec<MemoryNote>) {
        let to_merged_edge = other.iter()
            .map(|x| (x.id().to_owned(),x.links().to_owned()))
            .collect::<Vec<_>>();

        self.merge_nodes(other);
        let to_merged_edge = to_merged_edge
            .into_iter()
            .filter_map(|(id,links)| {
                if let Some(&node_index) = self.id_to_index.get(&id) {
                    Some((node_index, links))
                }else {
                    None
                }
            })
            .collect::<Vec<_>>();
        self.merge_batch_edges(to_merged_edge);

    }
    pub fn merge_cluster(&mut self, other: MemoryCluster) {
        todo!()
    }
    fn merge_node(&mut self, node: MemoryNote) -> NodeIndex {
        let node_id = node.id().to_owned();

        match self.id_to_index.get(&node_id) {
            Some(&index) if self.graph.contains_node(index) => {
                // 节点存在且有效
                if let Some(existing_node) = self.graph.node_weight_mut(index) {
                    existing_node.retrieval_increment();
                }
                index
            }
            _ => {
                // 节点不存在或索引无效
                self.add_new_node(node)
            }
        }
    }
    fn add_new_node(&mut self, node: MemoryNote) -> NodeIndex {
        let node_id = node.id().to_owned();
        let embeddings = node.get_embedding(&self.embedding_model);
        if let Ok(embeddings) = embeddings {
            self.add_embeddings(node_id.clone(), embeddings);
        }else {
            log::warn!("Failed to get embeddings for node {node_id}");
        }

        let index = self.graph.add_node(node);

        // 清理可能存在的无效索引
        //self.id_to_index.remove(&node_id);
        self.id_to_index.insert(node_id.clone(), index);

        // 处理悬挂边
        self.process_pending_edges(&node_id);

        index
    }
    fn process_pending_edges(&mut self, node_id: &NodeRefId) {
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
    fn merge_nodes(&mut self, nodes: Vec<MemoryNote>) -> Vec<NodeIndex> {
        nodes.into_iter()
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

        let target_id = edge.id.clone();
        if let Some(&target_index) = self.id_to_index.get(&edge.id) {
            if !self.graph.contains_node(target_index) {
                self.id_to_index.remove(&target_id);
                self.add_pending_edge(target_id, (source, edge));
                return;
            }
            let relation = edge.relation.clone();
            if !self.relation_map.contains_key(&(source, relation.clone(), target_index)) {
               let edge_index = self.graph.add_edge(source, target_index, GraphMemoryLink::from(edge));
               self.relation_map.insert((source, relation, target_index), edge_index);
            }
        } else {
           self.add_pending_edge(target_id, (source, edge))
        }
    }
    fn add_pending_edge(&mut self, target_id: NodeRefId, edge: (NodeIndex, MemoryLink)) {
        self.incompletely_linked_note.entry(target_id)
            .or_default()
            .push(edge);
    }
}
impl Debug for MemoryCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryCluster")
            .field("graph", &self.graph)
            .field("id_to_index", &self.id_to_index)
            .field("incompletely_linked_note", &self.incompletely_linked_note)
            .field("relation_map", &self.relation_map)
            .finish()
    }
}


pub struct MemoryNoteBuilder {
    content: String,
    mem_id: Option<NodeRefId>,
    keywords: Option<Vec<String>>,
    links: Option<Vec<MemoryLink>>,
    retrieval_count: Option<u32>,
    timestamp: Option<u64>,
    last_accessed: Option<u64>,
    context: Option<String>,
    evolution_history: Option<Vec<String>>,
    category: Option<String>,
    tags: Option<Vec<String>>,
    base_emotion: Option<String>,
}
impl MemoryNoteBuilder {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            mem_id: None,
            keywords: None,
            links: None,
            retrieval_count: None,
            timestamp: None,
            last_accessed: None,
            context: None,
            evolution_history: None,
            category: None,
            tags: None,
            base_emotion: None,
        }
    }
    pub fn id(mut self, id: impl Into<NodeRefId>) -> Self {
        self.mem_id = Some(id.into());
        self
    }
    pub fn keywords(mut self, keywords: impl Into<Vec<String>>) -> Self {
        self.keywords = Some(keywords.into());
        self
    }
    pub fn links(mut self, links: impl Into<Vec<MemoryLink>>) -> Self {
        self.links = Some(links.into());
        self
    }
    pub fn retrieval_count(mut self, retrieval_count: impl Into<u32>) -> Self {
        self.retrieval_count = Some(retrieval_count.into());
        self
    }
    pub fn timestamp(mut self, timestamp: impl Into<u64>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }
    pub fn last_accessed(mut self, last_accessed: impl Into<u64>) -> Self {
        self.last_accessed = Some(last_accessed.into());
        self
    }
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
    pub fn evolution_history(mut self, evolution_history: impl Into<Vec<String>>) -> Self {
        self.evolution_history = Some(evolution_history.into());
        self
    }
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
    pub fn tags(mut self, tags: impl Into<Vec<String>>) -> Self {
        self.tags = Some(tags.into());
        self
    }
    pub fn base_emotion(mut self, base_emotion: impl Into<String>) -> Self {
        self.base_emotion = Some(base_emotion.into());
        self
    }
    pub fn build(self) -> MemoryNote {
        let now = Utc::now().format("%Y%m%d%H%M%S").to_string().parse().unwrap_or(0);
        MemoryNote {
            content: self.content,
            mem_id:self.mem_id.unwrap_or(Uuid::new_v4().into()),
            keywords: self.keywords.unwrap_or_default(),
            links:  self.links.unwrap_or_default(),
            retrieval_count: self.retrieval_count.unwrap_or_default(),
            timestamp: self.timestamp.unwrap_or(now),
            last_accessed: self.last_accessed.unwrap_or(now),
            context: self.context.unwrap_or_default(),
            evolution_history: self.evolution_history.unwrap_or_default(),
            category: self.category.unwrap_or_default(),
            tags: self.tags.unwrap_or_default(),
            base_emotion: self.base_emotion.unwrap_or("无明显情感".to_string()),
        }
    }
}
pub struct MemoryQuery {
    pub text: String,
    pub concept_vec: Option<Embedding>,
    pub emotion_vec: Option<Embedding>,
    pub context_vec: Option<Embedding>
}
impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            concept_vec: None,
            emotion_vec: None,
            context_vec: None,
        }
    }
    pub fn with_concept(mut self, concept: Embedding) -> Self {
        self.concept_vec = Some(concept);
        self
    }
    pub fn with_emotion(mut self, emotion: Embedding) -> Self {
        self.emotion_vec = Some(emotion);
        self
    }

    pub fn with_context(mut self, context: Embedding) -> Self {
        self.context_vec = Some(context);
        self
    }
}

mod test {
    use fastembed::InitOptions;
    #[allow(unused_imports)]
    use super::*;
    fn prepare_vec_graph() -> Vec<MemoryNote> {
        vec![
            MemoryNoteBuilder::new("test1")
                .id("test1")
                .links(
                    vec![
                        MemoryLink::new("test2", None::<String>, "test".to_string(),100.0),
                        MemoryLink::new("test3", None::<String>, "test".to_string(),100.0),
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test2")
                .id("test2")
                .links(
                    vec![
                        MemoryLink::new("test1", None::<String>, "test".to_string(),100.0),
                        MemoryLink::new("test3", None::<String>, "test".to_string(),100.0),
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test3")
                .id("test3")
                .links(
                    vec![
                        MemoryLink::new("test1", None::<String>, "test".to_string(),100.0),
                        MemoryLink::new("test2", None::<String>, "test".to_string(),100.0),
                    ]
                )
                .build(),
        ]
    }
    fn prepare_merge_graph() -> Vec<MemoryNote> {
        vec![
            MemoryNoteBuilder::new("test4")
                .id("test4")
                .links(
                    vec![
                        MemoryLink::new("test8", None::<String>, "test".to_string(),100.0),
                        MemoryLink::new("test9", None::<String>, "test".to_string(),100.0),
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test5")
                .id("test5")
                .links(
                    vec![
                        MemoryLink::new("test4", None::<String>, "test".to_string(),100.0),
                        MemoryLink::new("test6", None::<String>, "test".to_string(),100.0),
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test6")
                .id("test6")
                .links(
                    vec![
                        MemoryLink::new("test10", None::<String>, "test".to_string(),100.0),
                        MemoryLink::new("test11", None::<String>, "test".to_string(),100.0),
                    ]
                )
                .build(),
        ]
    }
    fn prepare_merge_graph_2() -> Vec<MemoryNote> {
        vec![
            MemoryNoteBuilder::new("test8")
                .id("test8")
                .build(),
            MemoryNoteBuilder::new("test9")
                .id("test9")
                .build(),
            MemoryNoteBuilder::new("test10")
                .id("test10")
                .build(),
            MemoryNoteBuilder::new("test11")
                .id("test11")
                .build(),
        ]
     }
    #[test]
    fn test_memory_note_create_simple() {
        let memory_note = MemoryNote::new("test");
        assert_eq!(memory_note.content, "test");
    }
    #[test]
    fn test_memory_note_create_builder() {
        let memory_note = MemoryNoteBuilder::new("test")
            .id("test_id")
            .keywords(vec!["test".to_string()])
            .links(vec![MemoryLink::new("test", None::<String>, "test".to_string(),100.0)])
            .retrieval_count(6u32)
            .timestamp(2025u64)
            .last_accessed(2025u64)
            .context("test")
            .evolution_history(vec!["test".to_string()])
            .category("test")
            .tags(vec!["test".to_string()])
            .build();

        assert_eq!(memory_note.content, "test");
        assert_eq!(memory_note.mem_id.as_str(), "test_id");
        assert_eq!(memory_note.keywords, vec!["test".to_string()]);
        assert_eq!(memory_note.links, vec![MemoryLink::new("test", None::<String>,"test".to_string(),100.0)]);
        assert_eq!(memory_note.retrieval_count, 6u32);
        assert_eq!(memory_note.timestamp, 2025u64);
        assert_eq!(memory_note.last_accessed, 2025u64);
        assert_eq!(memory_note.context, "test");
        assert_eq!(memory_note.evolution_history, vec!["test".to_string()]);
        assert_eq!(memory_note.category, "test");
        assert_eq!(memory_note.tags, vec!["test".to_string()]);
    }
    #[test]
    fn test_memory_cluster_create() {
        let mem_vec = prepare_vec_graph();
        let mut cluster = MemoryCluster::new(
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
            ).unwrap()
        );
        cluster.merge(mem_vec);
        assert_eq!(cluster.graph.node_count(),3);
        assert_eq!(cluster.graph.edge_count(),6);
        println!("{:?}",cluster);
    }
    #[test]
    fn test_memory_merge() {
        let mem_vec = prepare_vec_graph();
        let mem_merge = prepare_merge_graph();
        let mut cluster = MemoryCluster::new(
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
            ).unwrap()
        );
        cluster.merge(mem_vec);
        cluster.merge(mem_merge);
        assert_eq!(cluster.graph.node_count(),6);
        assert_eq!(cluster.graph.edge_count(),8);
        assert_eq!(cluster.incompletely_linked_note.len(), 4);
        assert_eq!(cluster.id_to_index.len(),6);
        assert_eq!(cluster.relation_map.len(),8);
        let mem_merge_2 = prepare_merge_graph_2();
        cluster.merge(mem_merge_2);
        assert_eq!(cluster.graph.node_count(),10);
        assert_eq!(cluster.graph.edge_count(),12);
        assert_eq!(cluster.incompletely_linked_note.len(), 0);
        assert_eq!(cluster.id_to_index.len(),10);
        assert_eq!(cluster.relation_map.len(),12);
        println!("{:?}",cluster);

    }
}