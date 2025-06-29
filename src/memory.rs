mod long_term;
mod working;
mod default_prompts;
mod probability;
mod temporary;
mod share;

use std::collections::HashMap;
use chrono::Utc;
use qdrant_client::Payload;
use serde::{Deserialize, Serialize};
use serde_json::Map;
use uuid::Uuid;
use anyhow::Result;
use fastembed::{Embedding, TextEmbedding};
use petgraph::prelude::StableDiGraph;
use crate::soul_embedding::CalcEmbedding;

///Stand for a link to another MemoryNote,the intensity mimics the strength of the link in human brains
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct MemoryLink {
    pub id : String,
    pub relation: String,
    pub intensity: u32
}
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct GraphMemoryLink {
    pub relation: String,
    pub intensity: u32
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
    pub fn new(id: impl Into<String>, relation: Option<impl Into<String>>, intensity: u32) -> Self {
        MemoryLink {
            id: id.into(),
            relation: relation.map(|x| x.into()).unwrap_or("Simple".to_string()),
            intensity,
        }
    }
}

///The basic unit of memory
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct MemoryNote {
    pub content : String,
    id : String,
    pub keywords : Vec<String>,
    links : Vec<MemoryLink>,
    retrieval_count : u32,
    timestamp : u64,
    last_accessed : u64,
    pub context : String,
    evolution_history : Vec<String>,
    pub category : String,
    pub tags : Vec<String>,
}
impl MemoryNote {
    pub fn new(content: impl Into<String>) -> Self {
        let now = Utc::now().format("%Y%m%d%H%M%S").to_string().parse().unwrap_or(0);
        MemoryNote {
            content: content.into(),
            id: Uuid::new_v4().into(),
            keywords: vec![],
            links: vec![],
            retrieval_count: 0,
            timestamp: now,
            last_accessed: now,
            context: String::default(),
            evolution_history: vec![],
            category: String::default(),
            tags: vec![],
        }
    }
    pub fn id(&self) -> &str {
        self.id.as_str()
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
        if let serde_json::Value::Object(map) = serde_json::to_value(self)? {
            Ok(Payload::from(map))
        } else {
            Err(anyhow::anyhow!("Can't convert MemoryNote to Payload, Not a valid memory note"))
        }
    }
}
impl CalcEmbedding for MemoryNote {
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
pub struct MemoryCluster(pub(crate) StableDiGraph<MemoryNote, GraphMemoryLink>);
impl MemoryCluster {
    pub fn new() -> Self {
        Self(StableDiGraph::new())
    }
    // 获取内部图的不可变引用
    pub fn graph(&self) -> &StableDiGraph<MemoryNote, GraphMemoryLink> {
        &self.0
    }

    // 获取内部图的可变引用
    pub fn graph_mut(&mut self) -> &mut StableDiGraph<MemoryNote, GraphMemoryLink> {
        &mut self.0
    }
}
impl Default for MemoryCluster {
    fn default() -> Self {
        Self::new()
    }
}
impl From<Vec<MemoryNote>> for MemoryCluster {
    fn from(val: Vec<MemoryNote>) -> MemoryCluster {
        let mut cluster = MemoryCluster::new();
        let mut id_to_index = HashMap::new();
        for note in val {
            let id = note.id().to_owned();
            let index = cluster.graph_mut().add_node(note);
            id_to_index.insert(id, index);
        }
        
        let graph = cluster.graph_mut();
        let indices: Vec<_> = graph.node_indices().collect();
        
        for source_index in indices {
            let links = graph[source_index]
                .links
                .iter()
                .map(|link| (link.id.clone(), GraphMemoryLink::from(link.clone())))
                .collect::<Vec<_>>();
            for (id, graph_link) in links {
                if let Some(target_index) = id_to_index.get(&id) {
                    graph.add_edge(source_index, *target_index, graph_link);
                }else { 
                    eprintln!("Error: Node linked with id {id} not found")
                }
            }
            
        }
        cluster
    }
}

pub struct MemoryNoteBuilder {
    content: String,
    id: Option<String>,
    keywords: Option<Vec<String>>,
    links: Option<Vec<MemoryLink>>,
    retrieval_count: Option<u32>,
    timestamp: Option<u64>,
    last_accessed: Option<u64>,
    context: Option<String>,
    evolution_history: Option<Vec<String>>,
    category: Option<String>,
    tags: Option<Vec<String>>,
}
impl MemoryNoteBuilder {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            id: None,
            keywords: None,
            links: None,
            retrieval_count: None,
            timestamp: None,
            last_accessed: None,
            context: None,
            evolution_history: None,
            category: None,
            tags: None,
        }
    }
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
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
    pub fn build(self) -> MemoryNote {
        let now = Utc::now().format("%Y%m%d%H%M%S").to_string().parse().unwrap_or(0);
        MemoryNote {
            content: self.content,
            id:self.id.unwrap_or(Uuid::new_v4().into()),
            keywords: self.keywords.unwrap_or_default(),
            links:  self.links.unwrap_or_default(),
            retrieval_count: self.retrieval_count.unwrap_or_default(),
            timestamp: self.timestamp.unwrap_or(now),
            last_accessed: self.last_accessed.unwrap_or(now),
            context: self.context.unwrap_or_default(),
            evolution_history: self.evolution_history.unwrap_or_default(),
            category: self.category.unwrap_or_default(),
            tags: self.tags.unwrap_or_default(),
        }
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;
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
            .links(vec![MemoryLink::new("test", None::<String>,100)])
            .retrieval_count(6u32)
            .timestamp(2025u64)
            .last_accessed(2025u64)
            .context("test")
            .evolution_history(vec!["test".to_string()])
            .category("test")
            .tags(vec!["test".to_string()])
            .build();

        assert_eq!(memory_note.content, "test");
        assert_eq!(memory_note.id, "test_id");
        assert_eq!(memory_note.keywords, vec!["test".to_string()]);
        assert_eq!(memory_note.links, vec![MemoryLink::new("test", None::<String>,100)]);
        assert_eq!(memory_note.retrieval_count, 6u32);
        assert_eq!(memory_note.timestamp, 2025u64);
        assert_eq!(memory_note.last_accessed, 2025u64);
        assert_eq!(memory_note.context, "test");
        assert_eq!(memory_note.evolution_history, vec!["test".to_string()]);
        assert_eq!(memory_note.category, "test");
        assert_eq!(memory_note.tags, vec!["test".to_string()]);
    }
}