use std::collections::HashSet;
use crate::memory::{MemoryNote};
use crate::db::qdrant_retriever::QdrantRetriever;
use anyhow::Result;
use qdrant_client::qdrant::Filter;
use crate::llm_driver::Llm;

#[allow(dead_code)]
pub struct MemoryLongTerm {
    retriever: QdrantRetriever,
}
#[allow(dead_code)]
impl MemoryLongTerm {
    pub fn new(retriever: QdrantRetriever) -> Self {
        Self {
            retriever,
        }
    }
    ///巩固为长期记忆
    pub async fn consolidate(&self, memory_notes: impl Into<Vec<MemoryNote>>) -> Result<()> {
        //self.retriever.add_points(memory_notes.into()).await
        todo!()
    }
    /// Retrieve memory clusters recursively with depth, the original retrieve is depth 0,or None
    pub async fn retrieve(&self, queries: &[impl AsRef<str>], k: u64, depth: Option<u32>, filter: Option<impl Into<Filter> + Clone>) -> Result<Vec<MemoryNote>> {
        todo!()
    }
    pub async fn evolve(&self, memory_notes: impl Into<Vec<MemoryNote>>, llm_driver: &impl Llm){
        todo!() // 接受一个可能的相关邻居记忆列表，交给LLM判断是否进化，具体为是否需要新增，加强联系，是否要修正记忆内容，更新记忆上下文（本节点和邻居节点）
        // 邻居记忆列表来自于qdrant数据库，和本地工作记忆（记录一个“提及次数”，取最高的几项）
    }
    pub async fn clean(&self) {
        todo!()
    }
}