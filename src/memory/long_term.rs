use std::collections::HashSet;
use crate::memory::{BatchLTMemoryQuery, BatchLTQueryType, MemoryNote};
use crate::db::hybrid_retriever::HybridRetriever;
use anyhow::Result;
use qdrant_client::qdrant::Filter;
use crate::db::surreal_retriever::NeighborQueryBuilder;
use crate::llm_driver::Llm;

#[allow(dead_code)]
pub struct MemoryLongTerm {
    retriever: HybridRetriever,
}
#[allow(dead_code)]
impl MemoryLongTerm {
    pub fn new(retriever: HybridRetriever) -> Self {
        Self {
            retriever,
        }
    }
    ///巩固为长期记忆
    pub async fn consolidate(&self, memory_notes: impl Into<Vec<MemoryNote>>) -> Result<()> {
        self.retriever.upsert_notes(memory_notes.into()).await
    }
    /// Retrieve memory clusters recursively with depth, the original retrieve is depth 0,or None
    //TODO: reduce the collect method call
    pub async fn retrieve(&self, queries: BatchLTMemoryQuery) -> Result<Vec<MemoryNote>> {
        let depth = queries.depth.unwrap_or(0) as usize;
        let vs_k = queries.vs_k.unwrap_or(5) as u64;
        let initial_notes = match queries.query_type {
            BatchLTQueryType::Text(texts) => {
                self.retriever.retrieve_related_notes(
                    &texts, vs_k, queries.filter
                ).await?
            }
            BatchLTQueryType::Id(ids) => {
                let mut res = Vec::new();
                res.push(
                    self.retriever.retrieve_notes_by_id(ids).await?
                );
                res
            }
            _ => return Err(anyhow::anyhow!("Invalid query type")),
        };
        if depth == 0 {
            return Ok(initial_notes.into_iter().flatten().collect());
        }
        let neighbors = self.retriever.retrieve_neighbor_notes_batch(
            initial_notes.iter().flatten()
                .map(|note| {
                    NeighborQueryBuilder::new(depth)
                        .start_node(note.surreal_id())
                        .build()
                })
                .collect()
        ).await?;
        Ok(initial_notes.into_iter().chain(neighbors.into_iter()).flatten().collect())
    }
    pub async fn evolve(&self, memory_notes: impl Into<Vec<MemoryNote>>, llm_driver: &impl Llm){
        unimplemented!("implement in the next version") // 接受一个可能的相关邻居记忆列表，交给LLM判断是否进化，具体为是否需要新增，加强联系，是否要修正记忆内容，更新记忆上下文（本节点和邻居节点）
        // 邻居记忆列表来自于qdrant数据库，和本地工作记忆（记录一个“提及次数”，取最高的几项）
    }
    pub async fn clean(&self) {
        unimplemented!("implement in the next version")
    }
    pub async fn integration(&self) {
        unimplemented!("implement in the next version")
    }
}