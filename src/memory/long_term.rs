use std::collections::HashSet;
use crate::memory::{MemoryNote};
use crate::qdrant_retriever::QdrantRetriever;
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
    pub async fn consolidate(&self, memory_notes: impl Into<Vec<MemoryNote>>) -> Result<()> {
        self.retriever.add_points(memory_notes.into()).await
    }
    /// Retrieve memory clusters recursively with depth, the original retrieve is depth 0,or None
    pub async fn retrieve(&self, queries: &[impl AsRef<str>], k: u64, depth: Option<u32>, filter: Option<impl Into<Filter> + Clone>) -> Result<Vec<MemoryNote>> {
        //获取并解析原始提取
        let (_,response) = self.retriever.query(queries, k, filter).await?;
        let mut response = QdrantRetriever::parse_response_to_notes(response)
            .into_iter()
            .flatten()
            .collect::<Vec<MemoryNote>>();

        let depth = match depth {
            Some(0) | None => return Ok(response),
            Some( d) => d,
        };
        
        let mut visited = HashSet::new();
        let mut next_ids = Vec::new();
        for note in &response {
            if visited.insert(note.id.clone()) {
                next_ids.extend(note.links().iter().map(|link| link.id.clone()));
            }
        }
        
        let mut recursive_results = Vec::new();
        
        for _ in 0..depth {
            if next_ids.is_empty() {
                break;
            }
            //递归查询
            let notes = self.retriever.query_by_ids(&next_ids).await?;
            let mut current_notes = QdrantRetriever::parse_response_to_notes(notes)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            next_ids.clear();
            //将结果进行展开
            for note in &current_notes {
                if visited.insert(note.id.clone()) {
                    next_ids.extend(note.links().iter().map(|link| link.id.clone()));
                }
            }
            recursive_results.append(&mut current_notes);
        }
        //将最终结果合并
        response.append(&mut recursive_results);
        Ok(response)
    }
    pub async fn evolve(&self, memory_notes: impl Into<Vec<MemoryNote>>, llm_driver: &impl Llm){
        todo!() // 接受一个可能的相关邻居记忆列表，交给LLM判断是否进化，具体为是否需要新增，加强联系，是否要修正记忆内容，更新记忆上下文（本节点和邻居节点）
        // 邻居记忆列表来自于qdrant数据库，和本地工作记忆（记录一个“提及次数”，取最高的几项）
    }
    pub async fn clean(&self) {
        todo!()
    }
}