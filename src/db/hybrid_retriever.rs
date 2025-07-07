//TODO:test it
mod config;


use super::{qdrant_retriever, surreal_retriever};
use anyhow::Result;
use qdrant_client::qdrant;
use crate::db::hybrid_retriever::config::{QdrantConfig, SurrealConfig};
use crate::memory::MemoryNote;

pub struct HybridRetriever {
    qdrant_retriever: qdrant_retriever::QdrantRetriever,
    surreal_retriever: surreal_retriever::SurrealGraphRetriever,
}

impl HybridRetriever{
    pub async fn new(qdrant_config: QdrantConfig, surreal_config: SurrealConfig) -> Result<Self> {
        Ok(HybridRetriever {
            qdrant_retriever: qdrant_retriever::QdrantRetriever::new(
                qdrant_config.collection_name(),
                Some(qdrant_config.port()),
                qdrant_config.keep_alive(),
                qdrant_config.embedding_model().to_owned(), //TODO:test this
            )?,
            surreal_retriever: surreal_retriever::SurrealGraphRetriever::new(
                surreal_config.local_address(),
                Some(surreal_config.capacity()),
            ).await?
        })
    }
    pub async fn upsert_notes(&mut self, notes: Vec<MemoryNote>) -> Result<()> {
        self.qdrant_retriever.add_points(&notes).await?;
        let surreal_res = self.surreal_retriever.upsert_notes(notes).await;
        if surreal_res.is_err() {
            return surreal_res
                .map_err(|e|e.into_iter()
                    .fold(anyhow::anyhow!("Error upserting notes to surreal:"), |acc, e| acc.context(e)))
            
        }
        Ok(())
    }
    pub async fn retrieve_related_notes(&self, queries: &[impl AsRef<str>], k: u64, filter: Option<impl Into<qdrant::Filter> + Clone>) -> Result<Vec<Vec<MemoryNote>>> {
        let raw_ids = self.qdrant_retriever.query(queries, k, filter).await?;
        let processed_ids = qdrant_retriever::QdrantRetriever::parse_response_to_record_id(raw_ids.1);
        let res = self.surreal_retriever.batch_query_ids(processed_ids).await;
        if res.is_err() {
            res
                .map_err(|e|e.into_iter()
                    .fold(anyhow::anyhow!("Error querying surreal:"), |acc, e| acc.context(e)))
        }else{
            Ok(res.unwrap())
        }
    }
    
}