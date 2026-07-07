// 数据库抽象接口，也就是仓储 trait


use async_trait::async_trait;

use crate::memory::{
    memory_links::MemoryLink,
    memory_note::{MemoryId, MemoryNote},
};

use super::{
    error::StorageResult,
    model::{
        FeedbackEventRecord, MemoryLinkRecord, MemoryNoteRecord, RetrievalEventRecord,
        SimilarityHit, SimilarityQuery,
    },
};

#[async_trait]
pub trait MemoryRepository: Send + Sync {
    async fn bootstrap(&self) -> StorageResult<()>;

    async fn upsert_note(&self, note: &MemoryNote) -> StorageResult<MemoryNoteRecord>;

    async fn upsert_notes(&self, notes: &[MemoryNote]) -> StorageResult<Vec<MemoryNoteRecord>> {
        let mut records = Vec::with_capacity(notes.len());
        for note in notes {
            records.push(self.upsert_note(note).await?);
        }
        Ok(records)
    }

    async fn get_note(&self, memory_id: MemoryId) -> StorageResult<Option<MemoryNoteRecord>>;
    async fn delete_note(&self, memory_id: MemoryId) -> StorageResult<bool>;

    async fn upsert_link(&self, link: &MemoryLink) -> StorageResult<MemoryLinkRecord>;
    async fn delete_link(&self, link_id: &str) -> StorageResult<bool>;
    async fn list_outbound_links(
        &self,
        memory_id: MemoryId,
    ) -> StorageResult<Vec<MemoryLinkRecord>>;
    async fn list_inbound_links(&self, memory_id: MemoryId)
    -> StorageResult<Vec<MemoryLinkRecord>>;

    async fn append_retrieval_event(&self, event: RetrievalEventRecord) -> StorageResult<()>;
    async fn append_feedback_event(&self, event: FeedbackEventRecord) -> StorageResult<()>;

    async fn query_similar_notes(
        &self,
        query: SimilarityQuery,
    ) -> StorageResult<Vec<SimilarityHit>>;
}
