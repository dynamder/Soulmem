// 数据库抽象接口，仓储 trait

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use soul_mem_core::{
    memory_links::MemoryLink,
    memory_note::{MemoryId, MemoryNote},
};

use super::{
    error::StorageResult,
    model::{
        ConsolidationBatchResult, EventStats, EventWindow, FeedbackEventRecord, MemoryLinkRecord,
        MemoryNoteRecord, RetrievalEventRecord, SimilarityHit, SimilarityQuery,
    },
};

#[async_trait]
pub trait MemoryRepository: Send + Sync {
    async fn bootstrap(&self) -> StorageResult<()>;

    async fn upsert_note(&self, note: &MemoryNote) -> StorageResult<MemoryNoteRecord>;

    async fn save_note_bundle(
        &self,
        note: &MemoryNote,
        embedding: Vec<f32>,
    ) -> StorageResult<MemoryNoteRecord>;

    async fn save_note_bundles(
        &self,
        bundles: &[(MemoryNote, Vec<f32>)],
    ) -> StorageResult<Vec<MemoryNoteRecord>>;

    async fn save_consolidation_batch(
        &self,
        bundles: &[(MemoryNote, Vec<f32>)],
        links: &[MemoryLink],
    ) -> StorageResult<ConsolidationBatchResult>;

    async fn find_note_by_content(
        &self,
        note: &MemoryNote,
    ) -> StorageResult<Option<MemoryNoteRecord>>;

    async fn upsert_notes(&self, notes: &[MemoryNote]) -> StorageResult<Vec<MemoryNoteRecord>>;

    async fn get_note(&self, memory_id: MemoryId) -> StorageResult<Option<MemoryNoteRecord>>;
    async fn load_note(&self, memory_id: MemoryId) -> StorageResult<Option<MemoryNote>>;
    async fn delete_note(&self, memory_id: MemoryId) -> StorageResult<bool>;
    async fn set_note_embedding(
        &self,
        memory_id: MemoryId,
        embedding: Vec<f32>,
    ) -> StorageResult<()>;
    async fn get_note_embedding(&self, memory_id: MemoryId) -> StorageResult<Option<Vec<f32>>>;

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
    async fn list_retrieval_events(
        &self,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<Vec<RetrievalEventRecord>>;
    async fn list_feedback_events(
        &self,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<Vec<FeedbackEventRecord>>;

    async fn get_retrieval_event_stats(
        &self,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<EventStats> {
        let events = self.list_retrieval_events(memory_id, window).await?;
        Ok(event_stats_from_times(
            events.into_iter().map(|event| event.occurred_at),
        ))
    }

    async fn get_feedback_event_stats(
        &self,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<EventStats> {
        let events = self.list_feedback_events(memory_id, window).await?;
        Ok(event_stats_from_times(
            events.into_iter().map(|event| event.occurred_at),
        ))
    }

    async fn query_similar_notes(
        &self,
        query: SimilarityQuery,
    ) -> StorageResult<Vec<SimilarityHit>>;
}

fn event_stats_from_times(times: impl IntoIterator<Item = DateTime<Utc>>) -> EventStats {
    EventStats::from_timestamps(times)
}
