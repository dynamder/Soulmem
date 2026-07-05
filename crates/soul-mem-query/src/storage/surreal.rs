use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::memory::{
    memory_links::MemoryLink,
    memory_note::{MemoryId, MemoryNote},
};

use super::{
    error::{StorageError, StorageResult},
    model::{
        FeedbackEventRecord, MemoryLinkRecord, MemoryNoteRecord, RetrievalEventRecord,
        SimilarityHit, SimilarityQuery,
    },
    repository::MemoryRepository,
    surql::BOOTSTRAP_STATEMENTS,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurrealConnectionConfig {
    pub endpoint: String,
    pub namespace: String,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl SurrealConnectionConfig {
    pub fn new(
        endpoint: impl Into<String>,
        namespace: impl Into<String>,
        database: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            namespace: namespace.into(),
            database: database.into(),
            username: None,
            password: None,
        }
    }

    pub fn with_auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }
}

impl Default for SurrealConnectionConfig {
    fn default() -> Self {
        Self {
            endpoint: "ws://127.0.0.1:8000/rpc".to_string(),
            namespace: "soulmem".to_string(),
            database: "memory".to_string(),
            username: None,
            password: None,
        }
    }
}

#[derive(Debug, Default)]
struct RepositoryState {
    connected: bool,
    bootstrapped: bool,
    notes: HashMap<String, MemoryNoteRecord>,
    links: HashMap<String, MemoryLinkRecord>,
    retrieval_events: Vec<RetrievalEventRecord>,
    feedback_events: Vec<FeedbackEventRecord>,
}

#[derive(Clone)]
pub struct SurrealMemoryRepository {
    config: SurrealConnectionConfig,
    state: Arc<RwLock<RepositoryState>>,
}

impl SurrealMemoryRepository {
    pub fn new(config: SurrealConnectionConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(RepositoryState::default())),
        }
    }

    pub fn config(&self) -> &SurrealConnectionConfig {
        &self.config
    }

    pub async fn connect(&self) -> StorageResult<()> {
        let mut guard = self.state.write();
        guard.connected = true;
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        self.state.read().connected
    }

    pub async fn bootstrap_schema(&self) -> StorageResult<()> {
        self.ensure_connected()?;
        let mut guard = self.state.write();
        guard.bootstrapped = true;
        Ok(())
    }

    pub fn bootstrap_statements(&self) -> &'static [&'static str] {
        BOOTSTRAP_STATEMENTS
    }

    fn ensure_connected(&self) -> StorageResult<()> {
        if self.state.read().connected {
            Ok(())
        } else {
            Err(StorageError::backend(
                "SurrealDB repository is not connected; call connect() first",
            ))
        }
    }

    fn ensure_bootstrapped(&self) -> StorageResult<()> {
        let guard = self.state.read();
        if !guard.connected {
            return Err(StorageError::backend(
                "SurrealDB repository is not connected; call connect() first",
            ));
        }
        if !guard.bootstrapped {
            return Err(StorageError::backend(
                "schema is not bootstrapped; call bootstrap() first",
            ));
        }
        Ok(())
    }
}

impl Default for SurrealMemoryRepository {
    fn default() -> Self {
        Self::new(SurrealConnectionConfig::default())
    }
}

#[async_trait]
impl MemoryRepository for SurrealMemoryRepository {
    async fn bootstrap(&self) -> StorageResult<()> {
        if !self.is_connected() {
            self.connect().await?;
        }
        self.bootstrap_schema().await
    }

    async fn upsert_note(&self, note: &MemoryNote) -> StorageResult<MemoryNoteRecord> {
        self.ensure_bootstrapped()?;
        let record = MemoryNoteRecord::from_note(note)?;

        self.state
            .write()
            .notes
            .insert(record.id.clone(), record.clone());
        Ok(record)
    }

    async fn get_note(&self, memory_id: MemoryId) -> StorageResult<Option<MemoryNoteRecord>> {
        self.ensure_bootstrapped()?;
        let key = memory_id.to_string();
        Ok(self.state.read().notes.get(&key).cloned())
    }

    async fn delete_note(&self, memory_id: MemoryId) -> StorageResult<bool> {
        self.ensure_bootstrapped()?;
        let key = memory_id.to_string();
        let mut guard = self.state.write();
        let removed_note = guard.notes.remove(&key).is_some();
        if removed_note {
            guard
                .links
                .retain(|_, link| link.from != key && link.to != key);
        }
        Ok(removed_note)
    }

    async fn upsert_link(&self, link: &MemoryLink) -> StorageResult<MemoryLinkRecord> {
        self.ensure_bootstrapped()?;
        let record = MemoryLinkRecord::from_link(link)?;

        self.state
            .write()
            .links
            .insert(record.id.clone(), record.clone());
        Ok(record)
    }

    async fn delete_link(&self, link_id: &str) -> StorageResult<bool> {
        self.ensure_bootstrapped()?;
        Ok(self.state.write().links.remove(link_id).is_some())
    }

    async fn list_outbound_links(
        &self,
        memory_id: MemoryId,
    ) -> StorageResult<Vec<MemoryLinkRecord>> {
        self.ensure_bootstrapped()?;
        let from = memory_id.to_string();
        let links = self
            .state
            .read()
            .links
            .values()
            .filter(|link| link.from == from)
            .cloned()
            .collect();
        Ok(links)
    }

    async fn list_inbound_links(
        &self,
        memory_id: MemoryId,
    ) -> StorageResult<Vec<MemoryLinkRecord>> {
        self.ensure_bootstrapped()?;
        let to = memory_id.to_string();
        let links = self
            .state
            .read()
            .links
            .values()
            .filter(|link| link.to == to)
            .cloned()
            .collect();
        Ok(links)
    }

    async fn append_retrieval_event(&self, mut event: RetrievalEventRecord) -> StorageResult<()> {
        self.ensure_bootstrapped()?;
        if event.id.is_none() {
            event.id = Some(Uuid::new_v4().to_string());
        }
        self.state.write().retrieval_events.push(event);
        Ok(())
    }

    async fn append_feedback_event(&self, mut event: FeedbackEventRecord) -> StorageResult<()> {
        self.ensure_bootstrapped()?;
        if event.id.is_none() {
            event.id = Some(Uuid::new_v4().to_string());
        }
        self.state.write().feedback_events.push(event);
        Ok(())
    }

    async fn query_similar_notes(
        &self,
        _query: SimilarityQuery,
    ) -> StorageResult<Vec<SimilarityHit>> {
        self.ensure_bootstrapped()?;
        Err(StorageError::unsupported(
            "vector similarity search is not wired yet; integrate SurrealDB vector index in this method",
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::memory_note::{
        MemoryNoteBuilder, MemoryType,
        sem_mem::{ConceptType, SemMemory},
    };

    use super::*;

    #[tokio::test]
    async fn test_upsert_and_get_note() {
        let repo = SurrealMemoryRepository::default();
        repo.bootstrap().await.expect("bootstrap");

        let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Rust".to_string(),
            ConceptType::Entity,
            "language".to_string(),
        )))
        .build()
        .expect("build note");

        let saved = repo.upsert_note(&note).await.expect("upsert");
        let loaded = repo.get_note(note.id()).await.expect("get");

        assert_eq!(loaded.expect("exists").id, saved.id);
    }
}
