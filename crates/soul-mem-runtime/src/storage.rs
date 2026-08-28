pub mod surreal;

use std::fmt;

use async_trait::async_trait;
use soul_mem_core::{memory_links::LinkId, memory_note::MemoryId};
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding};
use thiserror::Error;

pub enum MergeMode {
    Replace,
    Merge,
}

#[async_trait]
pub trait MemoryRepository: Send + Sync {
    async fn upsert_notes(
        &self,
        mem_notes: Vec<EmbeddedMemoryNote>,
        merge_mode: MergeMode,
    ) -> StorageResult<()>;

    async fn fetch_neighbors(
        &self,
        source_ids: &[MemoryId],
        depth: usize,
    ) -> StorageResult<Vec<EmbeddedMemoryNote>>;

    async fn fetch_notes(&self, mem_ids: &[MemoryId]) -> StorageResult<Vec<EmbeddedMemoryNote>>;

    async fn similarity_fetch(
        &self,
        embeddings: Vec<MemoryEmbedding>,
        top_k: usize,
    ) -> StorageResult<Vec<EmbeddedMemoryNote>>;

    async fn remove_notes(&self, mem_ids: &[MemoryId]) -> StorageResult<()>;

    async fn remove_links(&self, link_ids: &[LinkId]) -> StorageResult<()>;

    async fn replace_notes(&self, mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()> {
        self.upsert_notes(mem_notes, MergeMode::Replace).await
    }

    async fn merge_notes(&self, mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()> {
        self.upsert_notes(mem_notes, MergeMode::Merge).await
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EntityKind {
    Note,
    Link,
}
impl fmt::Display for EntityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityKind::Note => write!(f, "memory note"),
            EntityKind::Link => write!(f, "memory link"),
        }
    }
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("{kind} {id} not found")]
    NotFound { kind: EntityKind, id: String },

    #[error("database error: {0}")]
    Db(#[from] surrealdb::Error),

    #[error("cannot serialize {kind}: {source}")]
    Serialize {
        kind: EntityKind,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// mapper 转换错误直接上浮为 `InvalidArgument`（`?` 自动转换）。
impl From<crate::storage::surreal::mapper::MapperError> for StorageError {
    fn from(e: crate::storage::surreal::mapper::MapperError) -> Self {
        StorageError::InvalidArgument(e.to_string())
    }
}

/// 行序列化错误上浮为 `Serialize`（`?` 自动转换）。
impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        StorageError::Serialize {
            kind: EntityKind::Note,
            source: e,
        }
    }
}

pub type StorageResult<T> = Result<T, StorageError>;
