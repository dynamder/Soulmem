pub mod surreal;

use std::fmt;

use async_trait::async_trait;
use soul_mem_core::{memory_links::LinkId, memory_note::MemoryId};
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
use thiserror::Error;

#[async_trait]
pub trait MemoryRepository: Send + Sync {
    /// 全量写入（替换）记忆：batch 内逐条以完整 `EmbeddedMemoryNote` 快照覆盖 DB 记录，
    /// 同一事务保证原子性（成功全成、失败全回滚）。
    ///
    /// **为什么不用 MERGE 语义**：SurrealDB 的 `MERGE` 是对象深合并
    /// （`surrealdb-core val/value/merge.rs`：payload 未提供的字段保留旧值、嵌套对象递归合并），
    /// 而 `mem_type`/`variant_emb` 是外部标签 enum 序列化的「整体快照」字段——深合并会在
    /// 变体切换（如 Semantic → Situation）时**残留旧变体键**，读回反序列化直接失败
    /// （`invalid value: map, expected map with a single key`，实测）；槽位向量为 None 时
    /// 也无法通过 MERGE 置回 NONE（旧向量残留 → KNN 误召回）。全量替换无此问题：
    /// 每次写入都以完整快照为准，槽位与变体严格一致，旧列一并清空。
    ///
    /// 调用方需持有完整 note（read-modify-write 场景先 `fetch_notes` 再改再写）。
    async fn upsert_notes(&self, mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()>;

    async fn fetch_neighbors(
        &self,
        source_ids: &[MemoryId],
        depth: usize,
    ) -> StorageResult<Vec<EmbeddedMemoryNote>>;

    async fn fetch_notes(&self, mem_ids: &[MemoryId]) -> StorageResult<Vec<EmbeddedMemoryNote>>;

    /// 按查询嵌入召回候选记忆：查询嵌入展平为槽位向量后，每个可索引槽位列做 HNSW KNN
    /// （每列取 `candidate_k` 条候选），去重 union 后返回候选集。
    /// **不做排序与截断**——精确重排（`compute_fused`）与最终 top-k 截断由调用方完成；
    /// `candidate_k` 即每槽位召回预算，需按槽位数自行放大余量
    /// （单个查询可能 fan-out 出多个槽位，见 mapper 的 `flatten_query_embedding`）。
    /// **所有权消费性接口**：接收 owned 查询并在内部解构移动（零隐式克隆）；
    /// 调用方如需复用查询，请在调用点显式 clone。
    async fn similarity_fetch(
        &self,
        queries: Vec<MemoryRetrieveQueryEmbedding>,
        candidate_k: usize,
    ) -> StorageResult<Vec<EmbeddedMemoryNote>>;

    async fn remove_notes(&self, mem_ids: &[MemoryId]) -> StorageResult<()>;

    async fn remove_links(&self, link_ids: &[LinkId]) -> StorageResult<()>;
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

impl From<crate::storage::surreal::mapper::MapperError> for StorageError {
    fn from(e: crate::storage::surreal::mapper::MapperError) -> Self {
        StorageError::InvalidArgument(e.to_string())
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        StorageError::Serialize {
            kind: EntityKind::Note,
            source: e,
        }
    }
}

pub type StorageResult<T> = Result<T, StorageError>;
