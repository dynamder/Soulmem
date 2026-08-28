//! SurrealDB 行类型与 soul-mem-core/soul-mem-query 类型之间的数据转换。
//!
//! 职责边界：
//! - 本模块只做**类型转换**（行 ↔ 核心类型），不做任何 SQL/仓储逻辑；
//! - `NoteRow`/`LinkRow` 是直接与 surrealdb SDK 交互的行类型；
//! - 向量部分采用**多列 HNSW 方案**（见 `.surreal-spike/SPIKE_RESULTS.md`）：
//!   每个可索引子向量一个 `option<array<float>>` 槽位列 + `variant_emb` 完整备份列。
//!   槽位枚举 `EmbeddingSlot` 是列名的单一事实来源。

pub mod link;
pub mod note;

use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use surrealdb::types::{RecordId, RecordIdKey};

pub use link::LinkRow;
pub use note::NoteRow;

/// 向量槽位：`MemoryEmbeddingVariant` 中每个可索引子向量在 `memory_note` 表中的列名。
///
/// 命名约定：`sem_*` = 语义记忆子向量；`sit_ctx_*` = 具体情境 context 子向量；
/// `sit_*` = 抽象情境子向量；`fused_self_emb` = 抽象情境叙事通道的派生向量（自身子向量 mean-pool）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingSlot {
    Tag,
    SemContent,
    SemAliases,
    SemDescription,
    SitNarrative,
    SitCtxLocName,
    SitCtxLocCoord,
    SitCtxPartName,
    SitCtxPartRole,
    SitCtxEnvAtmosphere,
    SitCtxEnvTone,
    SitCtxEventAction,
    SitCtxEventInitiator,
    SitCtxEventTarget,
    SitLocName,
    SitLocCoord,
    SitPartName,
    SitPartRole,
    SitEnvAtmosphere,
    SitEnvTone,
    SitEventAction,
    SitEventInitiator,
    SitEventTarget,
    FusedSelf,
}

impl EmbeddingSlot {
    /// SurrealDB 列名（仓储层 KNN 查询与 SCHEMAFULL 字段声明都取这里）。
    pub fn column(self) -> &'static str {
        match self {
            EmbeddingSlot::Tag => "tag_emb",
            EmbeddingSlot::SemContent => "sem_content_emb",
            EmbeddingSlot::SemAliases => "sem_aliases_emb",
            EmbeddingSlot::SemDescription => "sem_description_emb",
            EmbeddingSlot::SitNarrative => "sit_narrative_emb",
            EmbeddingSlot::SitCtxLocName => "sit_ctx_loc_name_emb",
            EmbeddingSlot::SitCtxLocCoord => "sit_ctx_loc_coord_emb",
            EmbeddingSlot::SitCtxPartName => "sit_ctx_part_name_emb",
            EmbeddingSlot::SitCtxPartRole => "sit_ctx_part_role_emb",
            EmbeddingSlot::SitCtxEnvAtmosphere => "sit_ctx_env_atmosphere_emb",
            EmbeddingSlot::SitCtxEnvTone => "sit_ctx_env_tone_emb",
            EmbeddingSlot::SitCtxEventAction => "sit_ctx_event_action_emb",
            EmbeddingSlot::SitCtxEventInitiator => "sit_ctx_event_initiator_emb",
            EmbeddingSlot::SitCtxEventTarget => "sit_ctx_event_target_emb",
            EmbeddingSlot::SitLocName => "sit_loc_name_emb",
            EmbeddingSlot::SitLocCoord => "sit_loc_coord_emb",
            EmbeddingSlot::SitPartName => "sit_part_name_emb",
            EmbeddingSlot::SitPartRole => "sit_part_role_emb",
            EmbeddingSlot::SitEnvAtmosphere => "sit_env_atmosphere_emb",
            EmbeddingSlot::SitEnvTone => "sit_env_tone_emb",
            EmbeddingSlot::SitEventAction => "sit_event_action_emb",
            EmbeddingSlot::SitEventInitiator => "sit_event_initiator_emb",
            EmbeddingSlot::SitEventTarget => "sit_event_target_emb",
            EmbeddingSlot::FusedSelf => "fused_self_emb",
        }
    }
}

/// mapper 错误。仓储层将其映射进 [`super::StorageError`]。
#[derive(Debug, thiserror::Error)]
pub enum MapperError {
    #[error("cannot build MemoryNote: {0}")]
    NoteBuild(#[from] soul_mem_core::memory_note::MemoryNoteBuildError),

    #[error("embedding calculation failed: {0}")]
    EmbeddingCalc(#[from] soul_mem_query::embedding::EmbeddingCalcError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("record id `{0}` is not a memory_note reference with a uuid key")]
    InvalidRecordKey(String),

    #[error("required field `{0}` missing on row")]
    MissingField(&'static str),
}

pub type MapperResult<T> = Result<T, MapperError>;

/// `MemoryId` → `memory_note:<uuid>` 记录 id 的转换 trait。
///
/// 用本地 trait 而非 std `From`：`From<MemoryId> for RecordId` 违反孤儿规则
/// （`MemoryId` 在 soul-mem-core、`RecordId` 在 surrealdb，均非本 crate 类型）。
pub trait MemoryIdCodec {
    /// 转为 `memory_note:<uuid>` 记录 id（MemoryId 以 uuid 字符串为键，不可失败）。
    fn to_record_id(self) -> RecordId;
}

/// `memory_note:<uuid>` 记录 id → `MemoryId` 的转换 trait（可能失败：表名/键不合法）。
pub trait RecordIdCodec {
    fn to_memory_id(&self) -> MapperResult<MemoryId>;
}

impl MemoryIdCodec for MemoryId {
    fn to_record_id(self) -> RecordId {
        RecordId::new("memory_note", self.to_string())
    }
}

impl RecordIdCodec for RecordId {
    fn to_memory_id(&self) -> MapperResult<MemoryId> {
        if self.table.as_str() != "memory_note" {
            return Err(MapperError::InvalidRecordKey(format!("{self:?}")));
        }
        match &self.key {
            RecordIdKey::Uuid(u) => Ok(MemoryId::from(uuid::Uuid::from(u.clone()))),
            RecordIdKey::String(s) => uuid::Uuid::parse_str(s)
                .map(MemoryId::from)
                .map_err(|_| MapperError::InvalidRecordKey(format!("{self:?}"))),
            _ => Err(MapperError::InvalidRecordKey(format!("{self:?}"))),
        }
    }
}

/// 解析 DB 返回的 record 字符串（``memory_note:`<uuid>` `` 或 `memory_note:<uuid>`）为 MemoryId。
/// DB 会把 record 值序列化为带反引号的字符串（uuid 键含 `-`），`RecordId` 的 Deserialize 不接受，
/// 读路径由此函数手动解析。
pub fn record_str_to_memory_id(s: &str) -> MapperResult<MemoryId> {
    let key = s
        .strip_prefix("memory_note:")
        .ok_or_else(|| MapperError::InvalidRecordKey(s.to_string()))?;
    let key = key.trim_matches('`');
    uuid::Uuid::parse_str(key)
        .map(MemoryId::from)
        .map_err(|_| MapperError::InvalidRecordKey(s.to_string()))
}

/// 把一条已嵌入的记忆拆成「note 行 + 全部链接行」，供仓储层同一事务写入。
pub fn split_embedded(embedded: EmbeddedMemoryNote) -> MapperResult<(NoteRow, Vec<LinkRow>)> {
    let links = embedded.note().links().iter().map(LinkRow::from).collect();
    let row = NoteRow::from_embedded(embedded)?;
    Ok((row, links))
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType, sem_mem::SemMemLink};
    use soul_mem_core::memory_note::MemoryNoteBuilder;

    #[test]
    fn slot_columns_are_unique_and_snake_case() {
        let mut cols = Vec::new();
        let slots = [
            EmbeddingSlot::Tag,
            EmbeddingSlot::SemContent,
            EmbeddingSlot::SitNarrative,
            EmbeddingSlot::SitCtxLocName,
            EmbeddingSlot::SitCtxPartRole,
            EmbeddingSlot::SitCtxEnvAtmosphere,
            EmbeddingSlot::SitCtxEventTarget,
            EmbeddingSlot::SitLocCoord,
            EmbeddingSlot::SitPartName,
            EmbeddingSlot::SitEnvTone,
            EmbeddingSlot::SitEventInitiator,
            EmbeddingSlot::FusedSelf,
        ];
        for s in slots {
            let c = s.column();
            assert!(!c.is_empty() && c.ends_with("_emb"));
            assert!(!cols.contains(&c), "duplicate column {c}");
            cols.push(c);
        }
    }

    #[test]
    fn split_embedded_returns_note_row_and_link_rows() {
        let from = MemoryId::new();
        let to = MemoryId::new();
        let link = MemoryLink::new(from, to, MemoryLinkType::Sem(SemMemLink::new("x".into(), 1.0)));
        let note = MemoryNoteBuilder::new(soul_mem_core::memory_note::MemoryType::Procedure(
            soul_mem_core::memory_note::proc_mem::ProcMemory::new(
                soul_mem_core::memory_note::proc_mem::Action::new(
                    "act".into(),
                    soul_mem_core::memory_note::proc_mem::ActionType::new_speak(),
                ),
            ),
        ))
        .mem_links(vec![link.clone()])
        .build()
        .unwrap();
        let embedding = soul_mem_query::embedding::note::MemoryEmbedding::new(
            soul_mem_query::embedding::EmbeddingVec::zero(4),
            soul_mem_query::embedding::note::MemoryEmbeddingVariant::Procedure(),
        );
        let embedded = EmbeddedMemoryNote { note, embedding };
        let note_id = embedded.note().id();
        let (row, links) = split_embedded(embedded).unwrap();
        assert_eq!(row.id, note_id);
        assert_eq!(links.len(), 1);
        // 链接行的 in/out 引用还原后应与链接端点一致
        let back: MemoryLink = links[0].clone().try_into().unwrap();
        assert_eq!(back.from(), link.from());
        assert_eq!(back.to(), link.to());
        // Procedure 变体的备份形状（0 元组变体外部标签）
        assert_eq!(row.variant_emb, serde_json::json!({"Procedure": []}));
    }
}
