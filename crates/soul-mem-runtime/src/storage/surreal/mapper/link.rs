//! `LinkRow`：`memory_link` 表的 SDK 交互行类型（SurrealDB 图边记录）。
//!
//! 边记录：记录键 = `memory_link:<uuid>`（LinkId）；`in`/`out` 为 `record<memory_note>` 引用
//! （图遍历运算符 `->memory_link->` 依赖这两个字段名，故用 `r#in` + serde rename）。
//! 边属性 = 链接权重/遗忘状态/类型。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use soul_mem_core::memory_links::{LinkId, MemoryLink, MemoryLinkBuilder, MemoryLinkType};
use surrealdb::types::RecordId;

use super::{MapperError, MapperResult, MemoryIdCodec, RecordIdCodec};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkRow {
    /// 链接 id。序列化为 `link_id` 字段（记录键是 `memory_link:<uuid>`，读写一致性由仓储层保证）。
    #[serde(rename = "link_id")]
    pub id: LinkId,

    /// 起点记忆引用 `memory_note:<uuid>`（`in` 是 Rust 关键字，用 `r#in`）。
    #[serde(rename = "in")]
    pub r#in: RecordId,

    /// 终点记忆引用 `memory_note:<uuid>`。
    #[serde(rename = "out")]
    pub out: RecordId,

    pub intensity: f64,
    pub missing_degree: f32,
    pub last_forget_time: DateTime<Utc>,

    /// 链接类型（嵌套对象，`object FLEXIBLE`）
    pub link_type: MemoryLinkType,
}

impl From<&MemoryLink> for LinkRow {
    fn from(link: &MemoryLink) -> Self {
        LinkRow {
            id: link.id(),
            r#in: link.from().to_record_id(),
            out: link.to().to_record_id(),
            intensity: link.intensity,
            missing_degree: link.missing_degree(),
            last_forget_time: link.last_forget_time(),
            link_type: link.link_type().clone(),
        }
    }
}

impl From<MemoryLink> for LinkRow {
    fn from(link: MemoryLink) -> Self {
        LinkRow::from(&link)
    }
}

impl TryFrom<LinkRow> for MemoryLink {
    type Error = MapperError;

    fn try_from(row: LinkRow) -> MapperResult<MemoryLink> {
        let from = row.r#in.to_memory_id()?;
        let to = row.out.to_memory_id()?;
        Ok(MemoryLinkBuilder::new(from, to, row.link_type)
            .id(row.id)
            .intensity(row.intensity)
            .missing_degree(row.missing_degree)
            .last_forget_time(row.last_forget_time)
            .build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_note::MemoryId;

    fn sample_link() -> MemoryLink {
        MemoryLinkBuilder::new(
            MemoryId::new(),
            MemoryId::new(),
            MemoryLinkType::Sem(SemMemLink::new("is_related_to".into(), 0.9)),
        )
        .id(LinkId::new())
        .intensity(0.7)
        .missing_degree(0.3)
        .last_forget_time(Utc.with_ymd_and_hms(2024, 5, 1, 8, 0, 0).unwrap())
        .build()
    }

    #[test]
    fn link_roundtrip() {
        let link = sample_link();
        let row = LinkRow::from(&link);

        // 图边字段名
        let json = serde_json::to_value(&row).unwrap();
        assert!(json.get("in").is_some(), "must serialize as `in`");
        assert!(json.get("out").is_some(), "must serialize as `out`");
        assert!(json.get("link_id").is_some(), "must serialize as `link_id`");

        let row2: LinkRow = serde_json::from_value(json).unwrap();
        assert_eq!(row, row2);

        let back: MemoryLink = row2.try_into().unwrap();
        assert_eq!(back, link);
    }

    #[test]
    fn record_id_conversion_roundtrip() {
        use super::super::{MemoryIdCodec, RecordIdCodec};
        let id = MemoryId::new();
        let rid = id.to_record_id();
        assert_eq!(rid.table.as_str(), "memory_note");
        assert_eq!(rid.to_memory_id().unwrap(), id);
    }

    #[test]
    fn bad_record_key_is_rejected() {
        use super::super::RecordIdCodec;
        let bad = RecordId::new("memory_note", 42);
        let err = bad.to_memory_id().unwrap_err();
        assert!(matches!(err, MapperError::InvalidRecordKey(_)));
    }

    #[test]
    fn wrong_table_is_rejected() {
        use super::super::RecordIdCodec;
        let bad = RecordId::new("other_table", "some-uuid");
        let err = bad.to_memory_id().unwrap_err();
        assert!(matches!(err, MapperError::InvalidRecordKey(_)));
    }
}
