//! `NoteRow`：`memory_note` 表的 SDK 交互行类型。
//!
//! 向量部分 = 多列 HNSW 方案：
//! - 每个可索引子向量一个 `option<array<float>>` 槽位列（ANN 召回用，`None` 缺省为 DB `NONE`）；
//! - `variant_emb` 完整备份列（还原 `MemoryEmbedding` 的唯一真相源，serde 直通）；
//! - `fused_self_emb` 为抽象情境叙事通道的派生向量（自身子向量 mean-pool，非分数混合）。
//! 还原（`into_embedded`）只读 `variant_emb` + `tag_emb`；槽位列不参与还原。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use soul_mem_core::memory_links::MemoryLink;
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant};
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::embedding::situation::context::ContextEmbedding;
use soul_mem_query::embedding::situation::environment::EnvironmentEmbedding;
use soul_mem_query::embedding::situation::event::EventEmbedding;
use soul_mem_query::embedding::situation::location::LocationEmbedding;
use soul_mem_query::embedding::situation::participant::ParticipantEmbedding;
use soul_mem_query::embedding::situation::{AbstractSituationEmbedding, SituationEmbedding, SpecificSituationEmbedding};
use soul_mem_query::embedding::EmbeddingVec;

use super::{EmbeddingSlot, MapperError, MapperResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteRow {
    /// 记忆 id。序列化为 `memory_id` 字段，避免与 SurrealDB 记录自带的 `id` 字段冲突
    /// （记录 id 是 `memory_note:<uuid>`，写入/读取时由仓储层用本字段保持一致）。
    #[serde(rename = "memory_id")]
    pub id: MemoryId,

    pub tags: Vec<String>,
    pub retrieval_count: usize,
    pub create_time: DateTime<Utc>,
    pub last_accessed_time: DateTime<Utc>,
    pub missing_degree: f32,
    pub last_forget_time: DateTime<Utc>,

    /// 记忆类型（嵌套对象，`object FLEXIBLE`）
    pub mem_type: MemoryType,

    /// 完整嵌入备份（还原唯一真相源）。
    /// 存 `serde_json::Value` 而非 `MemoryEmbeddingVariant`：`Procedure()`（0 元组变体）
    /// 无法被 serde 反序列化，因此 Procedure 记忆的还原不走 JSON（见 `into_embedded` 特判）。
    pub variant_emb: serde_json::Value,

    // —— 槽位列（ANN 召回用；写入时与 variant_emb 同一转换函数产出）——
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sem_content_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sem_aliases_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sem_description_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_narrative_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_loc_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_loc_coord_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_part_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_part_role_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_env_atmosphere_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_env_tone_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_event_action_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_event_initiator_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_ctx_event_target_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_loc_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_loc_coord_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_part_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_part_role_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_env_atmosphere_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_env_tone_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_event_action_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_event_initiator_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sit_event_target_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused_self_emb: Option<EmbeddingVec>,
}

impl NoteRow {
    /// 写方向：从已嵌入记忆构建行（flatten）。接收 owned（解构 move，嵌入零拷贝）。
    /// `mem_links` 不在此列（独立 LinkRow 表）。
    pub fn from_embedded(embedded: EmbeddedMemoryNote) -> MapperResult<NoteRow> {
        let EmbeddedMemoryNote { note, embedding } = embedded;
        let MemoryEmbedding { tag, variant } = embedding;
        let mut row = NoteRow {
            id: note.id(),
            tags: note.tags().to_vec(),
            retrieval_count: note.retrieval_count(),
            create_time: note.creation_time(),
            last_accessed_time: note.last_accessed_time(),
            missing_degree: note.missing_degree(),
            last_forget_time: note.last_forget_time(),
            mem_type: note.into_mem_type(),
            variant_emb: serde_json::to_value(&variant)?,
            tag_emb: None,
            sem_content_emb: None,
            sem_aliases_emb: None,
            sem_description_emb: None,
            sit_narrative_emb: None,
            sit_ctx_loc_name_emb: None,
            sit_ctx_loc_coord_emb: None,
            sit_ctx_part_name_emb: None,
            sit_ctx_part_role_emb: None,
            sit_ctx_env_atmosphere_emb: None,
            sit_ctx_env_tone_emb: None,
            sit_ctx_event_action_emb: None,
            sit_ctx_event_initiator_emb: None,
            sit_ctx_event_target_emb: None,
            sit_loc_name_emb: None,
            sit_loc_coord_emb: None,
            sit_part_name_emb: None,
            sit_part_role_emb: None,
            sit_env_atmosphere_emb: None,
            sit_env_tone_emb: None,
            sit_event_action_emb: None,
            sit_event_initiator_emb: None,
            sit_event_target_emb: None,
            fused_self_emb: None,
        };
        row.apply_slot(EmbeddingSlot::Tag, tag);
        for (slot, vec) in flatten_variant(variant)? {
            row.apply_slot(slot, vec);
        }
        Ok(row)
    }

    fn apply_slot(&mut self, slot: EmbeddingSlot, vec: EmbeddingVec) {
        match slot {
            EmbeddingSlot::Tag => self.tag_emb = Some(vec),
            EmbeddingSlot::SemContent => self.sem_content_emb = Some(vec),
            EmbeddingSlot::SemAliases => self.sem_aliases_emb = Some(vec),
            EmbeddingSlot::SemDescription => self.sem_description_emb = Some(vec),
            EmbeddingSlot::SitNarrative => self.sit_narrative_emb = Some(vec),
            EmbeddingSlot::SitCtxLocName => self.sit_ctx_loc_name_emb = Some(vec),
            EmbeddingSlot::SitCtxLocCoord => self.sit_ctx_loc_coord_emb = Some(vec),
            EmbeddingSlot::SitCtxPartName => self.sit_ctx_part_name_emb = Some(vec),
            EmbeddingSlot::SitCtxPartRole => self.sit_ctx_part_role_emb = Some(vec),
            EmbeddingSlot::SitCtxEnvAtmosphere => self.sit_ctx_env_atmosphere_emb = Some(vec),
            EmbeddingSlot::SitCtxEnvTone => self.sit_ctx_env_tone_emb = Some(vec),
            EmbeddingSlot::SitCtxEventAction => self.sit_ctx_event_action_emb = Some(vec),
            EmbeddingSlot::SitCtxEventInitiator => self.sit_ctx_event_initiator_emb = Some(vec),
            EmbeddingSlot::SitCtxEventTarget => self.sit_ctx_event_target_emb = Some(vec),
            EmbeddingSlot::SitLocName => self.sit_loc_name_emb = Some(vec),
            EmbeddingSlot::SitLocCoord => self.sit_loc_coord_emb = Some(vec),
            EmbeddingSlot::SitPartName => self.sit_part_name_emb = Some(vec),
            EmbeddingSlot::SitPartRole => self.sit_part_role_emb = Some(vec),
            EmbeddingSlot::SitEnvAtmosphere => self.sit_env_atmosphere_emb = Some(vec),
            EmbeddingSlot::SitEnvTone => self.sit_env_tone_emb = Some(vec),
            EmbeddingSlot::SitEventAction => self.sit_event_action_emb = Some(vec),
            EmbeddingSlot::SitEventInitiator => self.sit_event_initiator_emb = Some(vec),
            EmbeddingSlot::SitEventTarget => self.sit_event_target_emb = Some(vec),
            EmbeddingSlot::FusedSelf => self.fused_self_emb = Some(vec),
        }
    }

    /// 读方向：还原完整 `EmbeddedMemoryNote`。`links` 由仓储层从 `memory_link` 表取回后合并。
    ///
    /// 只读 `variant_emb`（嵌入真相源）+ `tag_emb`（tag 通道）；槽位列不参与还原。
    pub fn into_embedded(self, links: Vec<MemoryLink>) -> MapperResult<EmbeddedMemoryNote> {
        // 先判断变体再 move mem_type 进 builder
        let is_procedure = matches!(&self.mem_type, MemoryType::Procedure(_));
        let note = MemoryNoteBuilder::new(self.mem_type)
            .id(self.id)
            .tags(self.tags)
            .retrieval_count(self.retrieval_count)
            .create_time(self.create_time)
            .last_accessed_time(self.last_accessed_time)
            .mem_links(links)
            .missing_degree(self.missing_degree)
            .last_forget_time(self.last_forget_time)
            .build()?;
        let tag = self.tag_emb.ok_or(MapperError::MissingField("tag_emb"))?;
        // Procedure 变体不携带嵌入数据且 `Procedure()` 无法 serde 反序列化，
        // 直接构造；其余变体从 `variant_emb`（JSON 备份）还原。
        let variant = if is_procedure {
            MemoryEmbeddingVariant::Procedure()
        } else {
            serde_json::from_value(self.variant_emb)?
        };
        let embedding = MemoryEmbedding::new(tag, variant);
        Ok(EmbeddedMemoryNote { note, embedding })
    }
}

impl TryFrom<NoteRow> for EmbeddedMemoryNote {
    type Error = MapperError;

    fn try_from(row: NoteRow) -> MapperResult<Self> {
        row.into_embedded(Vec::new())
    }
}

/// 把 `MemoryEmbeddingVariant` 的每个可索引子向量展平为 (槽位, 向量) 对。
/// 接收 owned 变体，解构 move 出字段（嵌入零拷贝）。缺失的槽位不产出（写入时列保持 NONE）。
pub(crate) fn flatten_variant(
    variant: MemoryEmbeddingVariant,
) -> MapperResult<Vec<(EmbeddingSlot, EmbeddingVec)>> {
    let mut out = Vec::new();
    match variant {
        MemoryEmbeddingVariant::Semantic(sem) => {
            let SemanticEmbedding { content, aliases, description } = sem;
            out.push((EmbeddingSlot::SemContent, content));
            out.push((EmbeddingSlot::SemAliases, aliases));
            out.push((EmbeddingSlot::SemDescription, description));
        }
        MemoryEmbeddingVariant::Situation(sit) => match sit {
            SituationEmbedding::Specific(specific) => {
                let SpecificSituationEmbedding { narrative, context } = specific;
                out.push((EmbeddingSlot::SitNarrative, narrative));
                flatten_context(context, &mut out);
            }
            SituationEmbedding::Abstract(abs) => flatten_abstract(abs, &mut out)?,
        },
        MemoryEmbeddingVariant::Procedure() => {}
    }
    Ok(out)
}

fn flatten_context(ctx: ContextEmbedding, out: &mut Vec<(EmbeddingSlot, EmbeddingVec)>) {
    let ContextEmbedding {
        location,
        fused_participant,
        environment,
        fused_event,
        ..
    } = ctx;
    if let Some(LocationEmbedding { name, coordinates }) = location {
        out.push((EmbeddingSlot::SitCtxLocName, name));
        out.push((EmbeddingSlot::SitCtxLocCoord, coordinates));
    }
    if let Some(ParticipantEmbedding { name, role, .. }) = fused_participant {
        out.push((EmbeddingSlot::SitCtxPartName, name));
        out.push((EmbeddingSlot::SitCtxPartRole, role));
    }
    let EnvironmentEmbedding { atmosphere, tone } = environment;
    out.push((EmbeddingSlot::SitCtxEnvAtmosphere, atmosphere));
    out.push((EmbeddingSlot::SitCtxEnvTone, tone));
    if let Some(EventEmbedding { action, initiator, target, .. }) = fused_event {
        out.push((EmbeddingSlot::SitCtxEventAction, action));
        out.push((EmbeddingSlot::SitCtxEventInitiator, initiator));
        out.push((EmbeddingSlot::SitCtxEventTarget, target));
    }
}

fn flatten_abstract(
    abs: AbstractSituationEmbedding,
    out: &mut Vec<(EmbeddingSlot, EmbeddingVec)>,
) -> MapperResult<()> {
    // 叙事通道派生向量（自身子向量 mean-pool，非分数混合）；先借用计算再 move 解构
    let fused_self = abs.fused_self()?;
    match abs {
        AbstractSituationEmbedding::Location(LocationEmbedding { name, coordinates }) => {
            out.push((EmbeddingSlot::SitLocName, name));
            out.push((EmbeddingSlot::SitLocCoord, coordinates));
        }
        AbstractSituationEmbedding::Participant(ParticipantEmbedding { name, role, .. }) => {
            out.push((EmbeddingSlot::SitPartName, name));
            out.push((EmbeddingSlot::SitPartRole, role));
        }
        AbstractSituationEmbedding::Environment(EnvironmentEmbedding { atmosphere, tone }) => {
            out.push((EmbeddingSlot::SitEnvAtmosphere, atmosphere));
            out.push((EmbeddingSlot::SitEnvTone, tone));
        }
        AbstractSituationEmbedding::Event(EventEmbedding {
            action,
            initiator,
            target,
            ..
        }) => {
            out.push((EmbeddingSlot::SitEventAction, action));
            out.push((EmbeddingSlot::SitEventInitiator, initiator));
            out.push((EmbeddingSlot::SitEventTarget, target));
        }
    }
    out.push((EmbeddingSlot::FusedSelf, fused_self));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{AbstractSituation, Location};
    use soul_mem_query::embedding::sem::SemanticEmbedding;

    fn sem_embedded() -> EmbeddedMemoryNote {
        let mem_type = MemoryType::Semantic(SemMemory {
            content: "Rust编程".into(),
            aliases: vec!["Rust".into()],
            concept_type: ConceptType::Entity,
            description: "系统编程语言".into(),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let variant = MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            EmbeddingVec::new(vec![1.0, 0.0]),
            EmbeddingVec::new(vec![0.9, 0.1]),
            EmbeddingVec::new(vec![0.7, 0.3]),
        ));
        let embedding = MemoryEmbedding::new(EmbeddingVec::new(vec![0.5, 0.5]), variant);
        EmbeddedMemoryNote { note, embedding }
    }

    /// 通过 serde_json 构造任意变体（query crate 的构造器多为 pub(crate)，测试不便直接构造）。
    fn abstract_location_variant() -> MemoryEmbeddingVariant {
        serde_json::from_value(serde_json::json!({
            "Situation": { "Abstract": { "Location": {
                "name": [1.0, 0.0],
                "coordinates": [0.8, 0.6]
            } } }
        }))
        .unwrap()
    }

    #[test]
    fn semantic_roundtrip() {
        let embedded = sem_embedded();
        let row = NoteRow::from_embedded(embedded.clone()).unwrap();

        // flatten：三个语义子向量 + tag
        assert_eq!(row.tag_emb.as_ref().map(|v| v.iter().copied().collect::<Vec<_>>()),
                   Some(vec![0.5, 0.5]));
        assert_eq!(row.sem_content_emb.as_ref().map(|v| v.iter().copied().collect::<Vec<_>>()),
                   Some(vec![1.0, 0.0]));
        assert!(row.sem_aliases_emb.is_some());
        assert!(row.sem_description_emb.is_some());
        assert!(row.sit_narrative_emb.is_none()); // 语义记忆没有情境列
        assert!(row.fused_self_emb.is_none());

        // serde_json 往返
        let json = serde_json::to_value(&row).unwrap();
        let row2: NoteRow = serde_json::from_value(json).unwrap();
        assert_eq!(row, row2);

        // 还原
        let back = row2.into_embedded(Vec::new()).unwrap();
        assert_eq!(back, embedded);
    }

    #[test]
    fn abstract_situation_roundtrip_with_fused_self() {
        let mem_type = MemoryType::Situation(AbstractSituation::Location(Location {
            name: "酒馆".into(),
            coordinates: "坐标".into(),
        })
        .into());
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let variant = abstract_location_variant();
        let embedding = MemoryEmbedding::new(EmbeddingVec::new(vec![0.0, 1.0]), variant);
        let embedded = EmbeddedMemoryNote { note, embedding };

        let row = NoteRow::from_embedded(embedded.clone()).unwrap();
        // 抽象 location：两个结构化槽位 + fused_self
        assert!(row.sit_loc_name_emb.is_some());
        assert!(row.sit_loc_coord_emb.is_some());
        let fused = row.fused_self_emb.as_ref().expect("fused_self_emb set");
        // query crate `fused_self()` = mean_pooling（算术平均，不归一化）：
        // mean([1,0],[0.8,0.6]) = [0.9, 0.3]
        let f: Vec<f32> = fused.iter().copied().collect();
        assert!((f[0] - 0.9).abs() < 1e-6, "fused[0]={}", f[0]);
        assert!((f[1] - 0.3).abs() < 1e-6, "fused[1]={}", f[1]);
        assert!(row.sem_content_emb.is_none()); // 情境记忆没有语义列

        let back = row.into_embedded(Vec::new()).unwrap();
        assert_eq!(back, embedded);
    }

    #[test]
    fn procedure_roundtrip_unit_variant() {
        let mem_type = MemoryType::Procedure(soul_mem_core::memory_note::proc_mem::ProcMemory::new(
            soul_mem_core::memory_note::proc_mem::Action::new(
                "act".into(),
                soul_mem_core::memory_note::proc_mem::ActionType::new_speak(),
            ),
        ));
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding =
            MemoryEmbedding::new(EmbeddingVec::new(vec![0.2, 0.8]), MemoryEmbeddingVariant::Procedure());
        let embedded = EmbeddedMemoryNote { note, embedding };

        let row = NoteRow::from_embedded(embedded.clone()).unwrap();
        // `Procedure()`（0 元组变体）外部标签序列化为对象 {"Procedure": []}
        let json = serde_json::to_value(&row).unwrap();
        assert_eq!(json["variant_emb"], serde_json::json!({"Procedure": []}));

        let row2: NoteRow = serde_json::from_value(json).unwrap();
        assert_eq!(row2, row);
        // 还原：Procedure 变体直接构造，不走 JSON 反序列化
        let back = row2.into_embedded(Vec::new()).unwrap();
        assert!(back.embedding().variant().clone().to_procedure().is_some());
        assert_eq!(back, embedded);
    }

    #[test]
    fn into_embedded_merges_links() {
        let link = MemoryLink::new(MemoryId::new(), MemoryId::new(),
            soul_mem_core::memory_links::MemoryLinkType::Sem(
                soul_mem_core::memory_links::sem_mem::SemMemLink::new("r".into(), 1.0)));
        let embedded = sem_embedded();
        let row = NoteRow::from_embedded(embedded.clone()).unwrap();
        let back = row.into_embedded(vec![link.clone()]).unwrap();
        assert_eq!(back.note().links(), &vec![link]);
    }

    #[test]
    fn time_conflict_fails_with_note_build_error() {
        let embedded = sem_embedded();
        let mut row = NoteRow::from_embedded(embedded.clone()).unwrap();
        // 构造 last_accessed < create_time 的非法行
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        row.create_time = base;
        row.last_accessed_time = base - chrono::Duration::hours(1);
        let err = row.into_embedded(Vec::new()).unwrap_err();
        assert!(matches!(err, MapperError::NoteBuild(_)));
    }
}
