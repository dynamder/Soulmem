//! `NoteRow`：`memory_note` 表的 SDK 交互行类型。
//!
//! 向量部分 = 多列 HNSW 方案：
//! - 每个可索引子向量一个 `option<array<float>>` 槽位列（ANN 召回用，`None` 缺省为 DB `NONE`）；
//! - `variant_emb` 完整备份列（还原 `MemoryEmbedding` 的唯一真相源，serde 直通）；
//! - `fused_self_emb` 为抽象情境叙事通道的派生向量（自身子向量 mean-pool，非分数混合）。
//!   还原（`into_embedded`）只读 `variant_emb` + `tag_emb`；槽位列不参与还原。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use soul_mem_core::memory_links::MemoryLink;
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant};
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::embedding::situation::context::ContextEmbedding;
use soul_mem_query::embedding::situation::emotion::EmotionEmbedding;
use soul_mem_query::embedding::situation::environment::EnvironmentEmbedding;
use soul_mem_query::embedding::situation::event::EventEmbedding;
use soul_mem_query::embedding::situation::location::LocationEmbedding;
use soul_mem_query::embedding::situation::participant::ParticipantEmbedding;
use soul_mem_query::embedding::situation::sensory_data::SensoryDataEmbedding;
use soul_mem_query::embedding::situation::{AbstractSituationEmbedding, SituationEmbedding, SpecificSituationEmbedding};
use soul_mem_query::embedding::EmbeddingVec;
use surrealdb::types::SurrealValue;

use super::{EmbeddingSlot, MapperError, MapperResult};

/// `NoteRow` 直接派生 `SurrealValue`（SDK 原生转换）：
/// - 非 SurrealValue 字段（`MemoryId`/`MemoryType`/`Option<EmbeddingVec>`）用 `#[surreal(wrap)]`
///   包 `SerdeWrapper` 走 serde（None → Value::None）；
/// - datetime 字段不 wrap → `Value::Datetime`；`variant_emb`（serde_json）→ 嵌套对象。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, SurrealValue)]
#[surreal(crate = "surrealdb::types")]
pub struct NoteRow {
    /// 记忆 id。序列化为 `memory_id` 字段，避免与 SurrealDB 记录自带的 `id` 字段冲突
    /// （记录 id 是 `memory_note:<uuid>`，写入/读取时由仓储层用本字段保持一致）。
    #[serde(rename = "memory_id")]
    #[surreal(rename = "memory_id", wrap)]
    pub id: MemoryId,

    pub tags: Vec<String>,
    pub retrieval_count: usize,
    pub create_time: DateTime<Utc>,
    pub last_accessed_time: DateTime<Utc>,
    pub missing_degree: f32,
    pub last_forget_time: DateTime<Utc>,

    /// 记忆类型（嵌套对象，`object FLEXIBLE`）
    #[surreal(wrap)]
    pub mem_type: MemoryType,

    /// 完整嵌入备份（还原唯一真相源）。
    /// 存 `serde_json::Value` 而非 `MemoryEmbeddingVariant`：`Procedure()`（0 元组变体）
    /// 无法被 serde 反序列化，因此 Procedure 记忆的还原不走 JSON（见 `into_embedded` 特判）。
    pub variant_emb: serde_json::Value,

    // —— 槽位列（ANN 召回用；写入时与 variant_emb 同一转换函数产出）——
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub tag_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sem_content_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sem_aliases_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sem_description_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_narrative_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_loc_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_loc_coord_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_part_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_part_role_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_env_atmosphere_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_env_tone_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_event_action_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_event_initiator_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_event_target_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_emotion_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_ctx_sensory_data_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_loc_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_loc_coord_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_part_name_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_part_role_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_env_atmosphere_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_env_tone_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_event_action_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_event_initiator_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub sit_event_target_emb: Option<EmbeddingVec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[surreal(wrap)]
    pub fused_self_emb: Option<EmbeddingVec>,
}

impl NoteRow {
    /// 写方向：从已嵌入记忆构建行（flatten）。接收 owned（解构 move，嵌入零拷贝）。
    /// `mem_links` 不在此列（独立 LinkRow 表）。
    pub fn from_embedded(embedded: EmbeddedMemoryNote) -> MapperResult<NoteRow> {
        let EmbeddedMemoryNote { note, embedding } = embedded;
        let MemoryEmbedding { tag, variant } = embedding;
        let variant_emb = serde_json::to_value(&variant)?;
        // 先取字段再 move note（into_mem_type 消费）
        let id = note.id();
        let tags = note.tags().to_vec();
        let retrieval_count = note.retrieval_count();
        let create_time = note.creation_time();
        let last_accessed_time = note.last_accessed_time();
        let missing_degree = note.missing_degree();
        let last_forget_time = note.last_forget_time();
        let mem_type = note.into_mem_type();
        let mut builder = NoteRowBuilder::new(id, mem_type, variant_emb)
            .tags(tags)
            .retrieval_count(retrieval_count)
            .create_time(create_time)
            .last_accessed_time(last_accessed_time)
            .missing_degree(missing_degree)
            .last_forget_time(last_forget_time)
            // tag 恒写入：还原（into_embedded）以 tag_emb 为 tag 通道真相源（tag 不在 variant_emb 备份中）。
            .slot(EmbeddingSlot::Tag, tag);
        // 变体槽位向量跳过零值（与查询侧 similarity_fetch 的 is_zero 跳过对称）：
        // 零向量的 cosine 距离未定义，写入 HNSW 索引是退化情形，且查询侧永远跳过零向量、
        // 存了也无法召回。variant_emb 备份列保留完整向量（还原真相源不受影响）。
        for (slot, vec) in flatten_variant(variant)? {
            if !vec.is_zero() {
                builder = builder.slot(slot, vec);
            }
        }
        Ok(builder.build())
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
            EmbeddingSlot::SitCtxEmotion => self.sit_ctx_emotion_emb = Some(vec),
            EmbeddingSlot::SitCtxSensoryData => self.sit_ctx_sensory_data_emb = Some(vec),
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

/// `NoteRow` 构建器：必填 id/类型/嵌入备份，其余链式覆盖；槽位列用统一 `slot()`（对应 `EmbeddingSlot`）。
pub struct NoteRowBuilder {
    row: NoteRow,
}

impl NoteRowBuilder {
    pub fn new(id: MemoryId, mem_type: MemoryType, variant_emb: serde_json::Value) -> Self {
        let now = Utc::now();
        Self {
            row: NoteRow {
                id,
                tags: Vec::new(),
                retrieval_count: 0,
                create_time: now,
                last_accessed_time: now,
                missing_degree: 0.0,
                last_forget_time: now,
                mem_type,
                variant_emb,
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
                sit_ctx_emotion_emb: None,
                sit_ctx_sensory_data_emb: None,
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
            },
        }
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.row.tags = tags;
        self
    }
    pub fn retrieval_count(mut self, v: usize) -> Self {
        self.row.retrieval_count = v;
        self
    }
    pub fn create_time(mut self, t: DateTime<Utc>) -> Self {
        self.row.create_time = t;
        self
    }
    pub fn last_accessed_time(mut self, t: DateTime<Utc>) -> Self {
        self.row.last_accessed_time = t;
        self
    }
    pub fn missing_degree(mut self, v: f32) -> Self {
        self.row.missing_degree = v;
        self
    }
    pub fn last_forget_time(mut self, t: DateTime<Utc>) -> Self {
        self.row.last_forget_time = t;
        self
    }
    /// 设置槽位列（`EmbeddingSlot` → 对应列）。
    pub fn slot(mut self, slot: EmbeddingSlot, vec: EmbeddingVec) -> Self {
        self.row.apply_slot(slot, vec);
        self
    }
    pub fn build(self) -> NoteRow {
        self.row
    }
}

impl TryFrom<NoteRow> for EmbeddedMemoryNote {
    type Error = MapperError;

    fn try_from(row: NoteRow) -> MapperResult<Self> {
        row.into_embedded(Vec::new())
    }
}

/// 把整个 `MemoryEmbedding`（tag + variant）展平为槽位对。
/// note 侧与查询侧共用：仓储层 `similarity_fetch` 用它把查询嵌入转成逐槽位 KNN。
pub(crate) fn flatten_embedding(
    embedding: MemoryEmbedding,
) -> MapperResult<Vec<(EmbeddingSlot, EmbeddingVec)>> {
    let MemoryEmbedding { tag, variant } = embedding;
    let mut slots = vec![(EmbeddingSlot::Tag, tag)];
    slots.extend(flatten_variant(variant)?);
    Ok(slots)
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
                out.extend(flatten_context(context));
            }
            SituationEmbedding::Abstract(abs) => flatten_abstract(abs, &mut out)?,
        },
        MemoryEmbeddingVariant::Procedure() => {}
    }
    Ok(out)
}

/// 把具体情境的 context 子向量展平为槽位对（迭代器链：Option → flat_map → chain → collect）。
fn flatten_context(ctx: ContextEmbedding) -> Vec<(EmbeddingSlot, EmbeddingVec)> {
    let ContextEmbedding {
        location,
        fused_participant,
        fused_emotion,
        fused_sensory_data,
        environment,
        fused_event,
    } = ctx;

    let location = location.into_iter().flat_map(|LocationEmbedding { name, coordinates }| {
        [
            (EmbeddingSlot::SitCtxLocName, name),
            (EmbeddingSlot::SitCtxLocCoord, coordinates),
        ]
    });
    let participant = fused_participant
        .into_iter()
        .flat_map(|ParticipantEmbedding { name, role, .. }| {
            [
                (EmbeddingSlot::SitCtxPartName, name),
                (EmbeddingSlot::SitCtxPartRole, role),
            ]
        });
    // 情绪/感官通道：weight_pooling 后的融合向量；intensity 是标量，不入槽位列
    let emotion = fused_emotion.into_iter().map(|e| {
        let EmotionEmbedding { emotion, .. } = e;
        (EmbeddingSlot::SitCtxEmotion, emotion)
    });
    let sensory = fused_sensory_data.into_iter().map(|s| {
        let SensoryDataEmbedding { sensory, .. } = s;
        (EmbeddingSlot::SitCtxSensoryData, sensory)
    });
    let EnvironmentEmbedding { atmosphere, tone } = environment;
    let environment = [
        (EmbeddingSlot::SitCtxEnvAtmosphere, atmosphere),
        (EmbeddingSlot::SitCtxEnvTone, tone),
    ];
    let event = fused_event.into_iter().flat_map(|EventEmbedding {
        action,
        initiator,
        target,
        ..
    }| {
        [
            (EmbeddingSlot::SitCtxEventAction, action),
            (EmbeddingSlot::SitCtxEventInitiator, initiator),
            (EmbeddingSlot::SitCtxEventTarget, target),
        ]
    });

    location
        .chain(participant)
        .chain(emotion)
        .chain(sensory)
        .chain(environment)
        .chain(event)
        .collect()
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
    fn note_row_builder_defaults_and_slot() {
        let id = MemoryId::new();
        let mem_type = MemoryType::Procedure(soul_mem_core::memory_note::proc_mem::ProcMemory::new(
            soul_mem_core::memory_note::proc_mem::Action::new(
                "act".into(),
                soul_mem_core::memory_note::proc_mem::ActionType::new_speak(),
            ),
        ));
        let row = NoteRowBuilder::new(id, mem_type, serde_json::json!({"Procedure": []}))
            .slot(EmbeddingSlot::Tag, EmbeddingVec::new(vec![1.0, 0.0]))
            .slot(EmbeddingSlot::SemContent, EmbeddingVec::new(vec![0.5, 0.5]))
            .build();
        assert_eq!(row.id, id);
        assert!(row.tags.is_empty());
        assert_eq!(row.retrieval_count, 0);
        assert_eq!(row.tag_emb.as_ref().map(|v| v.iter().copied().collect::<Vec<_>>()), Some(vec![1.0, 0.0]));
        assert!(row.sem_content_emb.is_some());
        assert!(row.sit_narrative_emb.is_none());
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

    /// 具体情境全通道往返：narrative + context（location/participant/emotion/sensory/environment/event）
    /// 全部产出槽位列，其中 emotion/sensory 通道此前被遗漏（疏漏修复），必须断言其列被写入。
    #[test]
    fn specific_situation_roundtrip_with_emotion_sensory() {
        use soul_mem_core::memory_note::situation_mem::{Context, SpecificSituation};
        let mem_type = MemoryType::Situation(
            SpecificSituation::new("叙事".into(), Utc::now(), Context::default()).into(),
        );
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        // 通过 serde_json 构造 Specific 变体（query crate 构造器多为 pub(crate)）
        let variant: MemoryEmbeddingVariant = serde_json::from_value(serde_json::json!({
            "Situation": { "Specific": {
                "narrative": [1.0, 0.0],
                "context": {
                    "location": { "name": [0.9, 0.1], "coordinates": [0.8, 0.2] },
                    "fused_participant": { "name": [0.7, 0.3], "role": [0.6, 0.4], "fused": [0.5, 0.5] },
                    "fused_emotion": { "emotion": [0.55, 0.45], "intensity": 1.0 },
                    "fused_sensory_data": { "sensory": [0.53, 0.47], "intensity": 1.0 },
                    "environment": { "atmosphere": [0.4, 0.6], "tone": [0.3, 0.7] },
                    "fused_event": { "action": [0.2, 0.8], "initiator": [0.1, 0.9], "target": [0.05, 0.95], "intensity": 1.0 }
                }
            } }
        }))
        .unwrap();
        let embedding = MemoryEmbedding::new(EmbeddingVec::new(vec![0.0, 1.0]), variant);
        let embedded = EmbeddedMemoryNote { note, embedding };

        let row = NoteRow::from_embedded(embedded.clone()).unwrap();
        assert!(row.sit_narrative_emb.is_some());
        assert!(row.sit_ctx_loc_name_emb.is_some());
        assert!(row.sit_ctx_part_name_emb.is_some());
        assert!(row.sit_ctx_emotion_emb.is_some(), "emotion 通道必须写入槽位列");
        assert!(row.sit_ctx_sensory_data_emb.is_some(), "sensory 通道必须写入槽位列");
        assert!(row.sit_ctx_env_atmosphere_emb.is_some());
        assert!(row.sit_ctx_event_action_emb.is_some());
        assert!(row.sit_loc_name_emb.is_none()); // 抽象情境列不写入
        assert!(row.fused_self_emb.is_none()); // 具体情境无 fused_self

        let back = row.into_embedded(Vec::new()).unwrap();
        assert_eq!(back, embedded);
    }

    /// 零向量对称跳过：写入侧变体槽位向量为零时不落槽位列（NONE），但 variant_emb 备份
    /// 保留完整向量，还原仍保真（与查询侧 similarity_fetch 的 is_zero 跳过一致）。
    /// tag 例外：tag_emb 是还原的 tag 通道真相源，恒写入（含零向量）。
    #[test]
    fn zero_vectors_skip_slot_columns_but_keep_backup() {
        let mem_type = MemoryType::Semantic(SemMemory {
            content: "内容".into(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "描述".into(),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let variant = MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            EmbeddingVec::zero(4), // 零向量 content
            EmbeddingVec::new(vec![0.9, 0.1, 0.0, 0.0]),
            EmbeddingVec::zero(4), // 零向量 description
        ));
        let embedding = MemoryEmbedding::new(EmbeddingVec::zero(4), variant); // 零向量 tag
        let embedded = EmbeddedMemoryNote { note, embedding };

        let row = NoteRow::from_embedded(embedded.clone()).unwrap();
        assert!(row.tag_emb.is_some(), "tag 恒写入（还原真相源）");
        assert!(row.sem_content_emb.is_none(), "零向量 content 不写槽位列");
        assert!(row.sem_description_emb.is_none(), "零向量 description 不写槽位列");
        assert!(row.sem_aliases_emb.is_some(), "非零向量照常写入");
        // 备份列是唯一真相源，保留完整嵌入（含零向量）
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
