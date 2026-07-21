// 数据库记录模型

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use soul_mem_core::{
    memory_links::{
        MemoryLink, MemoryLinkType, proc_mem::ProcMemLink, sem_mem::SemMemLink,
        situation_mem::SituationMemLink,
    },
    memory_note::{
        MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType,
        proc_mem::ProcMemory,
        sem_mem::SemMemory,
        situation_mem::{AbstractSituation, SituationType, SpecificSituation},
    },
};
use uuid::Uuid;

use super::error::{StorageError, StorageResult};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryNoteKind {
    Semantic,
    SituationAbstract,
    SituationSpecific,
    Procedure,
}

impl MemoryNoteKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::SituationAbstract => "situation_abstract",
            Self::SituationSpecific => "situation_specific",
            Self::Procedure => "procedure",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SituationSubtype {
    Location,
    Participant,
    Environment,
    Event,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLinkKind {
    Semantic,
    Situation,
    Procedure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryNoteRecord {
    pub id: String,
    pub tags: Vec<String>,
    pub retrieval_count: usize,
    pub create_time: DateTime<Utc>,
    pub last_accessed_time: DateTime<Utc>,
    pub kind: MemoryNoteKind,
    pub situation_subtype: Option<SituationSubtype>,
    pub embedding: Option<Vec<f32>>,
    pub payload: Value,
}

impl MemoryNoteRecord {
    pub fn from_note(note: &MemoryNote) -> StorageResult<Self> {
        let (kind, situation_subtype, payload) = match note.mem_type() {
            MemoryType::Semantic(sem) => {
                (MemoryNoteKind::Semantic, None, serde_json::to_value(sem)?)
            }
            MemoryType::Procedure(proc_mem) => (
                MemoryNoteKind::Procedure,
                None,
                serde_json::to_value(proc_mem)?,
            ),
            MemoryType::Situation(situation) => match situation {
                SituationType::SpecificSituation(specific) => (
                    MemoryNoteKind::SituationSpecific,
                    None,
                    serde_json::to_value(specific)?,
                ),
                SituationType::AbstractSituation(abstract_situation) => {
                    let subtype = Some(match abstract_situation {
                        AbstractSituation::Location(_) => SituationSubtype::Location,
                        AbstractSituation::Participant(_) => SituationSubtype::Participant,
                        AbstractSituation::Environment(_) => SituationSubtype::Environment,
                        AbstractSituation::Event(_) => SituationSubtype::Event,
                    });
                    (
                        MemoryNoteKind::SituationAbstract,
                        subtype,
                        serde_json::to_value(abstract_situation)?,
                    )
                }
            },
        };

        Ok(Self {
            id: note.id().to_string(),
            tags: note.tags().to_vec(),
            retrieval_count: note.retrieval_count(),
            create_time: note.creation_time(),
            last_accessed_time: note.last_accessed_time(),
            kind,
            situation_subtype,
            embedding: None,
            payload,
        })
    }

    pub fn parse_memory_id(&self) -> StorageResult<MemoryId> {
        let uuid = Uuid::parse_str(&self.id)?;
        Ok(uuid.into())
    }

    pub fn to_note(&self, links: Vec<MemoryLink>) -> StorageResult<MemoryNote> {
        let mem_type = self.to_memory_type()?;

        MemoryNoteBuilder::new(mem_type)
            .id(self.parse_memory_id()?)
            .tags(self.tags.clone())
            .retrieval_count(self.retrieval_count)
            .create_time(self.create_time)
            .last_accessed_time(self.last_accessed_time)
            .mem_links(links)
            .build()
            .map_err(|err| {
                StorageError::invalid_data(format!(
                    "failed to rebuild memory note {}: {err}",
                    self.id
                ))
            })
    }

    fn to_memory_type(&self) -> StorageResult<MemoryType> {
        match self.kind {
            MemoryNoteKind::Semantic => {
                Ok(MemoryType::Semantic(serde_json::from_value::<SemMemory>(
                    self.payload.clone(),
                )?))
            }
            MemoryNoteKind::Procedure => {
                Ok(MemoryType::Procedure(serde_json::from_value::<ProcMemory>(
                    self.payload.clone(),
                )?))
            }
            MemoryNoteKind::SituationSpecific => {
                Ok(MemoryType::Situation(SituationType::SpecificSituation(
                    serde_json::from_value::<SpecificSituation>(self.payload.clone())?,
                )))
            }
            MemoryNoteKind::SituationAbstract => {
                let abstract_situation =
                    serde_json::from_value::<AbstractSituation>(self.payload.clone())?;

                if let Some(subtype) = &self.situation_subtype
                    && !matches_situation_subtype(&abstract_situation, subtype)
                {
                    return Err(StorageError::invalid_data(format!(
                        "memory note {} has mismatched situation_subtype",
                        self.id
                    )));
                }

                Ok(MemoryType::Situation(SituationType::AbstractSituation(
                    abstract_situation,
                )))
            }
        }
    }
}

fn matches_situation_subtype(situation: &AbstractSituation, subtype: &SituationSubtype) -> bool {
    matches!(
        (situation, subtype),
        (AbstractSituation::Location(_), SituationSubtype::Location)
            | (
                AbstractSituation::Participant(_),
                SituationSubtype::Participant
            )
            | (
                AbstractSituation::Environment(_),
                SituationSubtype::Environment
            )
            | (AbstractSituation::Event(_), SituationSubtype::Event)
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryLinkRecord {
    pub id: String,
    pub from: String,
    pub to: String,
    pub intensity: f64,
    pub kind: MemoryLinkKind,
    pub payload: Value,
}

impl MemoryLinkRecord {
    pub fn from_link(link: &MemoryLink) -> StorageResult<Self> {
        let (kind, payload) = match link.link_type() {
            MemoryLinkType::Sem(sem) => (MemoryLinkKind::Semantic, serde_json::to_value(sem)?),
            MemoryLinkType::Sit(sit) => (MemoryLinkKind::Situation, serde_json::to_value(sit)?),
            MemoryLinkType::Proc(proc_mem) => {
                (MemoryLinkKind::Procedure, serde_json::to_value(proc_mem)?)
            }
        };

        Ok(Self {
            id: link.id().to_string(),
            from: link.from().to_string(),
            to: link.to().to_string(),
            intensity: link.intensity,
            kind,
            payload,
        })
    }

    pub fn to_link(&self) -> StorageResult<MemoryLink> {
        let link_type = self.to_link_type()?;

        Ok(serde_json::from_value(json!({
            "id": self.id,
            "from": self.from,
            "to": self.to,
            "intensity": self.intensity,
            "link_type": link_type,
        }))?)
    }

    fn to_link_type(&self) -> StorageResult<MemoryLinkType> {
        match self.kind {
            MemoryLinkKind::Semantic => {
                Ok(MemoryLinkType::Sem(serde_json::from_value::<SemMemLink>(
                    self.payload.clone(),
                )?))
            }
            MemoryLinkKind::Situation => {
                Ok(MemoryLinkType::Sit(serde_json::from_value::<
                    SituationMemLink,
                >(self.payload.clone())?))
            }
            MemoryLinkKind::Procedure => {
                Ok(MemoryLinkType::Proc(serde_json::from_value::<ProcMemLink>(
                    self.payload.clone(),
                )?))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalEventRecord {
    pub id: Option<String>,
    pub memory_id: String,
    pub occurred_at: DateTime<Utc>,
    pub score: Option<f32>,
}

impl RetrievalEventRecord {
    pub fn new(memory_id: MemoryId) -> Self {
        Self {
            id: None,
            memory_id: memory_id.to_string(),
            occurred_at: Utc::now(),
            score: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackValue {
    Positive,
    Negative,
    Neutral,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeedbackEventRecord {
    pub id: Option<String>,
    pub memory_id: String,
    pub occurred_at: DateTime<Utc>,
    pub feedback: FeedbackValue,
}

impl FeedbackEventRecord {
    pub fn new(memory_id: MemoryId, feedback: FeedbackValue) -> Self {
        Self {
            id: None,
            memory_id: memory_id.to_string(),
            occurred_at: Utc::now(),
            feedback,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityQuery {
    pub embedding: Vec<f32>,
    pub limit: usize,
    pub min_score: f32,
    pub tags_any: Vec<String>,
    pub kinds: Vec<MemoryNoteKind>,
}

impl SimilarityQuery {
    pub fn new(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            limit: 8,
            min_score: 0.0,
            tags_any: Vec::new(),
            kinds: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EventWindow {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

impl EventWindow {
    pub fn all() -> Self {
        Self::default()
    }

    pub fn new(start: Option<DateTime<Utc>>, end: Option<DateTime<Utc>>) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EventStats {
    pub total: usize,
    pub first_occurred_at: Option<DateTime<Utc>>,
    pub last_occurred_at: Option<DateTime<Utc>>,
}

impl EventStats {
    pub fn from_timestamps(timestamps: impl IntoIterator<Item = DateTime<Utc>>) -> Self {
        let mut total = 0usize;
        let mut first_occurred_at: Option<DateTime<Utc>> = None;
        let mut last_occurred_at: Option<DateTime<Utc>> = None;

        for occurred_at in timestamps {
            total += 1;
            first_occurred_at = Some(match first_occurred_at {
                Some(existing) => existing.min(occurred_at),
                None => occurred_at,
            });
            last_occurred_at = Some(match last_occurred_at {
                Some(existing) => existing.max(occurred_at),
                None => occurred_at,
            });
        }

        Self {
            total,
            first_occurred_at,
            last_occurred_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityHit {
    pub memory_id: String,
    pub score: f32,
}

impl SimilarityHit {
    pub fn parse_memory_id(&self) -> StorageResult<MemoryId> {
        let uuid = Uuid::parse_str(&self.memory_id)?;
        Ok(uuid.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use soul_mem_core::{
        memory_links::{MemoryLinkType, sem_mem::SemMemLink},
        memory_note::{
            MemoryNoteBuilder,
            sem_mem::{ConceptType, SemMemory},
        },
    };

    #[test]
    fn test_memory_note_record_from_semantic_note() {
        let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Rust".to_string(),
            ConceptType::Entity,
            "A programming language".to_string(),
        )))
        .tags(vec!["lang".to_string()])
        .build()
        .expect("build semantic memory note");

        let record = MemoryNoteRecord::from_note(&note).expect("convert to record");
        assert_eq!(record.kind, MemoryNoteKind::Semantic);
        assert_eq!(record.tags, vec!["lang".to_string()]);
        assert!(record.parse_memory_id().is_ok());
    }

    #[test]
    fn test_memory_note_record_roundtrip() {
        let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Rust".to_string(),
            ConceptType::Entity,
            "A programming language".to_string(),
        )))
        .tags(vec!["lang".to_string()])
        .retrieval_count(2)
        .build()
        .expect("build semantic memory note");

        let record = MemoryNoteRecord::from_note(&note).expect("convert to record");
        let restored = record.to_note(Vec::new()).expect("restore note");

        assert_eq!(restored.id(), note.id());
        assert_eq!(restored.tags(), note.tags());
        assert_eq!(restored.retrieval_count(), note.retrieval_count());
        assert_eq!(restored.mem_type(), note.mem_type());
        assert!(restored.links().is_empty());
    }

    #[test]
    fn test_memory_link_record_roundtrip() {
        let from = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Rust".to_string(),
            ConceptType::Entity,
            "A programming language".to_string(),
        )))
        .build()
        .expect("build source note")
        .id();
        let to = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Cargo".to_string(),
            ConceptType::Entity,
            "Rust package manager".to_string(),
        )))
        .build()
        .expect("build target note")
        .id();

        let link = MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("mentions".to_string(), 0.8, 0.9)),
        );

        let record = MemoryLinkRecord::from_link(&link).expect("convert link to record");
        let restored = record.to_link().expect("restore link");

        assert_eq!(restored, link);
    }

    #[test]
    fn test_event_stats_from_timestamps() {
        let first = Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap();
        let second = first + Duration::hours(6);
        let third = first + Duration::days(1);

        let stats = EventStats::from_timestamps([second, third, first]);

        assert_eq!(stats.total, 3);
        assert_eq!(stats.first_occurred_at, Some(first));
        assert_eq!(stats.last_occurred_at, Some(third));
    }
}
