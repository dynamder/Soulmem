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
pub enum MemoryLinkKind {
    Semantic,
    Situation,
    Procedure,
}

impl MemoryLinkKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Situation => "situation",
            Self::Procedure => "procedure",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryNoteRecord {
    pub id: String,
    pub tags: Vec<String>,
    pub retrieval_count: usize,
    pub create_time: DateTime<Utc>,
    pub last_accessed_time: DateTime<Utc>,
    pub kind: MemoryNoteKind,
    #[serde(default)]
    pub identity_content: String,
    pub embedding: Option<Vec<f32>>,
    pub payload: Value,
}

impl MemoryNoteRecord {
    pub fn from_note(note: &MemoryNote) -> StorageResult<Self> {
        let (kind, payload) = match note.mem_type() {
            MemoryType::Semantic(sem) => (MemoryNoteKind::Semantic, serde_json::to_value(sem)?),
            MemoryType::Procedure(proc_mem) => {
                (MemoryNoteKind::Procedure, serde_json::to_value(proc_mem)?)
            }
            MemoryType::Situation(situation) => match situation {
                SituationType::SpecificSituation(specific) => (
                    MemoryNoteKind::SituationSpecific,
                    serde_json::to_value(specific)?,
                ),
                SituationType::AbstractSituation(abstract_situation) => (
                    MemoryNoteKind::SituationAbstract,
                    serde_json::to_value(abstract_situation)?,
                ),
            },
        };

        let identity_content = Self::build_identity_content(&kind, &payload)?;

        Ok(Self {
            id: note.id().to_string(),
            tags: note.tags().to_vec(),
            retrieval_count: note.retrieval_count(),
            create_time: note.creation_time(),
            last_accessed_time: note.last_accessed_time(),
            kind,
            identity_content,
            embedding: None,
            payload,
        })
    }

    pub fn build_identity_content(kind: &MemoryNoteKind, payload: &Value) -> StorageResult<String> {
        let content = match kind {
            MemoryNoteKind::Semantic => payload.get("content").and_then(Value::as_str),
            MemoryNoteKind::Procedure => payload.pointer("/action/content").and_then(Value::as_str),
            MemoryNoteKind::SituationAbstract | MemoryNoteKind::SituationSpecific => None,
        };

        match content {
            Some(content) if !content.trim().is_empty() => Ok(content.trim().to_string()),
            Some(_) => Err(StorageError::invalid_data(
                "memory note identity content must not be empty",
            )),
            None if matches!(
                kind,
                MemoryNoteKind::SituationAbstract | MemoryNoteKind::SituationSpecific
            ) =>
            {
                Ok(serde_json::to_string(payload)?)
            }
            None => Err(StorageError::invalid_data(
                "memory note payload is missing identity content",
            )),
        }
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

                Ok(MemoryType::Situation(SituationType::AbstractSituation(
                    abstract_situation,
                )))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryLinkRecord {
    pub id: String,
    pub from: String,
    pub to: String,
    pub intensity: f64,
    pub confidence: Option<f32>,
    pub kind: MemoryLinkKind,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationBatchResult {
    pub notes: Vec<MemoryNoteRecord>,
    pub links: Vec<MemoryLinkRecord>,
}

impl MemoryLinkRecord {
    pub fn from_link(link: &MemoryLink) -> StorageResult<Self> {
        let (kind, payload, confidence) = match link.link_type() {
            MemoryLinkType::Sem(sem) => (
                MemoryLinkKind::Semantic,
                json!({ "verb": sem.verb }),
                Some(sem.confidence),
            ),
            MemoryLinkType::Sit(sit) => {
                (MemoryLinkKind::Situation, serde_json::to_value(sit)?, None)
            }
            MemoryLinkType::Proc(_) => (MemoryLinkKind::Procedure, json!({}), None),
        };

        Ok(Self {
            id: link.id().to_string(),
            from: link.from().to_string(),
            to: link.to().to_string(),
            intensity: link.intensity,
            confidence,
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
                let verb = self
                    .payload
                    .get("verb")
                    .and_then(Value::as_str)
                    .filter(|verb| !verb.trim().is_empty())
                    .ok_or_else(|| {
                        StorageError::invalid_data("semantic memory_link payload is missing `verb`")
                    })?;
                Ok(MemoryLinkType::Sem(SemMemLink::new(
                    verb.to_string(),
                    self.confidence.unwrap_or(self.intensity as f32),
                )))
            }
            MemoryLinkKind::Situation => {
                Ok(MemoryLinkType::Sit(serde_json::from_value::<
                    SituationMemLink,
                >(self.payload.clone())?))
            }
            MemoryLinkKind::Procedure => Ok(MemoryLinkType::Proc(ProcMemLink::TrigToAction(
                soul_mem_core::memory_links::proc_mem::TrigToAction::new(self.intensity),
            ))),
        }
    }

    pub fn has_same_identity_as(&self, other: &Self) -> bool {
        self.from == other.from
            && self.to == other.to
            && self.kind == other.kind
            && self.payload == other.payload
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
        assert_eq!(record.identity_content, "Rust");
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

        let mut link = MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("mentions".to_string(), 0.9)),
        );
        link.intensity = 0.4;

        let record = MemoryLinkRecord::from_link(&link).expect("convert link to record");
        assert_eq!(record.confidence, Some(0.9));
        let restored = record.to_link().expect("restore link");

        assert_eq!(restored, link);
    }

    #[test]
    fn test_memory_link_identity_matches_unique_index() {
        let from = Uuid::new_v4().into();
        let to = Uuid::new_v4().into();
        let mut first = MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("uses".to_string(), 0.2)),
        );
        first.intensity = 0.2;
        let mut second = MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("uses".to_string(), 0.9)),
        );
        second.intensity = 0.9;

        let first = MemoryLinkRecord::from_link(&first).expect("convert first link");
        let second = MemoryLinkRecord::from_link(&second).expect("convert second link");

        assert!(first.has_same_identity_as(&second));

        let different_relation = MemoryLinkRecord {
            payload: json!({ "verb": "depends_on" }),
            ..second
        };
        assert!(!first.has_same_identity_as(&different_relation));
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
