use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::memory::{
    memory_links::{MemoryLink, MemoryLinkType},
    memory_note::{
        MemoryId, MemoryNote, MemoryType,
        situation_mem::{AbstractSituation, SituationType},
    },
    working_memory::record::UserFeedback,
};

use super::error::StorageResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryNoteKind {
    Semantic,
    SituationAbstract,
    SituationSpecific,
    Procedure,
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
            payload,
        })
    }

    pub fn parse_memory_id(&self) -> StorageResult<MemoryId> {
        let uuid = Uuid::parse_str(&self.id)?;
        Ok(uuid.into())
    }
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalEventRecord {
    pub id: Option<String>,
    pub memory_id: String,
    pub occurred_at: DateTime<Utc>,
    pub query_fingerprint: Option<String>,
    pub score: Option<f32>,
}

impl RetrievalEventRecord {
    pub fn new(memory_id: MemoryId) -> Self {
        Self {
            id: None,
            memory_id: memory_id.to_string(),
            occurred_at: Utc::now(),
            query_fingerprint: None,
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

impl From<UserFeedback> for FeedbackValue {
    fn from(value: UserFeedback) -> Self {
        match value {
            UserFeedback::Positive => FeedbackValue::Positive,
            UserFeedback::Negative => FeedbackValue::Negative,
            UserFeedback::Neutral => FeedbackValue::Neutral,
            UserFeedback::None => FeedbackValue::None,
        }
    }
}

impl From<&UserFeedback> for FeedbackValue {
    fn from(value: &UserFeedback) -> Self {
        value.clone().into()
    }
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
    use crate::memory::memory_note::{
        MemoryNoteBuilder,
        sem_mem::{ConceptType, SemMemory},
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
}
