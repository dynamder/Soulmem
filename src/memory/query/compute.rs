use crate::memory::{
    embedding::{
        note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant},
        query::{
            note::{MemoryRetrieveQueryEmbedding, MemoryRetrieveQueryVariantEmbedding},
            sem::SemanticQueryUnitEmbedding,
            situation::{
                environment::EnvironmentQueryUnitEmbedding, event::EventQueryUnitEmbedding,
                location::LocationQueryUnitEmbedding, participant::ParticipantQueryUnitEmbedding,
                SituationQueryUnitEmbedding,
            },
        },
        sem::SemanticEmbedding,
        situation::{
            environment::EnvironmentEmbedding, event::EventEmbedding, location::LocationEmbedding,
            participant::ParticipantEmbedding, AbstractSituationEmbedding, SituationEmbedding,
            SpecificSituationEmbedding,
        },
        Embeddable, EmbeddingCalcResult, EmbeddingModel,
    },
    memory_note::situation_mem::{Environment, Event, Location, Participant},
    memory_note::{MemoryId, MemoryNote},
    query::{
        self,
        retrieve::{
            EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
            MemoryRetrieveQueryVariant, ParticipantQueryUnit, PrioritizedMemoryRetrieveQuery,
            SemanticQueryUnit,
        },
    },
};

pub trait AnonymousQueryCompute {
    type Query;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32>;
}

pub trait QueryCompute: AnonymousQueryCompute {
    fn compute(&self, query: &Self::Query) -> EmbeddingCalcResult<QueryComputeResult>;
}

pub struct QueryComputeResult {
    pub id: MemoryId,
    pub score: f32,
}

impl QueryComputeResult {
    pub fn new(id: MemoryId, score: f32) -> Self {
        QueryComputeResult { id, score }
    }
}
////////////////////////////////////////////////////////////
impl AnonymousQueryCompute for LocationEmbedding {
    type Query = LocationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let name_score = self.name().cosine_similarity(query.name())?;
        let coordinates_score = query
            .coordinates()
            .map(|coordinate| coordinate.cosine_similarity(self.coordinates()))
            .transpose()?;

        if let Some(coord_score) = coordinates_score {
            Ok(name_score * 0.6 + coord_score * 0.4)
        } else {
            Ok(name_score)
        }
    }
}

impl AnonymousQueryCompute for ParticipantEmbedding {
    type Query = ParticipantQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let name_score = query
            .name()
            .map(|name| name.cosine_similarity(self.name()))
            .transpose()?;

        let role_score = query
            .role()
            .map(|role| role.cosine_similarity(self.role()))
            .transpose()?;

        match (name_score, role_score) {
            (Some(name_score), Some(role_score)) => Ok(name_score * 0.6 + role_score * 0.4),
            (Some(name_score), None) => Ok(name_score),
            (None, Some(role_score)) => Ok(role_score),
            (None, None) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for EnvironmentEmbedding {
    type Query = EnvironmentQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let atmosphere_score = query
            .atmosphere()
            .map(|atmosphere| atmosphere.cosine_similarity(self.atmosphere()))
            .transpose()?;

        let tone_score = query
            .tone()
            .map(|tone| tone.cosine_similarity(self.tone()))
            .transpose()?;

        match (atmosphere_score, tone_score) {
            (Some(atmosphere_score), Some(tone_score)) => {
                Ok(atmosphere_score * 0.5 + tone_score * 0.5)
            }
            (Some(atmosphere_score), None) => Ok(atmosphere_score),
            (None, Some(tone_score)) => Ok(tone_score),
            (None, None) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for EventEmbedding {
    type Query = EventQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let action_score = self.action().cosine_similarity(query.action())?;

        let initiator_score = query
            .initiator()
            .map(|initiator| initiator.cosine_similarity(self.initiator()))
            .transpose()?;

        let target_score = query
            .target()
            .map(|target| target.cosine_similarity(self.target()))
            .transpose()?;

        match (initiator_score, target_score) {
            (Some(initiator_score), Some(target_score)) => {
                Ok(initiator_score * 0.3 + target_score * 0.3 + action_score * 0.4)
            }
            (Some(initiator_score), None) => Ok(initiator_score * 0.4 + action_score * 0.6),
            (None, Some(target_score)) => Ok(target_score * 0.4 + action_score * 0.6),
            (None, None) => Ok(action_score),
        }
    }
}

impl AnonymousQueryCompute for SpecificSituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let narrative_score = query
            .narrative()
            .map(|narrative| narrative.cosine_similarity(self.narrative()))
            .transpose()?;

        //location
        let location_score = if let Some(query_location) = query.location() {
            self.context()
                .location()
                .map(|location| location.anonymous_compute(query_location))
                .transpose()?
        } else {
            None
        };

        //participants
        let participants_score = if let Some(query_participants) = query.participants() {
            self.context()
                .fused_participant()
                .map(|participants| participants.anonymous_compute(query_participants))
                .transpose()?
        } else {
            None
        };

        //environment
        let environment_score = query
            .environment()
            .map(|env| self.context().environment().anonymous_compute(env))
            .transpose()?;

        //event
        let event_score = if let Some(query_event) = query.event() {
            self.context()
                .fused_event()
                .map(|event| event.anonymous_compute(query_event))
                .transpose()?
        } else {
            None
        };

        //fuse score
        let score_vec = narrative_score
            .into_iter()
            .chain(location_score.into_iter())
            .chain(participants_score.into_iter())
            .chain(environment_score.into_iter())
            .chain(event_score.into_iter())
            .collect::<Vec<_>>();

        let len = score_vec.len();
        Ok(score_vec.into_iter().map(|i| i / len as f32).sum::<f32>())
    }
}

impl AnonymousQueryCompute for AbstractSituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match self {
            AbstractSituationEmbedding::Location(loc) => query
                .location()
                .map(|q_loc| loc.anonymous_compute(q_loc))
                .unwrap_or(Ok(0.0)),
            AbstractSituationEmbedding::Environment(env) => query
                .environment()
                .map(|q_env| env.anonymous_compute(q_env))
                .unwrap_or(Ok(0.0)),
            AbstractSituationEmbedding::Event(event) => query
                .event()
                .map(|q_event| event.anonymous_compute(q_event))
                .unwrap_or(Ok(0.0)),
            AbstractSituationEmbedding::Participant(participant) => query
                .participants()
                .map(|q_participant| participant.anonymous_compute(q_participant))
                .unwrap_or(Ok(0.0)),
        }
    }
}

impl AnonymousQueryCompute for SituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    //TODO: add time span score count
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match self {
            Self::Specific(specific) => specific.anonymous_compute(query),
            Self::Abstract(abstract_sit) => abstract_sit.anonymous_compute(query),
        }
    }
}

impl AnonymousQueryCompute for SemanticEmbedding {
    type Query = SemanticQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let concept_main_score = query
            .concept_identifier()
            .map(|con| con.cosine_similarity(self.content()))
            .transpose()?;
        let concept_aliases_score = query
            .concept_identifier()
            .map(|con| con.cosine_similarity(self.fused_aliases()))
            .transpose()?;

        let description_score = query
            .description()
            .map(|description| description.cosine_similarity(self.description()))
            .transpose()?;

        let concept_score = match (concept_main_score, concept_aliases_score) {
            (Some(main_score), Some(aliases_score)) => 0.7 * main_score + 0.3 * aliases_score,
            (None, None) => 0.0,
            _ => unreachable!(
                "main_score and aliases_score all compute from query.concept_identifier(), so they must be Some or None simultaneously"
            ),
        };

        if let Some(description_score) = description_score {
            Ok(concept_score * 0.5 + description_score * 0.5)
        } else {
            Ok(concept_score)
        }
    }
}

impl AnonymousQueryCompute for MemoryEmbeddingVariant {
    type Query = MemoryRetrieveQueryVariantEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match (self, query) {
            (Self::Semantic(sem), MemoryRetrieveQueryVariantEmbedding::Semantic(q_sem)) => {
                let score_vec = q_sem
                    .into_iter()
                    .map(|q_sem_unit| sem.anonymous_compute(q_sem_unit))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(score_vec.into_iter().sum::<f32>())
            }
            (Self::Situation(sit), MemoryRetrieveQueryVariantEmbedding::Situation(q_sit)) => {
                let score_vec = q_sit
                    .into_iter()
                    .map(|q_sit_unit| sit.anonymous_compute(q_sit_unit))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(score_vec.into_iter().sum::<f32>())
            }
            (_, _) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for MemoryEmbedding {
    type Query = MemoryRetrieveQueryEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let tag_score = self.tag().cosine_similarity(query.tag())?;
        let variant_score = self.variant().anonymous_compute(query.variant())?;
        Ok(0.4 * tag_score + 0.6 * variant_score)
    }
}

//TODO: take common fields in MemoryNote into computation
impl AnonymousQueryCompute for EmbeddedMemoryNote {
    type Query = MemoryRetrieveQueryEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        self.embedding().anonymous_compute(query)
    }
}

impl QueryCompute for EmbeddedMemoryNote {
    fn compute(&self, query: &Self::Query) -> EmbeddingCalcResult<QueryComputeResult> {
        Ok(QueryComputeResult {
            id: self.note().id(),
            score: self.anonymous_compute(query)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding::embedding_model::bge::BgeSmallZh;
    use crate::memory::memory_note::sem_mem::ConceptType;
    use crate::memory::memory_note::situation_mem::{Environment, Event, Location, Participant};

    #[test]
    fn test_query_compute_result() {
        let memory_id = MemoryId::new();
        let result = QueryComputeResult::new(memory_id, 0.85);
        assert_eq!(result.id, memory_id);
        assert_eq!(result.score, 0.85);
    }

    #[test]
    fn test_semantic_embedding_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let memory = crate::memory::memory_note::sem_mem::SemMemory {
            content: "Rust编程语言".to_string(),
            aliases: vec!["Rust".to_string()],
            concept_type: ConceptType::Entity,
            description: "一种注重安全性的系统编程语言".to_string(),
        };

        let sem_embedding = memory.embed(&model).unwrap();

        let query = SemanticQueryUnit::new()
            .with_concept_identifier("Rust".to_string())
            .with_description("系统编程语言".to_string());

        let query_emb = query.embed(&model).unwrap();

        let score = sem_embedding.anonymous_compute(&query_emb).unwrap();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_location_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let location = Location {
            name: "北京".to_string(),
            coordinates: "中国".to_string(),
        };
        let location_emb = location.embed(&model).unwrap();

        let location_query = LocationQueryUnit::new("北京").with_coordinates("中国".to_string());
        let location_query_emb = location_query.embed(&model).unwrap();

        let score = location_emb.anonymous_compute(&location_query_emb).unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn test_participant_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let participant = Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        };
        let participant_emb = participant.embed(&model).unwrap();

        let participant_query = ParticipantQueryUnit::new()
            .with_name("张三".to_string())
            .with_role("学生".to_string());
        let participant_query_emb = participant_query.embed(&model).unwrap();

        let score = participant_emb
            .anonymous_compute(&participant_query_emb)
            .unwrap();
        assert!(score > 0.5);
    }

    #[test]
    fn test_environment_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let environment = Environment {
            atmosphere: "安静".to_string(),
            tone: "舒适".to_string(),
        };
        let environment_emb = environment.embed(&model).unwrap();

        let environment_query = EnvironmentQueryUnit::new()
            .with_atmosphere("安静".to_string())
            .with_tone("舒适".to_string());
        let environment_query_emb = environment_query.embed(&model).unwrap();

        let score = environment_emb
            .anonymous_compute(&environment_query_emb)
            .unwrap();
        assert!(score > 0.5);
    }

    #[test]
    fn test_event_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let event = Event {
            action: "跑步".to_string(),
            action_intensity: 0.8,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        };
        let event_emb = event.embed(&model).unwrap();

        let event_query = EventQueryUnit::new("跑步".to_string())
            .with_initiator("张三".to_string())
            .with_target("操场".to_string());
        let event_query_emb = event_query.embed(&model).unwrap();

        let score = event_emb.anonymous_compute(&event_query_emb).unwrap();
        assert!(score > 0.0);
    }
}
