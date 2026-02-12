use crate::memory::{
    embedding::{
        EmbeddingCalcResult, MemoryEmbedding, query::{sem::SemanticQueryUnitEmbedding, situation::{
            SituationQueryUnitEmbedding, environment::EnvironmentQueryUnitEmbedding,
            event::EventQueryUnitEmbedding, location::LocationQueryUnitEmbedding,
            participant::ParticipantQueryUnitEmbedding,
        }}, sem::SemanticEmbedding, situation::{
            AbstractSituationEmbedding, SituationEmbedding, SpecificSituationEmbedding,
            environment::EnvironmentEmbedding, event::EventEmbedding, location::LocationEmbedding,
            participant::ParticipantEmbedding,
        }
    },
    memory_note::{EmbedMemoryNote, MemoryNote},
    query::{
        self,
        retrieve::{LocationQueryUnit, MemoryRetrieveQuery, PrioritizedMemoryRetrieveQuery, SemanticQueryUnit},
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
            .map(|env| self.context().environment().anonymous_compute(env)?);

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
        Ok(score_vec.into_iter().map(|i| i / len).sum::<f32>())
    }
}

impl AnonymousQueryCompute for AbstractSituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match self {
            AbstractSituationEmbedding::Environment(env) => query
                .environment()
                .map(|q_env| env.anonymous_compute(q_env))
                .unwrap_or(Ok(0.0)),
            AbstractSituationEmbedding::Event(event) => query
                .event()
                .map(|q_event| event.anonymous_compute(q_event))
                .unwrap_or(Ok(0.0)),
            AbstractSituationEmbedding::Situation(situation) => query
                .situation()
                .map(|q_situation| situation.anonymous_compute(q_situation))
                .unwrap_or(Ok(0.0)),
            AbstractSituationEmbedding::Participant(participant) => query
                .participant()
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
            Self::Abstract(abstract) => abstract.anonymous_compute(query),
        }
    }
}

impl AnonymousQueryCompute for SemanticEmbedding {
    type Query = SemanticQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let concept_main_score = query.concept_identifier().map(|con| con.cosine_similarity(self.content())).transpose()?;
        let concept_aliases_score = query.concept_identifier().map(|con| con.cosine_similarity(self.fused_aliases())).transpose()?;

        let description_score = query.description().map(|description| description.cosine_similarity(self.description())).transpose()?;

        let concept_score = match (concept_main_score, concept_aliases_score) {
            (Some(main_score), Some(aliases_score)) => 0.7 * main_score + 0.3 * aliases_score,
            (None, None) => 0.0,
            _ => unreachable!("main_score and aliases_score all compute from query.concept_identifier(), so they must be Some or None simultaneously")
        };

        if let Some(description_score) = description_score {
            Ok(concept_score * 0.5 + description_score * 0.5)
        } else {
            Ok(concept_score)
        }
    }
}

impl AnonymousQueryCompute for MemoryEmbedding {
    type Query = MemoryRetrieveQuery;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match (self, query) {
            (Self::Semantic(sem), MemoryRetrieveQuery::Semantic(q_sem)) => {
                let score_vec = q_sem.into_iter().map(|q_sem_unit| sem.anonymous_compute(q_sem_unit)).collect::<Result<Vec<_>, _>>()?;
                Ok(score_vec.into_iter().sum::<f32>())
            },
            (Self::Situation(sit), MemoryRetrieveQuery::Situation(q_sit)) => {
                let score_vec = q_sit.into_iter().map(|q_sit_unit| sit.anonymous_compute(q_sit_unit)).collect::<Result<Vec<_>, _>>()?;
                Ok(score_vec.into_iter().sum::<f32>())
            }
            (_, _) => Ok(0.0)
        }
    }
}

//TODO: take common fields in MemoryNote into computation
impl AnonymousQueryCompute for EmbedMemoryNote {
    type Query = MemoryRetrieveQuery;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        self.embedding().anonymous_compute(query)
    }
}

impl QueryCompute for EmbedMemoryNote {
    fn compute(&self, query: &Self::Query) -> EmbeddingCalcResult<QueryComputeResult> {
        Ok(QueryComputeResult { id: self.note().id(), score: self.anonymous_compute(query)? })
    }
}
