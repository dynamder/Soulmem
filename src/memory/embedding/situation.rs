pub mod context;
pub mod emotion;
pub mod environment;
pub mod event;
pub mod location;
pub mod participant;
pub mod sensory_data;
use crate::memory::{
    embedding::{
        Embeddable, EmbeddingVec,
        situation::{
            context::ContextEmbedding, environment::EnvironmentEmbedding, event::EventEmbedding,
            participant::ParticipantEmbedding,
        },
    },
    memory_note::situation_mem::{AbstractSituation, SpecificSituation},
};
use location::LocationEmbedding;
#[derive(Debug, Clone, PartialEq)]
pub enum SituationEmbedding {
    Specific(SpecificSituationEmbedding),
    Abstract(AbstractSituationEmbedding),
}
impl SituationEmbedding {
    pub fn to_specific(&self) -> Option<&SpecificSituationEmbedding> {
        match self {
            SituationEmbedding::Specific(embedding) => Some(embedding),
            _ => None,
        }
    }
    pub fn to_abstract(&self) -> Option<&AbstractSituationEmbedding> {
        match self {
            SituationEmbedding::Abstract(embedding) => Some(embedding),
            _ => None,
        }
    }
}
impl From<AbstractSituationEmbedding> for SituationEmbedding {
    fn from(value: AbstractSituationEmbedding) -> Self {
        SituationEmbedding::Abstract(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpecificSituationEmbedding {
    narrative: EmbeddingVec,
    context: ContextEmbedding,
}
impl SpecificSituationEmbedding {
    pub fn narrative(&self) -> &EmbeddingVec {
        &self.narrative
    }
    pub fn context(&self) -> &ContextEmbedding {
        &self.context
    }
}
impl Embeddable for SpecificSituation {
    type EmbeddingGen = SpecificSituationEmbedding;
    type EmbeddingFused = EmbeddedSpecificSituation;
    fn embed(
        &self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingGen> {
        let narrative_vec = model.infer_with_chunk(&self.get_narrative().as_str())?;
        let context_vec = self.get_context().embed(model)?;
        Ok(SpecificSituationEmbedding {
            narrative: narrative_vec,
            context: context_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSpecificSituation {
            embedding: self.embed(model)?,
            specific_situation: self,
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedSpecificSituation {
    pub embedding: SpecificSituationEmbedding,
    pub specific_situation: SpecificSituation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AbstractSituationEmbedding {
    Location(LocationEmbedding),
    Participant(ParticipantEmbedding),
    Environment(EnvironmentEmbedding),
    Event(EventEmbedding),
}

impl AbstractSituationEmbedding {
    pub fn to_location(&self) -> Option<&LocationEmbedding> {
        match self {
            AbstractSituationEmbedding::Location(location) => Some(location),
            _ => None,
        }
    }
    pub fn to_participant(&self) -> Option<&ParticipantEmbedding> {
        match self {
            AbstractSituationEmbedding::Participant(participant) => Some(participant),
            _ => None,
        }
    }
    pub fn to_environment(&self) -> Option<&EnvironmentEmbedding> {
        match self {
            AbstractSituationEmbedding::Environment(environment) => Some(environment),
            _ => None,
        }
    }
    pub fn to_event(&self) -> Option<&EventEmbedding> {
        match self {
            AbstractSituationEmbedding::Event(event) => Some(event),
            _ => None,
        }
    }
}

impl Embeddable for AbstractSituation {
    type EmbeddingGen = AbstractSituationEmbedding;
    type EmbeddingFused = EmbeddedAbstractSituation;
    fn embed(
        &self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingGen> {
        match self {
            Self::Environment(env) => {
                Ok(AbstractSituationEmbedding::Environment(env.embed(model)?))
            }
            Self::Event(eve) => Ok(AbstractSituationEmbedding::Event(eve.embed(model)?)),
            Self::Location(loc) => Ok(AbstractSituationEmbedding::Location(loc.embed(model)?)),
            Self::Participant(par) => {
                Ok(AbstractSituationEmbedding::Participant(par.embed(model)?))
            }
        }
    }
    fn embed_and_fuse(
        self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedAbstractSituation {
            embedding: self.embed(model)?,
            abstract_situation: self,
        })
    }
}

impl From<LocationEmbedding> for AbstractSituationEmbedding {
    fn from(location: LocationEmbedding) -> Self {
        AbstractSituationEmbedding::Location(location)
    }
}
impl From<ParticipantEmbedding> for AbstractSituationEmbedding {
    fn from(participant: ParticipantEmbedding) -> Self {
        AbstractSituationEmbedding::Participant(participant)
    }
}
impl From<EnvironmentEmbedding> for AbstractSituationEmbedding {
    fn from(environment: EnvironmentEmbedding) -> Self {
        AbstractSituationEmbedding::Environment(environment)
    }
}
impl From<EventEmbedding> for AbstractSituationEmbedding {
    fn from(event: EventEmbedding) -> Self {
        AbstractSituationEmbedding::Event(event)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedAbstractSituation {
    pub embedding: AbstractSituationEmbedding,
    pub abstract_situation: AbstractSituation,
}

#[cfg(test)]
mod test {
    use super::*;
}
