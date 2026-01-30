pub mod context;
pub mod emotion;
pub mod environment;
pub mod location;
pub mod participant;
pub mod sensory_data;
use crate::memory::embedding::EmbeddingVec;
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
    embedding: EmbeddingVec,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AbstractSituationEmbedding {
    Location(LocationEmbedding),
}
impl AbstractSituationEmbedding {
    pub fn to_location(&self) -> Option<&LocationEmbedding> {
        match self {
            AbstractSituationEmbedding::Location(location) => Some(location),
            _ => None,
        }
    }
}
impl From<LocationEmbedding> for AbstractSituationEmbedding {
    fn from(location: LocationEmbedding) -> Self {
        AbstractSituationEmbedding::Location(location)
    }
}
