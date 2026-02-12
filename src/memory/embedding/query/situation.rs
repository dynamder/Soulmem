use crate::memory::{
    embedding::{
        Embeddable,
        query::situation::{
            environment::EnvironmentQueryUnitEmbedding, event::EventQueryUnitEmbedding,
            location::LocationQueryUnitEmbedding, participant::ParticipantQueryUnitEmbedding,
        },
        vec_batch_embed,
    },
    query::retrieve::SituationQueryUnit,
};

pub mod environment;
pub mod event;
pub mod location;
pub mod participant;

#[derive(Debug, Clone, PartialEq)]
pub struct SituationQueryUnitEmbedding {
    location: Option<LocationQueryUnitEmbedding>,
    participants: Option<ParticipantQueryUnitEmbedding>,
    environment: Option<EnvironmentQueryUnitEmbedding>,
    event: Option<EventQueryUnitEmbedding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbbdedSituationQueryUnit {
    pub embedding: SituationQueryUnitEmbedding,
    pub query: SituationQueryUnit,
}

impl Embeddable for SituationQueryUnit {
    type EmbeddingGen = SituationQueryUnitEmbedding;
    type EmbeddingFused = EmbbdedSituationQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        //location
        let location_vecs = self
            .location()
            .map(|locations| vec_batch_embed(locations, model))
            .transpose()?;

        let fused_location_vec = location_vecs
            .map(|vecs| LocationQueryUnitEmbedding::mean_pooling(&vecs))
            .transpose()?
            .flatten();

        //participant
        let participant_vecs = self
            .participants()
            .map(|participants| vec_batch_embed(participants, model))
            .transpose()?;

        let fused_participant_vec = participant_vecs
            .map(|vecs| ParticipantQueryUnitEmbedding::mean_pooling(&vecs))
            .transpose()?
            .flatten();

        //environment
        let environment_vec = self
            .environment()
            .map(|environments| environments.embed(model))
            .transpose()?;

        //event
        let event_vecs = self
            .event()
            .map(|events| vec_batch_embed(events, model))
            .transpose()?;

        let fused_event_vec = event_vecs
            .map(|vecs| EventQueryUnitEmbedding::mean_pooling(&vecs))
            .transpose()?
            .flatten();

        Ok(SituationQueryUnitEmbedding {
            location: fused_location_vec,
            participants: fused_participant_vec,
            environment: environment_vec,
            event: fused_event_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbbdedSituationQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;

    use crate::memory::{
        embedding::embedding_model::bge::BgeSmallZh,
        query::retrieve::{
            EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, ParticipantQueryUnit,
            TimeSpanQueryUnit,
        },
    };

    use super::*;

    #[test]
    fn test_situation_query_unit_embed() {
        let situation = SituationQueryUnit::new()
            .with_environment(
                EnvironmentQueryUnit::new()
                    .with_atmosphere("atmosphere")
                    .with_tone("tone"),
            )
            .with_event(vec![
                EventQueryUnit::new("action")
                    .with_initiator("initiator")
                    .with_target("target"),
                EventQueryUnit::new("action")
                    .with_initiator("initiator")
                    .with_target("target"),
                EventQueryUnit::new("action")
                    .with_initiator("initiator")
                    .with_target("target"),
            ])
            .with_location(vec![
                LocationQueryUnit::new("name").with_coordinates("coordinates"),
                LocationQueryUnit::new("name").with_coordinates("coordinates"),
            ])
            .with_participants(vec![
                ParticipantQueryUnit::new()
                    .with_name("name")
                    .with_role("role"),
                ParticipantQueryUnit::new()
                    .with_name("name")
                    .with_role("role"),
            ])
            .with_time_span(vec![
                TimeSpanQueryUnit::new()
                    .with_start(DateTime::from_timestamp_nanos(100))
                    .with_end(DateTime::from_timestamp_nanos(1000)),
                TimeSpanQueryUnit::new()
                    .with_start(DateTime::from_timestamp_nanos(200))
                    .with_end(DateTime::from_timestamp_nanos(2000)),
            ]);

        let model = BgeSmallZh::default_cpu().unwrap();

        situation.embed_and_fuse(&model).unwrap();
    }
}
