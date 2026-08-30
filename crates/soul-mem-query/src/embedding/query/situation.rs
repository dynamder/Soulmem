use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{
    query::situation::{
        environment::EnvironmentQueryUnitEmbedding, event::EventQueryUnitEmbedding,
        location::LocationQueryUnitEmbedding, participant::ParticipantQueryUnitEmbedding,
    },
    vec_batch_embed, Embeddable, EmbeddingVec,
};
use crate::query::retrieve::SituationQueryUnit;

pub mod environment;
pub mod event;
pub mod location;
pub mod participant;

#[derive(Debug, Clone, PartialEq)]
pub struct SituationQueryUnitEmbedding {
    narrative: Option<EmbeddingVec>,
    location: Option<LocationQueryUnitEmbedding>,
    participants: Option<ParticipantQueryUnitEmbedding>,
    environment: Option<EnvironmentQueryUnitEmbedding>,
    event: Option<EventQueryUnitEmbedding>,
    pub blend_weights: BlendWeights,
}
impl SituationQueryUnitEmbedding {
    /// 公开构造（外部/测试构造用；blend_weights 取默认值）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        narrative: Option<EmbeddingVec>,
        location: Option<LocationQueryUnitEmbedding>,
        participants: Option<ParticipantQueryUnitEmbedding>,
        environment: Option<EnvironmentQueryUnitEmbedding>,
        event: Option<EventQueryUnitEmbedding>,
    ) -> Self {
        Self {
            narrative,
            location,
            participants,
            environment,
            event,
            blend_weights: BlendWeights::default(),
        }
    }

    pub fn narrative(&self) -> Option<&EmbeddingVec> {
        self.narrative.as_ref()
    }
    pub fn location(&self) -> Option<&LocationQueryUnitEmbedding> {
        self.location.as_ref()
    }
    pub fn participants(&self) -> Option<&ParticipantQueryUnitEmbedding> {
        self.participants.as_ref()
    }
    pub fn environment(&self) -> Option<&EnvironmentQueryUnitEmbedding> {
        self.environment.as_ref()
    }
    pub fn event(&self) -> Option<&EventQueryUnitEmbedding> {
        self.event.as_ref()
    }
    /// 递归设置 blend weights 到所有子单元
    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
        if let Some(ref mut loc) = self.location {
            loc.set_blend_weights(bw);
        }
        if let Some(ref mut part) = self.participants {
            part.set_blend_weights(bw);
        }
        if let Some(ref mut env) = self.environment {
            env.set_blend_weights(bw);
        }
        if let Some(ref mut evt) = self.event {
            evt.set_blend_weights(bw);
        }
    }

    /// 解构取所有权：narrative 与各子单元（移动而非克隆）。
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Option<EmbeddingVec>,
        Option<LocationQueryUnitEmbedding>,
        Option<ParticipantQueryUnitEmbedding>,
        Option<EnvironmentQueryUnitEmbedding>,
        Option<EventQueryUnitEmbedding>,
    ) {
        let Self {
            narrative,
            location,
            participants,
            environment,
            event,
            blend_weights: _,
        } = self;
        (narrative, location, participants, environment, event)
    }
}
#[cfg(test)]
impl SituationQueryUnitEmbedding {
    pub(crate) fn test_new(
        narrative: Option<EmbeddingVec>,
        location: Option<LocationQueryUnitEmbedding>,
        participants: Option<ParticipantQueryUnitEmbedding>,
        environment: Option<EnvironmentQueryUnitEmbedding>,
        event: Option<EventQueryUnitEmbedding>,
        blend_weights: BlendWeights,
    ) -> Self {
        Self {
            narrative,
            location,
            participants,
            environment,
            event,
            blend_weights,
        }
    }
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
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        //narrative
        let narrative_vec = self
            .narrative()
            .map(|narrative| model.infer_query_with_chunk(narrative))
            .transpose()?;

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
            narrative: narrative_vec,
            location: fused_location_vec,
            participants: fused_participant_vec,
            environment: environment_vec,
            event: fused_event_vec,
            blend_weights: BlendWeights::default(),
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbbdedSituationQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;

    use crate::{
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

    #[test]
    fn test_set_blend_weights_propagates() {
        let mut bw = BlendWeights::default();
        bw.tag = 0.9;

        let mut embedding = SituationQueryUnitEmbedding {
            narrative: Some(EmbeddingVec::new(vec![1.0])),
            location: Some(LocationQueryUnitEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                None,
                BlendWeights::default(),
            )),
            participants: Some(ParticipantQueryUnitEmbedding::test_new(
                Some(EmbeddingVec::new(vec![1.0])),
                None,
                BlendWeights::default(),
            )),
            environment: Some(EnvironmentQueryUnitEmbedding::test_new(
                Some(EmbeddingVec::new(vec![1.0])),
                None,
                BlendWeights::default(),
            )),
            event: Some(EventQueryUnitEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                None,
                None,
                BlendWeights::default(),
            )),
            blend_weights: BlendWeights::default(),
        };

        embedding.set_blend_weights(&bw);
        assert_eq!(embedding.blend_weights.tag, 0.9);
        assert_eq!(embedding.location.as_ref().unwrap().blend_weights.tag, 0.9);
        assert_eq!(embedding.participants.as_ref().unwrap().blend_weights.tag, 0.9);
        assert_eq!(embedding.environment.as_ref().unwrap().blend_weights.tag, 0.9);
        assert_eq!(embedding.event.as_ref().unwrap().blend_weights.tag, 0.9);
    }

    #[test]
    fn test_situation_query_unit_accessors() {
        let embedding = SituationQueryUnitEmbedding {
            narrative: Some(EmbeddingVec::new(vec![1.0])),
            location: Some(LocationQueryUnitEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                None,
                BlendWeights::default(),
            )),
            participants: Some(ParticipantQueryUnitEmbedding::test_new(
                Some(EmbeddingVec::new(vec![1.0])),
                None,
                BlendWeights::default(),
            )),
            environment: Some(EnvironmentQueryUnitEmbedding::test_new(
                Some(EmbeddingVec::new(vec![1.0])),
                None,
                BlendWeights::default(),
            )),
            event: Some(EventQueryUnitEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                None,
                None,
                BlendWeights::default(),
            )),
            blend_weights: BlendWeights::default(),
        };
        assert!(embedding.narrative().is_some());
        assert!(embedding.location().is_some());
        assert!(embedding.participants().is_some());
        assert!(embedding.environment().is_some());
        assert!(embedding.event().is_some());
    }
}
