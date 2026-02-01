use crate::memory::{
    embedding::{
        Embeddable, EmbeddingVec,
        situation::{
            emotion::EmotionEmbedding, environment::EnvironmentEmbedding, event::EventEmbedding,
            location::LocationEmbedding, participant::ParticipantEmbedding,
            sensory_data::SensoryDataEmbedding,
        },
    },
    memory_note::situation_mem::Context,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ContextEmbedding {
    location: Option<LocationEmbedding>,
    fused_participant: Option<ParticipantEmbedding>,
    fused_emotion: Option<EmotionEmbedding>,
    fused_sensory_data: Option<SensoryDataEmbedding>,
    environment: EnvironmentEmbedding,
    fused_event: Option<EventEmbedding>,
}
impl ContextEmbedding {
    pub fn location(&self) -> &Option<LocationEmbedding> {
        &self.location
    }
    pub fn fused_participant(&self) -> &Option<ParticipantEmbedding> {
        &self.fused_participant
    }
    pub fn fused_emotion(&self) -> &Option<EmotionEmbedding> {
        &self.fused_emotion
    }
    pub fn fused_event(&self) -> &Option<EventEmbedding> {
        &self.fused_event
    }
    pub fn fused_sensory_data(&self) -> &Option<SensoryDataEmbedding> {
        &self.fused_sensory_data
    }
    pub fn environment(&self) -> &EnvironmentEmbedding {
        &self.environment
    }
}

impl Embeddable for Context {
    type EmbeddingFused = EmbeddedContext;
    type EmbeddingGen = ContextEmbedding;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let location_vec = self
            .location()
            .as_ref()
            .map(|location| location.embed(model))
            .transpose()?;

        let participants_vecs = self
            .participants()
            .iter()
            .map(|p| p.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_participant_vec = ParticipantEmbedding::mean_pooling(&participants_vecs)?;

        let emotions_vecs = self
            .emotions()
            .iter()
            .map(|e| e.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_emotion_vec = EmotionEmbedding::weight_pooling(&emotions_vecs)?;

        let sensory_data_vecs = self
            .sensory_data()
            .iter()
            .map(|s| s.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_sensory_data_vec = SensoryDataEmbedding::weight_pooling(&sensory_data_vecs)?;

        let environment_vec = self.environment().embed(model)?;

        let event_vecs = self
            .events()
            .iter()
            .map(|e| e.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_event_vec = EventEmbedding::weight_pooling(&event_vecs)?;

        Ok(ContextEmbedding {
            location: location_vec,
            fused_participant: fused_participant_vec,
            fused_emotion: fused_emotion_vec,
            fused_sensory_data: fused_sensory_data_vec,
            environment: environment_vec,
            fused_event: fused_event_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedContext {
            embedding: self.embed(model)?,
            context: self,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedContext {
    pub embedding: ContextEmbedding,
    pub context: Context,
}
