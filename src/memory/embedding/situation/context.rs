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
            .get_location()
            .as_ref()
            .map(|location| location.embed(model))
            .transpose()?;

        let participants_vecs = self
            .get_participants()
            .iter()
            .map(|p| p.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_participant_vec = ParticipantEmbedding::mean_pooling(&participants_vecs)?;

        let emotions_vecs = self
            .get_emotions()
            .iter()
            .map(|e| e.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_emotion_vec = EmotionEmbedding::weight_pooling(&emotions_vecs)?;

        let sensory_data_vecs = self
            .get_sensory_data()
            .iter()
            .map(|s| s.embed(model))
            .collect::<Result<Vec<_>, _>>()?;
        let fused_sensory_data_vec = SensoryDataEmbedding::weight_pooling(&sensory_data_vecs)?;

        let environment_vec = self.get_environment().embed(model)?;

        let event_vecs = self
            .get_event()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{
        embedding::embedding_model::bge::BgeSmallZh,
        memory_note::situation_mem::{
            Emotion, Environment, Event, Location, Participant, SensoryData,
        },
    };

    fn test_context() -> Context {
        let location = Location {
            name: "北京".to_string(),
            coordinates: "亚洲，中国".to_string(),
        };
        let environment = Environment {
            atmosphere: "轻松".to_string(),
            tone: "黄色".to_string(),
        };
        let emotions = vec![
            Emotion {
                name: "快乐".to_string(),
                intensity: 0.8,
            },
            Emotion {
                name: "紧张".to_string(),
                intensity: 0.3,
            },
            Emotion {
                name: "悲伤".to_string(),
                intensity: 0.1,
            },
        ];
        let participants = vec![
            Participant {
                name: "小明".to_string(),
                role: "学生".to_string(),
            },
            Participant {
                name: "小红".to_string(),
                role: "老师".to_string(),
            },
        ];
        let sensory_data = vec![
            SensoryData {
                name: "花香".to_string(),
                intensity: 0.5,
            },
            SensoryData {
                name: "鸟鸣".to_string(),
                intensity: 0.8,
            },
        ];
        let event = Event {
            action: "上课".to_string(),
            action_intensity: 0.7,
            initiator: "小明".to_string(),
            target: "小红".to_string(),
        };
        Context::new(
            Some(location),
            participants,
            emotions,
            sensory_data,
            environment,
            vec![event],
        )
    }

    #[test]
    fn test_embed() {
        let context = test_context();
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedding = context.embed(&model).unwrap();
    }
}
