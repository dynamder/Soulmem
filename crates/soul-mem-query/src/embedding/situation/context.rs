use serde::{Deserialize, Serialize};

use crate::embedding::{
    Embeddable,
    situation::{
        emotion::EmotionEmbedding, environment::EnvironmentEmbedding, event::EventEmbedding,
        location::LocationEmbedding, participant::ParticipantEmbedding,
        sensory_data::SensoryDataEmbedding,
    },
};
use soul_mem_core::memory_note::situation_mem::Context;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextEmbedding {
    location: Option<LocationEmbedding>,
    fused_participant: Option<ParticipantEmbedding>,
    fused_emotion: Option<EmotionEmbedding>,
    fused_sensory_data: Option<SensoryDataEmbedding>,
    environment: EnvironmentEmbedding,
    fused_event: Option<EventEmbedding>,
}
impl ContextEmbedding {
    pub fn location(&self) -> Option<&LocationEmbedding> {
        self.location.as_ref()
    }
    pub fn fused_participant(&self) -> Option<&ParticipantEmbedding> {
        self.fused_participant.as_ref()
    }
    pub fn fused_emotion(&self) -> Option<&EmotionEmbedding> {
        self.fused_emotion.as_ref()
    }
    pub fn fused_event(&self) -> Option<&EventEmbedding> {
        self.fused_event.as_ref()
    }
    pub fn fused_sensory_data(&self) -> Option<&SensoryDataEmbedding> {
        self.fused_sensory_data.as_ref()
    }
    pub fn environment(&self) -> &EnvironmentEmbedding {
        &self.environment
    }
}

#[cfg(test)]
impl ContextEmbedding {
    pub(crate) fn test_new(
        location: Option<LocationEmbedding>,
        fused_participant: Option<ParticipantEmbedding>,
        fused_emotion: Option<EmotionEmbedding>,
        fused_sensory_data: Option<SensoryDataEmbedding>,
        environment: EnvironmentEmbedding,
        fused_event: Option<EventEmbedding>,
    ) -> Self {
        Self {
            location,
            fused_participant,
            fused_emotion,
            fused_sensory_data,
            environment,
            fused_event,
        }
    }
}

impl Embeddable for Context {
    type EmbeddingFused = EmbeddedContext;
    type EmbeddingGen = ContextEmbedding;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
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
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
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
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use crate::embedding::EmbeddingVec;
    use soul_mem_core::memory_note::situation_mem::{
        Emotion, Environment, Event, Location, Participant, SensoryData,
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
        let _embedding = context.embed(&model).unwrap();
    }

    fn sample_embedding() -> ContextEmbedding {
        ContextEmbedding {
            location: Some(LocationEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                EmbeddingVec::new(vec![2.0]),
            )),
            fused_participant: Some(ParticipantEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                EmbeddingVec::new(vec![2.0]),
                EmbeddingVec::new(vec![1.5]),
            )),
            fused_emotion: Some(EmotionEmbedding {
                emotion: EmbeddingVec::new(vec![1.0]),
                intensity: 0.5,
            }),
            fused_sensory_data: Some(SensoryDataEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                0.5,
            )),
            environment: EnvironmentEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                EmbeddingVec::new(vec![2.0]),
            ),
            fused_event: Some(EventEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                EmbeddingVec::new(vec![2.0]),
                EmbeddingVec::new(vec![3.0]),
                0.5,
            )),
        }
    }

    #[test]
    fn test_context_embedding_accessors() {
        let ctx = sample_embedding();
        assert!(ctx.location().is_some());
        assert!(ctx.fused_participant().is_some());
        assert!(ctx.fused_emotion().is_some());
        assert!(ctx.fused_sensory_data().is_some());
        assert!(ctx.fused_event().is_some());
        assert_eq!(ctx.environment().atmosphere().shape(), 1);

        let empty = ContextEmbedding {
            location: None,
            fused_participant: None,
            fused_emotion: None,
            fused_sensory_data: None,
            environment: EnvironmentEmbedding::test_new(
                EmbeddingVec::new(vec![1.0]),
                EmbeddingVec::new(vec![2.0]),
            ),
            fused_event: None,
        };
        assert!(empty.location().is_none());
        assert!(empty.fused_participant().is_none());
        assert!(empty.fused_emotion().is_none());
        assert!(empty.fused_sensory_data().is_none());
        assert!(empty.fused_event().is_none());
    }
}
