use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcError, EmbeddingCalcResult, EmbeddingVec},
    memory_note::situation_mem::Event,
};

#[derive(Debug, Clone, PartialEq)]
pub struct EventEmbedding {
    action: EmbeddingVec,
    initiator: EmbeddingVec,
    target: EmbeddingVec,
    intensity: f32,
}
impl EventEmbedding {
    pub fn action(&self) -> &EmbeddingVec {
        &self.action
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
    pub fn initiator(&self) -> &EmbeddingVec {
        &self.initiator
    }
    pub fn target(&self) -> &EmbeddingVec {
        &self.target
    }
    pub fn weight_pooling(events: &[EventEmbedding]) -> EmbeddingCalcResult<Option<Self>> {
        if events.is_empty() {
            return Ok(None);
        }
        let intensity_sum = events.iter().map(|e| e.intensity).sum::<f32>();
        let len = events[0].action.shape();
        if !events.iter().all(|vec| vec.action.shape() == len) {
            return Err(EmbeddingCalcError::ShapeMismatch);
        }
        let fused_action = events.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.action.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });
        let fused_initiator = events.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.initiator.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });
        let fused_target = events.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.target.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });

        Ok(Some(EventEmbedding {
            action: EmbeddingVec::new(fused_action),
            intensity: intensity_sum,
            initiator: EmbeddingVec::new(fused_initiator),
            target: EmbeddingVec::new(fused_target),
        }))
    }
}
impl Embeddable for Event {
    type EmbeddingGen = EventEmbedding;
    type EmbeddingFused = EmbeddedEvent;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [action_vec] = model
            .infer_batch(&vec![self.action.as_str()])?
            .try_into()
            .unwrap(); // SAFEUNWRAP: 此处长度必为1

        let [initiator_vec] = model
            .infer_batch(&vec![self.initiator.as_str()])?
            .try_into()
            .unwrap(); // SAFEUNWRAP: 此处长度必为1

        let [target_vec] = model
            .infer_batch(&vec![self.target.as_str()])?
            .try_into()
            .unwrap(); // SAFEUNWRAP: 此处长度必为1

        Ok(EventEmbedding {
            action: action_vec,
            intensity: self.action_intensity,
            initiator: initiator_vec,
            target: target_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedEvent {
            embedding: self.embed(model)?,
            event: self,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedEvent {
    pub embedding: EventEmbedding,
    pub event: Event,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding::embedding_model::bge::BgeSmallZh;

    #[test]
    fn test_event_embed() {
        let event = Event {
            action: "跑步".to_string(),
            action_intensity: 0.8,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        };
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedding = event.embed(&model).unwrap();
        assert_eq!(embedding.action.shape(), 512);
        assert_eq!(embedding.initiator.shape(), 512);
        assert_eq!(embedding.target.shape(), 512);
        assert_eq!(embedding.intensity(), 0.8);
    }

    #[test]
    fn test_event_weight_pooling() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let event1 = Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        };
        let event2 = Event {
            action: "走路".to_string(),
            action_intensity: 0.3,
            initiator: "张三".to_string(),
            target: "教室".to_string(),
        };
        let event3 = Event {
            action: "跳跃".to_string(),
            action_intensity: 0.2,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        };

        let emb1 = event1.embed(&model).unwrap();
        let emb2 = event2.embed(&model).unwrap();
        let emb3 = event3.embed(&model).unwrap();

        let pooled = EventEmbedding::weight_pooling(&[emb1, emb2, emb3])
            .unwrap()
            .unwrap();

        assert_eq!(pooled.action.shape(), 512);
        assert_eq!(pooled.initiator.shape(), 512);
        assert_eq!(pooled.target.shape(), 512);
        assert_eq!(pooled.intensity(), 1.0);
    }

    #[test]
    fn test_event_weight_pooling_empty() {
        let result = EventEmbedding::weight_pooling(&[]);
        assert!(result.unwrap().is_none());
    }
}
