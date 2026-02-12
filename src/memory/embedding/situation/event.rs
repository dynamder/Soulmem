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
