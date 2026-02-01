use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcError, EmbeddingCalcResult, EmbeddingVec},
    memory_note::situation_mem::Event,
};

#[derive(Debug, Clone, PartialEq)]
pub struct EventEmbedding {
    action: EmbeddingVec,
    intensity: f32,
}
impl EventEmbedding {
    pub fn action(&self) -> &EmbeddingVec {
        &self.action
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
    pub fn weight_pooling(events: &[EventEmbedding]) -> EmbeddingCalcResult<Option<Self>> {
        if events.is_empty() {
            return Ok(None);
        }
        let intensity_sum = events.iter().map(|e| e.intensity).sum::<f32>();
        let len = events[0].action.len();
        if !events.iter().all(|vec| vec.action.len() == len) {
            return Err(EmbeddingCalcError::ShapeMismatch);
        }
        let fused_action = events.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.action.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });

        Ok(Some(EventEmbedding {
            action: fused_action,
            intensity: intensity_sum,
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
        Ok(EventEmbedding {
            action: action_vec,
            intensity: self.action_intensity,
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
