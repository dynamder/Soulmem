use serde::{Deserialize, Serialize};

use crate::embedding::{Embeddable, EmbeddingCalcError, EmbeddingCalcResult, EmbeddingVec};
use soul_mem_core::memory_note::situation_mem::Event;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventEmbedding {
    pub action: EmbeddingVec,
    pub initiator: EmbeddingVec,
    pub target: EmbeddingVec,
    pub intensity: f32,
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
        //intensity_sum为0时0/0会产生NaN，直接报错，防止NaN污染后续检索
        if intensity_sum == 0.0 {
            return Err(EmbeddingCalcError::InvalidNumValue);
        }
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
#[cfg(test)]
impl EventEmbedding {
    pub(crate) fn test_new(
        action: EmbeddingVec,
        initiator: EmbeddingVec,
        target: EmbeddingVec,
        intensity: f32,
    ) -> Self {
        Self {
            action,
            initiator,
            target,
            intensity,
        }
    }
}
impl Embeddable for Event {
    type EmbeddingGen = EventEmbedding;
    type EmbeddingFused = EmbeddedEvent;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [action_vec] = model
            .infer_batch(&[self.action.as_str()])?
            .try_into()
            .unwrap(); // SAFEUNWRAP: 此处长度必为1

        let [initiator_vec] = model
            .infer_batch(&[self.initiator.as_str()])?
            .try_into()
            .unwrap(); // SAFEUNWRAP: 此处长度必为1

        let [target_vec] = model
            .infer_batch(&[self.target.as_str()])?
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
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
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
    use crate::embedding::embedding_model::bge::BgeSmallZh;

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

    fn embed_event(action: Vec<f32>, initiator: Vec<f32>, target: Vec<f32>, intensity: f32) -> EventEmbedding {
        EventEmbedding {
            action: EmbeddingVec::new(action),
            initiator: EmbeddingVec::new(initiator),
            target: EmbeddingVec::new(target),
            intensity,
        }
    }

    #[test]
    fn test_event_weight_pooling_values() {
        // 两事件：intensity 0.5 + 0.5 = 1.0，权重各为 0.5
        let e1 = embed_event(vec![1.0, 10.0], vec![1.0, 10.0], vec![1.0, 10.0], 0.5);
        let e2 = embed_event(vec![3.0, 20.0], vec![3.0, 20.0], vec![3.0, 20.0], 0.5);
        let pooled = EventEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        assert_close(pooled.action.iter().copied().collect::<Vec<_>>()[0], 2.0);
        assert_close(pooled.action.iter().copied().collect::<Vec<_>>()[1], 15.0);
        assert_close(pooled.initiator.iter().copied().collect::<Vec<_>>()[0], 2.0);
        assert_close(pooled.target.iter().copied().collect::<Vec<_>>()[0], 2.0);
        assert_close(pooled.intensity(), 1.0);
    }

    #[test]
    fn test_event_weight_pooling_weighted() {
        // 权重不对称：0.75 / 0.25
        let e1 = embed_event(vec![4.0], vec![0.0], vec![2.0], 3.0);
        let e2 = embed_event(vec![0.0], vec![0.0], vec![0.0], 1.0);
        let pooled = EventEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        // action: 4*0.75 + 0*0.25 = 3.0; target: 2*0.75 + 0*0.25 = 1.5
        assert_close(pooled.action.iter().copied().next().unwrap(), 3.0);
        assert_close(pooled.target.iter().copied().next().unwrap(), 1.5);
        assert_close(pooled.intensity(), 4.0);
    }

    #[test]
    fn test_event_weight_pooling_three_components() {
        // 三个字段都带非零值，验证每个字段的加权融合
        let e1 = embed_event(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], 1.0);
        let e2 = embed_event(vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0], 1.0);
        let pooled = EventEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        let action_vals = pooled.action.iter().copied().collect::<Vec<_>>();
        let initiator_vals = pooled.initiator.iter().copied().collect::<Vec<_>>();
        let target_vals = pooled.target.iter().copied().collect::<Vec<_>>();
        assert_close(action_vals[0], 4.0);
        assert_close(action_vals[1], 5.0);
        assert_close(initiator_vals[0], 6.0);
        assert_close(target_vals[0], 8.0);
        assert_close(target_vals[1], 9.0);
    }

    #[test]
    fn test_event_weight_pooling_zero_intensity() {
        let e1 = embed_event(vec![1.0], vec![1.0], vec![1.0], 0.0);
        let e2 = embed_event(vec![1.0], vec![1.0], vec![1.0], 0.0);
        assert!(matches!(
            EventEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::InvalidNumValue)
        ));
    }

    #[test]
    fn test_event_weight_pooling_shape_mismatch() {
        let e1 = embed_event(vec![1.0], vec![1.0], vec![1.0], 1.0);
        let e2 = embed_event(vec![1.0, 2.0], vec![1.0], vec![1.0], 1.0);
        assert!(matches!(
            EventEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "expected {actual} close to {expected}");
    }
}
