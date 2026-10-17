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
        let Some(first) = events.first() else {
            return Ok(None);
        };
        let intensity_sum = events.iter().map(|e| e.intensity).sum::<f32>();
        //intensity_sum为0时0/0会产生NaN，直接报错，防止NaN污染后续检索
        if intensity_sum == 0.0 {
            return Err(EmbeddingCalcError::InvalidNumValue);
        }

        Ok(Some(EventEmbedding {
            action: weighted_pool(events, first.action.shape(), intensity_sum, |e| &e.action)?,
            initiator: weighted_pool(events, first.initiator.shape(), intensity_sum, |e| {
                &e.initiator
            })?,
            target: weighted_pool(events, first.target.shape(), intensity_sum, |e| &e.target)?,
            intensity: intensity_sum,
        }))
    }
}

/// 同一字段在所有事件里的维度必须一致；不同字段之间不要求一致。
fn weighted_pool(
    events: &[EventEmbedding],
    len: usize,
    intensity_sum: f32,
    field: impl Fn(&EventEmbedding) -> &EmbeddingVec,
) -> EmbeddingCalcResult<EmbeddingVec> {
    if events.iter().any(|event| field(event).shape() != len) {
        return Err(EmbeddingCalcError::ShapeMismatch);
    }
    let mut pooled = vec![0.0f32; len];
    for event in events {
        let weight = event.intensity / intensity_sum;
        for (acc, value) in pooled.iter_mut().zip(field(event).iter()) {
            *acc += value * weight;
        }
    }
    Ok(EmbeddingVec::new(pooled))
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
        let action_vec = if self.action.trim().is_empty() {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_with_chunk(self.action.as_str())?
        };

        let initiator_vec = if self.initiator.trim().is_empty() {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_with_chunk(self.initiator.as_str())?
        };

        let target_vec = if self.target.trim().is_empty() {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_with_chunk(self.target.as_str())?
        };

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

    fn embed_event(
        action: Vec<f32>,
        initiator: Vec<f32>,
        target: Vec<f32>,
        intensity: f32,
    ) -> EventEmbedding {
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
        let pooled = EventEmbedding::weight_pooling(&[e1, e2]).unwrap().unwrap();
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
        let pooled = EventEmbedding::weight_pooling(&[e1, e2]).unwrap().unwrap();
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
        let pooled = EventEmbedding::weight_pooling(&[e1, e2]).unwrap().unwrap();
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

    #[test]
    fn test_event_weight_pooling_allows_different_field_lengths() {
        // 字段之间的维度互不相等是合法的：各自按自己的长度池化
        let e1 = embed_event(vec![1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0], 1.0);
        let e2 = embed_event(vec![7.0, 8.0], vec![9.0, 10.0, 11.0], vec![12.0], 1.0);
        let pooled = EventEmbedding::weight_pooling(&[e1, e2]).unwrap().unwrap();
        assert_eq!(pooled.action.shape(), 2);
        assert_eq!(pooled.initiator.shape(), 3);
        assert_eq!(pooled.target.shape(), 1);
        assert_close(pooled.initiator.iter().copied().collect::<Vec<_>>()[2], 8.0);
    }

    #[test]
    fn test_event_weight_pooling_rejects_initiator_length_mismatch() {
        let e1 = embed_event(vec![1.0], vec![1.0, 2.0], vec![1.0], 1.0);
        let e2 = embed_event(vec![1.0], vec![1.0], vec![1.0], 1.0);
        assert!(matches!(
            EventEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_event_weight_pooling_rejects_target_length_mismatch() {
        let e1 = embed_event(vec![1.0], vec![1.0], vec![1.0, 2.0], 1.0);
        let e2 = embed_event(vec![1.0], vec![1.0], vec![1.0], 1.0);
        assert!(matches!(
            EventEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {actual} close to {expected}"
        );
    }
}
