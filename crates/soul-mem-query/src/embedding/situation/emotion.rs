use serde::{Deserialize, Serialize};

use crate::embedding::{Embeddable, EmbeddingCalcError, EmbeddingCalcResult, EmbeddingVec};
use soul_mem_core::memory_note::situation_mem::Emotion;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionEmbedding {
    pub emotion: EmbeddingVec,
    pub intensity: f32,
}
impl EmotionEmbedding {
    pub fn emotion(&self) -> &EmbeddingVec {
        &self.emotion
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
    pub fn weight_pooling(emotions: &[EmotionEmbedding]) -> EmbeddingCalcResult<Option<Self>> {
        if emotions.is_empty() {
            return Ok(None);
        }
        let intensity_sum = emotions.iter().map(|e| e.intensity).sum::<f32>();
        //intensity_sum为0时0/0会产生NaN，直接报错，防止NaN污染后续检索
        if intensity_sum == 0.0 {
            return Err(EmbeddingCalcError::InvalidNumValue);
        }
        let len = emotions[0].emotion.shape();
        if !emotions.iter().all(|vec| vec.emotion.shape() == len) {
            return Err(EmbeddingCalcError::ShapeMismatch);
        }
        let fused_emotion = emotions.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.emotion.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });

        Ok(Some(EmotionEmbedding {
            emotion: EmbeddingVec::new(fused_emotion),
            intensity: intensity_sum,
        }))
    }
}
impl Embeddable for Emotion {
    type EmbeddingGen = EmotionEmbedding;
    type EmbeddingFused = EmbeddedEmotion;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [emotion_vec] = model
            .infer_batch(&[self.name.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处长度必为1
        Ok(EmotionEmbedding {
            emotion: emotion_vec,
            intensity: self.intensity,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedEmotion {
            embedding: self.embed(model)?,
            emotion: self,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedEmotion {
    pub embedding: EmotionEmbedding,
    pub emotion: Emotion,
}

#[cfg(test)]
mod tests {
    use crate::embedding::embedding_model::bge::BgeSmallZh;

    use super::*;

    #[test]
    fn test_embed() {
        let emotion = Emotion {
            name: "happy".to_string(),
            intensity: 1.0,
        };
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedding = emotion.embed(&model).unwrap();
        assert_eq!(embedding.intensity, 1.0);
    }

    fn embed_emotion(vec: Vec<f32>, intensity: f32) -> EmotionEmbedding {
        EmotionEmbedding {
            emotion: EmbeddingVec::new(vec),
            intensity,
        }
    }

    #[test]
    fn test_weight_pooling_values() {
        let e1 = embed_emotion(vec![1.0, 10.0], 0.5);
        let e2 = embed_emotion(vec![3.0, 20.0], 0.5);
        let pooled = EmotionEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        let vals = pooled.emotion().iter().copied().collect::<Vec<_>>();
        assert!((vals[0] - 2.0).abs() < 1e-5, "got {}", vals[0]);
        assert!((vals[1] - 15.0).abs() < 1e-5, "got {}", vals[1]);
        assert!((pooled.intensity() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_weight_pooling_weighted() {
        let e1 = embed_emotion(vec![4.0], 3.0);
        let e2 = embed_emotion(vec![0.0], 1.0);
        let pooled = EmotionEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        assert!((pooled.emotion().iter().copied().next().unwrap() - 3.0).abs() < 1e-5);
        assert!((pooled.intensity() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_weight_pooling_zero_intensity() {
        let e1 = embed_emotion(vec![1.0], 0.0);
        let e2 = embed_emotion(vec![1.0], 0.0);
        assert!(matches!(
            EmotionEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::InvalidNumValue)
        ));
    }

    #[test]
    fn test_weight_pooling_shape_mismatch() {
        let e1 = embed_emotion(vec![1.0], 1.0);
        let e2 = embed_emotion(vec![1.0, 2.0], 1.0);
        assert!(matches!(
            EmotionEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_weight_pooling_empty() {
        assert!(EmotionEmbedding::weight_pooling(&[]).unwrap().is_none());
    }
}
