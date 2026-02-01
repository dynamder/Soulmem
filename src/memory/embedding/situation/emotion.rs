use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcError, EmbeddingCalcResult, EmbeddingVec, mean_pooling},
    memory_note::situation_mem::Emotion,
};

#[derive(Debug, Clone, PartialEq)]
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
        let len = emotions[0].emotion.len();
        if !emotions.iter().all(|vec| vec.emotion.len() == len) {
            return Err(EmbeddingCalcError::ShapeMismatch);
        }
        let fused_emotion = emotions.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.emotion.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });

        Ok(Some(EmotionEmbedding {
            emotion: fused_emotion,
            intensity: intensity_sum,
        }))
    }
}
impl Embeddable for Emotion {
    type EmbeddingGen = EmotionEmbedding;
    type EmbeddingFused = EmbeddedEmotion;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [emotion_vec] = model
            .infer_batch(&vec![self.name.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处长度必为1
        Ok(EmotionEmbedding {
            emotion: emotion_vec,
            intensity: self.intensity,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
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
