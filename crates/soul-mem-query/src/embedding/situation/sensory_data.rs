use serde::{Deserialize, Serialize};

use crate::embedding::{Embeddable, EmbeddingCalcError, EmbeddingCalcResult, EmbeddingVec};
use soul_mem_core::memory_note::situation_mem::SensoryData;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensoryDataEmbedding {
    pub sensory: EmbeddingVec,
    pub intensity: f32,
}
impl SensoryDataEmbedding {
    pub fn sensory(&self) -> &EmbeddingVec {
        &self.sensory
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
    pub fn weight_pooling(datas: &[SensoryDataEmbedding]) -> EmbeddingCalcResult<Option<Self>> {
        if datas.is_empty() {
            return Ok(None);
        }
        let intensity_sum = datas.iter().map(|e| e.intensity).sum::<f32>();
        //intensity_sum为0时0/0会产生NaN，直接报错，防止NaN污染后续检索
        if intensity_sum == 0.0 {
            return Err(EmbeddingCalcError::InvalidNumValue);
        }
        let len = datas[0].sensory.shape();
        if !datas.iter().all(|vec| vec.sensory.shape() == len) {
            return Err(EmbeddingCalcError::ShapeMismatch);
        }
        let fused_emotion = datas.iter().fold(vec![0.0; len], |acc, vec| {
            acc.iter()
                .zip(vec.sensory.iter())
                .map(|(&a, &b)| a + b * vec.intensity / intensity_sum)
                .collect()
        });

        Ok(Some(SensoryDataEmbedding {
            sensory: EmbeddingVec::new(fused_emotion),
            intensity: intensity_sum,
        }))
    }
}
#[cfg(test)]
impl SensoryDataEmbedding {
    pub(crate) fn test_new(sensory: EmbeddingVec, intensity: f32) -> Self {
        Self { sensory, intensity }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedSensoryData {
    pub embedding: SensoryDataEmbedding,
    pub sensory_data: SensoryData,
}

impl Embeddable for SensoryData {
    type EmbeddingGen = SensoryDataEmbedding;
    type EmbeddingFused = EmbeddedSensoryData;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [sensory_vec] = model
            .infer_batch(&[self.name.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处长度必为1
        Ok(SensoryDataEmbedding {
            sensory: sensory_vec,
            intensity: self.intensity,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSensoryData {
            embedding: self.embed(model)?,
            sensory_data: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingCalcError;

    fn embed_data(vec: Vec<f32>, intensity: f32) -> SensoryDataEmbedding {
        SensoryDataEmbedding {
            sensory: EmbeddingVec::new(vec),
            intensity,
        }
    }

    #[test]
    fn test_weight_pooling_values() {
        let e1 = embed_data(vec![1.0, 10.0], 0.5);
        let e2 = embed_data(vec![3.0, 20.0], 0.5);
        let pooled = SensoryDataEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        let vals = pooled.sensory().iter().copied().collect::<Vec<_>>();
        assert!((vals[0] - 2.0).abs() < 1e-5, "got {}", vals[0]);
        assert!((vals[1] - 15.0).abs() < 1e-5, "got {}", vals[1]);
        assert!((pooled.intensity() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_weight_pooling_weighted() {
        let e1 = embed_data(vec![4.0], 3.0);
        let e2 = embed_data(vec![0.0], 1.0);
        let pooled = SensoryDataEmbedding::weight_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        assert!((pooled.sensory().iter().copied().next().unwrap() - 3.0).abs() < 1e-5);
        assert!((pooled.intensity() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_weight_pooling_zero_intensity() {
        let e1 = embed_data(vec![1.0], 0.0);
        let e2 = embed_data(vec![1.0], 0.0);
        assert!(matches!(
            SensoryDataEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::InvalidNumValue)
        ));
    }

    #[test]
    fn test_weight_pooling_shape_mismatch() {
        let e1 = embed_data(vec![1.0], 1.0);
        let e2 = embed_data(vec![1.0, 2.0], 1.0);
        assert!(matches!(
            SensoryDataEmbedding::weight_pooling(&[e1, e2]),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_weight_pooling_empty() {
        assert!(SensoryDataEmbedding::weight_pooling(&[]).unwrap().is_none());
    }
}
