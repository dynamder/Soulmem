use crate::memory::{
    embedding::{Embeddable, EmbeddingVec},
    memory_note::situation_mem::SensoryData,
};
#[derive(Debug, Clone, PartialEq)]
pub struct SensoryDataEmbedding {
    sensory: EmbeddingVec,
    intensity: f32,
}
impl SensoryDataEmbedding {
    pub fn sensory(&self) -> &EmbeddingVec {
        &self.sensory
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
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
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [sensory_vec] = model
            .infer_batch(&vec![self.name.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处长度必为1
        Ok(SensoryDataEmbedding {
            sensory: sensory_vec,
            intensity: self.intensity,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSensoryData {
            embedding: self.embed(model)?,
            sensory_data: self,
        })
    }
}
