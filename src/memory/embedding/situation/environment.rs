use crate::memory::{
    embedding::{Embeddable, EmbeddingVec},
    memory_note::situation_mem::Environment,
};

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentEmbedding {
    atmosphere: EmbeddingVec,
    tone: EmbeddingVec,
}
impl EnvironmentEmbedding {
    pub fn atmosphere(&self) -> &EmbeddingVec {
        &self.atmosphere
    }

    pub fn tone(&self) -> &EmbeddingVec {
        &self.tone
    }
}
impl Embeddable for Environment {
    type EmbeddingGen = EnvironmentEmbedding;
    type EmbeddingFused = EmbeddedEnvironment;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [atmosphere_vec, tone_vec] = model
            .infer_batch(&vec![self.atmosphere.as_str(), self.tone.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处返回的Vec长度必为2
        Ok(EnvironmentEmbedding {
            atmosphere: atmosphere_vec,
            tone: tone_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedEnvironment {
            embedding: self.embed(model)?,
            environment: self,
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedEnvironment {
    pub embedding: EnvironmentEmbedding,
    pub environment: Environment,
}
