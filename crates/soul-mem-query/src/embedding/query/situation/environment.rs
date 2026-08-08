use crate::{
    embedding::{Embeddable, EmbeddingVec},
    query::retrieve::EnvironmentQueryUnit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentQueryUnitEmbedding {
    atmosphere: Option<EmbeddingVec>,
    tone: Option<EmbeddingVec>,
}
impl EnvironmentQueryUnitEmbedding {
    pub fn atmosphere(&self) -> Option<&EmbeddingVec> {
        self.atmosphere.as_ref()
    }
    pub fn tone(&self) -> Option<&EmbeddingVec> {
        self.tone.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedEnvironmentQueryUnit {
    pub embedding: EnvironmentQueryUnitEmbedding,
    pub query: EnvironmentQueryUnit,
}

impl Embeddable for EnvironmentQueryUnit {
    type EmbeddingGen = EnvironmentQueryUnitEmbedding;
    type EmbeddingFused = EmbedEnvironmentQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let atmosphere_batch_vec = self
            .atmosphere()
            .map(|atmosphere| model.infer_batch(&[atmosphere]))
            .transpose()?;

        let atmosphere_vec = atmosphere_batch_vec.and_then(|vec| vec.into_iter().next());

        let tone_batch_vec = self
            .tone()
            .map(|tone| model.infer_batch(&[tone]))
            .transpose()?;

        let tone_vec = tone_batch_vec.and_then(|vec| vec.into_iter().next());

        Ok(EnvironmentQueryUnitEmbedding {
            atmosphere: atmosphere_vec,
            tone: tone_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedEnvironmentQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
