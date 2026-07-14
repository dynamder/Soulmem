use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{Embeddable, EmbeddingVec};
use crate::query::retrieve::EnvironmentQueryUnit;

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentQueryUnitEmbedding {
    atmosphere: Option<EmbeddingVec>,
    tone: Option<EmbeddingVec>,
    pub blend_weights: BlendWeights,
}
impl EnvironmentQueryUnitEmbedding {
    pub fn atmosphere(&self) -> Option<&EmbeddingVec> {
        self.atmosphere.as_ref()
    }
    pub fn tone(&self) -> Option<&EmbeddingVec> {
        self.tone.as_ref()
    }
    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
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
            .map(|atmosphere| model.infer_batch(&vec![atmosphere]))
            .transpose()?;

        let atmosphere_vec = atmosphere_batch_vec
            .map(|vec| vec.into_iter().next())
            .flatten();

        let tone_batch_vec = self
            .tone()
            .map(|tone| model.infer_batch(&vec![tone]))
            .transpose()?;

        let tone_vec = tone_batch_vec.map(|vec| vec.into_iter().next()).flatten();

        Ok(EnvironmentQueryUnitEmbedding {
            atmosphere: atmosphere_vec,
            tone: tone_vec,
            blend_weights: BlendWeights::default(),
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
