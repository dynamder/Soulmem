use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{Embeddable, EmbeddingVec};
use crate::query::retrieve::SemanticQueryUnit;

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticQueryUnitEmbedding {
    concept_identifier: Option<EmbeddingVec>,
    description: Option<EmbeddingVec>,
    pub blend_weights: BlendWeights,
}
impl SemanticQueryUnitEmbedding {
    pub fn concept_identifier(&self) -> Option<&EmbeddingVec> {
        self.concept_identifier.as_ref()
    }

    pub fn description(&self) -> Option<&EmbeddingVec> {
        self.description.as_ref()
    }

    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedSemanticQueryUnit {
    pub embedding: SemanticQueryUnitEmbedding,
    pub query: SemanticQueryUnit,
}

impl Embeddable for SemanticQueryUnit {
    type EmbeddingGen = SemanticQueryUnitEmbedding;
    type EmbeddingFused = EmbeddedSemanticQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let concept_identifier_batch_vec = self
            .concept_identifier()
            .map(|concept_identifier| model.infer_batch(&vec![concept_identifier]))
            .transpose()?;

        let concept_identifier_vec = concept_identifier_batch_vec
            .map(|vec| vec.into_iter().next())
            .flatten();

        let description_batch_vec = self
            .description()
            .map(|description| model.infer_batch(&vec![description]))
            .transpose()?;

        let description_vec = description_batch_vec
            .map(|vec| vec.into_iter().next())
            .flatten();

        Ok(SemanticQueryUnitEmbedding {
            concept_identifier: concept_identifier_vec,
            description: description_vec,
            blend_weights: BlendWeights::default(),
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSemanticQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
