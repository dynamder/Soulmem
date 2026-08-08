use crate::{
    embedding::{Embeddable, EmbeddingVec},
    query::retrieve::SemanticQueryUnit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticQueryUnitEmbedding {
    concept_identifier: Option<EmbeddingVec>,
    description: Option<EmbeddingVec>,
}
impl SemanticQueryUnitEmbedding {
    pub fn concept_identifier(&self) -> Option<&EmbeddingVec> {
        self.concept_identifier.as_ref()
    }

    pub fn description(&self) -> Option<&EmbeddingVec> {
        self.description.as_ref()
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
            .map(|concept_identifier| model.infer_batch(&[concept_identifier]))
            .transpose()?;

        let concept_identifier_vec =
            concept_identifier_batch_vec.and_then(|vec| vec.into_iter().next());

        let description_batch_vec = self
            .description()
            .map(|description| model.infer_batch(&[description]))
            .transpose()?;

        let description_vec = description_batch_vec.and_then(|vec| vec.into_iter().next());

        Ok(SemanticQueryUnitEmbedding {
            concept_identifier: concept_identifier_vec,
            description: description_vec,
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
