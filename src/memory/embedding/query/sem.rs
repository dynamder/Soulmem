use crate::memory::{
    embedding::{Embeddable, EmbeddingVec},
    query::retrieve::SemanticQueryUnit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticQueryUnitEmbedding {
    concept_identifier: Option<EmbeddingVec>,
    description: Option<EmbeddingVec>,
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
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
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
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSemanticQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
