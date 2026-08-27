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
#[cfg(test)]
impl SemanticQueryUnitEmbedding {
    pub(crate) fn test_new(
        concept_identifier: Option<EmbeddingVec>,
        description: Option<EmbeddingVec>,
        blend_weights: BlendWeights,
    ) -> Self {
        Self {
            concept_identifier,
            description,
            blend_weights,
        }
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
            .map(|concept_identifier| model.infer_query_batch(&vec![concept_identifier]))
            .transpose()?;

        let concept_identifier_vec =
            concept_identifier_batch_vec.and_then(|vec| vec.into_iter().next());

        let description_batch_vec = self
            .description()
            .map(|description| model.infer_query_batch(&vec![description]))
            .transpose()?;

        let description_vec = description_batch_vec.and_then(|vec| vec.into_iter().next());

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_query_unit_embedding_accessors() {
        let mut embedding = SemanticQueryUnitEmbedding::test_new(
            Some(EmbeddingVec::new(vec![1.0])),
            Some(EmbeddingVec::new(vec![2.0])),
            BlendWeights::default(),
        );
        assert_eq!(embedding.concept_identifier().unwrap().shape(), 1);
        assert_eq!(embedding.description().unwrap().shape(), 1);

        let mut bw = BlendWeights::default();
        bw.tag = 0.8;
        embedding.set_blend_weights(&bw);
        assert_eq!(embedding.blend_weights.tag, 0.8);
    }

    #[test]
    fn test_semantic_query_unit_embedding_none() {
        let embedding = SemanticQueryUnitEmbedding::test_new(None, None, BlendWeights::default());
        assert!(embedding.concept_identifier().is_none());
        assert!(embedding.description().is_none());
    }
}
