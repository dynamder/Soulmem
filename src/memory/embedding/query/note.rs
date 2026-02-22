use crate::memory::{
    embedding::{
        query::{sem::SemanticQueryUnitEmbedding, situation::SituationQueryUnitEmbedding},
        Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
    },
    query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant},
};

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryRetrieveQueryVariantEmbedding {
    Semantic(Vec<SemanticQueryUnitEmbedding>),
    Situation(Vec<SituationQueryUnitEmbedding>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedMemoryRetrieveQueryVariant {
    pub embedding: MemoryRetrieveQueryVariantEmbedding,
    pub query: MemoryRetrieveQueryVariant,
}

impl Embeddable for MemoryRetrieveQueryVariant {
    type EmbeddingGen = MemoryRetrieveQueryVariantEmbedding;
    type EmbeddingFused = EmbeddedMemoryRetrieveQueryVariant;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        match self {
            Self::Semantic(sem_units) => {
                let embedding = sem_units
                    .iter()
                    .map(|sem_unit| sem_unit.embed(model))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(MemoryRetrieveQueryVariantEmbedding::Semantic(embedding))
            }
            Self::Situation(sit_units) => {
                let embedding = sit_units
                    .iter()
                    .map(|sit_unit| sit_unit.embed(model))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(MemoryRetrieveQueryVariantEmbedding::Situation(embedding))
            }
        }
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedMemoryRetrieveQueryVariant {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
///////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRetrieveQueryEmbedding {
    tag: EmbeddingVec,
    variant: MemoryRetrieveQueryVariantEmbedding,
}
impl MemoryRetrieveQueryEmbedding {
    pub fn tag(&self) -> &EmbeddingVec {
        &self.tag
    }
    pub fn variant(&self) -> &MemoryRetrieveQueryVariantEmbedding {
        &self.variant
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedMemoryRetrieveQuery {
    pub embedding: MemoryRetrieveQueryEmbedding,
    pub query: MemoryRetrieveQuery,
}

impl Embeddable for MemoryRetrieveQuery {
    type EmbeddingGen = MemoryRetrieveQueryEmbedding;
    type EmbeddingFused = EmbeddedMemoryRetrieveQuery;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        let tag_strs: Vec<_> = self.tag().iter().map(|s| s.as_str()).collect();
        let tag_vec = model.infer_and_fuse(&tag_strs)?;

        let variant_vec = self.variant().embed(model)?;

        Ok(MemoryRetrieveQueryEmbedding {
            tag: tag_vec,
            variant: variant_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedMemoryRetrieveQuery {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding::embedding_model::bge::BgeSmallZh;
    use crate::memory::query::retrieve::MemoryRetrieveQueryVariant;

    #[test]
    fn test_memory_retrieve_query_variant_embedding_semantic() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let query_variant = MemoryRetrieveQueryVariant::Semantic(vec![
            crate::memory::query::retrieve::SemanticQueryUnit::new()
                .with_concept_identifier("测试".to_string()),
        ]);

        let embedding = query_variant.embed(&model).unwrap();

        assert!(matches!(
            embedding,
            MemoryRetrieveQueryVariantEmbedding::Semantic(_)
        ));
    }

    #[test]
    fn test_memory_retrieve_query_variant_embedding_situation() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let query_variant = MemoryRetrieveQueryVariant::Situation(vec![
            crate::memory::query::retrieve::SituationQueryUnit::new()
                .with_narrative("在学校学习".to_string()),
        ]);

        let embedding = query_variant.embed(&model).unwrap();

        assert!(matches!(
            embedding,
            MemoryRetrieveQueryVariantEmbedding::Situation(_)
        ));
    }

    #[test]
    fn test_memory_retrieve_query_variant_with_multiple_semantic_units() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let query_variant = MemoryRetrieveQueryVariant::Semantic(vec![
            crate::memory::query::retrieve::SemanticQueryUnit::new()
                .with_concept_identifier("Rust".to_string()),
            crate::memory::query::retrieve::SemanticQueryUnit::new()
                .with_concept_identifier("编程".to_string()),
        ]);

        let embedding = query_variant.embed(&model).unwrap();

        if let MemoryRetrieveQueryVariantEmbedding::Semantic(units) = embedding {
            assert_eq!(units.len(), 2);
        } else {
            panic!("Expected Semantic variant");
        }
    }
}
