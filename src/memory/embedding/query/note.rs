use crate::memory::{
    embedding::{
        Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
        query::{sem::SemanticQueryUnitEmbedding, situation::SituationQueryUnitEmbedding},
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
