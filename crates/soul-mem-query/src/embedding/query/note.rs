use crate::embedding::{
    blend_weights::BlendWeights,
    query::{sem::SemanticQueryUnitEmbedding, situation::SituationQueryUnitEmbedding},
    Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
};
use crate::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryRetrieveQueryVariantEmbedding {
    Semantic(Vec<SemanticQueryUnitEmbedding>),
    Situation(Vec<SituationQueryUnitEmbedding>),
}
impl MemoryRetrieveQueryVariantEmbedding {
    /// 将 blend weights 递归传播到所有子单元
    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        match self {
            Self::Semantic(units) => {
                for u in units.iter_mut() {
                    u.set_blend_weights(bw);
                }
            }
            Self::Situation(units) => {
                for u in units.iter_mut() {
                    u.set_blend_weights(bw);
                }
            }
        }
    }
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
    pub tag_weight: f32,
    pub variant_weight: f32,
    /// embedding 在最终混合分中的权重，`1 - string_blend_alpha` 为字符串得分权重
    pub string_blend_alpha: f32,
}
impl MemoryRetrieveQueryEmbedding {
    pub fn tag(&self) -> &EmbeddingVec {
        &self.tag
    }
    pub fn variant(&self) -> &MemoryRetrieveQueryVariantEmbedding {
        &self.variant
    }

    pub fn new(tag: EmbeddingVec) -> Self {
        Self {
            tag,
            variant: MemoryRetrieveQueryVariantEmbedding::Semantic(vec![]),
            tag_weight: 0.4,
            variant_weight: 0.6,
            string_blend_alpha: 0.6,
        }
    }

    /// 设置自定义 blend weights 并传播到所有子单元
    pub fn with_weights(mut self, bw: BlendWeights) -> Self {
        self.tag_weight = bw.tag;
        self.variant_weight = bw.variant;
        self.string_blend_alpha = bw.string_blend_alpha;
        self.variant.set_blend_weights(&bw);
        self
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
        //tag为空时跳过模型调用，用零向量填充，避免空输入导致嵌入失败
        let tag_vec = if tag_strs.is_empty() {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_and_fuse(&tag_strs)?
        };

        let variant_vec = self.variant().embed(model)?;

        Ok(MemoryRetrieveQueryEmbedding {
            tag: tag_vec,
            variant: variant_vec,
            tag_weight: 0.4,
            variant_weight: 0.6,
            string_blend_alpha: 0.6,
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
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use crate::query::retrieve::MemoryRetrieveQueryVariant;

    #[test]
    fn test_memory_retrieve_query_variant_embedding_semantic() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let query_variant = MemoryRetrieveQueryVariant::Semantic(vec![
            crate::query::retrieve::SemanticQueryUnit::new()
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
            crate::query::retrieve::SituationQueryUnit::new()
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
            crate::query::retrieve::SemanticQueryUnit::new()
                .with_concept_identifier("Rust".to_string()),
            crate::query::retrieve::SemanticQueryUnit::new()
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
