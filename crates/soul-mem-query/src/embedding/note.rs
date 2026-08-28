use crate::embedding::{
    Embeddable, EmbeddingCalcResult, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
    sem::SemanticEmbedding, situation::SituationEmbedding,
};
use serde::{Deserialize, Serialize};
use soul_mem_core::memory_note::{MemoryNote, MemoryType};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEmbedding {
    pub tag: EmbeddingVec,
    pub variant: MemoryEmbeddingVariant,
}
impl MemoryEmbedding {
    pub fn tag(&self) -> &EmbeddingVec {
        &self.tag
    }
    pub fn variant(&self) -> &MemoryEmbeddingVariant {
        &self.variant
    }
    pub fn new(tag: EmbeddingVec, variant: MemoryEmbeddingVariant) -> Self {
        Self { tag, variant }
    }
}

#[allow(clippy::large_enum_variant)] // Box 化会改变公开 API 与 serde 布局，暂保持现状
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryEmbeddingVariant {
    Situation(SituationEmbedding),
    Procedure(),
    Semantic(SemanticEmbedding),
}
impl MemoryEmbeddingVariant {
    pub fn to_situation(self) -> Option<SituationEmbedding> {
        match self {
            MemoryEmbeddingVariant::Situation(embedding) => Some(embedding),
            _ => None,
        }
    }
    pub fn to_procedure(self) -> Option<()> {
        match self {
            MemoryEmbeddingVariant::Procedure() => Some(()),
            _ => None,
        }
    }
    pub fn to_semantic(self) -> Option<SemanticEmbedding> {
        match self {
            MemoryEmbeddingVariant::Semantic(embedding) => Some(embedding),
            _ => None,
        }
    }
}

pub struct EmbeddedMemoryType {
    pub embedding: MemoryEmbeddingVariant,
    pub mem_type: MemoryType,
}

impl EmbeddedMemoryType {
    pub fn new(mem_type: MemoryType, embedding: MemoryEmbeddingVariant) -> Self {
        Self {
            mem_type,
            embedding,
        }
    }
}

impl MemoryEmbedding {
    pub fn euclidean_distance(
        &self,
        _other: &MemoryEmbedding,
        _hyperparams: VecBlendHyperParams,
    ) -> EmbeddingCalcResult<f32> {
        todo!("Euclidean distance")
    }
    pub fn cosine_similarity(
        &self,
        _other: &MemoryEmbedding,
        _hyperparams: VecBlendHyperParams,
    ) -> EmbeddingCalcResult<f32> {
        todo!("Cosine similarity")
    }
    pub fn manhattan_distance(
        &self,
        _other: &MemoryEmbedding,
        _hyperparams: VecBlendHyperParams,
    ) -> EmbeddingCalcResult<f32> {
        todo!("Manhattan distance")
    }
    pub fn linear_blend(
        &self,
        _other: &MemoryEmbeddingVariant,
        _blend_factor: f32,
    ) -> EmbeddingCalcResult<MemoryEmbeddingVariant> {
        todo!("linear blend")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VecBlendHyperParams {
    // Placeholder for vector blending hyperparameters
}

////////////////////////////////////////////////////////
impl Embeddable for MemoryType {
    type EmbeddingGen = MemoryEmbeddingVariant;
    type EmbeddingFused = EmbeddedMemoryType;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        match self {
            Self::Semantic(sem) => Ok(MemoryEmbeddingVariant::Semantic(sem.embed(model)?)),
            Self::Situation(sit) => Ok(MemoryEmbeddingVariant::Situation(sit.embed(model)?)),
            Self::Procedure(_) => Ok(MemoryEmbeddingVariant::Procedure()),
        }
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedMemoryType {
            embedding: self.embed(model)?,
            mem_type: self,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedMemoryNote {
    pub embedding: MemoryEmbedding,
    pub note: MemoryNote,
}
impl EmbeddedMemoryNote {
    pub fn note(&self) -> &MemoryNote {
        &self.note
    }
    pub fn embedding(&self) -> &MemoryEmbedding {
        &self.embedding
    }
    pub fn into_tuple(self) -> (MemoryNote, MemoryEmbedding) {
        (self.note, self.embedding)
    }
}

impl Embeddable for MemoryNote {
    type EmbeddingGen = MemoryEmbedding;
    type EmbeddingFused = EmbeddedMemoryNote;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        let tag_strs: Vec<_> = self.tags().iter().map(|s| s.as_str()).collect();
        //tags为空时跳过模型调用，用零向量填充，避免空输入导致嵌入失败
        let tag_vec = if tag_strs.is_empty() {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_and_fuse(&tag_strs)?
        };

        let mem_type_vec = self.mem_type().embed(model)?;
        Ok(MemoryEmbedding {
            tag: tag_vec,
            variant: mem_type_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedMemoryNote {
            embedding: self.embed(model)?,
            note: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use soul_mem_core::memory_note::MemoryNoteBuilder;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};

    #[test]
    fn test_memory_note_embedding() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let mem_type = soul_mem_core::memory_note::MemoryType::Semantic(SemMemory {
            content: "Rust编程".to_string(),
            aliases: vec!["Rust".to_string()],
            concept_type: ConceptType::Entity,
            description: "系统编程语言".to_string(),
            ..Default::default()
        });

        let note = MemoryNoteBuilder::new(mem_type)
            .tags(vec!["编程".to_string(), "语言".to_string()])
            .build()
            .unwrap();

        let embedding = note.embed(&model).unwrap();

        assert_eq!(embedding.tag().shape(), 512);
        let variant = embedding.variant().clone();
        assert!(variant.to_semantic().is_some());
    }

    #[test]
    fn test_memory_embedding_variant_semantic() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let mem_type = soul_mem_core::memory_note::MemoryType::Semantic(SemMemory {
            content: "测试".to_string(),
            aliases: vec!["test".to_string()],
            concept_type: ConceptType::Entity,
            description: "测试描述".to_string(),
            ..Default::default()
        });

        let variant_emb = mem_type.embed(&model).unwrap();

        assert!(variant_emb.to_semantic().is_some());
    }

    #[test]
    fn test_memory_embedding_variant_situation() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let location = soul_mem_core::memory_note::situation_mem::Location {
            name: "学校".to_string(),
            coordinates: "北京".to_string(),
        };

        let mem_type = soul_mem_core::memory_note::MemoryType::Situation(
            soul_mem_core::memory_note::situation_mem::SituationType::AbstractSituation(
                location.into(),
            ),
        );

        let variant_emb = mem_type.embed(&model).unwrap();

        assert!(variant_emb.to_situation().is_some());
    }

    #[test]
    fn test_memory_embedding_variant_accessors() {
        // Procedure 变体
        let proc = MemoryEmbeddingVariant::Procedure();
        assert!(proc.to_procedure().is_some());
        let proc = MemoryEmbeddingVariant::Procedure();
        assert!(proc.to_semantic().is_none());
        let proc = MemoryEmbeddingVariant::Procedure();
        assert!(proc.to_situation().is_none());

        // Semantic 变体（通过 MemoryEmbedding::new 直接构造，避免模型依赖）
        let sem_emb = crate::embedding::sem::SemanticEmbedding::new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
            EmbeddingVec::new(vec![3.0]),
        );
        let sem = MemoryEmbeddingVariant::Semantic(sem_emb);
        assert!(sem.to_semantic().is_some());
        let sem = MemoryEmbeddingVariant::Semantic(crate::embedding::sem::SemanticEmbedding::new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
            EmbeddingVec::new(vec![3.0]),
        ));
        assert!(sem.to_procedure().is_none());
        let sem = MemoryEmbeddingVariant::Semantic(crate::embedding::sem::SemanticEmbedding::new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
            EmbeddingVec::new(vec![3.0]),
        ));
        assert!(sem.to_situation().is_none());
    }

    #[test]
    fn test_memory_embedding_new_and_accessors() {
        let tag = EmbeddingVec::new(vec![1.0, 2.0]);
        let variant = MemoryEmbeddingVariant::Procedure();
        let embedding = MemoryEmbedding::new(tag.clone(), variant);
        assert_eq!(embedding.tag(), &tag);
        assert!(embedding.variant().clone().to_procedure().is_some());
    }

    #[test]
    fn test_embedded_memory_note_tuple() {
        let mem_type = soul_mem_core::memory_note::MemoryType::Semantic(SemMemory {
            content: "测试".to_string(),
            aliases: vec!["test".to_string()],
            concept_type: ConceptType::Entity,
            description: "测试描述".to_string(),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding = MemoryEmbedding {
            tag: EmbeddingVec::new(vec![1.0]),
            variant: MemoryEmbeddingVariant::Procedure(),
        };
        let embedded = EmbeddedMemoryNote { note, embedding };
        let (n, e) = embedded.clone().into_tuple();
        assert_eq!(n.id(), embedded.note().id());
        assert_eq!(e.tag(), embedded.embedding().tag());
    }
}
