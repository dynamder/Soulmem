use crate::memory::{
    embedding::{
        Embeddable, EmbeddingCalcResult, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
        sem::SemanticEmbedding, situation::SituationEmbedding,
    },
    memory_note::{MemoryNote, MemoryType},
};

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEmbedding {
    tag: EmbeddingVec,
    variant: MemoryEmbeddingVariant,
}
impl MemoryEmbedding {
    pub fn tag(&self) -> &EmbeddingVec {
        &self.tag
    }
    pub fn variant(&self) -> &MemoryEmbeddingVariant {
        &self.variant
    }
}

#[derive(Debug, Clone, PartialEq)]
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
        other: &MemoryEmbeddingVariant,
        blend_factor: f32,
    ) -> EmbeddingCalcResult<MemoryEmbeddingVariant> {
        todo!("linear blend")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VecBlendHyperParams {
    // Placeholder for vector blending hyperparameters
}
impl Default for VecBlendHyperParams {
    fn default() -> Self {
        VecBlendHyperParams {
            // Placeholder for default values
        }
    }
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

#[derive(Debug, Clone, PartialEq)]
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
        let tag_vec = model.infer_and_fuse(&tag_strs)?;

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
