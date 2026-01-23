use super::EmbeddingGenResult;
use super::EmbeddingModel;
use super::EmbeddingVec;
use super::MemoryEmbedding;
use crate::memory::embedding::Embeddable;
use crate::memory::memory_note::sem_mem::SemMemory;
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticEmbedding {
    content: EmbeddingVec,
    fused_aliases: EmbeddingVec,
    description: EmbeddingVec,
}

impl Embeddable for SemMemory {
    type EmbeddingFused = EmbeddedSemanticMemory;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<MemoryEmbedding> {
        todo!()
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        todo!()
    }
}

pub struct EmbeddedSemanticMemory {
    pub embedding: SemanticEmbedding,
    pub memory: SemMemory,
}
