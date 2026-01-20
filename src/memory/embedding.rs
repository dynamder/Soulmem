use thiserror::Error;

use crate::memory::memory_note::EmbedMemoryNote;

pub struct EmbeddingVector {
    // Placeholder for embedding vector
}
impl EmbeddingVector {
    pub fn euclidean_distance(&self, other: &EmbeddingVector) -> Result<f32, EmbeddingCalcError> {
        todo!("Euclidean distance")
    }
    pub fn cosine_similarity(&self, other: &EmbeddingVector) -> Result<f32, EmbeddingCalcError> {
        todo!("Cosine similarity")
    }
    pub fn manhattan_distance(&self, other: &EmbeddingVector) -> Result<f32, EmbeddingCalcError> {
        todo!("Manhattan distance")
    }
}

pub trait Embeddable {
    fn embed(&self, model: &EmbeddingModel) -> Result<EmbedMemoryNote, EmbeddingGenError>;
    fn embed_vec(&self, model: &EmbeddingModel) -> Result<MemoryEmbedding, EmbeddingGenError>;
}
type EmbedCalcResult = Result<MemoryEmbedding, EmbeddingCalcError>;
type EmbeddingGenResult = Result<EmbeddingVector, EmbeddingGenError>;

#[derive(Debug, Error)]
pub enum EmbeddingGenError {
    #[error("Invalid input")] //缺失了某些必要字段
    InvalidInput,
    #[error("Embedding failed")]
    EmbeddingFailed,
}

//Only a placeholder for now
#[derive(Debug, Error)]
pub enum EmbeddingCalcError {
    #[error("Invalid vec")] //缺失了某些必要字段
    InvalidVec,
    #[error("Shape mismatch")] //维度不匹配
    ShapeMismatch,
    #[error("Invalid number value")] //数值无效，例如NaN，Inf等
    InvalidNumValue,
}

pub struct EmbeddingModel {
    // Placeholder for embedding model wrapper
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEmbedding {
    //Placeholder for embedding holder
}
impl MemoryEmbedding {
    pub fn euclidean_distance(
        &self,
        other: &MemoryEmbedding,
        hyperparams: VecBlendHyperParams,
    ) -> Result<f32, EmbeddingCalcError> {
        todo!("Euclidean distance")
    }
    pub fn cosine_similarity(
        &self,
        other: &MemoryEmbedding,
        hyperparams: VecBlendHyperParams,
    ) -> Result<f32, EmbeddingCalcError> {
        todo!("Cosine similarity")
    }
    pub fn manhattan_distance(
        &self,
        other: &MemoryEmbedding,
        hyperparams: VecBlendHyperParams,
    ) -> Result<f32, EmbeddingCalcError> {
        todo!("Manhattan distance")
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
