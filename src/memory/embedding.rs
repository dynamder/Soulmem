use thiserror::Error;

use crate::memory::memory_note::EmbedMemoryNote;

pub struct EmbeddingVector {
    // Placeholder for embedding vector
}

pub trait Embeddable {
    fn embed(&self, model: &EmbeddingModel) -> Result<EmbedMemoryNote, EmbeddingError>;
    fn embed_vec(&self, model: &EmbeddingModel) -> Result<MemoryEmbedding, EmbeddingError>;
}

//Only a placeholder for now
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Invalid input")]
    InvalidInput,
    #[error("Embedding failed")]
    EmbeddingFailed,
}

pub struct EmbeddingModel {
    // Placeholder for embedding model wrapper
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEmbedding {
    //Placeholder for embedding holder
}
