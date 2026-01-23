use thiserror::Error;
pub mod embedding_model;
pub mod sem_embedding;

pub trait Embeddable {
    type EmbeddingFused;
    fn embed_and_fuse(self, model: &dyn EmbeddingModel)
    -> EmbeddingGenResult<Self::EmbeddingFused>;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<MemoryEmbedding>;
}
type EmbedCalcResult = Result<MemoryEmbedding, EmbeddingCalcError>;
type EmbeddingGenResult<T> = Result<T, EmbeddingGenError>;

#[derive(Debug, Error)]
pub enum EmbeddingGenError {
    #[error("Invalid input")] //缺失了某些必要字段
    InvalidInput,
    #[error("Embedding failed")]
    EmbeddingFailed(#[from] candle_core::Error),
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
pub type EmbeddingVec = Vec<f32>;

pub trait EmbeddingModel {
    fn infer(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>>;
}

pub trait GenericEmbeddingModel {
    fn infer<S: AsRef<str>>(&self, input: &[S]) -> EmbeddingGenResult<Vec<EmbeddingVec>>;
}

impl<T> EmbeddingModel for T
where
    T: GenericEmbeddingModel,
{
    fn infer(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        GenericEmbeddingModel::infer(self, input)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryEmbedding {
    Situation(),
    Procedure(),
    Semantic(),
}

impl MemoryEmbedding {
    pub fn euclidean_distance(
        &self,
        _other: &MemoryEmbedding,
        _hyperparams: VecBlendHyperParams,
    ) -> Result<f32, EmbeddingCalcError> {
        todo!("Euclidean distance")
    }
    pub fn cosine_similarity(
        &self,
        _other: &MemoryEmbedding,
        _hyperparams: VecBlendHyperParams,
    ) -> Result<f32, EmbeddingCalcError> {
        todo!("Cosine similarity")
    }
    pub fn manhattan_distance(
        &self,
        _other: &MemoryEmbedding,
        _hyperparams: VecBlendHyperParams,
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
