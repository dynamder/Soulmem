use async_trait::async_trait;
use petgraph::Direction::Outgoing;
use thiserror::Error;

use crate::memory::embedding::{sem::SemanticEmbedding, situation::SituationEmbedding};
pub mod embedding_model;
pub mod sem;
pub mod situation;

pub trait Embeddable {
    type EmbeddingFused;
    type EmbeddingGen;
    fn embed_and_fuse(self, model: &dyn EmbeddingModel)
    -> EmbeddingGenResult<Self::EmbeddingFused>;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen>;
}
pub type EmbeddingCalcResult<T> = Result<T, EmbeddingCalcError>;
pub type EmbeddingGenResult<T> = Result<T, EmbeddingGenError>;

#[derive(Debug, Error)]
pub enum EmbeddingGenError {
    #[error("Invalid input")] //缺失了某些必要字段
    InvalidInput,
    #[error("Embedding failed")]
    EmbeddingFailed(#[from] candle_core::Error),
    #[error("Post calculation failed")]
    PostCalcFailed(#[from] EmbeddingCalcError),
    #[error("{0}")]
    Anyhow(#[from] anyhow::Error),
}

//Only a placeholder for now
#[derive(Debug, Error)]
pub enum EmbeddingCalcError {
    #[error("Invalid vec")] //缺失了某些必要字段
    InvalidVec,
    #[error("Shape mismatch")] //维度不匹配
    ShapeMismatch,
    #[error("Incompatible embedding types")] //不兼容的嵌入类型
    IncompatibleEmbeddingTypes,
    #[error("Invalid number value")] //数值无效，例如NaN，Inf等
    InvalidNumValue,
}
pub type EmbeddingVec = Vec<f32>;

#[async_trait]
pub trait EmbeddingModel {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>>;
    fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec>;
    fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec>;
    fn max_input_token(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryEmbedding {
    Situation(SituationEmbedding),
    Procedure(),
    Semantic(SemanticEmbedding),
}
impl MemoryEmbedding {
    pub fn to_situation(self) -> Option<SituationEmbedding> {
        match self {
            MemoryEmbedding::Situation(embedding) => Some(embedding),
            _ => None,
        }
    }
    pub fn to_procedure(self) {
        todo!()
    }
    pub fn to_semantic(self) -> Option<SemanticEmbedding> {
        match self {
            MemoryEmbedding::Semantic(embedding) => Some(embedding),
            _ => None,
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
        other: &MemoryEmbedding,
        blend_factor: f32,
    ) -> EmbeddingCalcResult<MemoryEmbedding> {
        match (self, other) {
            (MemoryEmbedding::Semantic(embedding1), MemoryEmbedding::Semantic(embedding2)) => {
                embedding1
                    .linear_blend(embedding2, blend_factor)
                    .map(MemoryEmbedding::from)
            }
            (MemoryEmbedding::Situation(embedding1), MemoryEmbedding::Situation(embedding2)) => {
                // embedding1
                //     .linear_blend(embedding2, blend_factor)
                //     .map(MemoryEmbedding::from)
                todo!()
            }

            _ => Err(EmbeddingCalcError::IncompatibleEmbeddingTypes),
        }
    }
}
fn raw_linear_blend(
    vec1: &EmbeddingVec,
    vec2: &EmbeddingVec,
    blend_factor: f32,
) -> EmbeddingCalcResult<EmbeddingVec> {
    if vec1.len() != vec2.len() {
        return Err(EmbeddingCalcError::ShapeMismatch);
    }
    Ok(vec1
        .iter()
        .zip(vec2.iter())
        .map(|(&a, &b)| a * blend_factor + b * (1.0 - blend_factor))
        .collect())
}
fn mean_pooling(vecs: &[&EmbeddingVec]) -> EmbeddingCalcResult<EmbeddingVec> {
    if vecs.is_empty() {
        return Ok(Vec::new());
    }
    let len = vecs[0].len();
    if !vecs.iter().all(|vec| vec.len() == len) {
        return Err(EmbeddingCalcError::ShapeMismatch);
    }
    Ok(vecs
        .iter()
        .fold(vec![0.0; len], |acc, vec| {
            acc.iter().zip(vec.iter()).map(|(&a, &b)| a + b).collect()
        })
        .iter()
        .map(|&sum| sum / vecs.len() as f32)
        .collect())
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
