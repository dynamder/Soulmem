use async_trait::async_trait;

use thiserror::Error;

pub mod blend_weights;
pub mod embedding_model;
pub mod query;
pub mod sem;
pub mod situation;
pub mod vec;
pub use vec::{EmbeddingVec, mean_pooling, raw_linear_blend};
pub mod note;

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

#[async_trait]
pub trait EmbeddingModel {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>>;
    fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec>;
    fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec>;
    /// 查询侧批量嵌入。默认与 `infer_batch` 相同；检索类模型（如 BGE v1.5）
    /// 应覆写为在输入前加查询指令，与 passage 侧的非对称训练用法保持一致。
    fn infer_query_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        self.infer_batch(input)
    }
    /// 查询侧多输入融合嵌入。默认与 `infer_and_fuse` 相同。
    fn infer_query_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        self.infer_and_fuse(input)
    }
    /// 查询侧长文本分块嵌入。默认与 `infer_with_chunk` 相同。
    fn infer_query_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        self.infer_with_chunk(input)
    }
    fn max_input_token(&self) -> usize;
    /// 模型输出向量的维度，用于在没有有效输入时构造零向量
    fn dim(&self) -> usize;
}

//util function
fn vec_batch_embed<T: Embeddable>(
    vecs: &[T],
    model: &dyn EmbeddingModel,
) -> EmbeddingGenResult<Vec<<T as Embeddable>::EmbeddingGen>> {
    vecs.iter()
        .map(|vec| vec.embed(model))
        .collect::<Result<Vec<_>, _>>()
}
