//! 嵌入层：把领域对象变成向量。
//!
//! # 两个 trait 的分工
//!
//! - [`Embeddable`]：**领域侧**——谁有向量表示、怎么从自己算出向量。实现分布在
//!   `note.rs` / `sem.rs` / `situation/`（记忆侧）与 `query/`（查询侧）。
//! - [`EmbeddingModel`]：**模型侧**——怎么把一批字符串变成向量。
//!   实现见 `embedding_model/{bge,qwen3}.rs`。
//!
//! # 查询侧与记忆侧是不对称的
//!
//! 检索类模型（如 BGE v1.5）训练时**查询带指令前缀、passage 不带**。
//! [`EmbeddingModel`] 为此提供 `infer_query_batch` / `infer_query_and_fuse` /
//! `infer_query_with_chunk` 三个方法，而它们的**默认实现等于非 query 版本**。
//! 需要指令前缀的模型必须覆写；**忘了覆写不会报错**，只会让检索质量下降。
//! 范例见 `embedding_model/bge.rs` 的 `QUERY_INSTRUCTION`。
//!
//! # 错误类型
//!
//! - [`EmbeddingGenError`]：生成向量阶段的失败（模型 / IO / 输入缺失）。
//! - [`EmbeddingCalcError`]：已拿到向量后的计算失败（维度不匹配 / 数值无效）。
//!
//! 二者可互相转换，调用方通常只需 `?`。

use async_trait::async_trait;

use thiserror::Error;

pub mod blend_weights;
pub mod embedding_model;
pub mod note;
pub mod query;
pub mod sem;
pub mod situation;
pub mod vec;

pub use vec::{EmbeddingVec, mean_pooling, raw_linear_blend};

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

// 计算阶段的错误。变体按已实现路径的需要而设，新增校验点时按需扩展。
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
