use fastembed::{Embedding, TextEmbedding};
use anyhow::Result;
pub trait CalcEmbedding {
    #[allow(dead_code)]
    fn calc_embedding(&self,embedding_model: &mut TextEmbedding) -> Result<Vec<Embedding>>;
}