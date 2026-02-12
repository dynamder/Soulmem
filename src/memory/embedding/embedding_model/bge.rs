use anyhow::Result;
use async_trait::async_trait;
use embed_anything::embeddings::{
    embed::{Embedder, EmbedderBuilder},
    local::bert::{BertEmbed, BertEmbedder},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use text_splitter::{Characters, TextSplitter};

use crate::memory::embedding::{
    EmbeddingGenError, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
};

pub struct BgeSmallZh {
    model: BertEmbedder,
    splitter: TextSplitter<Characters>,
}
impl BgeSmallZh {
    pub fn default_cpu() -> Result<Self> {
        Ok(Self {
            splitter: TextSplitter::new(200), //should be 6000
            model: BertEmbedder::new("BAAI/bge-small-zh-v1.5".to_string(), None, None)?,
        })
    }
}
impl BgeSmallZh {
    //简单的批量生成，单个句子过长则截断
    pub fn embed_gen_simple_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        Ok(self
            .model
            .embed(&input, None, None)?
            .into_iter()
            .map(|e| EmbeddingVec::new(e.to_dense().unwrap())) //SAFEUNWRAP: qwen3 embedder在embed_anything的
            .collect())
    }
    //对于长文本，分块向量化后平均池化
    pub fn embed_gen_with_chunk_pooling(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        //分块文本
        let chunked_input = self.splitter.chunks(input).collect::<Vec<_>>();
        //println!("chunked_input: {:?}", chunked_input);
        if chunked_input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }

        self.embed_gen_with_mean_pooling(&chunked_input)
    }
    //将输入的所有句子向量化后平均池化，如果单个句子长度过长，会被截断
    pub fn embed_gen_with_mean_pooling(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        if input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }

        //生成embedding
        let embeddings = self
            .model
            .embed(input, None, None)?
            .into_iter()
            .map(|e| e.to_dense().unwrap()) //SAFEUNWRAP: qwen3 embedder在embed_anything的源码中永远返回dense
            .collect::<Vec<_>>();

        //融合embedding，平均池化
        let embedding_dimension = embeddings[0].len();
        let fused_embedding = (0..embedding_dimension)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for embedding in &embeddings {
                    sum += embedding[i];
                }
                sum / embeddings.len() as f32
            })
            .collect::<Vec<_>>();
        Ok(EmbeddingVec::new(fused_embedding))
    }
}

impl EmbeddingModel for BgeSmallZh {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        self.embed_gen_simple_batch(input)
    }
    fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        self.embed_gen_with_mean_pooling(input)
    }
    fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        self.embed_gen_with_chunk_pooling(input)
    }
    fn max_input_token(&self) -> usize {
        512
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bge_small_zh_cpu() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let input = "SoulMem是一个专为角色扮演任务设计的记忆系统，它旨在使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。它不旨在精确无误地记忆事件的细节，或事实性知识。请注意！：SoulMem是针对于个人用户，在家用电脑上运行的记忆系统，并非企业级解决方案。";
        let embeddings = model.embed_gen_with_chunk_pooling(&input).unwrap();
        assert_eq!(embeddings.shape(), 512);
    }
}
