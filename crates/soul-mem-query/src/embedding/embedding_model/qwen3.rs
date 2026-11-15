use anyhow::Result;

use embed_anything::embeddings::local::qwen3::{Qwen3Embed, Qwen3Embedder};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use text_splitter::{Characters, TextSplitter};

use crate::embedding::{EmbeddingGenError, EmbeddingGenResult, EmbeddingModel, EmbeddingVec};
pub struct Qwen3Embedding600M {
    model: Qwen3Embedder,
    splitter: TextSplitter<Characters>,
}
impl Qwen3Embedding600M {
    pub fn default_cpu() -> Result<Self> {
        Ok(Self {
            splitter: TextSplitter::new(6000), //should be 6000
            model: Qwen3Embedder::new(
                "Qwen/Qwen3-Embedding-0.6B",
                None,
                None,
                Some(embed_anything::Dtype::F32),
            )?,
        })
    }
}
impl Qwen3Embedding600M {
    pub fn embed_gen_simple_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        Ok(self
            .model
            .embed(input, None, None)?
            .into_iter()
            .map(|e| EmbeddingVec::new(e.to_dense().unwrap())) //SAFEUNWRAP: qwen3 embedder 在 embed_anything 源码中永远返回 dense
            .collect())
    }
    pub fn embed_gen_with_chunk_pooling(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        let chunked_input = self.splitter.chunks(input).collect::<Vec<_>>();
        if chunked_input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }

        self.embed_gen_with_mean_pooling(&chunked_input)
    }
    pub fn embed_gen_with_mean_pooling(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        if input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }

        let embeddings = self
            .model
            .embed(input, None, None)?
            .into_iter()
            .map(|e| e.to_dense().unwrap()) //SAFEUNWRAP: qwen3 embedder在embed_anything的源码中永远返回dense
            .collect::<Vec<_>>();

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

/// Qwen3-Embedding 的查询侧指令。模型卡要求查询带任务指令、段落不带；
/// 指令内容可按场景定制，这里用模型卡的检索默认值。
pub const QUERY_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

fn prepend_query_instruction(input: &[&str]) -> Vec<String> {
    input
        .iter()
        .map(|s| format!("Instruct: {QUERY_INSTRUCTION}\nQuery: {s}"))
        .collect()
}

impl EmbeddingModel for Qwen3Embedding600M {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        self.embed_gen_simple_batch(input)
    }
    fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        self.embed_gen_with_mean_pooling(input)
    }
    fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        self.embed_gen_with_chunk_pooling(input)
    }
    fn infer_query_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        let prefixed = prepend_query_instruction(input);
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed_gen_simple_batch(&refs)
    }
    fn infer_query_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        let prefixed = prepend_query_instruction(input);
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed_gen_with_mean_pooling(&refs)
    }
    fn infer_query_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        let chunked_input = self.splitter.chunks(input).collect::<Vec<_>>();
        if chunked_input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }
        // 指令加在每个分块上，保证所有分块都落在 query 空间。
        let prefixed = prepend_query_instruction(&chunked_input);
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed_gen_with_mean_pooling(&refs)
    }
    fn max_input_token(&self) -> usize {
        32768
    }
    fn dim(&self) -> usize {
        1024
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_query_instruction_format() {
        assert_eq!(
            prepend_query_instruction(&["a", "b"]),
            vec![
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: a".to_string(),
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: b".to_string(),
            ]
        );
    }

    #[tokio::test]
    #[ignore = "downloads ~1.2GB Qwen3-Embedding-0.6B model; run manually or in a dedicated job"]
    async fn test_qwen3_embedding_600m_cpu() {
        let model = Qwen3Embedding600M::default_cpu().unwrap();
        let input = "SoulMem是一个专为角色扮演任务设计的记忆系统，它旨在使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。它不旨在精确无误地记忆事件的细节，或事实性知识。请注意！：SoulMem是针对于个人用户，在家用电脑上运行的记忆系统，并非企业级解决方案。";
        let embeddings = model.embed_gen_with_chunk_pooling(input).unwrap();
        assert_eq!(embeddings.shape(), 1024);
    }
}
