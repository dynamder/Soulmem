use std::sync::{Arc, OnceLock};

use anyhow::Result;

use embed_anything::embeddings::local::bert::{BertEmbed, BertEmbedder};
use embed_anything::embeddings::local::pooling::Pooling;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use text_splitter::{Characters, TextSplitter};

use crate::embedding::{EmbeddingGenError, EmbeddingGenResult, EmbeddingModel, EmbeddingVec};

/// BGE v1.5 官方 s2p 检索指令：仅查询侧添加，passage 侧不加。
/// 参见 BAAI/bge-small-zh-v1.5 README（"for s2p retrieval task, add an instruction to query"）。
pub const QUERY_INSTRUCTION: &str = "为这个句子生成表示以用于检索相关文章：";

pub struct BgeSmallZh {
    model: Arc<BertEmbedder>,
    splitter: Arc<TextSplitter<Characters>>,
}
impl Clone for BgeSmallZh {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            splitter: Arc::clone(&self.splitter),
        }
    }
}
impl BgeSmallZh {
    /// 全局共享的BGE模型实例。embed_anything依赖的candle在Windows上对同一safetensors
    /// 文件进行并发mmap会产生文件锁冲突，因此整个进程只构造一次底层模型。
    pub fn default_cpu() -> Result<Self> {
        static MODEL: OnceLock<Arc<BertEmbedder>> = OnceLock::new();
        let model = MODEL.get_or_init(|| {
            Arc::new(
                BertEmbedder::new(
                    "BAAI/bge-small-zh-v1.5".to_string(),
                    None,
                    None,
                    Some(Pooling::Cls),
                )
                .expect("BGE model init failed"),
            )
        });
        Ok(Self {
            model: Arc::clone(model),
            splitter: Arc::new(TextSplitter::new(200)), //should be 6000
        })
    }
}
impl BgeSmallZh {
    //简单的批量生成，单个句子过长则截断
    pub fn embed_gen_simple_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        Ok(self
            .model
            .embed(input, None, None)?
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
        // 指令加在每个分块上，保证所有分块都落在 query 空间（分块均值不会被 passage 空间向量稀释）。
        let prefixed = prepend_query_instruction(&chunked_input);
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed_gen_with_mean_pooling(&refs)
    }
    fn max_input_token(&self) -> usize {
        512
    }
    fn dim(&self) -> usize {
        512
    }
}

fn prepend_query_instruction(input: &[&str]) -> Vec<String> {
    input
        .iter()
        .map(|s| format!("{QUERY_INSTRUCTION}{s}"))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bge_small_zh_cpu() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let input = "SoulMem是一个专为角色扮演任务设计的记忆系统，它旨在使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。它不旨在精确无误地记忆事件的细节，或事实性知识。请注意！：SoulMem是针对于个人用户，在家用电脑上运行的记忆系统，并非企业级解决方案。";
        let embeddings = model.embed_gen_with_chunk_pooling(input).unwrap();
        assert_eq!(embeddings.shape(), 512);
    }

    #[test]
    fn test_query_instruction_prepended_batch() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let text = "酒馆";
        let direct = model.infer_query_batch(&[text]).unwrap();
        let manual = model
            .infer_batch(&[&format!("{QUERY_INSTRUCTION}{text}")])
            .unwrap();
        assert_eq!(direct[0], manual[0]);
    }

    #[test]
    fn test_query_instruction_prepended_fuse() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let texts = ["酒馆", "酒吧"];
        let direct = model.infer_query_and_fuse(&texts).unwrap();
        let manual = model
            .infer_and_fuse(
                &texts
                    .iter()
                    .map(|s| format!("{QUERY_INSTRUCTION}{s}"))
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(direct, manual);
    }

    #[test]
    fn test_query_instruction_prepended_per_chunk() {
        let model = BgeSmallZh::default_cpu().unwrap();
        // 超过 200 字符，确保走分块路径
        let long = "在太阳系边缘沉睡了很久之后被符华唤醒，她带着我参与了福洛斯的战斗，让我认识了薇塔和七位小代理人。那之后我重新找回了用画笔描绘世界的感觉，把星星都画成温暖的光。".repeat(3);
        let direct = model.infer_query_with_chunk(&long).unwrap();

        let chunks = model.splitter.chunks(&long).collect::<Vec<_>>();
        assert!(chunks.len() > 1, "长文本应被分成多个分块");
        let manual = model
            .embed_gen_with_mean_pooling(
                &chunks
                    .iter()
                    .map(|c| format!("{QUERY_INSTRUCTION}{c}"))
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(direct, manual);
    }
}
