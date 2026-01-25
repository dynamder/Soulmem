use anyhow::Result;
use async_trait::async_trait;
use embed_anything::embeddings::embed::{Embedder, EmbedderBuilder};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use text_splitter::{Characters, TextSplitter};

use crate::memory::embedding::{
    EmbeddingGenError, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
};
pub struct Qwen3Embedding600M {
    model: Embedder,
    splitter: TextSplitter<Characters>,
}
impl Qwen3Embedding600M {
    pub fn default_cpu() -> Result<Self> {
        Ok(Self {
            splitter: TextSplitter::new(6000), //should be 6000
            model: EmbedderBuilder::new()
                .model_id(Some("Qwen/Qwen3-Embedding-0.6B"))
                .from_pretrained_hf()?,
        })
    }
    // pub fn default_gpu_cuda() -> Result<Self> {
    //     let device = candle_core::Device::new_cuda(0)?;
    //     Ok(Self {
    //         model: Qwen3TextEmbedding::from_hf(
    //             "Qwen/Qwen3-Embedding-0.6B",
    //             &device,
    //             candle_core::DType::F32,
    //             1024,
    //         )?,
    //         device,
    //     })
    // }
    // pub fn default_gpu_metal() -> Result<Self> {
    //     let device = candle_core::Device::new_metal(0)?;
    //     Ok(Self {
    //         model: Qwen3TextEmbedding::from_hf(
    //             "Qwen/Qwen3-Embedding-0.6B",
    //             &device,
    //             candle_core::DType::F32,
    //             1024,
    //         )?,
    //         device,
    //     })
    // }
}
impl Qwen3Embedding600M {
    //简单的批量生成，单个句子过长则截断
    pub async fn embed_gen_simple_batch(
        &self,
        input: &[&str],
    ) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        Ok(self
            .model
            .embed(&input, None, None)
            .await?
            .into_iter()
            .map(|e| e.to_dense().unwrap()) //SAFEUNWRAP: qwen3 embedder在embed_anything的
            .collect())
    }
    //对于长文本，分块向量化后平均池化
    pub async fn embed_gen_with_chunk_pooling(
        &self,
        input: &str,
    ) -> EmbeddingGenResult<EmbeddingVec> {
        //分块文本
        let chunked_input = self.splitter.chunks(&input).collect::<Vec<_>>();
        //println!("chunked_input: {:?}", chunked_input);
        if chunked_input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }

        self.embed_gen_with_mean_pooling(&chunked_input).await
    }
    //将输入的所有句子向量化后平均池化，如果单个句子长度过长，会被截断
    pub async fn embed_gen_with_mean_pooling(
        &self,
        input: &[&str],
    ) -> EmbeddingGenResult<EmbeddingVec> {
        if input.is_empty() {
            return Err(EmbeddingGenError::InvalidInput);
        }

        //生成embedding
        let embeddings = self
            .model
            .embed(input, None, None)
            .await?
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
            .collect::<EmbeddingVec>();
        Ok(fused_embedding)
    }
}
#[async_trait]
impl EmbeddingModel for Qwen3Embedding600M {
    async fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        self.embed_gen_simple_batch(input).await
    }
    async fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        self.embed_gen_with_mean_pooling(input).await
    }
    async fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        self.embed_gen_with_chunk_pooling(input).await
    }
    fn max_input_token(&self) -> usize {
        32768
    }
}
// pub struct Qwen3Embedding600MBuilder {
//     device: candle_core::Device,
//     dtype: Option<candle_core::DType>,
//     dimension: Option<usize>,
// }

// impl Qwen3Embedding600MBuilder {
//     pub fn new(device: candle_core::Device) -> Self {
//         Self {
//             device,
//             dtype: None,
//             dimension: None,
//         }
//     }

//     pub fn dtype(mut self, dtype: candle_core::DType) -> Self {
//         self.dtype = Some(dtype);
//         self
//     }

//     pub fn dimension(mut self, dimension: usize) -> Self {
//         self.dimension = Some(dimension);
//         self
//     }

//     pub fn build_from_hf(self) -> Result<Qwen3Embedding600M> {
//         let dtype = self.dtype.unwrap_or(candle_core::DType::F32);
//         let dimension = self.dimension.unwrap_or(1024);
//         let model = Qwen3TextEmbedding::from_hf(
//             "Qwen/Qwen3-Embedding-0.6B",
//             &self.device,
//             dtype,
//             dimension,
//         )?;
//         Ok(Qwen3Embedding600M {
//             model,
//             device: self.device,
//         })
//     }
// }

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_qwen3_embedding_600m_cpu() {
        let model = Qwen3Embedding600M::default_cpu().unwrap();
        let input = "SoulMem是一个专为角色扮演任务设计的记忆系统，它旨在使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。它不旨在精确无误地记忆事件的细节，或事实性知识。请注意！：SoulMem是针对于个人用户，在家用电脑上运行的记忆系统，并非企业级解决方案。";
        let embeddings = model.embed_gen_with_chunk_pooling(&input).await.unwrap();
        assert_eq!(embeddings.len(), 1024);
    }
}
