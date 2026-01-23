use anyhow::Result;
use fastembed::Qwen3TextEmbedding;

use crate::memory::embedding::{
    EmbeddingGenError, EmbeddingGenResult, EmbeddingVec, GenericEmbeddingModel,
};
pub struct Qwen3Embedding600M {
    device: candle_core::Device,
    model: Qwen3TextEmbedding,
}
impl Qwen3Embedding600M {
    pub fn default_cpu() -> Result<Self> {
        let device = candle_core::Device::Cpu;
        Ok(Self {
            model: Qwen3TextEmbedding::from_hf(
                "Qwen/Qwen3-Embedding-0.6B",
                &device,
                candle_core::DType::F32,
                1024,
            )?,
            device,
        })
    }
    pub fn default_gpu_cuda() -> Result<Self> {
        let device = candle_core::Device::new_cuda(0)?;
        Ok(Self {
            model: Qwen3TextEmbedding::from_hf(
                "Qwen/Qwen3-Embedding-0.6B",
                &device,
                candle_core::DType::F32,
                1024,
            )?,
            device,
        })
    }
    pub fn default_gpu_metal() -> Result<Self> {
        let device = candle_core::Device::new_metal(0)?;
        Ok(Self {
            model: Qwen3TextEmbedding::from_hf(
                "Qwen/Qwen3-Embedding-0.6B",
                &device,
                candle_core::DType::F32,
                1024,
            )?,
            device,
        })
    }
}
impl Qwen3Embedding600M {
    pub fn embed_gen<S: AsRef<str>>(&self, input: &[S]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        self.model.embed(input).map_err(EmbeddingGenError::from)
    }
}
impl GenericEmbeddingModel for Qwen3Embedding600M {
    fn infer<S: AsRef<str>>(&self, input: &[S]) -> super::EmbeddingGenResult<Vec<EmbeddingVec>> {
        self.embed_gen(input)
    }
}
pub struct Qwen3Embedding600MBuilder {
    device: candle_core::Device,
    dtype: Option<candle_core::DType>,
    dimension: Option<usize>,
}

impl Qwen3Embedding600MBuilder {
    pub fn new(device: candle_core::Device) -> Self {
        Self {
            device,
            dtype: None,
            dimension: None,
        }
    }

    pub fn dtype(mut self, dtype: candle_core::DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    pub fn build_from_hf(self) -> Result<Qwen3Embedding600M> {
        let dtype = self.dtype.unwrap_or(candle_core::DType::F32);
        let dimension = self.dimension.unwrap_or(1024);
        let model = Qwen3TextEmbedding::from_hf(
            "Qwen/Qwen3-Embedding-0.6B",
            &self.device,
            dtype,
            dimension,
        )?;
        Ok(Qwen3Embedding600M {
            model,
            device: self.device,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_qwen3_embedding_600m_cpu() {
        let device = Device::Cpu;
        let builder = Qwen3Embedding600MBuilder::new(device).dimension(1024);
        let model = builder.build_from_hf().unwrap();
        let input = vec![
            "Hello",
            "World",
            "你好！",
            "SoulMem是一个专为角色扮演任务设计的记忆系统，它旨在使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。它不旨在精确无误地记忆事件的细节，或事实性知识。请注意！：SoulMem是针对于个人用户，在家用电脑上运行的记忆系统，并非企业级解决方案。",
        ];
        let embeddings = model.embed_gen(&input).unwrap();
        assert_eq!(embeddings.len(), 4);
        for i in 0..embeddings.len() {
            assert_eq!(embeddings[i].len(), 1024);
        }
    }
    #[test]
    fn test_qwen3_embedding_600m_gpu() {
        let device = Device::new_cuda(0).unwrap();
        let builder = Qwen3Embedding600MBuilder::new(device).dimension(1024);
        let model = builder.build_from_hf().unwrap();
        let input = vec![
            "Hello",
            "World",
            "你好！",
            "SoulMem是一个专为角色扮演任务设计的记忆系统，它旨在使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。它不旨在精确无误地记忆事件的细节，或事实性知识。请注意！：SoulMem是针对于个人用户，在家用电脑上运行的记忆系统，并非企业级解决方案。",
        ];
        let embeddings = model.embed_gen(&input).unwrap();
        assert_eq!(embeddings.len(), 4);
        for i in 0..embeddings.len() {
            assert_eq!(embeddings[i].len(), 1024);
        }
    }
}
