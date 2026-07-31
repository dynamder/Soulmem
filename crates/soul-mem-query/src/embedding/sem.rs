use serde::{Deserialize, Serialize};

use super::EmbeddingGenResult;
use super::EmbeddingModel;
use super::EmbeddingVec;

use crate::embedding::note::MemoryEmbeddingVariant;
use crate::embedding::raw_linear_blend;
use crate::embedding::Embeddable;
use crate::embedding::EmbeddingCalcResult;
use soul_mem_core::memory_note::sem_mem::SemMemory;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticEmbedding {
    content: EmbeddingVec,
    aliases: EmbeddingVec,
    description: EmbeddingVec,
}
impl SemanticEmbedding {
    pub fn linear_blend(
        &self,
        other: &SemanticEmbedding,
        blend_factor: f32,
    ) -> EmbeddingCalcResult<SemanticEmbedding> {
        Ok(SemanticEmbedding {
            content: raw_linear_blend(&self.content, &other.content, blend_factor)?,
            aliases: raw_linear_blend(&self.aliases, &other.aliases, blend_factor)?,
            description: raw_linear_blend(&self.description, &other.description, blend_factor)?,
        })
    }
    pub fn content(&self) -> &EmbeddingVec {
        &self.content
    }
    pub fn aliases(&self) -> &EmbeddingVec {
        &self.aliases
    }
    pub fn description(&self) -> &EmbeddingVec {
        &self.description
    }

    pub fn new(content: EmbeddingVec, aliases: EmbeddingVec, description: EmbeddingVec) -> Self {
        Self {
            content,
            aliases,
            description,
        }
    }
}

impl Embeddable for SemMemory {
    type EmbeddingGen = SemanticEmbedding;
    type EmbeddingFused = EmbeddedSemanticMemory;
    //TODO: use multi-thread if necessary
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        let content_vec = model.infer_with_chunk(&self.content)?;

        let description_vec = model.infer_with_chunk(&self.description)?;

        if self.aliases.is_empty() {
            return Ok(SemanticEmbedding {
                content: content_vec.clone(),
                aliases: content_vec,
                description: description_vec,
            });
        }

        let aliases_vec = model.infer_and_fuse(
            &self
                .aliases
                .iter()
                .map(|alias| alias.as_str())
                .collect::<Vec<&str>>(),
        )?;

        Ok(SemanticEmbedding {
            content: content_vec,
            aliases: aliases_vec,
            description: description_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        let embedding = self.embed(model)?;
        Ok(EmbeddedSemanticMemory {
            embedding,
            memory: self,
        })
    }
}

pub struct EmbeddedSemanticMemory {
    pub embedding: SemanticEmbedding,
    pub memory: SemMemory,
}

impl From<SemanticEmbedding> for MemoryEmbeddingVariant {
    fn from(value: SemanticEmbedding) -> Self {
        MemoryEmbeddingVariant::Semantic(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use soul_mem_core::memory_note::sem_mem::ConceptType;

    #[test]
    fn test_semantic_embed() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let memory = SemMemory {
            content: "Rustacean".to_string(),
            aliases: vec!["Ruster".to_string(), "Rust programmer".to_string(), "Rust开发者".to_string(), "Rust程序员".to_string(), "Rust爱好者".to_string()],
            description: "使用Rust编程语言的程序员。它们中的一些热衷于重写各类本来由其他语言编写的代码，以提高性能。他们偶尔会被Rust编译器的严格而搞得脑子疼。部分狂热分子热衷于在社交媒体上宣传Rust，引起了部分开发者对这些狂热分子的不满。".to_string(),
            concept_type: ConceptType::Entity,
        };

        let sem_embedding = memory.embed(&model).unwrap();
        let dimension = sem_embedding.content.shape();
        assert_eq!(dimension, 512);
        assert_eq!(sem_embedding.description.shape(), dimension);
        assert_eq!(sem_embedding.aliases.shape(), dimension);
    }

    #[test]
    fn test_semantic_embed_linear_blend() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let memory1 = SemMemory {
            content: "人工智能".to_string(),
            aliases: vec!["AI".to_string()],
            concept_type: ConceptType::Abstract,
            description: "让机器具有人类智能的技术".to_string(),
        };

        let memory2 = SemMemory {
            content: "机器学习".to_string(),
            aliases: vec!["ML".to_string()],
            concept_type: ConceptType::Abstract,
            description: "人工智能的一个分支".to_string(),
        };

        let embedding1 = memory1.embed(&model).unwrap();
        let embedding2 = memory2.embed(&model).unwrap();

        let blended = embedding1.linear_blend(&embedding2, 0.5).unwrap();

        assert_eq!(blended.content.shape(), 512);
        assert_eq!(blended.aliases.shape(), 512);
        assert_eq!(blended.description.shape(), 512);
    }

    #[test]
    fn test_semantic_embed_with_aliases() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let memory = SemMemory {
            content: "测试内容".to_string(),
            aliases: vec!["alias1".to_string(), "alias2".to_string()],
            concept_type: ConceptType::Entity,
            description: "测试描述".to_string(),
        };

        let sem_embedding = memory.embed(&model).unwrap();

        assert_eq!(sem_embedding.content.shape(), 512);
        assert_eq!(sem_embedding.aliases.shape(), 512);
    }
}
