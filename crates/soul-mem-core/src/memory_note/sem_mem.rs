use serde::{Deserialize, Serialize};

// 概念类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConceptType {
    Entity,
    Abstract,
}

// 语义记忆节点
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemMemory {
    pub content: String,
    pub aliases: Vec<String>,
    pub concept_type: ConceptType,
    pub description: String,
    /// 遗忘缺失度（0.0 新鲜 ~ 1.0 完全遗忘）
    #[serde(default)]
    pub missing_degree: f32,
}

impl SemMemory {
    pub fn new(content: String, concept_type: ConceptType, description: String) -> Self {
        Self {
            content,
            aliases: Vec::new(),
            concept_type,
            description,
            missing_degree: 0.0,
        }
    }
    pub fn missing_degree(&self) -> f32 {
        self.missing_degree
    }
    pub fn set_missing_degree(&mut self, missing_degree: f32) {
        self.missing_degree = missing_degree.clamp(0.0, 1.0);
    }
}

impl Default for SemMemory {
    fn default() -> Self {
        Self {
            content: String::new(),
            aliases: Vec::new(),
            concept_type: ConceptType::Entity,
            description: String::new(),
            missing_degree: 0.0,
        }
    }
}
