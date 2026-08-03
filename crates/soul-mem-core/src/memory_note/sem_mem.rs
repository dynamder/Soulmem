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
}

impl SemMemory {
    pub fn new(content: String, concept_type: ConceptType, description: String) -> Self {
        Self {
            content,
            aliases: Vec::new(),
            concept_type,
            description,
        }
    }
}

impl Default for SemMemory {
    fn default() -> Self {
        Self {
            content: String::new(),
            aliases: Vec::new(),
            concept_type: ConceptType::Entity,
            description: String::new(),
        }
    }
}
