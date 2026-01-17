use serde::{Deserialize, Serialize};

// 概念类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConceptType {
    Entity,
    Abstract,
}

impl ConceptType {
    pub fn new_entity() -> Self {
        Self::Entity
    }

    pub fn new_abstract() -> Self {
        Self::Abstract
    }
}

// 语义记忆节点
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemMemory {
    content: String,
    aliases: Vec<String>,
    concept_type: ConceptType,
    description: String,
}

impl SemMemory {
    pub fn new(content: String, concept_type: ConceptType, description: String) -> Self {
        SemMemory {
            content,
            aliases: Vec::new(),
            concept_type,
            description,
        }
    }

    pub fn get_content(&self) -> &str {
        &self.content
    }

    pub fn get_aliases(&self) -> &[String] {
        &self.aliases
    }

    pub fn get_concept_type(&self) -> &ConceptType {
        &self.concept_type
    }

    pub fn get_description(&self) -> &str {
        &self.description
    }
}
