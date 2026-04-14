use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidationOutput {
    pub version: String,
    pub nodes: Vec<ConsolidatedNode>,
    pub edges: Vec<ConsolidatedEdge>,
}

impl ConsolidationOutput {
    pub fn validate(&self) -> Result<()> {
        if self.version.trim().is_empty() {
            return Err(anyhow!("version must not be empty"));
        }

        if self.nodes.is_empty() {
            return Err(anyhow!("nodes must not be empty"));
        }

        let mut node_ids = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if node.node_id.trim().is_empty() {
                return Err(anyhow!("node_id must not be empty"));
            }
            if node.title.trim().is_empty() {
                return Err(anyhow!("node title must not be empty"));
            }
            if !(0.0..=1.0).contains(&node.confidence) {
                return Err(anyhow!(
                    "node confidence must be in [0, 1], got {} for node {}",
                    node.confidence,
                    node.node_id
                ));
            }
            if !node_ids.insert(node.node_id.as_str()) {
                return Err(anyhow!("duplicated node_id: {}", node.node_id));
            }
        }

        for edge in &self.edges {
            if edge.from.trim().is_empty() || edge.to.trim().is_empty() {
                return Err(anyhow!("edge from/to must not be empty"));
            }
            if edge.relation.trim().is_empty() {
                return Err(anyhow!("edge relation must not be empty"));
            }
            if !(0.0..=1.0).contains(&edge.intensity) {
                return Err(anyhow!(
                    "edge intensity must be in [0, 1], got {} for edge {} -> {}",
                    edge.intensity,
                    edge.from,
                    edge.to
                ));
            }
            if !(0.0..=1.0).contains(&edge.confidence) {
                return Err(anyhow!(
                    "edge confidence must be in [0, 1], got {} for edge {} -> {}",
                    edge.confidence,
                    edge.from,
                    edge.to
                ));
            }
            if !node_ids.contains(edge.from.as_str()) || !node_ids.contains(edge.to.as_str()) {
                return Err(anyhow!(
                    "edge references unknown node: {} -> {}",
                    edge.from,
                    edge.to
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidatedNode {
    pub node_id: String,
    pub memory_type: ConsolidatedMemoryType,
    pub title: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub confidence: f32,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidatedEdge {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub intensity: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidatedMemoryType {
    Semantic,
    Situation,
    Procedure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SemanticNodePayload {
    pub content: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub description: Option<String>,
    pub concept_type: Option<SemanticConceptType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticConceptType {
    Entity,
    Abstract,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProcedureNodePayload {
    pub content: Option<String>,
    pub action_type: Option<ProcedureActionType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureActionType {
    Speak,
    Think,
    Skill,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SituationNodePayload {
    AbstractLocation {
        name: String,
        coordinates: String,
    },
    AbstractParticipant {
        name: String,
        role: String,
    },
    AbstractEnvironment {
        atmosphere: String,
        tone: String,
    },
    AbstractEvent {
        action: String,
        action_intensity: f32,
        initiator: String,
        target: String,
    },
}
