use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationOutput {
    pub nodes: Vec<ConsolidatedNode>,
    pub edges: Vec<ConsolidatedEdge>,
}

impl ConsolidationOutput {
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(anyhow!("nodes must not be empty"));
        }

        let mut node_ids = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if node.node_id.trim().is_empty() {
                return Err(anyhow!("node_id must not be empty"));
            }
            if !node_ids.insert(node.node_id.as_str()) {
                return Err(anyhow!("duplicated node_id: {}", node.node_id));
            }
            validate_node_payload(node)?;
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

fn validate_node_payload(node: &ConsolidatedNode) -> Result<()> {
    if !node.payload.is_object() {
        return Err(anyhow!(
            "payload must be an object for node {}",
            node.node_id
        ));
    }

    match &node.memory_type {
        ConsolidatedMemoryType::Semantic => {
            let payload: SemanticNodePayload = serde_json::from_value(node.payload.clone())
                .map_err(|err| {
                    anyhow!("invalid semantic payload for node {}: {err}", node.node_id)
                })?;
            validate_required_text("semantic content", &payload.content)?;
            validate_required_text("semantic description", &payload.description)?;
            for alias in &payload.aliases {
                validate_required_text("semantic alias", alias)?;
            }
        }
        ConsolidatedMemoryType::Procedure => {
            let payload: ProcedureNodePayload = serde_json::from_value(node.payload.clone())
                .map_err(|err| {
                    anyhow!("invalid procedure payload for node {}: {err}", node.node_id)
                })?;
            validate_required_text("procedure content", &payload.content)?;
        }
        ConsolidatedMemoryType::Situation => {
            let payload: SituationNodePayload = serde_json::from_value(node.payload.clone())
                .map_err(|err| {
                    anyhow!("invalid situation payload for node {}: {err}", node.node_id)
                })?;
            match payload {
                SituationNodePayload::AbstractLocation { name, coordinates } => {
                    validate_required_text("situation name", &name)?;
                    validate_required_text("situation coordinates", &coordinates)?;
                }
                SituationNodePayload::AbstractParticipant { name, role } => {
                    validate_required_text("situation name", &name)?;
                    validate_required_text("situation role", &role)?;
                }
                SituationNodePayload::AbstractEnvironment { atmosphere, tone } => {
                    validate_required_text("situation atmosphere", &atmosphere)?;
                    validate_required_text("situation tone", &tone)?;
                }
                SituationNodePayload::AbstractEvent {
                    action,
                    action_intensity,
                    initiator,
                    target,
                } => {
                    validate_required_text("situation action", &action)?;
                    validate_required_text("situation initiator", &initiator)?;
                    validate_required_text("situation target", &target)?;
                    if !(0.0..=1.0).contains(&action_intensity) {
                        return Err(anyhow!(
                            "situation action_intensity must be in [0, 1] for node {}",
                            node.node_id
                        ));
                    }
                }
                SituationNodePayload::SpecificSituation {
                    narrative,
                    time_span: _,
                    context,
                } => {
                    validate_required_text("situation narrative", &narrative)?;
                    validate_situation_context(&context, &node.node_id)?;
                }
            }
        }
    }

    Ok(())
}

fn validate_required_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || is_placeholder(value) {
        return Err(anyhow!("{field} must contain meaningful text"));
    }
    Ok(())
}

fn validate_situation_context(context: &SituationContextPayload, node_id: &str) -> Result<()> {
    if let Some(location) = &context.location {
        validate_required_text("situation location name", &location.name)?;
        validate_required_text("situation location coordinates", &location.coordinates)?;
    }

    for participant in &context.participants {
        validate_required_text("situation participant name", &participant.name)?;
        validate_required_text("situation participant role", &participant.role)?;
    }
    for emotion in &context.emotions {
        validate_required_text("situation emotion name", &emotion.name)?;
        validate_unit_number("situation emotion intensity", emotion.intensity, node_id)?;
    }
    for sensory_data in &context.sensory_data {
        validate_required_text("situation sensory data name", &sensory_data.name)?;
        validate_unit_number(
            "situation sensory data intensity",
            sensory_data.intensity,
            node_id,
        )?;
    }

    validate_required_text(
        "situation environment atmosphere",
        &context.environment.atmosphere,
    )?;
    validate_required_text("situation environment tone", &context.environment.tone)?;

    for event in &context.event {
        validate_required_text("situation event action", &event.action)?;
        validate_required_text("situation event initiator", &event.initiator)?;
        validate_required_text("situation event target", &event.target)?;
        validate_unit_number(
            "situation event action_intensity",
            event.action_intensity,
            node_id,
        )?;
    }

    Ok(())
}

fn validate_unit_number(field: &str, value: f32, node_id: &str) -> Result<()> {
    if !(0.0..=1.0).contains(&value) {
        return Err(anyhow!(
            "{field} must be in [0, 1] for node {node_id}, got {value}"
        ));
    }
    Ok(())
}

fn is_placeholder(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "unknown" | "n/a" | "null" | "none" | "不详" | "未知"
    )
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsolidatedNode {
    pub node_id: String,
    pub memory_type: ConsolidatedMemoryType,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticNodePayload {
    pub content: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub concept_type: SemanticConceptType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticConceptType {
    Entity,
    Abstract,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureNodePayload {
    pub content: String,
    pub action_type: ProcedureActionType,
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
#[serde(deny_unknown_fields)]
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
    SpecificSituation {
        narrative: String,
        time_span: DateTime<Utc>,
        context: SituationContextPayload,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationContextPayload {
    pub location: Option<SituationLocationPayload>,
    pub participants: Vec<SituationParticipantPayload>,
    pub emotions: Vec<SituationEmotionPayload>,
    pub sensory_data: Vec<SituationSensoryDataPayload>,
    pub environment: SituationEnvironmentPayload,
    pub event: Vec<SituationEventPayload>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationLocationPayload {
    pub name: String,
    pub coordinates: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationParticipantPayload {
    pub name: String,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationEnvironmentPayload {
    pub atmosphere: String,
    pub tone: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationEventPayload {
    pub action: String,
    pub action_intensity: f32,
    pub initiator: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationEmotionPayload {
    pub name: String,
    pub intensity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SituationSensoryDataPayload {
    pub name: String,
    pub intensity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_payload_requires_every_struct_field() {
        let output: ConsolidationOutput = serde_json::from_value(serde_json::json!({
            "nodes": [{
                "node_id": "n1",
                "memory_type": "semantic",
                "payload": {
                    "content": "原神",
                    "aliases": [],
                    "concept_type": "entity"
                }
            }],
            "edges": []
        }))
        .expect("top-level output should deserialize");

        assert!(output.validate().is_err());
    }

    #[test]
    fn payload_rejects_fields_outside_the_schema() {
        let output: ConsolidationOutput = serde_json::from_value(serde_json::json!({
            "nodes": [{
                "node_id": "n1",
                "memory_type": "procedure",
                "payload": {
                    "content": "回答前先核对数据库记录",
                    "action_type": "think",
                    "unexpected": true
                }
            }],
            "edges": []
        }))
        .expect("top-level output should deserialize");

        assert!(output.validate().is_err());
    }
}
