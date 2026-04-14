use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};

use crate::memory::{
    memory_links::{
        MemoryLink, MemoryLinkType,
        proc_mem::{ProcMemLink, TrigToAction},
        sem_mem::SemMemLink,
    },
    memory_note::{
        MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType,
        proc_mem::{Action, ActionType, ProcMemory, SkillRecord},
        sem_mem::{ConceptType, SemMemory},
        situation_mem::{
            AbstractSituation, Environment, Event, Location, Participant, SituationType,
        },
    },
};

use super::dto::{
    ConsolidatedEdge, ConsolidatedMemoryType, ConsolidatedNode, ConsolidationOutput,
    ProcedureActionType, ProcedureNodePayload, SemanticConceptType, SemanticNodePayload,
    SituationNodePayload,
};

pub struct MappedConsolidation {
    pub notes: Vec<MemoryNote>,
    pub id_map: HashMap<String, MemoryId>,
}

pub fn map_output_to_notes(output: ConsolidationOutput) -> Result<MappedConsolidation> {
    output.validate()?;

    let mut id_map = HashMap::with_capacity(output.nodes.len());
    let mut type_map = HashMap::with_capacity(output.nodes.len());

    for node in &output.nodes {
        let mem_id = MemoryId::new();
        id_map.insert(node.node_id.clone(), mem_id);
        type_map.insert(node.node_id.clone(), node.memory_type.clone());
    }

    let mut links_by_source: HashMap<MemoryId, Vec<MemoryLink>> = HashMap::new();
    for edge in &output.edges {
        let link = map_edge(edge, &id_map, &type_map)?;
        links_by_source.entry(link.from()).or_default().push(link);
    }

    let mut notes = Vec::with_capacity(output.nodes.len());
    for node in output.nodes {
        let mem_id = id_map
            .get(&node.node_id)
            .copied()
            .ok_or_else(|| anyhow!("node_id missing from id map: {}", node.node_id))?;

        let mem_type = map_node_type(&node)?;
        let links = links_by_source.remove(&mem_id).unwrap_or_default();

        let note = MemoryNoteBuilder::new(mem_type)
            .id(mem_id)
            .tags(node.tags)
            .mem_links(links)
            .build()
            .with_context(|| format!("failed to build MemoryNote for node {}", node.node_id))?;

        notes.push(note);
    }

    Ok(MappedConsolidation { notes, id_map })
}

fn map_node_type(node: &ConsolidatedNode) -> Result<MemoryType> {
    match node.memory_type {
        ConsolidatedMemoryType::Semantic => map_semantic(node),
        ConsolidatedMemoryType::Procedure => map_procedure(node),
        ConsolidatedMemoryType::Situation => map_situation(node),
    }
}

fn map_semantic(node: &ConsolidatedNode) -> Result<MemoryType> {
    let payload: SemanticNodePayload = if node.payload.is_null() {
        SemanticNodePayload::default()
    } else {
        serde_json::from_value(node.payload.clone())
            .with_context(|| format!("invalid semantic payload for node {}", node.node_id))?
    };

    let concept_type = match payload.concept_type.unwrap_or(SemanticConceptType::Entity) {
        SemanticConceptType::Entity => ConceptType::Entity,
        SemanticConceptType::Abstract => ConceptType::Abstract,
    };

    let content = payload.content.unwrap_or_else(|| node.title.clone());
    let description = payload.description.unwrap_or_else(|| node.title.clone());

    let mut sem = SemMemory::new(content, concept_type, description);
    sem.aliases = payload.aliases;

    Ok(MemoryType::Semantic(sem))
}

fn map_procedure(node: &ConsolidatedNode) -> Result<MemoryType> {
    let payload: ProcedureNodePayload = if node.payload.is_null() {
        ProcedureNodePayload::default()
    } else {
        serde_json::from_value(node.payload.clone())
            .with_context(|| format!("invalid procedure payload for node {}", node.node_id))?
    };

    let action_type = match payload.action_type.unwrap_or(ProcedureActionType::Think) {
        ProcedureActionType::Speak => ActionType::new_speak(),
        ProcedureActionType::Think => ActionType::new_think(),
        ProcedureActionType::Skill => ActionType::new_skill(SkillRecord {}),
    };

    let content = payload.content.unwrap_or_else(|| node.title.clone());
    let action = Action::new(content, action_type);
    Ok(MemoryType::Procedure(ProcMemory::new(action)))
}

fn map_situation(node: &ConsolidatedNode) -> Result<MemoryType> {
    let payload: SituationNodePayload = serde_json::from_value(node.payload.clone())
        .with_context(|| format!("invalid situation payload for node {}", node.node_id))?;

    let sit = match payload {
        SituationNodePayload::AbstractLocation { name, coordinates } => {
            SituationType::AbstractSituation(AbstractSituation::Location(Location {
                name,
                coordinates,
            }))
        }
        SituationNodePayload::AbstractParticipant { name, role } => {
            SituationType::AbstractSituation(AbstractSituation::Participant(Participant {
                name,
                role,
            }))
        }
        SituationNodePayload::AbstractEnvironment { atmosphere, tone } => {
            SituationType::AbstractSituation(AbstractSituation::Environment(Environment {
                atmosphere,
                tone,
            }))
        }
        SituationNodePayload::AbstractEvent {
            action,
            action_intensity,
            initiator,
            target,
        } => SituationType::AbstractSituation(AbstractSituation::Event(Event {
            action,
            action_intensity,
            initiator,
            target,
        })),
    };

    Ok(MemoryType::Situation(sit))
}

fn map_edge(
    edge: &ConsolidatedEdge,
    id_map: &HashMap<String, MemoryId>,
    type_map: &HashMap<String, ConsolidatedMemoryType>,
) -> Result<MemoryLink> {
    let from = id_map
        .get(&edge.from)
        .copied()
        .ok_or_else(|| anyhow!("edge from node not found: {}", edge.from))?;
    let to = id_map
        .get(&edge.to)
        .copied()
        .ok_or_else(|| anyhow!("edge to node not found: {}", edge.to))?;

    let link_type = match type_map.get(&edge.from) {
        Some(ConsolidatedMemoryType::Procedure) => {
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(edge.intensity)))
        }
        _ => MemoryLinkType::Sem(SemMemLink::new(
            edge.relation.clone(),
            edge.intensity,
            edge.confidence,
        )),
    };

    let mut link = MemoryLink::new(from, to, link_type);
    link.intensity = edge.intensity as f64;
    Ok(link)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::memory_note::MemoryType;

    #[test]
    fn map_basic_semantic_output() {
        let output = ConsolidationOutput {
            version: "1.0".to_string(),
            nodes: vec![
                ConsolidatedNode {
                    node_id: "n1".to_string(),
                    memory_type: ConsolidatedMemoryType::Semantic,
                    title: "user likes coffee".to_string(),
                    tags: vec!["preference".to_string()],
                    confidence: 0.9,
                    payload: serde_json::json!({
                        "content": "likes coffee",
                        "aliases": ["coffee lover"],
                        "description": "stable preference",
                        "concept_type": "entity"
                    }),
                },
                ConsolidatedNode {
                    node_id: "n2".to_string(),
                    memory_type: ConsolidatedMemoryType::Semantic,
                    title: "coffee can calm user".to_string(),
                    tags: vec!["effect".to_string()],
                    confidence: 0.8,
                    payload: serde_json::json!({
                        "content": "coffee calms user",
                        "concept_type": "abstract"
                    }),
                },
            ],
            edges: vec![ConsolidatedEdge {
                from: "n1".to_string(),
                to: "n2".to_string(),
                relation: "related_to".to_string(),
                intensity: 0.7,
                confidence: 0.8,
            }],
        };

        let mapped = map_output_to_notes(output).expect("mapping should succeed");
        assert_eq!(mapped.notes.len(), 2);
        assert_eq!(mapped.id_map.len(), 2);
        assert!(matches!(
            mapped.notes[0].mem_type(),
            MemoryType::Semantic(_)
        ));
    }
}
