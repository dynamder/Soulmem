use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow};

use soul_mem_core::{
    memory_links::{
        MemoryLink, MemoryLinkType,
        proc_mem::{ProcMemLink, TrigToAction},
        sem_mem::SemMemLink,
        situation_mem::{AbstractToSpecific, SituationMemLink},
    },
    memory_note::{
        MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType,
        proc_mem::{Action, ActionType, ProcMemory, SkillRecord},
        sem_mem::{ConceptType, SemMemory},
        situation_mem::{
            AbstractSituation, Context as SituationContext, Emotion, Environment, Event, Location,
            Participant, SensoryData, SituationType, SpecificSituation,
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
}

pub fn map_output_to_notes(output: ConsolidationOutput) -> Result<MappedConsolidation> {
    output.validate()?;

    let mut id_map = HashMap::with_capacity(output.nodes.len());
    let mut type_map = HashMap::with_capacity(output.nodes.len());
    let mut specific_situation_ids = HashSet::new();

    for node in &output.nodes {
        let mem_id = MemoryId::new();
        id_map.insert(node.node_id.clone(), mem_id);
        type_map.insert(node.node_id.clone(), node.memory_type.clone());
        if node.memory_type == ConsolidatedMemoryType::Situation {
            let payload: SituationNodePayload = serde_json::from_value(node.payload.clone())
                .with_context(|| format!("invalid situation payload for node {}", node.node_id))?;
            if matches!(payload, SituationNodePayload::SpecificSituation { .. }) {
                specific_situation_ids.insert(node.node_id.clone());
            }
        }
    }

    let mut links_by_source: HashMap<MemoryId, Vec<MemoryLink>> = HashMap::new();
    for edge in &output.edges {
        let link = map_edge(edge, &id_map, &type_map, &specific_situation_ids)?;
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
            .mem_links(links)
            .build()
            .with_context(|| format!("failed to build MemoryNote for node {}", node.node_id))?;

        notes.push(note);
    }

    Ok(MappedConsolidation { notes })
}

fn map_node_type(node: &ConsolidatedNode) -> Result<MemoryType> {
    match node.memory_type {
        ConsolidatedMemoryType::Semantic => map_semantic(node),
        ConsolidatedMemoryType::Procedure => map_procedure(node),
        ConsolidatedMemoryType::Situation => map_situation(node),
    }
}

fn map_semantic(node: &ConsolidatedNode) -> Result<MemoryType> {
    let payload: SemanticNodePayload = serde_json::from_value(node.payload.clone())
        .with_context(|| format!("invalid semantic payload for node {}", node.node_id))?;

    let concept_type = match payload.concept_type {
        SemanticConceptType::Entity => ConceptType::Entity,
        SemanticConceptType::Abstract => ConceptType::Abstract,
    };

    let mut sem = SemMemory::new(payload.content, concept_type, payload.description);
    sem.aliases = payload.aliases;

    Ok(MemoryType::Semantic(sem))
}

fn map_procedure(node: &ConsolidatedNode) -> Result<MemoryType> {
    let payload: ProcedureNodePayload = serde_json::from_value(node.payload.clone())
        .with_context(|| format!("invalid procedure payload for node {}", node.node_id))?;

    let action_type = match payload.action_type {
        ProcedureActionType::Speak => ActionType::new_speak(),
        ProcedureActionType::Think => ActionType::new_think(),
        ProcedureActionType::Skill => ActionType::new_skill(SkillRecord {}),
    };

    let action = Action::new(payload.content, action_type);
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
        SituationNodePayload::SpecificSituation {
            narrative,
            time_span,
            context,
        } => {
            let location = context.location.map(|location| Location {
                name: location.name,
                coordinates: location.coordinates,
            });
            let participants = context
                .participants
                .into_iter()
                .map(|participant| Participant {
                    name: participant.name,
                    role: participant.role,
                })
                .collect();
            let emotions = context
                .emotions
                .into_iter()
                .map(|emotion| Emotion {
                    name: emotion.name,
                    intensity: emotion.intensity,
                })
                .collect();
            let sensory_data = context
                .sensory_data
                .into_iter()
                .map(|sensory_data| SensoryData {
                    name: sensory_data.name,
                    intensity: sensory_data.intensity,
                })
                .collect();
            let environment = Environment {
                atmosphere: context.environment.atmosphere,
                tone: context.environment.tone,
            };
            let events = context
                .event
                .into_iter()
                .map(|event| Event {
                    action: event.action,
                    action_intensity: event.action_intensity,
                    initiator: event.initiator,
                    target: event.target,
                })
                .collect();
            let context = SituationContext::new(
                location,
                participants,
                emotions,
                sensory_data,
                environment,
                events,
            );

            SituationType::SpecificSituation(SpecificSituation::new(narrative, time_span, context))
        }
    };

    Ok(MemoryType::Situation(sit))
}

fn map_edge(
    edge: &ConsolidatedEdge,
    id_map: &HashMap<String, MemoryId>,
    type_map: &HashMap<String, ConsolidatedMemoryType>,
    specific_situation_ids: &HashSet<String>,
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
        Some(ConsolidatedMemoryType::Procedure) => MemoryLinkType::Proc(ProcMemLink::TrigToAction(
            TrigToAction::new(edge.intensity.into()),
        )),
        Some(ConsolidatedMemoryType::Situation)
            if !specific_situation_ids.contains(&edge.from)
                && specific_situation_ids.contains(&edge.to) =>
        {
            MemoryLinkType::Sit(SituationMemLink::AbstractToSpecific(
                AbstractToSpecific::new(),
            ))
        }
        _ => MemoryLinkType::Sem(SemMemLink::new(edge.relation.clone(), edge.confidence)),
    };

    let mut link = MemoryLink::new(from, to, link_type);
    link.intensity = edge.intensity as f64;
    Ok(link)
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_note::MemoryType;

    #[test]
    fn map_basic_semantic_output() {
        let output = ConsolidationOutput {
            nodes: vec![
                ConsolidatedNode {
                    node_id: "n1".to_string(),
                    memory_type: ConsolidatedMemoryType::Semantic,
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
                    payload: serde_json::json!({
                        "content": "coffee calms user",
                        "aliases": [],
                        "description": "coffee has a calming effect on the user",
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
        assert!(matches!(
            mapped.notes[0].mem_type(),
            MemoryType::Semantic(_)
        ));
    }

    #[test]
    fn map_procedure_output() {
        let output = ConsolidationOutput {
            nodes: vec![ConsolidatedNode {
                node_id: "p1".to_string(),
                memory_type: ConsolidatedMemoryType::Procedure,
                payload: serde_json::json!({
                    "content": "回答前先核对数据库记录",
                    "action_type": "think"
                }),
            }],
            edges: Vec::new(),
        };

        let mapped = map_output_to_notes(output).expect("mapping should succeed");
        assert!(matches!(
            mapped.notes[0].mem_type(),
            MemoryType::Procedure(_)
        ));
    }

    #[test]
    fn map_situation_edge_to_situation_link() {
        let output = ConsolidationOutput {
            nodes: vec![
                ConsolidatedNode {
                    node_id: "s1".to_string(),
                    memory_type: ConsolidatedMemoryType::Situation,
                    payload: serde_json::json!({
                        "kind": "abstract_location",
                        "name": "library",
                        "coordinates": "campus"
                    }),
                },
                ConsolidatedNode {
                    node_id: "s2".to_string(),
                    memory_type: ConsolidatedMemoryType::Situation,
                    payload: serde_json::json!({
                        "kind": "specific_situation",
                        "narrative": "alice studied at the campus library",
                        "time_span": "2026-08-16T10:00:00Z",
                        "context": {
                            "location": {
                                "name": "library",
                                "coordinates": "campus"
                            },
                            "participants": [{
                                "name": "alice",
                                "role": "student"
                            }],
                            "emotions": [],
                            "sensory_data": [],
                            "environment": {
                                "atmosphere": "quiet",
                                "tone": "focused"
                            },
                            "event": [{
                                "action": "study",
                                "action_intensity": 0.6,
                                "initiator": "alice",
                                "target": "Rust"
                            }]
                        }
                    }),
                },
            ],
            edges: vec![ConsolidatedEdge {
                from: "s1".to_string(),
                to: "s2".to_string(),
                relation: "co_occurs".to_string(),
                intensity: 0.5,
                confidence: 0.7,
            }],
        };

        let mapped = map_output_to_notes(output).expect("mapping should succeed");
        assert_eq!(mapped.notes.len(), 2);
        assert!(matches!(
            mapped.notes[1].mem_type(),
            MemoryType::Situation(SituationType::SpecificSituation(_))
        ));

        let source_note = mapped
            .notes
            .iter()
            .find(|note| note.links().len() == 1)
            .expect("source note should exist");
        assert_eq!(source_note.links().len(), 1);
        assert!(matches!(
            source_note.links()[0].link_type(),
            MemoryLinkType::Sit(SituationMemLink::AbstractToSpecific(_))
        ));
    }
}
