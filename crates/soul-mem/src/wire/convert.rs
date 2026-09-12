//! protobuf 消息 ⇄ 内部领域模型的转换与校验。
//!
//! 对外协议（`pb`）与内部类型（core/query/runtime）字段一一对应；
//! 所有解析/校验错误统一为 `Error::InvalidArgument`，便于远端得到稳定错误码。

use crate::error::{Error, Result};
use crate::wire::pb;
use chrono::{DateTime, Utc};
use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_links::situation_mem::{
    AbstractToSpecific, SituationMemLink, SpecificToAbstract,
};
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory, SkillRecord};
use soul_mem_core::memory_note::sem_mem::{ConceptType as CoreConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::{
    AbstractSituation, Context, Emotion, Environment, Event, Location, Participant, SensoryData,
    SituationType, SpecificSituation,
};
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_query::query::retrieve::{
    EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
    MemoryRetrieveQueryVariant, ParticipantQueryUnit, SemanticQueryUnit, SituationQueryUnit,
    TimeSpanQueryUnit,
};
use soul_mem_runtime::working_memory::record::UserFeedback;

/// 单条信息内容最大长度。
pub const MAX_DELTA_CHARS: usize = 32_000;

// ---------------- 时间 ----------------

pub fn time_to_string(t: DateTime<Utc>) -> String {
    t.to_rfc3339()
}

pub fn time_from_string(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| Error::InvalidArgument(format!("invalid RFC3339 time {s:?}: {e}")))
}

fn parse_time_or_now(s: &str) -> Result<DateTime<Utc>> {
    if s.is_empty() {
        Ok(Utc::now())
    } else {
        time_from_string(s)
    }
}

pub fn parse_memory_id(s: &str) -> Result<MemoryId> {
    let uuid = uuid::Uuid::parse_str(s)
        .map_err(|_| Error::InvalidArgument(format!("invalid MemoryNote id: {s:?}")))?;
    Ok(MemoryId::from(uuid))
}

// ---------------- 简单枚举 ----------------

/// 信息角色：proto → 滑动窗口使用的角色字符串。
pub fn role_to_str(role: i32) -> Result<&'static str> {
    match pb::MessageRole::try_from(role).unwrap_or(pb::MessageRole::RoleUnspecified) {
        pb::MessageRole::RoleUser => Ok("user"),
        pb::MessageRole::RoleAssistant => Ok("assistant"),
        pb::MessageRole::RoleUnspecified => {
            Err(Error::InvalidArgument("delta role is unspecified".into()))
        }
    }
}

pub fn feedback_from_proto(kind: i32) -> Result<UserFeedback> {
    match pb::FeedbackKind::try_from(kind).unwrap_or(pb::FeedbackKind::FeedbackUnspecified) {
        pb::FeedbackKind::FeedbackPositive => Ok(UserFeedback::Positive),
        pb::FeedbackKind::FeedbackNegative => Ok(UserFeedback::Negative),
        pb::FeedbackKind::FeedbackNeutral => Ok(UserFeedback::Neutral),
        pb::FeedbackKind::FeedbackUnspecified => Err(Error::InvalidArgument(
            "feedback kind is unspecified".into(),
        )),
    }
}

// ---------------- 检索 query ----------------

/// protobuf 检索 query → 内部 `MemoryRetrieveQuery`。
pub fn query_from_proto(query: &pb::MemoryRetrieveQuery) -> Result<MemoryRetrieveQuery> {
    let variant = match query.variant.as_ref() {
        Some(pb::memory_retrieve_query::Variant::Semantic(list)) => {
            let units = list
                .units
                .iter()
                .map(|u| {
                    let mut unit = SemanticQueryUnit::new();
                    if let Some(v) = &u.concept_identifier {
                        unit = unit.with_concept_identifier(v.clone());
                    }
                    if let Some(v) = &u.description {
                        unit = unit.with_description(v.clone());
                    }
                    unit
                })
                .collect();
            MemoryRetrieveQueryVariant::make_semantic(units)
        }
        Some(pb::memory_retrieve_query::Variant::Situation(list)) => {
            let units = list.units.iter().map(situation_unit_from_proto).collect();
            MemoryRetrieveQueryVariant::make_situation(units)
        }
        None => {
            return Err(Error::InvalidArgument(
                "MemoryRetrieveQuery.variant is required".into(),
            ));
        }
    };
    Ok(MemoryRetrieveQuery::new(query.tag.clone(), variant))
}

fn situation_unit_from_proto(u: &pb::SituationQueryUnit) -> SituationQueryUnit {
    let mut unit = SituationQueryUnit::new();
    if let Some(v) = &u.narrative {
        unit = unit.with_narrative(v.clone());
    }
    if !u.location.is_empty() {
        unit = unit.with_location(
            u.location
                .iter()
                .map(|l| {
                    let mut loc = LocationQueryUnit::new(l.name.clone());
                    if let Some(c) = &l.coordinates {
                        loc = loc.with_coordinates(c.clone());
                    }
                    loc
                })
                .collect(),
        );
    }
    if !u.participants.is_empty() {
        unit = unit.with_participants(
            u.participants
                .iter()
                .map(|p| {
                    let mut part = ParticipantQueryUnit::new();
                    if let Some(n) = &p.name {
                        part = part.with_name(n.clone());
                    }
                    if let Some(r) = &p.role {
                        part = part.with_role(r.clone());
                    }
                    part
                })
                .collect(),
        );
    }
    if !u.time_span.is_empty() {
        let spans: Result<Vec<TimeSpanQueryUnit>> = u
            .time_span
            .iter()
            .map(|t| {
                let mut span = TimeSpanQueryUnit::new();
                if let Some(s) = &t.start {
                    span = span.with_start(time_from_string(s)?);
                }
                if let Some(e) = &t.end {
                    span = span.with_end(time_from_string(e)?);
                }
                Ok(span)
            })
            .collect();
        if let Ok(spans) = spans {
            unit = unit.with_time_span(spans);
        }
    }
    if let Some(env) = &u.environment {
        let mut e = EnvironmentQueryUnit::new();
        if let Some(a) = &env.atmosphere {
            e = e.with_atmosphere(a.clone());
        }
        if let Some(t) = &env.tone {
            e = e.with_tone(t.clone());
        }
        unit = unit.with_environment(e);
    }
    if !u.event.is_empty() {
        unit = unit.with_event(
            u.event
                .iter()
                .map(|ev| {
                    let mut e = EventQueryUnit::new(ev.action.clone());
                    if let Some(i) = &ev.initiator {
                        e = e.with_initiator(i.clone());
                    }
                    if let Some(t) = &ev.target {
                        e = e.with_target(t.clone());
                    }
                    e
                })
                .collect(),
        );
    }
    unit
}

// ---------------- MemoryNote ----------------

/// 内部 `MemoryNote` → protobuf。
pub fn note_to_proto(note: &MemoryNote) -> Result<pb::MemoryNote> {
    Ok(pb::MemoryNote {
        id: note.id().to_string(),
        tags: note.tags().to_vec(),
        retrieval_count: note.retrieval_count() as u64,
        create_time: time_to_string(note.creation_time()),
        last_accessed_time: time_to_string(note.last_accessed_time()),
        mem_type: Some(mem_type_to_proto(note.mem_type())?),
        mem_links: note
            .links()
            .iter()
            .map(link_to_proto)
            .collect::<Result<_>>()?,
        missing_degree: note.missing_degree(),
        last_forget_time: time_to_string(note.last_forget_time()),
    })
}

/// protobuf → 内部 `MemoryNote`。
pub fn note_from_proto(note: pb::MemoryNote) -> Result<MemoryNote> {
    let mem_type = note
        .mem_type
        .ok_or_else(|| Error::InvalidArgument("MemoryNote.mem_type is required".into()))
        .and_then(mem_type_from_proto)?;
    let links = note
        .mem_links
        .into_iter()
        .map(link_from_proto)
        .collect::<Result<Vec<_>>>()?;

    let create_time = parse_time_or_now(&note.create_time)?;
    let last_accessed_time = parse_time_or_now(&note.last_accessed_time)?;
    let last_forget_time = parse_time_or_now(&note.last_forget_time)?;

    let mut builder = MemoryNoteBuilder::new(mem_type)
        .tags(note.tags)
        .retrieval_count(note.retrieval_count as usize)
        .create_time(create_time)
        .last_accessed_time(last_accessed_time)
        .mem_links(links)
        .missing_degree(note.missing_degree)
        .last_forget_time(last_forget_time);
    if !note.id.is_empty() {
        builder = builder.id(parse_memory_id(&note.id)?);
    }
    builder
        .build()
        .map_err(|e| Error::InvalidArgument(format!("invalid MemoryNote: {e}")))
}

fn mem_type_to_proto(mem_type: &MemoryType) -> Result<pb::memory_note::MemType> {
    Ok(match mem_type {
        MemoryType::Semantic(sem) => pb::memory_note::MemType::Semantic(pb::SemMemory {
            content: sem.content.clone(),
            aliases: sem.aliases.clone(),
            concept_type: concept_to_proto(&sem.concept_type) as i32,
            description: sem.description.clone(),
        }),
        MemoryType::Situation(situation) => {
            pb::memory_note::MemType::Situation(situation_to_proto(situation))
        }
        MemoryType::Procedure(proc_mem) => {
            pb::memory_note::MemType::Procedure(proc_to_proto(proc_mem))
        }
    })
}

fn mem_type_from_proto(value: pb::memory_note::MemType) -> Result<MemoryType> {
    Ok(match value {
        pb::memory_note::MemType::Semantic(sem) => MemoryType::Semantic(SemMemory {
            content: sem.content,
            aliases: sem.aliases,
            concept_type: concept_from_proto(sem.concept_type),
            description: sem.description,
        }),
        pb::memory_note::MemType::Situation(situation) => {
            MemoryType::Situation(situation_from_proto(situation)?)
        }
        pb::memory_note::MemType::Procedure(proc_mem) => {
            MemoryType::Procedure(proc_from_proto(proc_mem))
        }
    })
}

fn concept_to_proto(value: &CoreConceptType) -> pb::ConceptType {
    match value {
        CoreConceptType::Entity => pb::ConceptType::ConceptEntity,
        CoreConceptType::Abstract => pb::ConceptType::ConceptAbstract,
    }
}

fn concept_from_proto(value: i32) -> CoreConceptType {
    match pb::ConceptType::try_from(value).unwrap_or(pb::ConceptType::ConceptUnspecified) {
        pb::ConceptType::ConceptAbstract => CoreConceptType::Abstract,
        _ => CoreConceptType::Entity,
    }
}

fn situation_to_proto(situation: &SituationType) -> pb::SituationNote {
    let kind = match situation {
        SituationType::AbstractSituation(abstract_situation) => {
            pb::situation_note::Kind::Abstract(abstract_situation_to_proto(abstract_situation))
        }
        SituationType::SpecificSituation(specific) => {
            pb::situation_note::Kind::Specific(specific_to_proto(specific))
        }
    };
    pb::SituationNote { kind: Some(kind) }
}

fn abstract_situation_to_proto(value: &AbstractSituation) -> pb::AbstractSituation {
    let kind = match value {
        AbstractSituation::Location(l) => {
            pb::abstract_situation::Kind::Location(location_to_proto(l))
        }
        AbstractSituation::Participant(p) => {
            pb::abstract_situation::Kind::Participant(participant_to_proto(p))
        }
        AbstractSituation::Environment(e) => {
            pb::abstract_situation::Kind::Environment(environment_to_proto(e))
        }
        AbstractSituation::Event(ev) => pb::abstract_situation::Kind::Event(event_to_proto(ev)),
    };
    pb::AbstractSituation { kind: Some(kind) }
}

fn specific_to_proto(value: &SpecificSituation) -> pb::SpecificSituation {
    pb::SpecificSituation {
        narrative: value.get_narrative().clone(),
        time_span: time_to_string(*value.get_time_span()),
        context: Some(context_to_proto(value.get_context())),
    }
}

fn context_to_proto(value: &Context) -> pb::Context {
    pb::Context {
        location: value.get_location().as_ref().map(location_to_proto),
        participants: value
            .get_participants()
            .iter()
            .map(participant_to_proto)
            .collect(),
        emotions: value
            .get_emotions()
            .iter()
            .map(|e| pb::Emotion {
                name: e.name.clone(),
                intensity: e.intensity,
            })
            .collect(),
        sensory_data: value
            .get_sensory_data()
            .iter()
            .map(|s| pb::SensoryData {
                name: s.name.clone(),
                intensity: s.intensity,
            })
            .collect(),
        environment: Some(environment_to_proto(value.get_environment())),
        event: value.get_event().iter().map(event_to_proto).collect(),
    }
}

fn situation_from_proto(value: pb::SituationNote) -> Result<SituationType> {
    match value.kind {
        Some(pb::situation_note::Kind::Abstract(abstract_situation)) => Ok(
            SituationType::AbstractSituation(abstract_situation_from_proto(abstract_situation)),
        ),
        Some(pb::situation_note::Kind::Specific(specific)) => Ok(SituationType::SpecificSituation(
            specific_from_proto(specific)?,
        )),
        None => Err(Error::InvalidArgument(
            "SituationNote.kind is required".into(),
        )),
    }
}

fn abstract_situation_from_proto(value: pb::AbstractSituation) -> AbstractSituation {
    match value.kind {
        Some(pb::abstract_situation::Kind::Location(l)) => {
            AbstractSituation::Location(location_from_proto(l))
        }
        Some(pb::abstract_situation::Kind::Participant(p)) => {
            AbstractSituation::Participant(participant_from_proto(p))
        }
        Some(pb::abstract_situation::Kind::Environment(e)) => {
            AbstractSituation::Environment(environment_from_proto(e))
        }
        Some(pb::abstract_situation::Kind::Event(ev)) => {
            AbstractSituation::Event(event_from_proto(ev))
        }
        None => AbstractSituation::Environment(Environment::default()),
    }
}

fn specific_from_proto(value: pb::SpecificSituation) -> Result<SpecificSituation> {
    let context = value.context.map(context_from_proto).unwrap_or_default();
    Ok(SpecificSituation::new(
        value.narrative,
        time_from_string(&value.time_span)?,
        context,
    ))
}

fn context_from_proto(value: pb::Context) -> Context {
    Context::new(
        value.location.map(location_from_proto),
        value
            .participants
            .into_iter()
            .map(participant_from_proto)
            .collect(),
        value
            .emotions
            .into_iter()
            .map(|e| Emotion {
                name: e.name,
                intensity: e.intensity,
            })
            .collect(),
        value
            .sensory_data
            .into_iter()
            .map(|s| SensoryData {
                name: s.name,
                intensity: s.intensity,
            })
            .collect(),
        value
            .environment
            .map(environment_from_proto)
            .unwrap_or_default(),
        value.event.into_iter().map(event_from_proto).collect(),
    )
}

fn location_to_proto(value: &Location) -> pb::Location {
    pb::Location {
        name: value.name.clone(),
        coordinates: value.coordinates.clone(),
    }
}
fn location_from_proto(value: pb::Location) -> Location {
    Location {
        name: value.name,
        coordinates: value.coordinates,
    }
}

fn participant_to_proto(value: &Participant) -> pb::Participant {
    pb::Participant {
        name: value.name.clone(),
        role: value.role.clone(),
    }
}
fn participant_from_proto(value: pb::Participant) -> Participant {
    Participant {
        name: value.name,
        role: value.role,
    }
}

fn environment_to_proto(value: &Environment) -> pb::Environment {
    pb::Environment {
        atmosphere: value.atmosphere.clone(),
        tone: value.tone.clone(),
    }
}
fn environment_from_proto(value: pb::Environment) -> Environment {
    Environment {
        atmosphere: value.atmosphere,
        tone: value.tone,
    }
}

fn event_to_proto(value: &Event) -> pb::Event {
    pb::Event {
        action: value.action.clone(),
        action_intensity: value.action_intensity,
        initiator: value.initiator.clone(),
        target: value.target.clone(),
    }
}
fn event_from_proto(value: pb::Event) -> Event {
    Event {
        action: value.action,
        action_intensity: value.action_intensity,
        initiator: value.initiator,
        target: value.target,
    }
}

fn proc_to_proto(value: &ProcMemory) -> pb::ProcMemory {
    let action = value.get_action();
    pb::ProcMemory {
        action: Some(pb::Action {
            content: action.get_content().to_string(),
            kind: action_kind_to_proto(action.get_action_type()) as i32,
        }),
    }
}

fn proc_from_proto(value: pb::ProcMemory) -> ProcMemory {
    let action = value.action.unwrap_or(pb::Action {
        content: String::new(),
        kind: pb::ActionKind::ActionUnspecified as i32,
    });
    ProcMemory::new(Action::new(
        action.content,
        action_kind_from_proto(action.kind),
    ))
}

fn action_kind_to_proto(value: &ActionType) -> pb::ActionKind {
    match value {
        ActionType::Speak => pb::ActionKind::ActionSpeak,
        ActionType::Think => pb::ActionKind::ActionThink,
        ActionType::Skill(_) => pb::ActionKind::ActionSkill,
    }
}

fn action_kind_from_proto(value: i32) -> ActionType {
    match pb::ActionKind::try_from(value).unwrap_or(pb::ActionKind::ActionUnspecified) {
        pb::ActionKind::ActionSkill => ActionType::Skill(SkillRecord {}),
        pb::ActionKind::ActionThink => ActionType::Think,
        _ => ActionType::Speak,
    }
}

// ---------------- MemoryLink ----------------

fn link_to_proto(link: &MemoryLink) -> Result<pb::MemoryLink> {
    Ok(pb::MemoryLink {
        from: link.from().to_string(),
        to: link.to().to_string(),
        intensity: link.intensity,
        missing_degree: link.missing_degree(),
        last_forget_time: time_to_string(link.last_forget_time()),
        link_type: Some(link_type_to_proto(link.link_type())),
    })
}

fn link_type_to_proto(value: &MemoryLinkType) -> pb::memory_link::LinkType {
    match value {
        MemoryLinkType::Sem(sem) => pb::memory_link::LinkType::Sem(pb::SemMemLink {
            verb: sem.verb.clone(),
            confidence: sem.confidence,
        }),
        MemoryLinkType::Proc(proc_link) => pb::memory_link::LinkType::Proc(match proc_link {
            ProcMemLink::TrigToAction(t) => pb::ProcMemLink {
                kind: Some(pb::proc_mem_link::Kind::TrigToAction(pb::TrigToAction {
                    prob: t.prob,
                })),
            },
        }),
        MemoryLinkType::Situation(situation) => {
            pb::memory_link::LinkType::Situation(match situation {
                SituationMemLink::AbstractToSpecific(_) => pb::SituationMemLink {
                    kind: Some(pb::situation_mem_link::Kind::AbstractToSpecific(
                        pb::Empty {},
                    )),
                },
                SituationMemLink::SpecificToAbstract(_) => pb::SituationMemLink {
                    kind: Some(pb::situation_mem_link::Kind::SpecificToAbstract(
                        pb::Empty {},
                    )),
                },
            })
        }
    }
}

fn link_from_proto(value: pb::MemoryLink) -> Result<MemoryLink> {
    let link_type = value
        .link_type
        .ok_or_else(|| Error::InvalidArgument("MemoryLink.link_type is required".into()))?;
    let from = parse_memory_id(&value.from)?;
    let to = parse_memory_id(&value.to)?;
    let link_type = link_type_from_proto(link_type);
    let mut link = MemoryLink::from_tuple(from, to, link_type, value.intensity);
    link.set_missing_degree(value.missing_degree);
    if !value.last_forget_time.is_empty() {
        link.set_last_forget_time(time_from_string(&value.last_forget_time)?);
    }
    Ok(link)
}

fn link_type_from_proto(value: pb::memory_link::LinkType) -> MemoryLinkType {
    match value {
        pb::memory_link::LinkType::Sem(sem) => {
            MemoryLinkType::Sem(SemMemLink::new(sem.verb, sem.confidence))
        }
        pb::memory_link::LinkType::Proc(proc_link) => {
            let prob = match proc_link.kind {
                Some(pb::proc_mem_link::Kind::TrigToAction(t)) => t.prob,
                None => 0.0,
            };
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(prob)))
        }
        pb::memory_link::LinkType::Situation(situation) => {
            MemoryLinkType::Situation(match situation.kind {
                Some(pb::situation_mem_link::Kind::SpecificToAbstract(_)) => {
                    SituationMemLink::SpecificToAbstract(SpecificToAbstract::new())
                }
                _ => SituationMemLink::AbstractToSpecific(AbstractToSpecific::new()),
            })
        }
    }
}
