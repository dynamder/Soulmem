//! proto 消息与领域 DTO 的互转。
//!
//! 核心与 `proto` 之间唯一的耦合点；适配器只调用这里的函数。

use chrono::{DateTime, Utc};
use prost_types::Timestamp;

use crate::error::{ServiceError, ServiceResult};
use crate::proto::v1;
use crate::service::{
    ControlAction, Delta as ServiceDelta, ServiceEvent, ServiceRequest, ServiceResponse,
    ServiceState,
};
use soul_mem_core::render::{MemoryKind, RenderedMemoryNote};
use soul_mem_query::query::retrieve::{
    EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
    MemoryRetrieveQueryVariant, ParticipantQueryUnit, PrioritizedMemoryRetrieveQuery,
    SemanticQueryUnit, SituationQueryUnit, TimeSpanQueryUnit,
};

/// proto 请求 -> 领域组合请求。
///
/// 未知/未指定的控制信号不再静默丢弃：未指定（0）视作无操作，无法识别的值报错，
/// 由适配器以 `Reply { ok: false }` 明确反馈给调用方。
pub fn service_request(message: v1::Request) -> ServiceResult<ServiceRequest> {
    let mut controls = Vec::new();
    for raw in message.controls {
        if let Some(action) = map_action(raw)? {
            controls.push(action);
        }
    }
    Ok(ServiceRequest {
        queries: message.queries.into_iter().map(query).collect(),
        deltas: message.deltas.into_iter().map(delta).collect(),
        controls,
    })
}

/// proto query -> 带优先级的内部查询。
fn query(message: v1::Query) -> PrioritizedMemoryRetrieveQuery {
    let variant = match message.variant {
        Some(v1::query::Variant::Semantic(semantic)) => MemoryRetrieveQueryVariant::Semantic(
            semantic.units.into_iter().map(semantic_unit).collect(),
        ),
        Some(v1::query::Variant::Situation(situation)) => MemoryRetrieveQueryVariant::Situation(
            situation.units.into_iter().map(situation_unit).collect(),
        ),
        None => MemoryRetrieveQueryVariant::Semantic(Vec::new()),
    };
    MemoryRetrieveQuery::new(message.tag, variant).with_priority(message.priority)
}

fn semantic_unit(message: v1::SemanticUnit) -> SemanticQueryUnit {
    let v1::SemanticUnit {
        concept_identifier,
        description,
    } = message;
    let mut unit = SemanticQueryUnit::new();
    if let Some(concept_identifier) = concept_identifier {
        unit = unit.with_concept_identifier(concept_identifier);
    }
    if let Some(description) = description {
        unit = unit.with_description(description);
    }
    unit
}

fn situation_unit(message: v1::SituationUnit) -> SituationQueryUnit {
    let v1::SituationUnit {
        narrative,
        location,
        participants,
        time_span,
        environment,
        event,
    } = message;
    let mut unit = SituationQueryUnit::new();
    if let Some(narrative) = narrative {
        unit = unit.with_narrative(narrative);
    }
    if !location.is_empty() {
        unit = unit.with_location(location.into_iter().map(location_unit).collect());
    }
    if !participants.is_empty() {
        unit = unit.with_participants(participants.into_iter().map(participant_unit).collect());
    }
    if !time_span.is_empty() {
        unit = unit.with_time_span(time_span.into_iter().map(time_span_unit).collect());
    }
    if let Some(environment) = environment {
        unit = unit.with_environment(environment_unit(environment));
    }
    if !event.is_empty() {
        unit = unit.with_event(event.into_iter().map(event_unit).collect());
    }
    unit
}

fn location_unit(message: v1::LocationUnit) -> LocationQueryUnit {
    let v1::LocationUnit { name, coordinates } = message;
    let mut unit = LocationQueryUnit::new(name);
    if let Some(coordinates) = coordinates {
        unit = unit.with_coordinates(coordinates);
    }
    unit
}

fn participant_unit(message: v1::ParticipantUnit) -> ParticipantQueryUnit {
    let v1::ParticipantUnit { name, role } = message;
    let mut unit = ParticipantQueryUnit::new();
    if let Some(name) = name {
        unit = unit.with_name(name);
    }
    if let Some(role) = role {
        unit = unit.with_role(role);
    }
    unit
}

fn environment_unit(message: v1::EnvironmentUnit) -> EnvironmentQueryUnit {
    let v1::EnvironmentUnit { atmosphere, tone } = message;
    let mut unit = EnvironmentQueryUnit::new();
    if let Some(atmosphere) = atmosphere {
        unit = unit.with_atmosphere(atmosphere);
    }
    if let Some(tone) = tone {
        unit = unit.with_tone(tone);
    }
    unit
}

fn event_unit(message: v1::EventUnit) -> EventQueryUnit {
    let v1::EventUnit {
        action,
        initiator,
        target,
    } = message;
    let mut unit = EventQueryUnit::new(action);
    if let Some(initiator) = initiator {
        unit = unit.with_initiator(initiator);
    }
    if let Some(target) = target {
        unit = unit.with_target(target);
    }
    unit
}

fn time_span_unit(message: v1::TimeSpanUnit) -> TimeSpanQueryUnit {
    let v1::TimeSpanUnit { start, end } = message;
    let mut unit = TimeSpanQueryUnit::new();
    if let Some(start) = start.and_then(from_timestamp) {
        unit = unit.with_start(start);
    }
    if let Some(end) = end.and_then(from_timestamp) {
        unit = unit.with_end(end);
    }
    unit
}

/// proto 增量 -> 领域增量。
fn delta(message: v1::Delta) -> ServiceDelta {
    ServiceDelta {
        statement: message.statement,
        role: message.role,
    }
}

/// 单个控制枚举映射。
///
/// 返回 `Ok(None)` 表示"未指定、无操作"；未知值返回 `Err`，由上层转成失败响应。
fn map_action(raw: i32) -> ServiceResult<Option<ControlAction>> {
    match v1::ControlAction::try_from(raw) {
        Ok(v1::ControlAction::Unspecified) => Ok(None),
        Ok(v1::ControlAction::Consolidate) => Ok(Some(ControlAction::Consolidate)),
        Ok(v1::ControlAction::Persist) => Ok(Some(ControlAction::Persist)),
        Ok(v1::ControlAction::Snapshot) => Ok(Some(ControlAction::Snapshot)),
        Ok(v1::ControlAction::Forget) => Ok(Some(ControlAction::Forget)),
        Ok(v1::ControlAction::Pause) => Ok(Some(ControlAction::Pause)),
        Ok(v1::ControlAction::Resume) => Ok(Some(ControlAction::Resume)),
        Ok(v1::ControlAction::Maintenance) => Ok(Some(ControlAction::Maintenance)),
        Err(_) => Err(ServiceError::BadRequest(format!("未知控制信号: {raw}"))),
    }
}

/// 领域状态 -> proto 状态。
pub fn service_state(state: ServiceState) -> v1::ServiceState {
    v1::ServiceState {
        working_state: if state.working {
            v1::WorkingState::Working as i32
        } else {
            v1::WorkingState::Idle as i32
        },
        node_count: state.node_count,
        llm_available: state.llm_available,
        last_consolidation_at: state.last_consolidation_at.map(to_timestamp),
    }
}

/// 组装一次处理响应：`ok` 由领域层的 `error` 决定，失败时仍带状态与已接受条数。
pub fn reply(request_id: String, response: ServiceResponse) -> v1::Reply {
    let ok = response.error.is_none();
    v1::Reply {
        request_id,
        ok,
        error: response.error.unwrap_or_default(),
        output: response.output,
        memories: response.memories.into_iter().map(rendered_note).collect(),
        state: Some(service_state(response.state)),
        accepted: response.accepted,
    }
}

/// 组装失败响应。
pub fn reply_error(request_id: String, message: impl Into<String>) -> v1::Reply {
    v1::Reply {
        request_id,
        ok: false,
        error: message.into(),
        output: None,
        memories: Vec::new(),
        state: None,
        accepted: 0,
    }
}

/// 领域结构化记忆 -> proto 结构化记忆。
fn rendered_note(note: RenderedMemoryNote) -> v1::RenderedMemoryNote {
    v1::RenderedMemoryNote {
        kind: memory_kind(note.kind) as i32,
        score: note.score,
        content: note.content,
    }
}

/// 领域 `MemoryKind` -> proto 枚举值。
fn memory_kind(kind: MemoryKind) -> v1::MemoryKind {
    match kind {
        MemoryKind::Semantic => v1::MemoryKind::Semantic,
        MemoryKind::SpecificSituation => v1::MemoryKind::SpecificSituation,
        MemoryKind::AbstractSituation => v1::MemoryKind::AbstractSituation,
        MemoryKind::Procedure => v1::MemoryKind::Procedure,
    }
}

/// 领域事件 -> proto 事件（时间戳取当前）。
pub fn event(event: ServiceEvent) -> v1::Event {
    let at = Some(to_timestamp(Utc::now()));
    let body = match event {
        ServiceEvent::Heartbeat(state) => v1::event::Body::Heartbeat(service_state(state)),
        ServiceEvent::ConsolidationDone { created_notes, .. } => {
            v1::event::Body::ConsolidationDone(v1::ConsolidationDone { created_notes })
        }
    };
    v1::Event {
        at,
        body: Some(body),
    }
}

/// `DateTime<Utc>` -> `google.protobuf.Timestamp`。
fn to_timestamp(dt: DateTime<Utc>) -> Timestamp {
    Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    }
}

/// `google.protobuf.Timestamp` -> `DateTime<Utc>`。
///
/// `Timestamp.nanos` 是 `i32`：epoch 之前的时间会带负纳秒，直接 `as u32` 会溢出成一个
/// 巨大的非法值而被 `from_timestamp` 判为 `None`。这里先把它归一化到 `[0, 1e9)`。
fn from_timestamp(ts: Timestamp) -> Option<DateTime<Utc>> {
    let (seconds, nanos) = if ts.nanos < 0 {
        (ts.seconds - 1, (ts.nanos as i64 + 1_000_000_000) as u32)
    } else {
        (ts.seconds, ts.nanos as u32)
    };
    DateTime::from_timestamp(seconds, nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unspecified_control_is_noop() {
        let request = v1::Request {
            controls: vec![v1::ControlAction::Unspecified as i32],
            ..v1::Request::default()
        };
        let converted = service_request(request).expect("未指定应视为无操作");
        assert!(converted.controls.is_empty());
    }

    #[test]
    fn unknown_control_is_rejected() {
        let request = v1::Request {
            controls: vec![9999],
            ..v1::Request::default()
        };
        assert!(service_request(request).is_err());
    }

    #[test]
    fn maintenance_control_maps() {
        let request = v1::Request {
            controls: vec![v1::ControlAction::Maintenance as i32],
            ..v1::Request::default()
        };
        let converted = service_request(request).expect("维护信号应可映射");
        assert_eq!(converted.controls, vec![ControlAction::Maintenance]);
    }

    #[test]
    fn negative_nanos_timestamp_is_parsed() {
        // nanos 为负表示 epoch 之前：归一化后应能解析，而不是因 u32 溢出变成 None。
        let ts = Timestamp {
            seconds: 0,
            nanos: -1,
        };
        assert!(from_timestamp(ts).is_some());
    }

    #[test]
    fn reply_carries_state_on_business_error() {
        let response = ServiceResponse {
            output: None,
            memories: Vec::new(),
            state: ServiceState {
                working: false,
                node_count: 0,
                edge_count: 0,
                llm_available: true,
                last_consolidation_at: None,
            },
            accepted: 2,
            error: Some("boom".into()),
        };
        let reply = reply("req-1".into(), response);
        assert!(!reply.ok);
        assert_eq!(reply.error, "boom");
        assert_eq!(reply.accepted, 2);
        assert!(reply.state.is_some());
    }

    #[test]
    fn rendered_note_maps_kind_score_content() {
        let note = RenderedMemoryNote {
            kind: MemoryKind::SpecificSituation,
            score: 0.75,
            content: "在咖啡馆聊天".to_string(),
            context: None,
        };
        let proto = rendered_note(note);
        assert_eq!(proto.kind, v1::MemoryKind::SpecificSituation as i32);
        assert_eq!(proto.score, 0.75);
        assert_eq!(proto.content, "在咖啡馆聊天");
    }
}
