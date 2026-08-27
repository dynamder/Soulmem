use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrioritizedMemoryRetrieveQuery {
    priority: u32, //优先级将决定最终混合一个MemoryNote的检索分数时的权重, similarity和cached_path检索策略会使用
    query: MemoryRetrieveQuery,
}

impl PrioritizedMemoryRetrieveQuery {
    pub fn new(priority: u32, query: MemoryRetrieveQuery) -> Self {
        PrioritizedMemoryRetrieveQuery { priority, query }
    }
    pub fn priority(&self) -> u32 {
        self.priority
    }
    pub fn query(&self) -> &MemoryRetrieveQuery {
        &self.query
    }
    pub fn downgrade(self) -> MemoryRetrieveQuery {
        self.query
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRetrieveQuery {
    tag: Vec<String>,
    variant: MemoryRetrieveQueryVariant,
}
impl MemoryRetrieveQuery {
    pub fn new(tag: Vec<String>, variant: MemoryRetrieveQueryVariant) -> Self {
        Self { tag, variant }
    }
    pub fn tag(&self) -> &[String] {
        &self.tag
    }
    pub fn variant(&self) -> &MemoryRetrieveQueryVariant {
        &self.variant
    }
    pub fn with_priority(self, priority: u32) -> PrioritizedMemoryRetrieveQuery {
        PrioritizedMemoryRetrieveQuery::new(priority, self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryRetrieveQueryVariant {
    Semantic(Vec<SemanticQueryUnit>),
    Situation(Vec<SituationQueryUnit>),
}
impl MemoryRetrieveQueryVariant {
    pub fn make_semantic(units: Vec<SemanticQueryUnit>) -> Self {
        MemoryRetrieveQueryVariant::Semantic(units)
    }
    pub fn make_situation(units: Vec<SituationQueryUnit>) -> Self {
        MemoryRetrieveQueryVariant::Situation(units)
    }
    pub fn as_semantic(&self) -> Option<&Vec<SemanticQueryUnit>> {
        match self {
            MemoryRetrieveQueryVariant::Semantic(units) => Some(units),
            _ => None,
        }
    }
    pub fn as_situation(&self) -> Option<&Vec<SituationQueryUnit>> {
        match self {
            MemoryRetrieveQueryVariant::Situation(units) => Some(units),
            _ => None,
        }
    }
}

//语义查询单元，一个单元代表一个概念或实体
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticQueryUnit {
    concept_identifier: Option<String>,
    description: Option<String>,
}
impl Default for SemanticQueryUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticQueryUnit {
    pub fn new() -> Self {
        SemanticQueryUnit {
            concept_identifier: None,
            description: None,
        }
    }
    pub fn with_concept_identifier(mut self, concept_identifier: String) -> Self {
        self.concept_identifier = Some(concept_identifier);
        self
    }
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn concept_identifier(&self) -> Option<&str> {
        self.concept_identifier.as_deref()
    }
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

//情境查询单元，一个单元代表一个情境或事件，一个单元内的信息在查询时是“与”关系
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SituationQueryUnit {
    narrative: Option<String>,
    location: Option<Vec<LocationQueryUnit>>,
    participants: Option<Vec<ParticipantQueryUnit>>,
    time_span: Option<Vec<TimeSpanQueryUnit>>,
    environment: Option<EnvironmentQueryUnit>,
    event: Option<Vec<EventQueryUnit>>,
}
impl Default for SituationQueryUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl SituationQueryUnit {
    pub fn new() -> Self {
        SituationQueryUnit {
            narrative: None,
            location: None,
            participants: None,
            time_span: None,
            environment: None,
            event: None,
        }
    }
    pub fn with_location(mut self, location: Vec<LocationQueryUnit>) -> Self {
        self.location = Some(location);
        self
    }
    pub fn with_participants(mut self, participants: Vec<ParticipantQueryUnit>) -> Self {
        self.participants = Some(participants);
        self
    }
    pub fn with_time_span(mut self, time_span: Vec<TimeSpanQueryUnit>) -> Self {
        self.time_span = Some(time_span);
        self
    }
    pub fn with_environment(mut self, environment: EnvironmentQueryUnit) -> Self {
        self.environment = Some(environment);
        self
    }
    pub fn with_event(mut self, event: Vec<EventQueryUnit>) -> Self {
        self.event = Some(event);
        self
    }
    pub fn with_narrative(mut self, narrative: String) -> Self {
        self.narrative = Some(narrative);
        self
    }
    pub fn narrative(&self) -> Option<&String> {
        self.narrative.as_ref()
    }
    pub fn location(&self) -> Option<&Vec<LocationQueryUnit>> {
        self.location.as_ref()
    }
    pub fn participants(&self) -> Option<&Vec<ParticipantQueryUnit>> {
        self.participants.as_ref()
    }
    pub fn time_span(&self) -> Option<&Vec<TimeSpanQueryUnit>> {
        self.time_span.as_ref()
    }
    pub fn environment(&self) -> Option<&EnvironmentQueryUnit> {
        self.environment.as_ref()
    }
    pub fn event(&self) -> Option<&Vec<EventQueryUnit>> {
        self.event.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocationQueryUnit {
    name: String,
    coordinates: Option<String>,
}
impl LocationQueryUnit {
    pub fn new(name: impl Into<String>) -> Self {
        LocationQueryUnit {
            name: name.into(),
            coordinates: None,
        }
    }
    pub fn with_coordinates(mut self, coordinates: impl Into<String>) -> Self {
        self.coordinates = Some(coordinates.into());
        self
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn coordinates(&self) -> Option<&str> {
        self.coordinates.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParticipantQueryUnit {
    name: Option<String>,
    role: Option<String>,
}
impl Default for ParticipantQueryUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticipantQueryUnit {
    pub fn new() -> Self {
        ParticipantQueryUnit {
            name: None,
            role: None,
        }
    }
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn role(&self) -> Option<&str> {
        self.role.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnvironmentQueryUnit {
    atmosphere: Option<String>,
    tone: Option<String>,
}
impl Default for EnvironmentQueryUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvironmentQueryUnit {
    pub fn new() -> Self {
        EnvironmentQueryUnit {
            atmosphere: None,
            tone: None,
        }
    }
    pub fn with_atmosphere(mut self, atmosphere: impl Into<String>) -> Self {
        self.atmosphere = Some(atmosphere.into());
        self
    }
    pub fn with_tone(mut self, tone: impl Into<String>) -> Self {
        self.tone = Some(tone.into());
        self
    }
    pub fn atmosphere(&self) -> Option<&str> {
        self.atmosphere.as_deref()
    }
    pub fn tone(&self) -> Option<&str> {
        self.tone.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventQueryUnit {
    action: String,
    initiator: Option<String>,
    target: Option<String>,
}
impl EventQueryUnit {
    pub fn new(action: impl Into<String>) -> Self {
        EventQueryUnit {
            action: action.into(),
            initiator: None,
            target: None,
        }
    }
    pub fn with_initiator(mut self, initiator: impl Into<String>) -> Self {
        self.initiator = Some(initiator.into());
        self
    }
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }
    pub fn action(&self) -> &str {
        &self.action
    }
    pub fn initiator(&self) -> Option<&str> {
        self.initiator.as_deref()
    }
    pub fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }
}

//TODO: time_span查询在MVP中暂未实现评分逻辑，字段保留以供后续使用
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeSpanQueryUnit {
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
}
impl Default for TimeSpanQueryUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSpanQueryUnit {
    pub fn new() -> Self {
        TimeSpanQueryUnit {
            start: None,
            end: None,
        }
    }
    pub fn with_start(mut self, start: DateTime<Utc>) -> Self {
        self.start = Some(start);
        self
    }
    pub fn with_end(mut self, end: DateTime<Utc>) -> Self {
        self.end = Some(end);
        self
    }
    pub fn start(&self) -> Option<&DateTime<Utc>> {
        self.start.as_ref()
    }
    pub fn end(&self) -> Option<&DateTime<Utc>> {
        self.end.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prioritized_query_roundtrip() {
        let variant = MemoryRetrieveQueryVariant::make_semantic(vec![SemanticQueryUnit::new()]);
        let query = MemoryRetrieveQuery::new(vec!["tag".to_string()], variant);
        let prioritized = query.clone().with_priority(7);
        assert_eq!(prioritized.priority(), 7);
        assert_eq!(prioritized.query(), &query);
        assert_eq!(prioritized.downgrade(), query);
    }

    #[test]
    fn test_retrieve_query_tag() {
        let query = MemoryRetrieveQuery::new(
            vec!["a".to_string(), "b".to_string()],
            MemoryRetrieveQueryVariant::make_semantic(vec![]),
        );
        assert_eq!(query.tag(), &["a".to_string(), "b".to_string()]);
        let empty = MemoryRetrieveQuery::new(vec![], MemoryRetrieveQueryVariant::make_semantic(vec![]));
        assert!(empty.tag().is_empty());
    }

    #[test]
    fn test_variant_as_semantic() {
        let variant = MemoryRetrieveQueryVariant::make_semantic(vec![
            SemanticQueryUnit::new().with_concept_identifier("c".to_string()),
        ]);
        let units = variant.as_semantic().expect("semantic units");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].concept_identifier(), Some("c"));

        let situation = MemoryRetrieveQueryVariant::make_situation(vec![SituationQueryUnit::new()]);
        assert!(situation.as_semantic().is_none());
    }

    #[test]
    fn test_variant_as_situation() {
        let variant = MemoryRetrieveQueryVariant::make_situation(vec![
            SituationQueryUnit::new().with_narrative("n".to_string()),
        ]);
        let units = variant.as_situation().expect("situation units");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].narrative().map(|s| s.as_str()), Some("n"));

        let semantic = MemoryRetrieveQueryVariant::make_semantic(vec![]);
        assert!(semantic.as_situation().is_none());
    }

    #[test]
    fn test_semantic_query_unit_getters() {
        let unit = SemanticQueryUnit::new();
        assert_eq!(unit.concept_identifier(), None);
        assert_eq!(unit.description(), None);

        let unit = unit
            .with_concept_identifier("concept".to_string())
            .with_description("desc".to_string());
        assert_eq!(unit.concept_identifier(), Some("concept"));
        assert_eq!(unit.description(), Some("desc"));
    }

    #[test]
    fn test_situation_query_unit_getters() {
        let unit = SituationQueryUnit::new();
        assert!(unit.narrative().is_none());
        assert!(unit.location().is_none());
        assert!(unit.participants().is_none());
        assert!(unit.time_span().is_none());
        assert!(unit.environment().is_none());
        assert!(unit.event().is_none());

        let unit = unit
            .with_narrative("nar".to_string())
            .with_location(vec![LocationQueryUnit::new("loc")])
            .with_participants(vec![ParticipantQueryUnit::new().with_name("p".to_string())])
            .with_time_span(vec![TimeSpanQueryUnit::new()])
            .with_environment(EnvironmentQueryUnit::new().with_atmosphere("a".to_string()))
            .with_event(vec![EventQueryUnit::new("e")]);
        assert_eq!(unit.narrative().map(|s| s.as_str()), Some("nar"));
        assert_eq!(unit.location().map(|v| v.len()), Some(1));
        assert_eq!(unit.participants().map(|v| v.len()), Some(1));
        assert_eq!(unit.time_span().map(|v| v.len()), Some(1));
        assert!(unit.environment().is_some());
        assert_eq!(unit.event().map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_location_query_unit() {
        let unit = LocationQueryUnit::new("北京");
        assert_eq!(unit.name(), "北京");
        assert_eq!(unit.coordinates(), None);
        let unit = unit.with_coordinates("中国");
        assert_eq!(unit.coordinates(), Some("中国"));
    }

    #[test]
    fn test_participant_query_unit() {
        let unit = ParticipantQueryUnit::new();
        assert_eq!(unit.name(), None);
        assert_eq!(unit.role(), None);
        let unit = unit.with_name("张三".to_string()).with_role("学生".to_string());
        assert_eq!(unit.name(), Some("张三"));
        assert_eq!(unit.role(), Some("学生"));
    }

    #[test]
    fn test_environment_query_unit() {
        let unit = EnvironmentQueryUnit::new();
        assert_eq!(unit.atmosphere(), None);
        assert_eq!(unit.tone(), None);
        let unit = unit.with_atmosphere("安静".to_string()).with_tone("温暖".to_string());
        assert_eq!(unit.atmosphere(), Some("安静"));
        assert_eq!(unit.tone(), Some("温暖"));
    }

    #[test]
    fn test_event_query_unit() {
        let unit = EventQueryUnit::new("跑步");
        assert_eq!(unit.action(), "跑步");
        assert_eq!(unit.initiator(), None);
        assert_eq!(unit.target(), None);
        let unit = unit.with_initiator("张三".to_string()).with_target("操场".to_string());
        assert_eq!(unit.initiator(), Some("张三"));
        assert_eq!(unit.target(), Some("操场"));
    }

    #[test]
    fn test_time_span_query_unit() {
        let unit = TimeSpanQueryUnit::new();
        assert!(unit.start().is_none());
        assert!(unit.end().is_none());
        let start = Utc::now();
        let end = start + chrono::Duration::hours(1);
        let unit = unit.with_start(start).with_end(end);
        assert_eq!(unit.start(), Some(&start));
        assert_eq!(unit.end(), Some(&end));
    }
}
