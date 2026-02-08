use chrono::{DateTime, Utc};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrioritizedMemoryRetrieveQuery {
    priority: u32,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryRetrieveQuery {
    Semantic(Vec<SemanticQueryUnit>),
    Situation(Vec<SituationQueryUnit>),
}
impl MemoryRetrieveQuery {
    pub fn make_semantic(units: Vec<SemanticQueryUnit>) -> Self {
        MemoryRetrieveQuery::Semantic(units)
    }
    pub fn make_situation(units: Vec<SituationQueryUnit>) -> Self {
        MemoryRetrieveQuery::Situation(units)
    }
    pub fn as_semantic(&self) -> Option<&Vec<SemanticQueryUnit>> {
        match self {
            MemoryRetrieveQuery::Semantic(units) => Some(units),
            _ => None,
        }
    }
    pub fn as_situation(&self) -> Option<&Vec<SituationQueryUnit>> {
        match self {
            MemoryRetrieveQuery::Situation(units) => Some(units),
            _ => None,
        }
    }
    pub fn with_priority(self, priority: u32) -> PrioritizedMemoryRetrieveQuery {
        PrioritizedMemoryRetrieveQuery::new(priority, self)
    }
}

//语义查询单元，一个单元代表一个概念或实体
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticQueryUnit {
    concept_identifier: Option<String>,
    description: Option<String>,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SituationQueryUnit {
    location: Option<Vec<LocationQueryUnit>>,
    participants: Option<Vec<ParticipantQueryUnit>>,
    time_span: Option<Vec<TimeSpanQueryUnit>>,
    environment: Option<EnvironmentQueryUnit>,
    event: Option<Vec<EventQueryUnit>>,
}
impl SituationQueryUnit {
    pub fn new() -> Self {
        SituationQueryUnit {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocationQueryUnit {
    name: String,
    coordinates: Option<String>,
}
impl LocationQueryUnit {
    pub fn new(name: String) -> Self {
        LocationQueryUnit {
            name,
            coordinates: None,
        }
    }
    pub fn with_coordinates(mut self, coordinates: String) -> Self {
        self.coordinates = Some(coordinates);
        self
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn coordinates(&self) -> Option<&str> {
        self.coordinates.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParticipantQueryUnit {
    name: Option<String>,
    role: Option<String>,
}
impl ParticipantQueryUnit {
    pub fn new() -> Self {
        ParticipantQueryUnit {
            name: None,
            role: None,
        }
    }
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    pub fn with_role(mut self, role: String) -> Self {
        self.role = Some(role);
        self
    }
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn role(&self) -> Option<&str> {
        self.role.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnvironmentQueryUnit {
    atmosphere: Option<String>,
    tone: Option<String>,
}
impl EnvironmentQueryUnit {
    pub fn new() -> Self {
        EnvironmentQueryUnit {
            atmosphere: None,
            tone: None,
        }
    }
    pub fn with_atmosphere(mut self, atmosphere: String) -> Self {
        self.atmosphere = Some(atmosphere);
        self
    }
    pub fn with_tone(mut self, tone: String) -> Self {
        self.tone = Some(tone);
        self
    }
    pub fn atmosphere(&self) -> Option<&str> {
        self.atmosphere.as_deref()
    }
    pub fn tone(&self) -> Option<&str> {
        self.tone.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventQueryUnit {
    action: String,
    initiator: Option<String>,
    target: Option<String>,
}
impl EventQueryUnit {
    pub fn new(action: String) -> Self {
        EventQueryUnit {
            action,
            initiator: None,
            target: None,
        }
    }
    pub fn with_initiator(mut self, initiator: String) -> Self {
        self.initiator = Some(initiator);
        self
    }
    pub fn with_target(mut self, target: String) -> Self {
        self.target = Some(target);
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimeSpanQueryUnit {
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
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
