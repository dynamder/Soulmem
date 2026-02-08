pub struct MemoryRetrieveQuery {}

//语义查询单元，一个单元代表一个概念或实体
pub struct SemanticQueryUnit {
    concept_identifier: Option<String>,
    description: Option<String>,
    priority: Option<usize>,
}
impl SemanticQueryUnit {
    pub fn new() -> Self {
        SemanticQueryUnit {
            concept_identifier: None,
            description: None,
            priority: None,
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
    pub fn with_priority(mut self, priority: usize) -> Self {
        self.priority = Some(priority);
        self
    }

    pub fn concept_identifier(&self) -> Option<&str> {
        self.concept_identifier.as_deref()
    }
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    pub fn priority(&self) -> Option<usize> {
        self.priority
    }
}

//情境查询单元，一个单元代表一个情境或事件，一个单元内的信息在查询时是“与”关系
pub struct SituationQueryUnit {}

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

pub struct EventQueryUnit {}
