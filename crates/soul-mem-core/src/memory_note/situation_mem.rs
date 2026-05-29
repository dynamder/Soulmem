use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

//一种抽象性情景记忆、一种具体性情景记忆
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub enum SituationType {
    AbstractSituation(AbstractSituation),
    SpecificSituation(SpecificSituation),
}

impl From<AbstractSituation> for SituationType {
    fn from(situation: AbstractSituation) -> Self {
        SituationType::AbstractSituation(situation)
    }
}
impl From<SpecificSituation> for SituationType {
    fn from(situation: SpecificSituation) -> Self {
        SituationType::SpecificSituation(situation)
    }
}

//抽象性情景记忆（地点、人物、情境、事件）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub enum AbstractSituation {
    Location(Location),
    Participant(Participant),
    Environment(Environment),
    Event(Event),
}

impl From<Location> for AbstractSituation {
    fn from(location: Location) -> Self {
        AbstractSituation::Location(location)
    }
}
impl From<Participant> for AbstractSituation {
    fn from(participant: Participant) -> Self {
        AbstractSituation::Participant(participant)
    }
}
impl From<Environment> for AbstractSituation {
    fn from(environment: Environment) -> Self {
        AbstractSituation::Environment(environment)
    }
}
impl From<Event> for AbstractSituation {
    fn from(event: Event) -> Self {
        AbstractSituation::Event(event)
    }
}

//具体性情景记忆（叙述、时间、描述）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct SpecificSituation {
    narrative: String,
    time_span: DateTime<Utc>,
    context: Context,
}

impl SpecificSituation {
    pub fn new(narrative: String, time_span: DateTime<Utc>, context: Context) -> Self {
        SpecificSituation {
            narrative,
            time_span,
            context,
        }
    }
    pub fn get_narrative(&self) -> &String {
        &self.narrative
    }
    pub fn get_mut_narrative(&mut self) -> &mut String {
        &mut self.narrative
    }
    pub fn get_time_span(&self) -> &DateTime<Utc> {
        &self.time_span
    }
    pub fn get_mut_time_span(&mut self) -> &mut DateTime<Utc> {
        &mut self.time_span
    }
    pub fn get_context(&self) -> &Context {
        &self.context
    }
    pub fn get_mut_context(&mut self) -> &mut Context {
        &mut self.context
    }
}

//描述（地点、人物、情感、感官数据、环境、事件）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Context {
    location: Option<Location>,
    participants: Vec<Participant>,
    emotions: Vec<Emotion>,
    sensory_data: Vec<SensoryData>,
    environment: Environment,
    event: Vec<Event>,
}

impl Context {
    pub fn new(
        location: Option<Location>,
        participants: Vec<Participant>,
        emotions: Vec<Emotion>,
        sensory_data: Vec<SensoryData>,
        environment: Environment,
        event: Vec<Event>,
    ) -> Self {
        Context {
            location,
            participants,
            emotions,
            sensory_data,
            environment,
            event,
        }
    }
    pub fn get_mut_location(&mut self) -> &mut Option<Location> {
        &mut self.location
    }
    pub fn get_location(&self) -> &Option<Location> {
        &self.location
    }
    pub fn get_mut_participants(&mut self) -> &mut Vec<Participant> {
        &mut self.participants
    }
    pub fn get_participants(&self) -> &Vec<Participant> {
        &self.participants
    }
    pub fn get_mut_emotions(&mut self) -> &mut Vec<Emotion> {
        &mut self.emotions
    }
    pub fn get_emotions(&self) -> &Vec<Emotion> {
        &self.emotions
    }
    pub fn get_mut_sensory_data(&mut self) -> &mut Vec<SensoryData> {
        &mut self.sensory_data
    }
    pub fn get_sensory_data(&self) -> &Vec<SensoryData> {
        &self.sensory_data
    }
    pub fn get_mut_environment(&mut self) -> &mut Environment {
        &mut self.environment
    }
    pub fn get_environment(&self) -> &Environment {
        &self.environment
    }
    pub fn get_mut_event(&mut self) -> &mut Vec<Event> {
        &mut self.event
    }
    pub fn get_event(&self) -> &Vec<Event> {
        &self.event
    }
}

//事件（动作，动作强度，单个发起者，单个目标）（抽象）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Event {
    pub action: String,
    pub action_intensity: f32,
    pub initiator: String,
    pub target: String,
}

//环境（氛围，环境色调）（抽象、描述）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub atmosphere: String,
    pub tone: String,
}

//智能体情绪（名称，强度）（描述）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Emotion {
    pub name: String,
    pub intensity: f32,
}

//记忆时间主动参与者（名称，角色）(抽象、描述)
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Participant {
    pub name: String,
    pub role: String,
}

//地点（名称，坐标）(抽象、描述)
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct Location {
    pub name: String,
    pub coordinates: String,
}

//传感数据（名称，强度）（描述）
#[derive(Debug, PartialEq, PartialOrd, Clone, Serialize, Deserialize)]
pub struct SensoryData {
    pub name: String,
    pub intensity: f32,
}
