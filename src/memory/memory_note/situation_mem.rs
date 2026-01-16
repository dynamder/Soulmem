use chrono::{DateTime, Utc};

//一种抽象性情景记忆、一种具体性情景记忆
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum SituationType {
    AbstractSituation(AbstractSituation),
    SpecificSituation(SpecificSituation),
}

//抽象性情景记忆（地点、人物、情境、事件）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum AbstractSituation {
    Location(Location),
    Participant(Participant),
    Situation(Situation),
    Event(Event),
}

//具体性情景记忆（叙述、时间、描述）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
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
    pub fn mut_narrative(&mut self) -> &mut String {
        &mut self.narrative
    }
    pub fn get_time_span(&self) -> &DateTime<Utc> {
        &self.time_span
    }
    pub fn mut_time_span(&mut self) -> &mut DateTime<Utc> {
        &mut self.time_span
    }
    pub fn get_context(&self) -> &Context {
        &self.context
    }
    pub fn mut_context(&mut self) -> &mut Context {
        &mut self.context
    }
}

//描述（地点、人物、情感、感官数据、情境）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Context {
    location: Option<Location>,
    participants: Vec<Participant>,
    emotions: Vec<Emotion>,
    sensory_data: Vec<SensoryData>,
    situation: String,
}

impl Context {
    pub fn new(
        location: Option<Location>,
        participants: Vec<Participant>,
        emotions: Vec<Emotion>,
        sensory_data: Vec<SensoryData>,
        situation: String,
    ) -> Self {
        Context {
            location,
            participants,
            emotions,
            sensory_data,
            situation,
        }
    }
    pub fn mut_location(&mut self) -> &mut Option<Location> {
        &mut self.location
    }
    pub fn mut_participants(&mut self) -> &mut Vec<Participant> {
        &mut self.participants
    }
    pub fn mut_emotions(&mut self) -> &mut Vec<Emotion> {
        &mut self.emotions
    }
    pub fn mut_sensory_data(&mut self) -> &mut Vec<SensoryData> {
        &mut self.sensory_data
    }
    pub fn mut_situation(&mut self) -> &mut String {
        &mut self.situation
    }
}

//事件（动作，动作强度）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Event {
    pub action: String,
    pub action_intensity: i32,
}

//情景（氛围，环境色调）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Situation {
    pub atmosphere: String,
    pub tone: String,
}

//智能体情绪（名称，强度）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Emotion {
    pub name: String,
    pub intensity: i32,
}

//记忆时间主动参与者（名称，角色）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Participant {
    pub name: String,
    pub role: String,
}

//地点（名称，坐标）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Location {
    pub name: String,
    pub coordinates: String,
}

//传感数据（名称，强度）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct SensoryData {
    pub name: String,
    pub intensity: i32,
}
