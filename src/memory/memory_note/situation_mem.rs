use chrono::{DateTime, Utc};

//一种抽象性情景记忆、一种具体性情景记忆
#[derive(Debug, PartialEq, PartialOrd, Clone)]
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
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub enum AbstractSituation {
    Location(Location),
    Participant(Participant),
    Situation(Environment),
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
    fn from(situation: Environment) -> Self {
        AbstractSituation::Situation(situation)
    }
}
impl From<Event> for AbstractSituation {
    fn from(event: Event) -> Self {
        AbstractSituation::Event(event)
    }
}

//具体性情景记忆（叙述、时间、描述）
#[derive(Debug, PartialEq, PartialOrd, Clone)]
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

//描述（地点、人物、情感、感官数据、情境）
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Context {
    location: Option<Location>,
    participants: Vec<Participant>,
    emotions: Vec<Emotion>,
    sensory_data: Vec<SensoryData>,
    situation: Environment,
}

impl Context {
    pub fn new(
        location: Option<Location>,
        participants: Vec<Participant>,
        emotions: Vec<Emotion>,
        sensory_data: Vec<SensoryData>,
        situation: Environment,
    ) -> Self {
        Context {
            location,
            participants,
            emotions,
            sensory_data,
            situation,
        }
    }
    pub fn get_mut_location(&mut self) -> &mut Option<Location> {
        &mut self.location
    }
    pub fn get_mut_participants(&mut self) -> &mut Vec<Participant> {
        &mut self.participants
    }
    pub fn get_mut_emotions(&mut self) -> &mut Vec<Emotion> {
        &mut self.emotions
    }
    pub fn get_mut_sensory_data(&mut self) -> &mut Vec<SensoryData> {
        &mut self.sensory_data
    }
    pub fn get_mut_situation(&mut self) -> &mut Environment {
        &mut self.situation
    }
}

//事件（动作，动作强度）（抽象）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Event {
    pub action: String,
    pub action_intensity: i32,
}
impl TryFrom<AbstractSituation> for Event {
    type Error = String;

    fn try_from(value: AbstractSituation) -> Result<Self, Self::Error> {
        match value {
            AbstractSituation::Event(Event {
                action,
                action_intensity,
            }) => Ok(Event {
                action,
                action_intensity,
            }),
            _ => Err("Cannot convert AbstractSituation to Event".to_string()),
        }
    }
}

//情景（氛围，环境色调）（抽象、描述）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Environment {
    pub atmosphere: String,
    pub tone: String,
}
impl TryFrom<AbstractSituation> for Environment {
    type Error = String;

    fn try_from(value: AbstractSituation) -> Result<Self, Self::Error> {
        match value {
            AbstractSituation::Situation(Environment { atmosphere, tone }) => {
                Ok(Environment { atmosphere, tone })
            }
            _ => Err("Cannot convert AbstractSituation to Situation".to_string()),
        }
    }
}
impl From<Context> for Environment {
    fn from(context: Context) -> Self {
        Environment {
            atmosphere: context.situation.atmosphere,
            tone: context.situation.tone,
        }
    }
}

//智能体情绪（名称，强度）（描述）
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Emotion {
    pub name: String,
    pub intensity: f32,
}
impl From<Context> for Vec<Emotion> {
    fn from(context: Context) -> Vec<Emotion> {
        context
            .emotions
            .into_iter()
            .map(|emotion| Emotion {
                name: emotion.name,
                intensity: emotion.intensity,
            })
            .collect()
    }
}

//记忆时间主动参与者（名称，角色）(抽象、描述)
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Participant {
    pub name: String,
    pub role: String,
}
impl TryFrom<AbstractSituation> for Participant {
    type Error = String;

    fn try_from(value: AbstractSituation) -> Result<Self, Self::Error> {
        match value {
            AbstractSituation::Participant(Participant { name, role }) => {
                Ok(Participant { name, role })
            }
            _ => Err("Cannot convert AbstractParticipant to Participant".to_string()),
        }
    }
}
impl From<Context> for Vec<Participant> {
    fn from(context: Context) -> Vec<Participant> {
        context
            .participants
            .into_iter()
            .map(|participant| Participant {
                name: participant.name,
                role: participant.role,
            })
            .collect()
    }
}

//地点（名称，坐标）(抽象、描述)
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Location {
    pub name: String,
    pub coordinates: String,
}
impl TryFrom<AbstractSituation> for Location {
    type Error = String;

    fn try_from(value: AbstractSituation) -> Result<Self, Self::Error> {
        match value {
            AbstractSituation::Location(Location { name, coordinates }) => {
                Ok(Location { name, coordinates })
            }
            _ => Err("Cannot convert AbstractSituation to Location".to_string()),
        }
    }
}
impl TryFrom<Context> for Location {
    type Error = String;

    fn try_from(context: Context) -> Result<Self, Self::Error> {
        match context.location {
            Some(Location { name, coordinates }) => Ok(Location { name, coordinates }),
            None => Err("Cannot convert Context to Location".to_string()),
        }
    }
}

//传感数据（名称，强度）（描述）
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct SensoryData {
    pub name: String,
    pub intensity: i32,
}
impl From<Context> for Vec<SensoryData> {
    fn from(context: Context) -> Vec<SensoryData> {
        context
            .sensory_data
            .into_iter()
            .map(|sensory_data| SensoryData {
                name: sensory_data.name,
                intensity: sensory_data.intensity,
            })
            .collect()
    }
}
