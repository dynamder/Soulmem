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

impl Default for SpecificSituation {
    fn default() -> Self {
        Self {
            narrative: String::new(),
            time_span: Utc::now(),
            context: Context::default(),
        }
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

impl Default for Context {
    fn default() -> Self {
        Self {
            location: None,
            participants: Vec::new(),
            emotions: Vec::new(),
            sensory_data: Vec::new(),
            environment: Environment::default(),
            event: Vec::new(),
        }
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

impl Default for Environment {
    fn default() -> Self {
        Self {
            atmosphere: String::new(),
            tone: String::new(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sample_context() -> Context {
        Context::new(
            Some(Location {
                name: "cafe".to_string(),
                coordinates: "0,0".to_string(),
            }),
            vec![Participant {
                name: "alice".to_string(),
                role: "friend".to_string(),
            }],
            vec![Emotion {
                name: "joy".to_string(),
                intensity: 0.8,
            }],
            vec![SensoryData {
                name: "warmth".to_string(),
                intensity: 0.5,
            }],
            Environment {
                atmosphere: "cozy".to_string(),
                tone: "warm".to_string(),
            },
            vec![Event {
                action: "talk".to_string(),
                action_intensity: 0.4,
                initiator: "alice".to_string(),
                target: "bob".to_string(),
            }],
        )
    }

    #[test]
    fn test_specific_situation_narrative_roundtrip() {
        let time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut situation = SpecificSituation::new(
            "original".to_string(),
            time,
            Context::new(None, vec![], vec![], vec![], Environment { atmosphere: "".to_string(), tone: "".to_string() }, vec![]),
        );
        assert_eq!(situation.get_narrative(), "original");
        assert_eq!(situation.get_time_span(), &time);

        situation.get_mut_narrative().push_str(" extended");
        assert_eq!(situation.get_narrative(), "original extended");
    }

    #[test]
    fn test_context_getters_roundtrip() {
        let mut ctx = sample_context();

        assert_eq!(ctx.get_location().as_ref().map(|l| l.name.as_str()), Some("cafe"));
        assert_eq!(ctx.get_participants().len(), 1);
        assert_eq!(ctx.get_emotions().len(), 1);
        assert_eq!(ctx.get_sensory_data().len(), 1);
        assert_eq!(ctx.get_environment().atmosphere, "cozy");
        assert_eq!(ctx.get_event().len(), 1);

        ctx.get_mut_location().as_mut().unwrap().name = "park".to_string();
        ctx.get_mut_participants().push(Participant {
            name: "bob".to_string(),
            role: "friend".to_string(),
        });
        ctx.get_mut_emotions().push(Emotion {
            name: "calm".to_string(),
            intensity: 0.2,
        });
        ctx.get_mut_sensory_data().push(SensoryData {
            name: "breeze".to_string(),
            intensity: 0.3,
        });
        ctx.get_mut_environment().tone = "cool".to_string();
        ctx.get_mut_event().push(Event {
            action: "walk".to_string(),
            action_intensity: 0.6,
            initiator: "bob".to_string(),
            target: "alice".to_string(),
        });

        assert_eq!(ctx.get_location().as_ref().map(|l| l.name.as_str()), Some("park"));
        assert_eq!(ctx.get_participants().len(), 2);
        assert_eq!(ctx.get_emotions().len(), 2);
        assert_eq!(ctx.get_sensory_data().len(), 2);
        assert_eq!(ctx.get_environment().tone, "cool");
        assert_eq!(ctx.get_event().len(), 2);
    }

    #[test]
    fn test_situation_type_from_abstract_and_specific() {
        let abstract_s = AbstractSituation::Location(Location {
            name: "home".to_string(),
            coordinates: "1,1".to_string(),
        });
        let st: SituationType = abstract_s.into();
        assert!(matches!(st, SituationType::AbstractSituation(_)));

        let time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let specific = SpecificSituation::new(
            "n".to_string(),
            time,
            Context::new(None, vec![], vec![], vec![], Environment { atmosphere: "".to_string(), tone: "".to_string() }, vec![]),
        );
        let st: SituationType = specific.into();
        assert!(matches!(st, SituationType::SpecificSituation(_)));
    }

    #[test]
    fn test_abstract_situation_from_variants() {
        let location = AbstractSituation::Location(Location {
            name: "x".to_string(),
            coordinates: "0,0".to_string(),
        });
        assert!(matches!(location, AbstractSituation::Location(_)));

        let participant = AbstractSituation::Participant(Participant {
            name: "p".to_string(),
            role: "r".to_string(),
        });
        assert!(matches!(participant, AbstractSituation::Participant(_)));

        let env = AbstractSituation::Environment(Environment {
            atmosphere: "a".to_string(),
            tone: "t".to_string(),
        });
        assert!(matches!(env, AbstractSituation::Environment(_)));

        let event = AbstractSituation::Event(Event {
            action: "a".to_string(),
            action_intensity: 1.0,
            initiator: "i".to_string(),
            target: "t".to_string(),
        });
        assert!(matches!(event, AbstractSituation::Event(_)));
    }
}
