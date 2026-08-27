pub mod context;
pub mod emotion;
pub mod environment;
pub mod event;
pub mod location;
pub mod participant;
pub mod sensory_data;
use serde::{Deserialize, Serialize};

use crate::embedding::{
    mean_pooling,
    situation::{
        context::ContextEmbedding, environment::EnvironmentEmbedding, event::EventEmbedding,
        participant::ParticipantEmbedding,
    },
    Embeddable, EmbeddingCalcResult, EmbeddingVec,
};
use location::LocationEmbedding;
use soul_mem_core::memory_note::situation_mem::{
    AbstractSituation, SituationType, SpecificSituation,
};
#[allow(clippy::large_enum_variant)] // Box 化会改变公开 API 与 serde 布局，暂保持现状
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SituationEmbedding {
    Specific(SpecificSituationEmbedding),
    Abstract(AbstractSituationEmbedding),
}
impl SituationEmbedding {
    pub fn to_specific(&self) -> Option<&SpecificSituationEmbedding> {
        match self {
            SituationEmbedding::Specific(embedding) => Some(embedding),
            _ => None,
        }
    }
    pub fn to_abstract(&self) -> Option<&AbstractSituationEmbedding> {
        match self {
            SituationEmbedding::Abstract(embedding) => Some(embedding),
            _ => None,
        }
    }
}
impl From<AbstractSituationEmbedding> for SituationEmbedding {
    fn from(value: AbstractSituationEmbedding) -> Self {
        SituationEmbedding::Abstract(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecificSituationEmbedding {
    narrative: EmbeddingVec,
    context: ContextEmbedding,
}
impl SpecificSituationEmbedding {
    pub fn narrative(&self) -> &EmbeddingVec {
        &self.narrative
    }
    pub fn context(&self) -> &ContextEmbedding {
        &self.context
    }
}
#[cfg(test)]
impl SpecificSituationEmbedding {
    pub(crate) fn test_new(narrative: EmbeddingVec, context: ContextEmbedding) -> Self {
        Self { narrative, context }
    }
}
impl Embeddable for SpecificSituation {
    type EmbeddingGen = SpecificSituationEmbedding;
    type EmbeddingFused = EmbeddedSpecificSituation;
    fn embed(
        &self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingGen> {
        let narrative_vec = model.infer_with_chunk(self.get_narrative().as_str())?;
        let context_vec = self.get_context().embed(model)?;
        Ok(SpecificSituationEmbedding {
            narrative: narrative_vec,
            context: context_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSpecificSituation {
            embedding: self.embed(model)?,
            specific_situation: self,
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedSpecificSituation {
    pub embedding: SpecificSituationEmbedding,
    pub specific_situation: SpecificSituation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AbstractSituationEmbedding {
    Location(LocationEmbedding),
    Participant(ParticipantEmbedding),
    Environment(EnvironmentEmbedding),
    Event(EventEmbedding),
}

impl AbstractSituationEmbedding {
    pub fn to_location(&self) -> Option<&LocationEmbedding> {
        match self {
            AbstractSituationEmbedding::Location(location) => Some(location),
            _ => None,
        }
    }
    pub fn to_participant(&self) -> Option<&ParticipantEmbedding> {
        match self {
            AbstractSituationEmbedding::Participant(participant) => Some(participant),
            _ => None,
        }
    }
    pub fn to_environment(&self) -> Option<&EnvironmentEmbedding> {
        match self {
            AbstractSituationEmbedding::Environment(environment) => Some(environment),
            _ => None,
        }
    }
    pub fn to_event(&self) -> Option<&EventEmbedding> {
        match self {
            AbstractSituationEmbedding::Event(event) => Some(event),
            _ => None,
        }
    }

    /// 由该抽象情境的各结构化字段融合出一个"自我"向量，
    /// 用于与query.narrative做语义匹配，缓解抽象情境对叙事型查询检出率低的问题。
    pub fn fused_self(&self) -> EmbeddingCalcResult<EmbeddingVec> {
        match self {
            AbstractSituationEmbedding::Location(loc) => {
                mean_pooling(&[loc.name(), loc.coordinates()])
            }
            AbstractSituationEmbedding::Participant(participant) => Ok(participant.fused().clone()),
            AbstractSituationEmbedding::Environment(env) => {
                mean_pooling(&[env.atmosphere(), env.tone()])
            }
            AbstractSituationEmbedding::Event(event) => {
                mean_pooling(&[event.action(), event.initiator(), event.target()])
            }
        }
    }
}

impl Embeddable for AbstractSituation {
    type EmbeddingGen = AbstractSituationEmbedding;
    type EmbeddingFused = EmbeddedAbstractSituation;
    fn embed(
        &self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingGen> {
        match self {
            Self::Environment(env) => {
                Ok(AbstractSituationEmbedding::Environment(env.embed(model)?))
            }
            Self::Event(eve) => Ok(AbstractSituationEmbedding::Event(eve.embed(model)?)),
            Self::Location(loc) => Ok(AbstractSituationEmbedding::Location(loc.embed(model)?)),
            Self::Participant(par) => {
                Ok(AbstractSituationEmbedding::Participant(par.embed(model)?))
            }
        }
    }
    fn embed_and_fuse(
        self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedAbstractSituation {
            embedding: self.embed(model)?,
            abstract_situation: self,
        })
    }
}

impl From<LocationEmbedding> for AbstractSituationEmbedding {
    fn from(location: LocationEmbedding) -> Self {
        AbstractSituationEmbedding::Location(location)
    }
}
impl From<ParticipantEmbedding> for AbstractSituationEmbedding {
    fn from(participant: ParticipantEmbedding) -> Self {
        AbstractSituationEmbedding::Participant(participant)
    }
}
impl From<EnvironmentEmbedding> for AbstractSituationEmbedding {
    fn from(environment: EnvironmentEmbedding) -> Self {
        AbstractSituationEmbedding::Environment(environment)
    }
}
impl From<EventEmbedding> for AbstractSituationEmbedding {
    fn from(event: EventEmbedding) -> Self {
        AbstractSituationEmbedding::Event(event)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedAbstractSituation {
    pub embedding: AbstractSituationEmbedding,
    pub abstract_situation: AbstractSituation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedSituationType {
    pub embedding: SituationEmbedding,
    pub situation: SituationType,
}

impl Embeddable for SituationType {
    type EmbeddingGen = SituationEmbedding;
    type EmbeddingFused = EmbeddedSituationType;
    fn embed(
        &self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingGen> {
        match self {
            Self::AbstractSituation(abstract_sit) => {
                Ok(SituationEmbedding::Abstract(abstract_sit.embed(model)?))
            }
            Self::SpecificSituation(specific) => {
                Ok(SituationEmbedding::Specific(specific.embed(model)?))
            }
        }
    }
    fn embed_and_fuse(
        self,
        model: &dyn super::EmbeddingModel,
    ) -> super::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSituationType {
            embedding: self.embed(model)?,
            situation: self,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use soul_mem_core::memory_note::situation_mem::{
        Context, Emotion, Environment, Event, Location, Participant, SensoryData,
    };

    #[test]
    fn test_specific_situation_embed() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let context = Context::new(
            Some(Location {
                name: "学校".to_string(),
                coordinates: "北京".to_string(),
            }),
            vec![Participant {
                name: "张三".to_string(),
                role: "学生".to_string(),
            }],
            vec![Emotion {
                name: "开心".to_string(),
                intensity: 0.8,
            }],
            vec![SensoryData {
                name: "明亮".to_string(),
                intensity: 0.6,
            }],
            Environment {
                atmosphere: "温暖".to_string(),
                tone: "舒适".to_string(),
            },
            vec![Event {
                action: "学习".to_string(),
                action_intensity: 0.7,
                initiator: "张三".to_string(),
                target: "知识".to_string(),
            }],
        );

        let situation = SpecificSituation::new(
            "今天在学校学习了很多知识".to_string(),
            chrono::Utc::now(),
            context,
        );

        let embedding = situation.embed(&model).unwrap();
        assert_eq!(embedding.narrative().shape(), 512);
    }

    #[test]
    fn test_abstract_situation_embed() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let location = AbstractSituation::Location(Location {
            name: "图书馆".to_string(),
            coordinates: "学校内".to_string(),
        });

        let embedding = location.embed(&model).unwrap();
        assert!(embedding.to_location().is_some());
    }

    #[test]
    fn test_environment_embed() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let env = Environment {
            atmosphere: "紧张".to_string(),
            tone: "严肃".to_string(),
        };

        let embedding = env.embed(&model).unwrap();
        assert_eq!(embedding.atmosphere().shape(), 512);
        assert_eq!(embedding.tone().shape(), 512);
    }

    #[test]
    fn test_situation_type_embed() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let location = AbstractSituation::Location(Location {
            name: "公园".to_string(),
            coordinates: "市中心".to_string(),
        });

        let situation_type: SituationType = location.into();
        let embedding = situation_type.embed(&model).unwrap();

        assert!(embedding.to_abstract().is_some());
    }

    #[test]
    fn test_situation_embedding_accessors() {
        let specific_emb = SpecificSituationEmbedding {
            narrative: EmbeddingVec::new(vec![1.0, 2.0]),
            context: ContextEmbedding::test_new(
                None,
                None,
                None,
                None,
                EnvironmentEmbedding::test_new(
                    EmbeddingVec::new(vec![0.0, 0.0]),
                    EmbeddingVec::new(vec![0.0, 0.0]),
                ),
                None,
            ),
        };
        let specific = SituationEmbedding::Specific(specific_emb.clone());
        assert!(specific.to_specific().is_some());
        assert!(specific.to_abstract().is_none());

        let abstract_emb = AbstractSituationEmbedding::Location(LocationEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
        ));
        let abstract_variant = SituationEmbedding::Abstract(abstract_emb.clone());
        assert!(abstract_variant.to_specific().is_none());
        assert!(abstract_variant.to_abstract().is_some());

        // From<AbstractSituationEmbedding>
        let from: SituationEmbedding = abstract_emb.clone().into();
        assert!(matches!(from, SituationEmbedding::Abstract(_)));
    }

    #[test]
    fn test_abstract_situation_embedding_accessors() {
        let loc = AbstractSituationEmbedding::Location(LocationEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
        ));
        assert!(loc.to_location().is_some());
        assert!(loc.to_participant().is_none());
        assert!(loc.to_environment().is_none());
        assert!(loc.to_event().is_none());

        let participant = AbstractSituationEmbedding::Participant(ParticipantEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
            EmbeddingVec::new(vec![1.5]),
        ));
        assert!(participant.to_location().is_none());
        assert!(participant.to_participant().is_some());

        let environment = AbstractSituationEmbedding::Environment(EnvironmentEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
        ));
        assert!(environment.to_environment().is_some());
        assert!(environment.to_participant().is_none());

        let event = AbstractSituationEmbedding::Event(EventEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            EmbeddingVec::new(vec![2.0]),
            EmbeddingVec::new(vec![3.0]),
            0.5,
        ));
        assert!(event.to_event().is_some());
        assert!(event.to_location().is_none());
    }

    #[test]
    fn test_abstract_situation_fused_self() {
        let loc = AbstractSituationEmbedding::Location(LocationEmbedding::test_new(
            EmbeddingVec::new(vec![1.0, 2.0]),
            EmbeddingVec::new(vec![3.0, 4.0]),
        ));
        let fused = loc.fused_self().unwrap();
        assert_eq!(fused.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
    }
}
