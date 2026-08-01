use crate::embedding::{
    note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant},
    query::{
        note::{MemoryRetrieveQueryEmbedding, MemoryRetrieveQueryVariantEmbedding},
        sem::SemanticQueryUnitEmbedding,
        situation::{
            environment::EnvironmentQueryUnitEmbedding, event::EventQueryUnitEmbedding,
            location::LocationQueryUnitEmbedding, participant::ParticipantQueryUnitEmbedding,
            SituationQueryUnitEmbedding,
        },
    },
    sem::SemanticEmbedding,
    situation::{
        environment::EnvironmentEmbedding, event::EventEmbedding, location::LocationEmbedding,
        participant::ParticipantEmbedding, AbstractSituationEmbedding, SituationEmbedding,
        SpecificSituationEmbedding,
    },
    EmbeddingCalcResult,
};

use soul_mem_core::memory_note::MemoryId;

pub trait AnonymousQueryCompute {
    type Query;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32>;
}

pub trait QueryCompute: AnonymousQueryCompute {
    fn compute(&self, query: &Self::Query) -> EmbeddingCalcResult<QueryComputeResult>;
}

pub struct QueryComputeResult {
    pub id: MemoryId,
    pub score: f32,
}

impl QueryComputeResult {
    pub fn new(id: MemoryId, score: f32) -> Self {
        QueryComputeResult { id, score }
    }
}
////////////////////////////////////////////////////////////
impl AnonymousQueryCompute for LocationEmbedding {
    type Query = LocationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let name_score = self.name().cosine_similarity(query.name())?;
        let coordinates_score = query
            .coordinates()
            .map(|coordinate| coordinate.cosine_similarity(self.coordinates()))
            .transpose()?;

        let bw = &query.blend_weights;
        if let Some(coord_score) = coordinates_score {
            Ok(bw.sit_location_name * name_score + bw.sit_location_coord * coord_score)
        } else {
            Ok(name_score)
        }
    }
}

impl AnonymousQueryCompute for ParticipantEmbedding {
    type Query = ParticipantQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let name_score = query
            .name()
            .map(|name| name.cosine_similarity(self.name()))
            .transpose()?;

        let role_score = query
            .role()
            .map(|role| role.cosine_similarity(self.role()))
            .transpose()?;

        let bw = &query.blend_weights;
        match (name_score, role_score) {
            (Some(name_score), Some(role_score)) => {
                Ok(bw.sit_participant_name * name_score + bw.sit_participant_role * role_score)
            }
            (Some(name_score), None) => Ok(name_score),
            (None, Some(role_score)) => Ok(role_score),
            (None, None) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for EnvironmentEmbedding {
    type Query = EnvironmentQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let atmosphere_score = query
            .atmosphere()
            .map(|atmosphere| atmosphere.cosine_similarity(self.atmosphere()))
            .transpose()?;

        let tone_score = query
            .tone()
            .map(|tone| tone.cosine_similarity(self.tone()))
            .transpose()?;

        let bw = &query.blend_weights;
        match (atmosphere_score, tone_score) {
            (Some(atmosphere_score), Some(tone_score)) => {
                Ok(bw.sit_env_atmosphere * atmosphere_score + bw.sit_env_tone * tone_score)
            }
            (Some(atmosphere_score), None) => Ok(atmosphere_score),
            (None, Some(tone_score)) => Ok(tone_score),
            (None, None) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for EventEmbedding {
    type Query = EventQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let action_score = self.action().cosine_similarity(query.action())?;

        let initiator_score = query
            .initiator()
            .map(|initiator| initiator.cosine_similarity(self.initiator()))
            .transpose()?;

        let target_score = query
            .target()
            .map(|target| target.cosine_similarity(self.target()))
            .transpose()?;

        let bw = &query.blend_weights;
        match (initiator_score, target_score) {
            (Some(initiator_score), Some(target_score)) => Ok(bw.sit_event_initiator
                * initiator_score
                + bw.sit_event_target * target_score
                + bw.sit_event_action * action_score),
            (Some(initiator_score), None) => {
                let a_w = bw.sit_event_initiator_only_action;
                let i_w = 1.0 - a_w;
                Ok(i_w * initiator_score + a_w * action_score)
            }
            (None, Some(target_score)) => {
                let a_w = bw.sit_event_target_only_action;
                let t_w = 1.0 - a_w;
                Ok(t_w * target_score + a_w * action_score)
            }
            (None, None) => Ok(action_score),
        }
    }
}

impl AnonymousQueryCompute for SpecificSituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let narrative_score = query
            .narrative()
            .map(|narrative| narrative.cosine_similarity(self.narrative()))
            .transpose()?;

        //location
        let location_score = if let Some(query_location) = query.location() {
            self.context()
                .location()
                .map(|location| location.anonymous_compute(query_location))
                .transpose()?
        } else {
            None
        };

        //participants
        let participants_score = if let Some(query_participants) = query.participants() {
            self.context()
                .fused_participant()
                .map(|participants| participants.anonymous_compute(query_participants))
                .transpose()?
        } else {
            None
        };

        //environment
        let environment_score = query
            .environment()
            .map(|env| self.context().environment().anonymous_compute(env))
            .transpose()?;

        //event
        let event_score = if let Some(query_event) = query.event() {
            self.context()
                .fused_event()
                .map(|event| event.anonymous_compute(query_event))
                .transpose()?
        } else {
            None
        };

        //fuse score
        let score_vec = narrative_score
            .into_iter()
            .chain(location_score.into_iter())
            .chain(participants_score.into_iter())
            .chain(environment_score.into_iter())
            .chain(event_score.into_iter())
            .collect::<Vec<_>>();

        let len = score_vec.len();
        Ok(score_vec.into_iter().map(|i| i / len as f32).sum::<f32>())
    }
}

impl AnonymousQueryCompute for AbstractSituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        //结构化匹配：只有当query提供了与抽象情境同类型的字段时才计分，否则为None
        let structured_score = match self {
            AbstractSituationEmbedding::Location(loc) => query
                .location()
                .map(|q_loc| loc.anonymous_compute(q_loc))
                .transpose()?,
            AbstractSituationEmbedding::Environment(env) => query
                .environment()
                .map(|q_env| env.anonymous_compute(q_env))
                .transpose()?,
            AbstractSituationEmbedding::Event(event) => query
                .event()
                .map(|q_event| event.anonymous_compute(q_event))
                .transpose()?,
            AbstractSituationEmbedding::Participant(participant) => query
                .participants()
                .map(|q_participant| participant.anonymous_compute(q_participant))
                .transpose()?,
        };

        //叙事匹配：抽象情境的"自我"向量与query.narrative的相似度
        let narrative_score = query
            .narrative()
            .map(|narrative| narrative.cosine_similarity(&self.fused_self()?))
            .transpose()?;

        let score_vec = structured_score
            .into_iter()
            .chain(narrative_score.into_iter())
            .collect::<Vec<_>>();

        let len = score_vec.len();
        if len == 0 {
            return Ok(0.0);
        }
        Ok(score_vec.into_iter().map(|i| i / len as f32).sum::<f32>())
    }
}

impl AnonymousQueryCompute for SituationEmbedding {
    type Query = SituationQueryUnitEmbedding;
    //TODO: add time span score count
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match self {
            Self::Specific(specific) => specific.anonymous_compute(query),
            Self::Abstract(abstract_sit) => abstract_sit.anonymous_compute(query),
        }
    }
}

impl AnonymousQueryCompute for SemanticEmbedding {
    type Query = SemanticQueryUnitEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let concept_main_score = query
            .concept_identifier()
            .map(|con| con.cosine_similarity(self.content()))
            .transpose()?;
        let concept_aliases_score = query
            .concept_identifier()
            .map(|con| con.cosine_similarity(self.aliases()))
            .transpose()?;

        let description_score = query
            .description()
            .map(|description| description.cosine_similarity(self.description()))
            .transpose()?;

        let bw = &query.blend_weights;
        //max_pooling: 命中的无论是content还是aliases，取更高者作为概念分数
        let concept_score = match (concept_main_score, concept_aliases_score) {
            (Some(main_score), Some(aliases_score)) => main_score.max(aliases_score),
            (None, None) => 0.0,
            _ => unreachable!(
                "main_score and aliases_score all compute from query.concept_identifier(), so they must be Some or None simultaneously"
            ),
        };

        if let Some(description_score) = description_score {
            Ok(bw.sem_concept * concept_score + bw.sem_description * description_score)
        } else {
            Ok(concept_score)
        }
    }
}

impl AnonymousQueryCompute for MemoryEmbeddingVariant {
    type Query = MemoryRetrieveQueryVariantEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        match (self, query) {
            (Self::Semantic(sem), MemoryRetrieveQueryVariantEmbedding::Semantic(q_sem)) => {
                let score_vec = q_sem
                    .into_iter()
                    .map(|q_sem_unit| sem.anonymous_compute(q_sem_unit))
                    .collect::<Result<Vec<_>, _>>()?;
                //按单元数归一化，避免长查询因sum而系统性占优，与Situation分支保持一致
                if score_vec.is_empty() {
                    return Ok(0.0);
                }
                let len = score_vec.len();
                Ok(score_vec.into_iter().sum::<f32>() / len as f32)
            }
            (Self::Situation(sit), MemoryRetrieveQueryVariantEmbedding::Situation(q_sit)) => {
                let score_vec = q_sit
                    .into_iter()
                    .map(|q_sit_unit| sit.anonymous_compute(q_sit_unit))
                    .collect::<Result<Vec<_>, _>>()?;
                //TODO: 检查该分支是否也需要按单元数归一化
                Ok(score_vec.into_iter().sum::<f32>())
            }
            (_, _) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for MemoryEmbedding {
    type Query = MemoryRetrieveQueryEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        let tag_score = self.tag().cosine_similarity(query.tag())?;
        let variant_score = self.variant().anonymous_compute(query.variant())?;
        Ok(query.tag_weight * tag_score + query.variant_weight * variant_score)
    }
}

//TODO: take common fields in MemoryNote into computation
impl AnonymousQueryCompute for EmbeddedMemoryNote {
    type Query = MemoryRetrieveQueryEmbedding;
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        self.embedding().anonymous_compute(query)
    }
}

impl QueryCompute for EmbeddedMemoryNote {
    fn compute(&self, query: &Self::Query) -> EmbeddingCalcResult<QueryComputeResult> {
        Ok(QueryComputeResult {
            id: self.note().id(),
            score: self.anonymous_compute(query)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use crate::embedding::Embeddable;
    use crate::query::retrieve::{
        EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, ParticipantQueryUnit,
        SemanticQueryUnit,
    };
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{Environment, Event, Location, Participant};

    #[test]
    fn test_query_compute_result() {
        let memory_id = MemoryId::new();
        let result = QueryComputeResult::new(memory_id, 0.85);
        assert_eq!(result.id, memory_id);
        assert_eq!(result.score, 0.85);
    }

    #[test]
    fn test_semantic_embedding_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let memory = SemMemory {
            content: "Rust编程语言".to_string(),
            aliases: vec!["Rust".to_string()],
            concept_type: ConceptType::Entity,
            description: "一种注重安全性的系统编程语言".to_string(),
        };

        let sem_embedding = memory.embed(&model).unwrap();

        let query = SemanticQueryUnit::new()
            .with_concept_identifier("Rust".to_string())
            .with_description("系统编程语言".to_string());

        let query_emb = query.embed(&model).unwrap();

        let score = sem_embedding.anonymous_compute(&query_emb).unwrap();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_location_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let location = Location {
            name: "北京".to_string(),
            coordinates: "中国".to_string(),
        };
        let location_emb = location.embed(&model).unwrap();

        let location_query = LocationQueryUnit::new("北京").with_coordinates("中国".to_string());
        let location_query_emb = location_query.embed(&model).unwrap();

        let score = location_emb.anonymous_compute(&location_query_emb).unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn test_participant_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let participant = Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        };
        let participant_emb = participant.embed(&model).unwrap();

        let participant_query = ParticipantQueryUnit::new()
            .with_name("张三".to_string())
            .with_role("学生".to_string());
        let participant_query_emb = participant_query.embed(&model).unwrap();

        let score = participant_emb
            .anonymous_compute(&participant_query_emb)
            .unwrap();
        assert!(score > 0.5);
    }

    #[test]
    fn test_environment_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let environment = Environment {
            atmosphere: "安静".to_string(),
            tone: "舒适".to_string(),
        };
        let environment_emb = environment.embed(&model).unwrap();

        let environment_query = EnvironmentQueryUnit::new()
            .with_atmosphere("安静".to_string())
            .with_tone("舒适".to_string());
        let environment_query_emb = environment_query.embed(&model).unwrap();

        let score = environment_emb
            .anonymous_compute(&environment_query_emb)
            .unwrap();
        assert!(score > 0.5);
    }

    #[test]
    fn test_event_query_compute() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let event = Event {
            action: "跑步".to_string(),
            action_intensity: 0.8,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        };
        let event_emb = event.embed(&model).unwrap();

        let event_query = EventQueryUnit::new("跑步".to_string())
            .with_initiator("张三".to_string())
            .with_target("操场".to_string());
        let event_query_emb = event_query.embed(&model).unwrap();

        let score = event_emb.anonymous_compute(&event_query_emb).unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn test_semantic_alias_max_pooling_hits_alias() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let memory = SemMemory {
            content: "Rust编程语言".to_string(),
            aliases: vec!["Rust".to_string()],
            concept_type: ConceptType::Entity,
            description: "一种注重安全性的系统编程语言".to_string(),
        };

        let sem_embedding = memory.embed(&model).unwrap();

        //别名命中查询：query.concept_identifier 与 alias 完全一致
        let alias_query = SemanticQueryUnit::new().with_concept_identifier("Rust".to_string());
        let alias_query_emb = alias_query.embed(&model).unwrap();

        //content命中查询：query.concept_identifier 与 content 完全一致
        let content_query =
            SemanticQueryUnit::new().with_concept_identifier("Rust编程语言".to_string());
        let content_query_emb = content_query.embed(&model).unwrap();

        let alias_score = sem_embedding.anonymous_compute(&alias_query_emb).unwrap();
        let content_score = sem_embedding.anonymous_compute(&content_query_emb).unwrap();

        //别名与content命中的分数都应显著高于0，且max_pooling保证别名命中不被content稀释
        assert!(alias_score > 0.5, "alias score too low: {alias_score}");
        assert!(
            content_score > 0.5,
            "content score too low: {content_score}"
        );
        assert!(
            (alias_score - content_score).abs() < 0.2,
            "alias ({alias_score}) and content ({content_score}) scores should be comparable"
        );
    }

    #[test]
    fn test_abstract_situation_narrative_fallback() {
        let model = BgeSmallZh::default_cpu().unwrap();

        use crate::query::retrieve::SituationQueryUnit;
        use soul_mem_core::memory_note::situation_mem::{AbstractSituation, SituationType};

        //抽象情境：事件节点，无narrative字段
        let abstract_event = AbstractSituation::Event(Event {
            action: "战斗".to_string(),
            action_intensity: 0.9,
            initiator: "我".to_string(),
            target: "对手".to_string(),
        });
        let situation_type: SituationType = abstract_event.into();
        let embedding = situation_type.embed(&model).unwrap();
        let abstract_emb = embedding.to_abstract().unwrap();

        //纯叙事查询：没有event结构化字段，只能靠narrative fallback
        let narrative_query = SituationQueryUnit::new()
            .with_narrative("享受战斗时的愉快氛围而不是单纯厮杀".to_string());
        let narrative_query_emb = narrative_query.embed(&model).unwrap();

        let score = abstract_emb
            .anonymous_compute(&narrative_query_emb)
            .unwrap();
        assert!(
            score > 0.3,
            "abstract situation should be matched by narrative, got {score}"
        );

        //结构化命中：提供event字段时分数应更高（结构化+叙事双重信号）
        let structured_query = SituationQueryUnit::new()
            .with_narrative("享受战斗时的愉快氛围".to_string())
            .with_event(vec![EventQueryUnit::new("战斗".to_string())]);
        let structured_query_emb = structured_query.embed(&model).unwrap();
        let structured_score = abstract_emb
            .anonymous_compute(&structured_query_emb)
            .unwrap();
        assert!(structured_score >= score);
    }
}
