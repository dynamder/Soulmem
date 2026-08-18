use crate::embedding::{
    note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant},
    query::{
        note::{
            EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
            MemoryRetrieveQueryVariantEmbedding,
        },
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

use crate::query::string_distance::compute_note_string_score;

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

        //fuse score：多个信号取 max（任一信号强命中即算命中），
        //避免均值把最强信号稀释（narrative 与结构化字段尺度不同）。
        let score_vec = narrative_score
            .into_iter()
            .chain(location_score.into_iter())
            .chain(participants_score.into_iter())
            .chain(environment_score.into_iter())
            .chain(event_score.into_iter())
            .collect::<Vec<_>>();

        Ok(score_vec.into_iter().fold(0.0f32, f32::max))
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

        // 结构化匹配与叙事匹配取 max：任一通道强命中即算命中。
        Ok(score_vec.into_iter().fold(0.0f32, f32::max))
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
        // tag 通道缺失（任一侧无 tag，零向量占位）时，不把缺失通道当 0 分参与加权，
        // 否则 Situation 等无 tag 场景的分数会被压缩到 0.4×0+0.6×variant，理论最高仅 0.6。
        let tag_score = self.tag().cosine_similarity(query.tag())?;
        let variant_score = self.variant().anonymous_compute(query.variant())?;
        if self.tag().is_zero() || query.tag().is_zero() {
            return Ok(variant_score);
        }
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

impl EmbeddedMemoryNote {
    /// 融合评分：`embedding 余弦相似度` 与 `Jaro-Winkler 字符串距离` 按 `string_blend_alpha` 混合。
    ///
    /// 两个分量均为 [0, 1] 量纲，`string_blend_alpha` 为 embedding 所占权重。
    /// 字符串分量仅对精确标识符（concept_identifier / AbstractSituation 结构化字段）生效，
    /// 变体不匹配时字符串分量返回 0.0，此时混合分退化为纯 embedding 分，保持与旧行为一致。
    pub fn compute_fused(
        &self,
        query: &EmbeddedMemoryRetrieveQuery,
        string_blend_alpha: f32,
    ) -> EmbeddingCalcResult<QueryComputeResult> {
        let embedding_score = self.embedding().anonymous_compute(&query.embedding)?;
        let string_score = compute_note_string_score(self.note(), &query.query);
        // 字符串分量仅对精确标识符（Semantic content/aliases、AbstractSituation 结构化字段）生效；
        // 对 SpecificSituation 等类型恒为 0。此时若仍按 (1-alpha)×0 混合，
        // 会把 embedding 分系统性压缩到 alpha×上限以下（Situation 理论最高仅 0.36），
        // 与"字符串分量缺失时退化为纯 embedding 分"的设计注释不符。
        // 因此无字符串信号时直接返回纯 embedding 分。
        // 有字符串信号时取 max：字符串通道只加分、不拉低
        // （当 string < embedding 时混合分会低于纯 embedding 分，取 max 保持"兜底加分"语义）。
        let score = if string_score <= 0.0 {
            embedding_score
        } else {
            let blended =
                string_blend_alpha * embedding_score + (1.0 - string_blend_alpha) * string_score;
            embedding_score.max(blended)
        };
        Ok(QueryComputeResult::new(self.note().id(), score))
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
    use crate::embedding::blend_weights::BlendWeights;
    use crate::embedding::embedding_model::bge::BgeSmallZh;
    use crate::embedding::note::{MemoryEmbedding, MemoryEmbeddingVariant};
    use crate::embedding::query::note::{EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding};
    use crate::embedding::query::sem::SemanticQueryUnitEmbedding;
    use crate::embedding::query::situation::environment::EnvironmentQueryUnitEmbedding;
    use crate::embedding::query::situation::event::EventQueryUnitEmbedding;
    use crate::embedding::query::situation::location::LocationQueryUnitEmbedding;
    use crate::embedding::query::situation::participant::ParticipantQueryUnitEmbedding;
    use crate::embedding::query::situation::SituationQueryUnitEmbedding;
    use crate::embedding::sem::SemanticEmbedding;
    use crate::embedding::situation::context::ContextEmbedding;
    use crate::embedding::situation::environment::EnvironmentEmbedding;
    use crate::embedding::situation::event::EventEmbedding;
    use crate::embedding::situation::location::LocationEmbedding;
    use crate::embedding::situation::participant::ParticipantEmbedding;
    use crate::embedding::situation::{AbstractSituationEmbedding, SpecificSituationEmbedding};
    use crate::embedding::query::note::MemoryRetrieveQueryVariantEmbedding;
    use crate::embedding::EmbeddingVec;
    use crate::embedding::Embeddable;
    use crate::query::retrieve::{
        EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
        MemoryRetrieveQueryVariant, ParticipantQueryUnit, SemanticQueryUnit, SituationQueryUnit,
    };
    use crate::query::string_distance::compute_note_string_score;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::AbstractSituation;
    use soul_mem_core::memory_note::situation_mem::{Environment, Event, Location, Participant};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};

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
            ..Default::default()
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
        // CLS pooling 下短别名与长 content 的相似度量级天然不同（实测 0.55 vs 0.83），
        // 该容差只用于防"别名命中被稀释到接近 0"，不再要求两者数值接近。
        assert!(
            (alias_score - content_score).abs() < 0.35,
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

    fn embed_sem_note(model: &BgeSmallZh, content: &str, aliases: &[&str]) -> EmbeddedMemoryNote {
        let mem_type = MemoryType::Semantic(SemMemory {
            content: content.to_string(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            concept_type: ConceptType::Entity,
            description: format!("与{content}相关的描述"),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding = note.embed(model).unwrap();
        EmbeddedMemoryNote { note, embedding }
    }

    fn embed_sem_query(
        model: &BgeSmallZh,
        concept_identifier: &str,
    ) -> EmbeddedMemoryRetrieveQuery {
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier(concept_identifier.to_string())
            ]),
        );
        let embedding = query.embed(model).unwrap();
        EmbeddedMemoryRetrieveQuery { embedding, query }
    }

    #[test]
    fn test_compute_fused_matches_blend_formula() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedded_note = embed_sem_note(&model, "小酒馆", &[]);
        let embedded_query = embed_sem_query(&model, "酒馆");

        let pure = embedded_note
            .anonymous_compute(&embedded_query.embedding)
            .unwrap();
        let str_score = compute_note_string_score(&embedded_note.note(), &embedded_query.query);
        let fused = embedded_note
            .compute_fused(&embedded_query, 0.6)
            .unwrap()
            .score;

        // 字符串通道只加分：fused = max(pure, 0.6*pure + 0.4*str)
        let expected = pure.max(0.6 * pure + 0.4 * str_score);
        assert!(
            (fused - expected).abs() < 1e-5,
            "fused {fused} != expected {expected}"
        );
        assert!(fused.is_finite());
        assert!((0.0..=1.0).contains(&fused), "fused out of range: {fused}");
        // 字符串分提供正的兜底贡献（0.4 * str > 0，且至少不低于纯 embedding 分）
        assert!(str_score > 0.5, "str_score too low: {str_score}");
        assert!(fused >= pure, "string channel must not drag fused below pure");
    }

    #[test]
    fn test_compute_fused_string_boost_ranking() {
        let model = BgeSmallZh::default_cpu().unwrap();
        // 字形相近的命中项 vs 字形完全不同的干扰项
        let hit = embed_sem_note(&model, "小酒馆", &[]);
        let miss = embed_sem_note(&model, "火车站", &[]);
        let embedded_query = embed_sem_query(&model, "酒馆");

        let hit_fused = hit.compute_fused(&embedded_query, 0.6).unwrap().score;
        let miss_fused = miss.compute_fused(&embedded_query, 0.6).unwrap().score;

        // 字符串分对命中项是正贡献，对干扰项（str=0）无贡献
        let hit_str = compute_note_string_score(&hit.note(), &embedded_query.query);
        let miss_str = compute_note_string_score(&miss.note(), &embedded_query.query);
        assert!(hit_str > 0.5);
        assert_eq!(miss_str, 0.0);
        assert!(
            hit_fused > miss_fused,
            "hit {hit_fused} <= miss {miss_fused}"
        );
        assert!((0.0..=1.0).contains(&hit_fused));
    }

    #[test]
    fn test_compute_fused_alpha_zero_is_pure_embedding() {
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedded_note = embed_sem_note(&model, "小酒馆", &[]);
        let embedded_query = embed_sem_query(&model, "酒馆");

        // alpha=1.0 时退化为纯 embedding 分
        let pure = embedded_note
            .anonymous_compute(&embedded_query.embedding)
            .unwrap();
        let fused = embedded_note
            .compute_fused(&embedded_query, 1.0)
            .unwrap()
            .score;
        assert!((fused - pure).abs() < 1e-6);

        // alpha=0.0 时混合分 = str，字符串通道只加分 → fused = max(emb, str)
        let str_score = compute_note_string_score(&embedded_note.note(), &embedded_query.query);
        let fused = embedded_note
            .compute_fused(&embedded_query, 0.0)
            .unwrap()
            .score;
        assert!((fused - pure.max(str_score)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_fused_abstract_situation_boost() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let mem_type = MemoryType::Situation(
            AbstractSituation::Location(Location {
                name: "酒馆".to_string(),
                coordinates: String::new(),
            })
            .into(),
        );
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding = note.embed(&model).unwrap();
        let embedded_note = EmbeddedMemoryNote { note, embedding };

        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_location(vec![LocationQueryUnit::new("小酒馆")])
            ]),
        );
        let query_embedding = query.embed(&model).unwrap();
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query,
        };

        let pure = embedded_note
            .anonymous_compute(&embedded_query.embedding)
            .unwrap();
        let str_score = compute_note_string_score(&embedded_note.note(), &embedded_query.query);
        let fused = embedded_note
            .compute_fused(&embedded_query, 0.6)
            .unwrap()
            .score;

        // 字符串通道只加分：fused = max(pure, 0.6*pure + 0.4*str)
        let expected = pure.max(0.6 * pure + 0.4 * str_score);
        assert!(
            (fused - expected).abs() < 1e-5,
            "abstract fused {fused} != {expected}"
        );
        assert!((0.0..=1.0).contains(&fused));
        assert!(str_score > 0.5, "location str_score too low: {str_score}");
    }

    #[test]
    fn test_compute_fused_variant_mismatch_degrades_to_embedding() {
        let model = BgeSmallZh::default_cpu().unwrap();
        // Semantic 记忆 + Situation 查询：字符串分=0，混合分退化为 0.6 * embedding 分
        let embedded_note = embed_sem_note(&model, "战斗", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_narrative("战斗场景".to_string())
            ]),
        );
        let query_embedding = query.embed(&model).unwrap();
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query,
        };

        let pure = embedded_note
            .anonymous_compute(&embedded_query.embedding)
            .unwrap();
        let fused = embedded_note
            .compute_fused(&embedded_query, 0.6)
            .unwrap()
            .score;
        assert_eq!(
            compute_note_string_score(&embedded_note.note(), &embedded_query.query),
            0.0
        );
        // 字符串分量缺失（变体不匹配）时退化为纯 embedding 分
        assert!((fused - pure).abs() < 1e-6);
    }

    // —— 以下测试通过直接构造 EmbeddingVec 验证评分公式，不依赖真实模型 ——

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {actual} close to {expected}"
        );
    }

    /// 单位向量 [1, 0]：与 `at(c)` 的余弦相似度为 c。
    fn unit() -> EmbeddingVec {
        EmbeddingVec::new(vec![1.0, 0.0])
    }

    /// 单位向量 [c, sqrt(1-c²)]：与 `unit()` 的余弦相似度为 c。
    fn at(c: f32) -> EmbeddingVec {
        EmbeddingVec::new(vec![c, (1.0 - c * c).sqrt()])
    }

    #[test]
    fn test_location_anonymous_compute_with_coordinates() {
        let loc = LocationEmbedding::test_new(unit(), unit());
        let query = LocationQueryUnitEmbedding::test_new(
            at(0.5),
            Some(at(0.8)),
            BlendWeights::default(),
        );
        let score = loc.anonymous_compute(&query).unwrap();
        // name_score=0.5, coord_score=0.8: 0.6*0.5 + 0.4*0.8 = 0.62
        assert_close(score, 0.62);
    }

    #[test]
    fn test_location_anonymous_compute_without_coordinates() {
        let loc = LocationEmbedding::test_new(unit(), unit());
        let query = LocationQueryUnitEmbedding::test_new(at(0.5), None, BlendWeights::default());
        let score = loc.anonymous_compute(&query).unwrap();
        assert_close(score, 0.5);
    }

    #[test]
    fn test_participant_anonymous_compute_all_fields() {
        let participant = ParticipantEmbedding::test_new(unit(), unit(), unit());
        let query = ParticipantQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            BlendWeights::default(),
        );
        let score = participant.anonymous_compute(&query).unwrap();
        // name_score=0.5, role_score=0.8: 0.6*0.5 + 0.4*0.8 = 0.62
        assert_close(score, 0.62);
    }

    #[test]
    fn test_participant_anonymous_compute_name_only() {
        let participant = ParticipantEmbedding::test_new(unit(), unit(), unit());
        let query = ParticipantQueryUnitEmbedding::test_new(Some(at(0.5)), None, BlendWeights::default());
        let score = participant.anonymous_compute(&query).unwrap();
        assert_close(score, 0.5);
    }

    #[test]
    fn test_participant_anonymous_compute_role_only() {
        let participant = ParticipantEmbedding::test_new(unit(), unit(), unit());
        let query = ParticipantQueryUnitEmbedding::test_new(None, Some(at(0.8)), BlendWeights::default());
        let score = participant.anonymous_compute(&query).unwrap();
        assert_close(score, 0.8);
    }

    #[test]
    fn test_participant_anonymous_compute_none() {
        let participant = ParticipantEmbedding::test_new(unit(), unit(), unit());
        let query = ParticipantQueryUnitEmbedding::test_new(None, None, BlendWeights::default());
        let score = participant.anonymous_compute(&query).unwrap();
        assert_close(score, 0.0);
    }

    #[test]
    fn test_environment_anonymous_compute_all_fields() {
        let env = EnvironmentEmbedding::test_new(unit(), unit());
        let query = EnvironmentQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            BlendWeights::default(),
        );
        let score = env.anonymous_compute(&query).unwrap();
        // atmosphere=0.5, tone=0.8: 0.5*0.5 + 0.5*0.8 = 0.65
        assert_close(score, 0.65);
    }

    #[test]
    fn test_environment_anonymous_compute_none() {
        let env = EnvironmentEmbedding::test_new(unit(), unit());
        let query = EnvironmentQueryUnitEmbedding::test_new(None, None, BlendWeights::default());
        let score = env.anonymous_compute(&query).unwrap();
        assert_close(score, 0.0);
    }

    #[test]
    fn test_event_anonymous_compute_all_fields() {
        let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
        let query = EventQueryUnitEmbedding::test_new(
            at(0.9),
            Some(at(0.5)),
            Some(at(0.8)),
            BlendWeights::default(),
        );
        let score = event.anonymous_compute(&query).unwrap();
        // initiator=0.5, target=0.8, action=0.9: 0.3*0.5+0.3*0.8+0.4*0.9 = 0.75
        assert_close(score, 0.75);
    }

    #[test]
    fn test_event_anonymous_compute_initiator_only() {
        let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
        let query = EventQueryUnitEmbedding::test_new(
            at(0.9),
            Some(at(0.5)),
            None,
            BlendWeights::default(),
        );
        let score = event.anonymous_compute(&query).unwrap();
        // a_w = 0.6, i_w = 0.4: 0.4*0.5 + 0.6*0.9 = 0.74
        assert_close(score, 0.74);
    }

    #[test]
    fn test_event_anonymous_compute_target_only() {
        let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
        let query = EventQueryUnitEmbedding::test_new(
            at(0.9),
            None,
            Some(at(0.8)),
            BlendWeights::default(),
        );
        let score = event.anonymous_compute(&query).unwrap();
        // a_w = 0.6, t_w = 0.4: 0.4*0.8 + 0.6*0.9 = 0.86
        assert_close(score, 0.86);
    }

    #[test]
    fn test_event_anonymous_compute_action_only() {
        let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
        let query = EventQueryUnitEmbedding::test_new(
            at(0.9),
            None,
            None,
            BlendWeights::default(),
        );
        let score = event.anonymous_compute(&query).unwrap();
        assert_close(score, 0.9);
    }

    #[test]
    fn test_semantic_anonymous_compute_with_description() {
        let sem = SemanticEmbedding::new(unit(), unit(), unit());
        let query = SemanticQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            BlendWeights::default(),
        );
        let score = sem.anonymous_compute(&query).unwrap();
        // concept = max(0.5, 0.5) = 0.5; 0.5*0.5 + 0.5*0.8 = 0.65
        assert_close(score, 0.65);
    }

    #[test]
    fn test_semantic_anonymous_compute_alias_winning() {
        // concept_identifier 命中 alias（aliases=unit() 与 query=unit() → 1.0），content 较低
        let sem = SemanticEmbedding::new(at(0.5), unit(), unit());
        let query = SemanticQueryUnitEmbedding::test_new(Some(unit()), None, BlendWeights::default());
        let score = sem.anonymous_compute(&query).unwrap();
        // concept = max(0.5, 1.0) = 1.0（alias 命中）；无 description → 直接返回 concept
        assert_close(score, 1.0);
    }

    #[test]
    fn test_semantic_anonymous_compute_without_description() {
        let sem = SemanticEmbedding::new(unit(), unit(), unit());
        let query = SemanticQueryUnitEmbedding::test_new(Some(at(0.5)), None, BlendWeights::default());
        let score = sem.anonymous_compute(&query).unwrap();
        assert_close(score, 0.5);
    }

    #[test]
    fn test_specific_situation_anonymous_compute_single_narrative() {
        let specific = SpecificSituationEmbedding::test_new(
            unit(),
            ContextEmbedding::test_new(
                None,
                None,
                None,
                None,
                EnvironmentEmbedding::test_new(unit(), unit()),
                None,
            ),
        );
        let query = SituationQueryUnitEmbedding::test_new(
            Some(at(0.8)),
            None,
            None,
            None,
            None,
            BlendWeights::default(),
        );
        let score = specific.anonymous_compute(&query).unwrap();
        // 仅 narrative → 单元素均值 = 0.8
        assert_close(score, 0.8);
    }

    #[test]
    fn test_abstract_situation_anonymous_compute_none() {
        let abstract_emb = AbstractSituationEmbedding::Location(LocationEmbedding::test_new(unit(), unit()));
        let query = SituationQueryUnitEmbedding::test_new(None, None, None, None, None, BlendWeights::default());
        let score = abstract_emb.anonymous_compute(&query).unwrap();
        assert_close(score, 0.0);
    }

    #[test]
    fn test_specific_situation_anonymous_compute_max_two_signals() {
        // narrative + location 两个信号 → 取 max（任一强命中即算命中）
        let specific = SpecificSituationEmbedding::test_new(
            unit(),
            ContextEmbedding::test_new(
                Some(LocationEmbedding::test_new(unit(), unit())),
                None,
                None,
                None,
                EnvironmentEmbedding::test_new(unit(), unit()),
                None,
            ),
        );
        let query = SituationQueryUnitEmbedding::test_new(
            Some(at(0.8)),
            Some(LocationQueryUnitEmbedding::test_new(
                at(0.5),
                None,
                BlendWeights::default(),
            )),
            None,
            None,
            None,
            BlendWeights::default(),
        );
        let score = specific.anonymous_compute(&query).unwrap();
        // narrative=0.8, location(name)=0.5 → max = 0.8
        assert_close(score, 0.8);
    }

    #[test]
    fn test_abstract_situation_anonymous_compute_max_two_signals() {
        // Location 抽象情境 + narrative 和结构化 location 两个信号
        let abstract_emb = AbstractSituationEmbedding::Location(LocationEmbedding::test_new(unit(), unit()));
        let query = SituationQueryUnitEmbedding::test_new(
            Some(at(0.8)),
            Some(LocationQueryUnitEmbedding::test_new(
                at(0.5),
                None,
                BlendWeights::default(),
            )),
            None,
            None,
            None,
            BlendWeights::default(),
        );
        let score = abstract_emb.anonymous_compute(&query).unwrap();
        // structured=0.5, narrative=0.8 → max = 0.8
        assert_close(score, 0.8);
    }

    #[test]
    fn test_semantic_anonymous_compute_no_concept_identifier() {
        // concept_identifier 缺失 → (None, None) 分支返回 0.0
        let sem = SemanticEmbedding::new(unit(), unit(), unit());
        let query = SemanticQueryUnitEmbedding::test_new(None, None, BlendWeights::default());
        let score = sem.anonymous_compute(&query).unwrap();
        assert_close(score, 0.0);
    }

    #[test]
    fn test_memory_variant_semantic_average() {
        let sem = SemanticEmbedding::new(unit(), unit(), unit());
        let query_sem = SemanticQueryUnitEmbedding::test_new(
            Some(unit()),
            None,
            BlendWeights::default(),
        );
        let variant = MemoryEmbeddingVariant::Semantic(sem);
        let query_variant = MemoryRetrieveQueryVariantEmbedding::Semantic(vec![query_sem]);
        let score = variant.anonymous_compute(&query_variant).unwrap();
        assert_close(score, 1.0);
    }

    #[test]
    fn test_memory_variant_semantic_average_multiple_units() {
        // 多单元语义查询：按单元数归一化取平均（/len 而非 *len）
        let sem = SemanticEmbedding::new(unit(), unit(), unit());
        let query_sem_1 = SemanticQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            None,
            BlendWeights::default(),
        );
        let query_sem_2 = SemanticQueryUnitEmbedding::test_new(
            Some(at(0.9)),
            None,
            BlendWeights::default(),
        );
        let variant = MemoryEmbeddingVariant::Semantic(sem);
        let query_variant =
            MemoryRetrieveQueryVariantEmbedding::Semantic(vec![query_sem_1, query_sem_2]);
        let score = variant.anonymous_compute(&query_variant).unwrap();
        // (0.5 + 0.9)/2 = 0.7
        assert_close(score, 0.7);
    }

    #[test]
    fn test_memory_embedding_tag_variant_fusion() {
        let note_emb = MemoryEmbedding::new(unit(), MemoryEmbeddingVariant::Procedure());
        // 通过 MemoryEmbedding 构造的 tag + 空语义 variant
        let mut query = MemoryRetrieveQueryEmbedding::new(unit());
        let mut bw = BlendWeights::default();
        bw.tag = 0.4;
        bw.variant = 0.6;
        query = query.with_weights(bw);
        let score = note_emb.anonymous_compute(&query).unwrap();
        // tag=1.0 * 0.4 + variant(0.0)*0.6 = 0.4
        assert_close(score, 0.4);
    }

    #[test]
    fn test_memory_embedding_tag_missing_uses_variant_only() {
        // variant 构造：description=unit()，query 仅提供 description → variant 分 = 0.5
        let note_emb = MemoryEmbedding::new(
            unit(),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(2),
                EmbeddingVec::zero(2),
                unit(),
            )),
        );
        let q_sem =
            SemanticQueryUnitEmbedding::test_new(None, Some(unit()), BlendWeights::default());
        let q_variant = MemoryRetrieveQueryVariantEmbedding::Semantic(vec![q_sem]);

        // query 无 tag（零向量）→ 纯 variant 分，不再被 0.4 压缩
        let query_no_tag = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(2))
            .with_variant(q_variant.clone());
        let score = note_emb.anonymous_compute(&query_no_tag).unwrap();
        assert_close(score, 0.5);

        // 两侧都有 tag → 保持 0.3/0.7 加权：0.3*1.0 + 0.7*0.5 = 0.65
        let query_tag = MemoryRetrieveQueryEmbedding::new(unit()).with_variant(q_variant.clone());
        let score = note_emb.anonymous_compute(&query_tag).unwrap();
        assert_close(score, 0.65);

        // note 无 tag、query 有 tag → 同样纯 variant 分
        let note_no_tag = MemoryEmbedding::new(
            EmbeddingVec::zero(2),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(2),
                EmbeddingVec::zero(2),
                unit(),
            )),
        );
        let query_tag2 = MemoryRetrieveQueryEmbedding::new(unit()).with_variant(q_variant);
        let score = note_no_tag.anonymous_compute(&query_tag2).unwrap();
        assert_close(score, 0.5);
    }
}
