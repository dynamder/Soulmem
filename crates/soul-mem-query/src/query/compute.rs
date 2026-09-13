use crate::embedding::{
    EmbeddingCalcResult,
    note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant},
    query::{
        note::{
            EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
            MemoryRetrieveQueryVariantEmbedding,
        },
        sem::SemanticQueryUnitEmbedding,
        situation::{
            SituationQueryUnitEmbedding, environment::EnvironmentQueryUnitEmbedding,
            event::EventQueryUnitEmbedding, location::LocationQueryUnitEmbedding,
            participant::ParticipantQueryUnitEmbedding,
        },
    },
    sem::SemanticEmbedding,
    situation::{
        AbstractSituationEmbedding, SituationEmbedding, SpecificSituationEmbedding,
        environment::EnvironmentEmbedding, event::EventEmbedding, location::LocationEmbedding,
        participant::ParticipantEmbedding,
    },
};

use crate::query::string_distance::compute_note_string_score_weighted;

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
            .chain(location_score)
            .chain(participants_score)
            .chain(environment_score)
            .chain(event_score)
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
            .chain(narrative_score)
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
    #[hotpath::measure]
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
                    .iter()
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
                    .iter()
                    .map(|q_sit_unit| sit.anonymous_compute(q_sit_unit))
                    .collect::<Result<Vec<_>, _>>()?;
                //按单元数归一化，与上面的 Semantic 分支保持一致。
                //此前该分支直接 sum，多单元情境查询得分可达 N×1.0，突破 [0,1] 量纲，
                //破坏 default_pipeline 依赖的"相似度与联想两路分数同量纲、可直接比较"契约。
                if score_vec.is_empty() {
                    return Ok(0.0);
                }
                let len = score_vec.len();
                Ok(score_vec.into_iter().sum::<f32>() / len as f32)
            }
            (_, _) => Ok(0.0),
        }
    }
}

impl AnonymousQueryCompute for MemoryEmbedding {
    type Query = MemoryRetrieveQueryEmbedding;
    #[hotpath::measure]
    fn anonymous_compute(&self, query: &Self::Query) -> EmbeddingCalcResult<f32> {
        // tag 通道缺失（任一侧无 tag，零向量占位）时，不把缺失通道当 0 分参与加权，
        // 否则 Situation 等无 tag 场景的分数会被压缩到 0.4×0+0.6×variant，理论最高仅 0.6。
        // 零向量标记由单遍融合 cosine 顺带给出，避免对 tag 再做两遍 is_zero 全扫。
        let (tag_score, tag_zero) = self.tag().cosine_similarity_and_zero(query.tag())?;
        let variant_score = self.variant().anonymous_compute(query.variant())?;
        if tag_zero {
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
    #[hotpath::measure]
    pub fn compute_fused(
        &self,
        query: &EmbeddedMemoryRetrieveQuery,
        string_blend_alpha: f32,
    ) -> EmbeddingCalcResult<QueryComputeResult> {
        let embedding_score = self.embedding().anonymous_compute(&query.embedding)?;
        // 字符串通道与 embedding 通道共用查询自带的同一份 BlendWeights：
        // 事件子字段（initiator/target/action）的加权必须两侧同源，
        // 否则 with_weights() 定制的权重只影响 embedding 侧，字符串侧静默沿用默认值。
        let string_score = compute_note_string_score_weighted(
            self.note(),
            &query.query,
            &query.embedding.blend_weights,
        );
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
mod tests;
