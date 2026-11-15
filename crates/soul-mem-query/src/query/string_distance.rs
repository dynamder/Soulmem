use soul_mem_core::memory_note::situation_mem::AbstractSituation;
use soul_mem_core::memory_note::{MemoryNote, MemoryType, sem_mem::SemMemory};

use crate::embedding::blend_weights::BlendWeights;
use crate::query::retrieve::{MemoryRetrieveQuery, SemanticQueryUnit, SituationQueryUnit};

/// Jaro-Winkler 字符串相似度，范围 [0, 1]；短串（≤4 字符）的插入可能退化为 0。
pub fn jaro_winkler_score(a: &str, b: &str) -> f32 {
    strsim::jaro_winkler(a, b) as f32
}

/// 归一化 Levenshtein 相似度，范围 [0, 1]；能捕捉插入/删除造成的错位。
pub fn normalized_levenshtein_score(a: &str, b: &str) -> f32 {
    strsim::normalized_levenshtein(a, b) as f32
}

/// 取 `max(Jaro-Winkler, normalized Levenshtein)`；双方为空视为一致，仅一方为空视为 0。
pub fn string_distance_score(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    jaro_winkler_score(a, b).max(normalized_levenshtein_score(a, b))
}

/// 使用默认权重的便捷入口。评分链路请用 [`compute_note_string_score_weighted`]，
/// 否则 `with_weights()` 定制的子字段权重不会作用到字符串通道。
pub fn compute_note_string_score(note: &MemoryNote, query: &MemoryRetrieveQuery) -> f32 {
    compute_note_string_score_weighted(note, query, &BlendWeights::default())
}

/// 只用精确标识符计算字符串分；描述性字段（`role`/`tone`/`description`/`narrative`）不参与。
///
/// 事件三路加权直接取 `weights`，与 embedding 通道同源。max pooling 聚合，变体不匹配返回 0.0。
#[hotpath::measure]
pub fn compute_note_string_score_weighted(
    note: &MemoryNote,
    query: &MemoryRetrieveQuery,
    weights: &BlendWeights,
) -> f32 {
    match (note.mem_type(), query.variant()) {
        (
            MemoryType::Semantic(sem),
            crate::query::retrieve::MemoryRetrieveQueryVariant::Semantic(units),
        ) => semantic_string_score(sem, units),
        (
            MemoryType::Situation(
                soul_mem_core::memory_note::situation_mem::SituationType::AbstractSituation(abs),
            ),
            crate::query::retrieve::MemoryRetrieveQueryVariant::Situation(units),
        ) => abstract_sit_string_score(abs, units, weights),
        _ => 0.0,
    }
}

/// Semantic 记忆字符串评分：`concept_identifier` 对 `content` 与每个 `alias` 做字符串距离，
/// 取 max pooling（与 embedding 侧 content/aliases 的 max_pooling 语义一致）。
fn semantic_string_score(sem: &SemMemory, units: &[SemanticQueryUnit]) -> f32 {
    units
        .iter()
        .filter_map(|unit| unit.concept_identifier())
        .map(|q_concept| {
            let content_score = string_distance_score(q_concept, &sem.content);
            let alias_score = sem
                .aliases
                .iter()
                .map(|alias| string_distance_score(q_concept, alias))
                .fold(0.0f32, f32::max);
            content_score.max(alias_score)
        })
        .fold(0.0f32, f32::max)
}

/// 抽象情境字符串评分：按变体类型匹配对应的精确标识符字段。
/// 事件的子字段加权取自 `weights`（`BlendWeights`），确保子字段混合与 embedding 侧同构。
fn abstract_sit_string_score(
    abs: &AbstractSituation,
    units: &[SituationQueryUnit],
    weights: &BlendWeights,
) -> f32 {
    match abs {
        AbstractSituation::Location(loc) => units
            .iter()
            .flat_map(|u| u.location().into_iter().flatten())
            .map(|q_loc| string_distance_score(q_loc.name(), &loc.name))
            .fold(0.0f32, f32::max),
        AbstractSituation::Participant(participant) => units
            .iter()
            .flat_map(|u| u.participants().into_iter().flatten())
            .filter_map(|q_p| q_p.name())
            .map(|q_name| string_distance_score(q_name, &participant.name))
            .fold(0.0f32, f32::max),
        AbstractSituation::Environment(env) => units
            .iter()
            .filter_map(|u| u.environment())
            .filter_map(|q_env| q_env.atmosphere())
            .map(|q_atm| string_distance_score(q_atm, &env.atmosphere))
            .fold(0.0f32, f32::max),
        AbstractSituation::Event(evt) => units
            .iter()
            .flat_map(|u| u.event().into_iter().flatten())
            .map(|q_evt| {
                let action_score = string_distance_score(q_evt.action(), &evt.action);
                let initiator_score = q_evt
                    .initiator()
                    .map(|i| string_distance_score(i, &evt.initiator))
                    .unwrap_or(0.0);
                let target_score = q_evt
                    .target()
                    .map(|t| string_distance_score(t, &evt.target))
                    .unwrap_or(0.0);
                match (q_evt.initiator(), q_evt.target()) {
                    (Some(_), Some(_)) => {
                        weights.sit_event_initiator * initiator_score
                            + weights.sit_event_target * target_score
                            + weights.sit_event_action * action_score
                    }
                    (Some(_), None) => {
                        (1.0 - weights.sit_event_initiator_only_action) * initiator_score
                            + weights.sit_event_initiator_only_action * action_score
                    }
                    (None, Some(_)) => {
                        (1.0 - weights.sit_event_target_only_action) * target_score
                            + weights.sit_event_target_only_action * action_score
                    }
                    (None, None) => action_score,
                }
            })
            .fold(0.0f32, f32::max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::retrieve::{
        EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
        MemoryRetrieveQueryVariant, ParticipantQueryUnit, SemanticQueryUnit, SituationQueryUnit,
    };
    use soul_mem_core::memory_note::situation_mem::{
        AbstractSituation, Environment, Event, Location, Participant,
    };
    use soul_mem_core::memory_note::{MemoryNoteBuilder, sem_mem::ConceptType};

    use crate::embedding::EmbeddingVec;
    use crate::embedding::query::note::MemoryRetrieveQueryEmbedding;

    fn sem_note(content: &str, aliases: &[&str]) -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: content.to_string(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            concept_type: ConceptType::Entity,
            description: String::new(),
        }))
        .build()
        .unwrap()
    }

    fn situation_note(abs: AbstractSituation) -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Situation(abs.into()))
            .build()
            .unwrap()
    }

    #[test]
    fn test_jaro_winkler_range() {
        assert!(jaro_winkler_score("", "") >= 0.0 && jaro_winkler_score("", "") <= 1.0);
        assert_eq!(jaro_winkler_score("Rust", "Rust"), 1.0);
        assert_eq!(jaro_winkler_score("Rust", "铁锈"), 0.0);
        for (a, b) in [("酒馆", "小酒馆"), ("张三", "张三丰"), ("战斗", "战斗")] {
            let s = jaro_winkler_score(a, b);
            assert!((0.0..=1.0).contains(&s), "{a} vs {b}: {s}");
        }
    }

    #[test]
    fn test_string_distance_score_prefix_insertion() {
        let s = string_distance_score("酒馆", "小酒馆");
        assert!(
            s > 0.5,
            "prefix insertion should be rescued by Levenshtein, got {s}"
        );
        assert!(s < 1.0, "prefix insertion is not identical, got {s}");
    }

    #[test]
    fn test_string_distance_score_suffix_insertion() {
        let s = string_distance_score("图书", "图书馆");
        assert!(s > 0.8, "suffix insertion got {s}");
    }

    #[test]
    fn test_string_distance_score_exact_and_disjoint() {
        assert_eq!(string_distance_score("战斗", "战斗"), 1.0);
        assert_eq!(string_distance_score("战斗", "冲突"), 0.0);
        assert_eq!(string_distance_score("Rust", "铁锈"), 0.0);
    }

    #[test]
    fn test_string_distance_score_empty_handling() {
        assert_eq!(string_distance_score("", ""), 1.0);
        assert_eq!(string_distance_score("", "酒馆"), 0.0);
        assert_eq!(string_distance_score("酒馆", ""), 0.0);
    }

    #[test]
    fn test_string_distance_score_dominates_jaro_on_shift() {
        for (a, b) in [("小酒馆", "酒馆"), ("张三丰", "张三"), ("战斗", "战斗")] {
            let combined = string_distance_score(a, b);
            assert!(combined >= jaro_winkler_score(a, b));
            assert!(combined >= normalized_levenshtein_score(a, b));
        }
    }

    #[test]
    fn test_semantic_exact_identifier_scores_higher() {
        let note_hit = sem_note("战斗", &[]);
        let note_miss = sem_note("冲突", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("战斗".to_string()),
            ]),
        );
        let hit = compute_note_string_score(&note_hit, &query);
        let miss = compute_note_string_score(&note_miss, &query);
        assert_eq!(hit, 1.0);
        assert_eq!(miss, 0.0);
        assert!(hit > miss);
    }

    #[test]
    fn test_semantic_alias_max_pooling() {
        let note = sem_note("Rust编程语言", &["Rust"]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("Rust".to_string()),
            ]),
        );
        let score = compute_note_string_score(&note, &query);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_semantic_partial_overlap_in_unit_range() {
        let note = sem_note("图书馆", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("图书".to_string()),
            ]),
        );
        let score = compute_note_string_score(&note, &query);
        assert!(score > 0.0 && score < 1.0, "partial overlap got {score}");
    }

    #[test]
    fn test_semantic_missing_concept_identifier_zero() {
        let note = sem_note("战斗", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_description("关于战争的描述".to_string()),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 0.0);
    }

    #[test]
    fn test_semantic_variant_mismatch_zero() {
        let note = sem_note("战斗", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_narrative("战斗场景".to_string()),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 0.0);
    }

    #[test]
    fn test_abstract_location_name_match() {
        let note = situation_note(AbstractSituation::Location(Location {
            name: "酒馆".to_string(),
            coordinates: String::new(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_location(vec![LocationQueryUnit::new("小酒馆")]),
            ]),
        );
        let score = compute_note_string_score(&note, &query);
        assert!(score > 0.5, "location partial match too low: {score}");
    }

    #[test]
    fn test_abstract_participant_name_only() {
        let note = situation_note(AbstractSituation::Participant(Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_participants(vec![
                    ParticipantQueryUnit::new().with_name("张三".to_string()),
                ]),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 1.0);
    }

    #[test]
    fn test_abstract_environment_atmosphere_only() {
        let note = situation_note(AbstractSituation::Environment(Environment {
            atmosphere: "安静".to_string(),
            tone: "温暖".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_environment(
                    EnvironmentQueryUnit::new().with_atmosphere("安静".to_string()),
                ),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 1.0);
    }

    #[test]
    fn test_abstract_event_action_weighting() {
        let note = situation_note(AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("跑步".to_string())]),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 1.0);

        let query2 = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("跑步".to_string()).with_initiator("张三".to_string())],
            )]),
        );
        let score2 = compute_note_string_score(&note, &query2);
        assert!(score2 >= 1.0 - 1e-6, "full event hit got {score2}");
    }

    #[test]
    fn test_abstract_situation_type_mismatch_zero() {
        let note = situation_note(AbstractSituation::Location(Location {
            name: "酒馆".to_string(),
            coordinates: String::new(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_narrative("在一个酒馆里".to_string()),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 0.0);
    }

    #[test]
    fn test_robustness_empty_inputs() {
        let note = sem_note("", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("".to_string()),
            ]),
        );
        let score = compute_note_string_score(&note, &query);
        assert!(score.is_finite() && (0.0..=1.0).contains(&score));

        let empty_query =
            MemoryRetrieveQuery::new(vec![], MemoryRetrieveQueryVariant::Semantic(vec![]));
        assert_eq!(compute_note_string_score(&note, &empty_query), 0.0);

        let sit_note = situation_note(AbstractSituation::Event(Event {
            action: String::new(),
            action_intensity: 0.0,
            initiator: String::new(),
            target: String::new(),
        }));
        let sit_query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("".to_string())]),
            ]),
        );
        let sit_score = compute_note_string_score(&sit_note, &sit_query);
        assert!(sit_score.is_finite() && (0.0..=1.0).contains(&sit_score));
    }

    #[test]
    fn test_multiple_units_max_pooling() {
        let note = sem_note("战斗", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("无关概念".to_string()),
                SemanticQueryUnit::new().with_concept_identifier("战斗".to_string()),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 1.0);
    }

    #[test]
    fn test_score_magnitude_consistency_with_embedding_scale() {
        let note = sem_note("小酒馆", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("酒馆".to_string()),
            ]),
        );
        let str_score = compute_note_string_score(&note, &query);
        let blended_floor = 0.4 * str_score;
        assert!(str_score > 0.5, "str_score too low: {str_score}");
        assert!(
            blended_floor > 0.2,
            "blended floor too low: {blended_floor}"
        );
        assert!((0.0..=1.0).contains(&str_score));
    }

    #[test]
    fn test_abstract_event_weights_are_exact() {
        let w = BlendWeights::default();
        let note = situation_note(AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        }));

        let q_both = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("无关")
                    .with_initiator("张三丰".to_string())
                    .with_target("操场".to_string())],
            )]),
        );
        let expected_both = w.sit_event_initiator * string_distance_score("张三丰", "张三")
            + w.sit_event_target * string_distance_score("操场", "操场")
            + w.sit_event_action * string_distance_score("无关", "跑步");
        assert_eq!(compute_note_string_score(&note, &q_both), expected_both);

        let q_initiator = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("无关").with_initiator("张三丰".to_string())],
            )]),
        );
        let a_w = w.sit_event_initiator_only_action;
        let expected_initiator = (1.0 - a_w) * string_distance_score("张三丰", "张三")
            + a_w * string_distance_score("无关", "跑步");
        assert_eq!(
            compute_note_string_score(&note, &q_initiator),
            expected_initiator
        );

        let q_target = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("跑").with_target("操".to_string())],
            )]),
        );
        let a_w = w.sit_event_target_only_action;
        let expected_target = (1.0 - a_w) * string_distance_score("操", "操场")
            + a_w * string_distance_score("跑", "跑步");
        assert_eq!(compute_note_string_score(&note, &q_target), expected_target);

        let q_action = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("跑步".to_string())]),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &q_action), 1.0);

        let q_full = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("跑步")
                    .with_initiator("张三".to_string())
                    .with_target("操场".to_string())],
            )]),
        );
        assert_eq!(compute_note_string_score(&note, &q_full), 1.0);
    }

    // 回归：事件三路加权必须来自传入的 `BlendWeights`，不能写死。
    #[test]
    fn test_abstract_event_weights_follow_blend_weights() {
        let note = situation_note(AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("无关")
                    .with_initiator("张三丰".to_string())
                    .with_target("操场".to_string())],
            )]),
        );

        let action_score = string_distance_score("无关", "跑步");
        let initiator_score = string_distance_score("张三丰", "张三");
        let target_score = string_distance_score("操场", "操场");

        let action_only = BlendWeights {
            sit_event_initiator: 0.0,
            sit_event_target: 0.0,
            sit_event_action: 1.0,
            ..BlendWeights::default()
        };
        assert_eq!(
            compute_note_string_score_weighted(&note, &query, &action_only),
            action_score,
            "事件加权未取自 BlendWeights（疑似仍为硬编码字面量）"
        );

        let target_only = BlendWeights {
            sit_event_initiator: 0.0,
            sit_event_target: 1.0,
            sit_event_action: 0.0,
            ..BlendWeights::default()
        };
        assert_eq!(
            compute_note_string_score_weighted(&note, &query, &target_only),
            target_score
        );

        let default = BlendWeights::default();
        let expected_default = default.sit_event_initiator * initiator_score
            + default.sit_event_target * target_score
            + default.sit_event_action * action_score;
        assert_eq!(
            compute_note_string_score_weighted(&note, &query, &default),
            expected_default
        );

        let q_initiator = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("无关").with_initiator("张三丰".to_string())],
            )]),
        );
        let initiator_only = BlendWeights {
            sit_event_initiator_only_action: 1.0,
            ..BlendWeights::default()
        };
        assert_eq!(
            compute_note_string_score_weighted(&note, &q_initiator, &initiator_only),
            action_score,
            "仅 initiator 时未按 sit_event_initiator_only_action 加权"
        );

        assert_eq!(
            compute_note_string_score(&note, &query),
            compute_note_string_score_weighted(&note, &query, &BlendWeights::default())
        );
    }

    // 回归：`with_weights()` 的定制权重必须到达字符串通道。
    #[test]
    fn test_query_embedding_carries_weights_into_string_channel() {
        let custom = BlendWeights {
            sit_event_initiator: 0.0,
            sit_event_target: 0.0,
            sit_event_action: 1.0,
            ..BlendWeights::default()
        };
        let embedding =
            MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(8)).with_weights(custom.clone());

        assert_eq!(embedding.blend_weights, custom);

        let note = situation_note(AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
                vec![EventQueryUnit::new("无关")
                    .with_initiator("张三丰".to_string())
                    .with_target("操场".to_string())],
            )]),
        );
        assert_eq!(
            compute_note_string_score_weighted(&note, &query, &embedding.blend_weights),
            string_distance_score("无关", "跑步"),
            "with_weights() 的权重未到达字符串通道"
        );

        assert_ne!(
            compute_note_string_score_weighted(&note, &query, &embedding.blend_weights),
            compute_note_string_score(&note, &query)
        );
    }
}
