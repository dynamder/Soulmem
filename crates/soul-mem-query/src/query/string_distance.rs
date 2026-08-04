use soul_mem_core::memory_note::situation_mem::AbstractSituation;
use soul_mem_core::memory_note::{sem_mem::SemMemory, MemoryNote, MemoryType};

use crate::query::retrieve::{MemoryRetrieveQuery, SemanticQueryUnit, SituationQueryUnit};

/// Jaro-Winkler 字符串相似度，范围 [0, 1]。
/// 注意：对短串（≤4字符）前缀/中间插入会因匹配窗口退化为0而得到 0 分。
pub fn jaro_winkler_score(a: &str, b: &str) -> f32 {
    strsim::jaro_winkler(a, b) as f32
}

/// 归一化 Levenshtein 相似度，范围 [0, 1]，1.0 表示完全一致。
/// 相比 Jaro-Winkler，能正确捕捉插入/删除造成的错位。
pub fn normalized_levenshtein_score(a: &str, b: &str) -> f32 {
    strsim::normalized_levenshtein(a, b) as f32
}

/// 字符串距离综合得分，范围 [0, 1]，与 embedding 余弦相似度同量纲，可直接线性混合。
///
/// 取 `max(Jaro-Winkler, normalized Levenshtein)`：
///   - 后缀插入（`"图书"` vs `"图书馆"`）、前缀插入（`"酒馆"` vs `"小酒馆"`）由 Levenshtein 兜底；
///   - 前缀加成与整体字形贴近程度由 Jaro-Winkler 主导。
/// 空串双方均空时视为完全一致；仅一方为空时视为无重叠（0.0）。
pub fn string_distance_score(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    jaro_winkler_score(a, b).max(normalized_levenshtein_score(a, b))
}

/// 计算一条记忆笔记相对查询的"精确标识符字符串匹配"得分。
///
/// 仅对以下精确标识符字段计算字符串距离（与 embedding 余弦相似度保持同一 [0,1] 量纲）：
///   - Semantic 的 `concept_identifier` vs 记忆的 `content` / `aliases`
///   - AbstractSituation 的 `Location.name` / `Participant.name` / `Environment.atmosphere` / `Event`(action/initiator/target)
///
/// 描述性字段（`role`、`tone`、`description`、`narrative` 等）不做字符串比较，
/// 以免多词描述导致得分系统性偏低、破坏与 embedding 得分的可比性。
///
/// 使用 max pooling 聚合：任一查询单元与任一目标字符串的最强命中即代表该笔记的字符串得分。
/// 变体不匹配（如 Semantic 记忆 vs Situation 查询）返回 0.0，与 embedding 侧行为一致。
pub fn compute_note_string_score(note: &MemoryNote, query: &MemoryRetrieveQuery) -> f32 {
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
        ) => abstract_sit_string_score(abs, units),
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
/// 各子字段的加权沿用 `BlendWeights` 中已定义的结构化权重，确保子字段混合与 embedding 侧同构。
fn abstract_sit_string_score(abs: &AbstractSituation, units: &[SituationQueryUnit]) -> f32 {
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
                        0.3 * initiator_score + 0.3 * target_score + 0.4 * action_score
                    }
                    (Some(_), None) => 0.4 * initiator_score + 0.6 * action_score,
                    (None, Some(_)) => 0.4 * target_score + 0.6 * action_score,
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
    use soul_mem_core::memory_note::{sem_mem::ConceptType, MemoryNoteBuilder};

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
        // Jaro-Winkler 对短串前缀插入得 0，归一化 Levenshtein 兜底
        let s = string_distance_score("酒馆", "小酒馆");
        assert!(
            s > 0.5,
            "prefix insertion should be rescued by Levenshtein, got {s}"
        );
        assert!(s < 1.0, "prefix insertion is not identical, got {s}");
    }

    #[test]
    fn test_string_distance_score_suffix_insertion() {
        // 后缀插入由 Jaro-Winkler 主导
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
        // 双方均为空视为一致；仅一方为空视为无重叠
        assert_eq!(string_distance_score("", ""), 1.0);
        assert_eq!(string_distance_score("", "酒馆"), 0.0);
        assert_eq!(string_distance_score("酒馆", ""), 0.0);
    }

    #[test]
    fn test_string_distance_score_dominates_jaro_on_shift() {
        // 综合得分不得低于任一单独指标
        for (a, b) in [("小酒馆", "酒馆"), ("张三丰", "张三"), ("战斗", "战斗")] {
            let combined = string_distance_score(a, b);
            assert!(combined >= jaro_winkler_score(a, b));
            assert!(combined >= normalized_levenshtein_score(a, b));
        }
    }

    #[test]
    fn test_semantic_exact_identifier_scores_higher() {
        // 同一 query 下：content 与 concept_identifier 字形一致应显著高于语义相关但字形不同的
        let note_hit = sem_note("战斗", &[]);
        let note_miss = sem_note("冲突", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("战斗".to_string())
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
        // content 不匹配，但 alias 完全命中
        let note = sem_note("Rust编程语言", &["Rust"]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("Rust".to_string())
            ]),
        );
        let score = compute_note_string_score(&note, &query);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_semantic_partial_overlap_in_unit_range() {
        // 部分重叠应落在 (0, 1)，而非 0 或 1，确保分数是连续可比的
        let note = sem_note("图书馆", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("图书".to_string())
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
                SemanticQueryUnit::new().with_description("关于战争的描述".to_string())
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 0.0);
    }

    #[test]
    fn test_semantic_variant_mismatch_zero() {
        // Semantic 记忆 + Situation 查询：变体不匹配，返回 0，与 embedding 侧一致
        let note = sem_note("战斗", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_narrative("战斗场景".to_string())
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
                SituationQueryUnit::new().with_location(vec![LocationQueryUnit::new("小酒馆")])
            ]),
        );
        let score = compute_note_string_score(&note, &query);
        assert!(score > 0.5, "location partial match too low: {score}");
    }

    #[test]
    fn test_abstract_participant_name_only() {
        // 只比较 name，role 不参与（role 是描述性字段）
        let note = situation_note(AbstractSituation::Participant(Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new()
                .with_participants(vec![
                    ParticipantQueryUnit::new().with_name("张三".to_string())
                ])]),
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
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new()
                .with_environment(
                    EnvironmentQueryUnit::new().with_atmosphere("安静".to_string()),
                )]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 1.0);
    }

    #[test]
    fn test_abstract_event_action_weighting() {
        // 仅 action 命中：得分为纯 action 分数
        let note = situation_note(AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("跑步".to_string())])
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 1.0);

        // action + initiator 均命中：加权混合仍应给出合理值
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
        // Location 记忆 + 无 location 查询单元：无法匹配 → 0
        let note = situation_note(AbstractSituation::Location(Location {
            name: "酒馆".to_string(),
            coordinates: String::new(),
        }));
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_narrative("在一个酒馆里".to_string())
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &query), 0.0);
    }

    #[test]
    fn test_robustness_empty_inputs() {
        // 空字符串 / 空单元列表不应 panic 或产生 NaN/Inf
        let note = sem_note("", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("".to_string())
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
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("".to_string())])
            ]),
        );
        let sit_score = compute_note_string_score(&sit_note, &sit_query);
        assert!(sit_score.is_finite() && (0.0..=1.0).contains(&sit_score));
    }

    #[test]
    fn test_multiple_units_max_pooling() {
        // 多单元查询：一个单元完全命中应驱动字符串得分，而不是被不匹配单元平均稀释
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
        // 数量级一致性验证：字符串得分与 embedding 余弦相似度同处 [0,1]，
        // 且对"字形相近"的精确标识符命中应不低于 embedding 分数，保证混合后不劣化排序。
        let note = sem_note("小酒馆", &[]);
        let query = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier("酒馆".to_string())
            ]),
        );
        let str_score = compute_note_string_score(&note, &query);
        // 0.6*emb + 0.4*str 中，即便 emb 为 0，str 也能提供 ≥ 0.4*str_score 的兜底分
        let blended_floor = 0.4 * str_score;
        assert!(str_score > 0.5, "str_score too low: {str_score}");
        assert!(
            blended_floor > 0.2,
            "blended floor too low: {blended_floor}"
        );
        // 字符串得分严格在 embedding 量纲 [0,1] 内
        assert!((0.0..=1.0).contains(&str_score));
    }
}
