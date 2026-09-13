use soul_mem_core::memory_note::situation_mem::AbstractSituation;
use soul_mem_core::memory_note::{MemoryNote, MemoryType, sem_mem::SemMemory};

use crate::embedding::blend_weights::BlendWeights;
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
///
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

/// 使用 [`BlendWeights::default()`] 的便捷入口。
///
/// 等价于 `compute_note_string_score_weighted(note, query, &BlendWeights::default())`。
/// 评分链路上请改用后者并传入查询自带的权重（`query.embedding.blend_weights`），
/// 否则 `with_weights()` 定制的子字段权重不会作用到字符串通道。
pub fn compute_note_string_score(note: &MemoryNote, query: &MemoryRetrieveQuery) -> f32 {
    compute_note_string_score_weighted(note, query, &BlendWeights::default())
}

/// 计算一条记忆笔记相对查询的"精确标识符字符串匹配"得分，显式接收权重集。
///
/// 仅对以下精确标识符字段计算字符串距离（与 embedding 余弦相似度保持同一 [0,1] 量纲）：
///   - Semantic 的 `concept_identifier` vs 记忆的 `content` / `aliases`
///   - AbstractSituation 的 `Location.name` / `Participant.name` / `Environment.atmosphere` / `Event`(action/initiator/target)
///
/// 描述性字段（`role`、`tone`、`description`、`narrative` 等）不做字符串比较，
/// 以免多词描述导致得分系统性偏低、破坏与 embedding 得分的可比性。
///
/// 事件的 `action` / `initiator` / `target` 三路加权**直接取自 `weights`**
/// （`sit_event_*` 系列），与 embedding 通道同源，保证两侧同构。
///
/// 使用 max pooling 聚合：任一查询单元与任一目标字符串的最强命中即代表该笔记的字符串得分。
/// 变体不匹配（如 Semantic 记忆 vs Situation 查询）返回 0.0，与 embedding 侧行为一致。
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
        // content 不匹配，但 alias 完全命中
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
        // 部分重叠应落在 (0, 1)，而非 0 或 1，确保分数是连续可比的
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
        // Semantic 记忆 + Situation 查询：变体不匹配，返回 0，与 embedding 侧一致
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
        // 只比较 name，role 不参与（role 是描述性字段）
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
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("跑步".to_string())]),
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
                SituationQueryUnit::new().with_narrative("在一个酒馆里".to_string()),
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
                SemanticQueryUnit::new().with_concept_identifier("酒馆".to_string()),
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

    #[test]
    fn test_abstract_event_weights_are_exact() {
        // 事件加权公式的精确值验证。系数**取自 `BlendWeights`**，推导方式与 embedding 侧
        // `EventEmbedding::anonymous_compute` 逐字一致：
        //   双方都存在: w_i*initiator + w_t*target + w_a*action
        //   仅 initiator: (1 - w_ia)*initiator + w_ia*action
        //   仅 target:    (1 - w_ta)*target + w_ta*action
        //
        // 注意 `1.0 - w` 在 f32 下不精确（1.0 - 0.6 != 0.4），因此期望值必须与实现同式推导，
        // 不能写成字面量 0.4 —— 那正是旧版把字符串侧钉在一个 embedding 侧从未产生过的数值上的原因。
        // 使用字形部分重叠的字符串使各分量落在 (0,1) 区间，从而区分 * 与 /、+ 与 -。
        let w = BlendWeights::default();
        let note = situation_note(AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        }));

        // initiator+target 命中，action 不相关
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

        // 仅 initiator 命中，action 不相关
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

        // 仅 target 命中（部分匹配），action 也部分匹配 → 公式重构验证
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

        // 无 initiator/target，action 命中
        let q_action = MemoryRetrieveQuery::new(
            vec![],
            MemoryRetrieveQueryVariant::Situation(vec![
                SituationQueryUnit::new().with_event(vec![EventQueryUnit::new("跑步".to_string())]),
            ]),
        );
        assert_eq!(compute_note_string_score(&note, &q_action), 1.0);

        // 全部命中
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

    /// 回归测试：事件字符串评分的三路加权必须来自传入的 `BlendWeights`，
    /// 而不是写死在函数体里的字面量。
    ///
    /// 手法：把权重全部压到单个分量上，评分就应**恰好**等于该分量的字符串分。
    /// 若实现里仍残留 `0.3/0.3/0.4` 硬编码，断言必然失败。
    ///
    /// 注意 `test_abstract_event_weights_are_exact` **无法**发现此问题——
    /// 它把同一组字面量（0.3/0.3/0.4、0.4/0.6）抄成了"期望值"，
    /// 等于用实现验证实现；只有**改变权重**才能区分"读权重"与"读常量"。
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

        // 权重全给 action → 评分恰好等于 action 分
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

        // 权重全给 target → 评分恰好等于 target 分
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

        // 默认权重下与"显式引用默认值"的加权一致（不再硬编码 0.3/0.3/0.4）
        let default = BlendWeights::default();
        let expected_default = default.sit_event_initiator * initiator_score
            + default.sit_event_target * target_score
            + default.sit_event_action * action_score;
        assert_eq!(
            compute_note_string_score_weighted(&note, &query, &default),
            expected_default
        );

        // 仅 initiator 的退化路径同样走权重
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

        // 便捷入口等价于默认权重
        assert_eq!(
            compute_note_string_score(&note, &query),
            compute_note_string_score_weighted(&note, &query, &BlendWeights::default())
        );
    }

    /// 回归测试：`with_weights()` 定制的权重必须能到达字符串通道。
    ///
    /// 此前 `MemoryRetrieveQueryEmbedding` 只保存 `tag_weight` / `variant_weight` /
    /// `string_blend_alpha` 三个标量，完整权重集无处可取，字符串侧只能硬编码，
    /// 导致 `with_weights()` 只对 embedding 侧生效、字符串侧静默沿用默认值。
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

        // 完整权重集必须原样保留在查询上
        assert_eq!(embedding.blend_weights, custom);

        // 复现 compute_fused 的取权重方式：字符串通道读到的是定制权重而非默认值
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

        // 与默认权重下的结果必须不同，证明定制的权重确实生效
        assert_ne!(
            compute_note_string_score_weighted(&note, &query, &embedding.blend_weights),
            compute_note_string_score(&note, &query)
        );
    }
}
