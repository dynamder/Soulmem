use super::*;
use crate::embedding::Embeddable;
use crate::embedding::EmbeddingVec;
use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::embedding_model::bge::BgeSmallZh;
use crate::embedding::note::{MemoryEmbedding, MemoryEmbeddingVariant};
use crate::embedding::query::note::MemoryRetrieveQueryVariantEmbedding;
use crate::embedding::query::note::{EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding};
use crate::embedding::query::sem::SemanticQueryUnitEmbedding;
use crate::embedding::query::situation::SituationQueryUnitEmbedding;
use crate::embedding::query::situation::environment::EnvironmentQueryUnitEmbedding;
use crate::embedding::query::situation::event::EventQueryUnitEmbedding;
use crate::embedding::query::situation::location::LocationQueryUnitEmbedding;
use crate::embedding::query::situation::participant::ParticipantQueryUnitEmbedding;
use crate::embedding::sem::SemanticEmbedding;
use crate::embedding::situation::context::ContextEmbedding;
use crate::embedding::situation::environment::EnvironmentEmbedding;
use crate::embedding::situation::event::EventEmbedding;
use crate::embedding::situation::location::LocationEmbedding;
use crate::embedding::situation::participant::ParticipantEmbedding;
use crate::embedding::situation::{AbstractSituationEmbedding, SpecificSituationEmbedding};
use crate::query::retrieve::{
    EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
    MemoryRetrieveQueryVariant, ParticipantQueryUnit, SemanticQueryUnit, SituationQueryUnit,
};
use crate::query::string_distance::compute_note_string_score;
use crate::query::string_distance::compute_note_string_score_weighted;
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
    let narrative_query =
        SituationQueryUnit::new().with_narrative("享受战斗时的愉快氛围而不是单纯厮杀".to_string());
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

fn embed_sem_query(model: &BgeSmallZh, concept_identifier: &str) -> EmbeddedMemoryRetrieveQuery {
    let query = MemoryRetrieveQuery::new(
        vec![],
        MemoryRetrieveQueryVariant::Semantic(vec![
            SemanticQueryUnit::new().with_concept_identifier(concept_identifier.to_string()),
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
    let str_score = compute_note_string_score(embedded_note.note(), &embedded_query.query);
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
    assert!(
        fused >= pure,
        "string channel must not drag fused below pure"
    );
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
    let hit_str = compute_note_string_score(hit.note(), &embedded_query.query);
    let miss_str = compute_note_string_score(miss.note(), &embedded_query.query);
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
    let str_score = compute_note_string_score(embedded_note.note(), &embedded_query.query);
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
            SituationQueryUnit::new().with_location(vec![LocationQueryUnit::new("小酒馆")]),
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
    let str_score = compute_note_string_score(embedded_note.note(), &embedded_query.query);
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

/// 端到端回归测试：`compute_fused` 的字符串通道必须使用查询自带的
/// `embedding.blend_weights`，而不是默认权重/硬编码常量。
///
/// 此前 `MemoryRetrieveQueryEmbedding` 不保存完整权重集，`with_weights()` 只作用于
/// embedding 侧；字符串侧静默沿用 0.3/0.3/0.4，导致同一查询的两个通道权重不一致。
/// 手法：用查询自身的 `blend_weights` 复算 `compute_fused` 的混合公式，
/// 与实现结果精确比对；权重不生效时二者必然发散。
#[test]
fn test_compute_fused_string_channel_follows_custom_weights() {
    let model = BgeSmallZh::default_cpu().unwrap();
    let alpha = 0.6f32;

    // 事件记忆：initiator/target 与查询高度重叠，action 完全不相关。
    // 于是"权重压到 action"与"默认权重"在字符串通道上会产生显著不同的得分。
    let mem_type = MemoryType::Situation(
        AbstractSituation::Event(Event {
            action: "跑步".to_string(),
            action_intensity: 0.5,
            initiator: "张三".to_string(),
            target: "操场".to_string(),
        })
        .into(),
    );
    let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
    let embedding = note.embed(&model).unwrap();
    let embedded_note = EmbeddedMemoryNote { note, embedding };

    let query = MemoryRetrieveQuery::new(
        vec![],
        MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new().with_event(
            vec![EventQueryUnit::new("无关")
                    .with_initiator("张三丰".to_string())
                    .with_target("操场".to_string())],
        )]),
    );

    // 定制权重：全部压到 action → 字符串通道应退化为 action 分（≈0）
    let custom = BlendWeights {
        sit_event_initiator: 0.0,
        sit_event_target: 0.0,
        sit_event_action: 1.0,
        ..BlendWeights::default()
    };
    let query_embedding = query.embed(&model).unwrap().with_weights(custom);
    assert_eq!(
        query_embedding.blend_weights.sit_event_action, 1.0,
        "with_weights() 未把权重保存在查询上"
    );
    let embedded_query = EmbeddedMemoryRetrieveQuery {
        embedding: query_embedding,
        query,
    };

    // 用查询自身的权重复算期望值（与 compute_fused 同一取权重方式）
    let pure = embedded_note
        .anonymous_compute(&embedded_query.embedding)
        .unwrap();
    let str_score = compute_note_string_score_weighted(
        embedded_note.note(),
        &embedded_query.query,
        &embedded_query.embedding.blend_weights,
    );
    let expected = if str_score <= 0.0 {
        pure
    } else {
        pure.max(alpha * pure + (1.0 - alpha) * str_score)
    };
    let fused = embedded_note
        .compute_fused(&embedded_query, alpha)
        .unwrap()
        .score;
    assert!(
        (fused - expected).abs() < 1e-5,
        "compute_fused={fused} 与按查询权重复算的期望={expected} 不一致（字符串通道未采用定制权重）"
    );

    // 定制权重确实改变了字符串通道（证明上面的比对不是恒等式）
    let default_str = compute_note_string_score(embedded_note.note(), &embedded_query.query);
    assert!(
        (str_score - default_str).abs() > 0.1,
        "定制权重未改变字符串得分：custom={str_score} default={default_str}"
    );
}

#[test]
fn test_compute_fused_variant_mismatch_degrades_to_embedding() {
    let model = BgeSmallZh::default_cpu().unwrap();
    // Semantic 记忆 + Situation 查询：字符串分=0，混合分退化为 0.6 * embedding 分
    let embedded_note = embed_sem_note(&model, "战斗", &[]);
    let query = MemoryRetrieveQuery::new(
        vec![],
        MemoryRetrieveQueryVariant::Situation(vec![
            SituationQueryUnit::new().with_narrative("战斗场景".to_string()),
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
        compute_note_string_score(embedded_note.note(), &embedded_query.query),
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

/// 回归：情境变体必须按**单元数归一化**，与 Semantic 分支保持一致。
///
/// 此前该分支直接 `sum`，多单元情境查询得分可达 N×1.0，突破 [0,1] 量纲，
/// 破坏 `default_pipeline` 依赖的"相似度与联想两路分数同量纲、可直接比较"契约
/// （也让 `similarity` 的"最低兜底分"语义失效）。
/// 现有测试全是单单元查询（sum/1 == sum），所以这个边界从未被覆盖。
#[test]
fn test_situation_variant_normalizes_by_unit_count() {
    let note_variant = MemoryEmbeddingVariant::Situation(SituationEmbedding::Abstract(
        AbstractSituationEmbedding::Location(LocationEmbedding::test_new(unit(), unit())),
    ));

    // 每个单元对 Location.name 都完全命中 → 单单元得分恒为 1.0
    let mk_unit = || {
        SituationQueryUnitEmbedding::test_new(
            None,
            Some(LocationQueryUnitEmbedding::test_new(
                at(1.0),
                None,
                BlendWeights::default(),
            )),
            None,
            None,
            None,
            BlendWeights::default(),
        )
    };
    let query =
        MemoryRetrieveQueryVariantEmbedding::Situation(vec![mk_unit(), mk_unit(), mk_unit()]);

    let score = note_variant.anonymous_compute(&query).unwrap();
    assert!(
        (0.0..=1.0).contains(&score),
        "多单元情境查询得分越界（应按单元数归一化为均值）: {score}"
    );
    assert_close(score, 1.0);
}

/// 回归：**所有**子通道的加权都必须取自传入的 `BlendWeights`，而不是实现内的字面量。
///
/// 此前每个通道测试都用 `BlendWeights::default()` 构造查询，期望值又是"按默认权重手算出的
/// 字面量"（如 `0.62`），因此**无法区分"读配置"与"读常量"**——把任何一处实现改成硬编码，
/// 这些测试依旧全绿。`BlendWeights` 的 16 个字段里有 9 个从未在任何测试中被赋过非默认值。
///
/// 手法：把权重全部压到单个子字段上，得分就该**恰好**等于该子字段的相似度。
/// 只有"改变权重并断言结果随之改变"才能区分"读权重"与"读常量"。
#[test]
fn test_channel_weights_are_read_from_config() {
    // —— Location：sit_location_name / sit_location_coord ——
    let loc = LocationEmbedding::test_new(unit(), unit());
    let w_name = BlendWeights {
        sit_location_name: 1.0,
        sit_location_coord: 0.0,
        ..BlendWeights::default()
    };
    assert_close(
        loc.anonymous_compute(&LocationQueryUnitEmbedding::test_new(
            at(0.5),
            Some(at(0.8)),
            w_name,
        ))
        .unwrap(),
        0.5,
    );
    let w_coord = BlendWeights {
        sit_location_name: 0.0,
        sit_location_coord: 1.0,
        ..BlendWeights::default()
    };
    assert_close(
        loc.anonymous_compute(&LocationQueryUnitEmbedding::test_new(
            at(0.5),
            Some(at(0.8)),
            w_coord,
        ))
        .unwrap(),
        0.8,
    );

    // —— Participant：sit_participant_name / sit_participant_role ——
    let part = ParticipantEmbedding::test_new(unit(), unit(), unit());
    let w_name = BlendWeights {
        sit_participant_name: 1.0,
        sit_participant_role: 0.0,
        ..BlendWeights::default()
    };
    assert_close(
        part.anonymous_compute(&ParticipantQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            w_name,
        ))
        .unwrap(),
        0.5,
    );
    let w_role = BlendWeights {
        sit_participant_name: 0.0,
        sit_participant_role: 1.0,
        ..BlendWeights::default()
    };
    assert_close(
        part.anonymous_compute(&ParticipantQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            w_role,
        ))
        .unwrap(),
        0.8,
    );

    // —— Environment：sit_env_atmosphere / sit_env_tone ——
    let env = EnvironmentEmbedding::test_new(unit(), unit());
    let w_atm = BlendWeights {
        sit_env_atmosphere: 1.0,
        sit_env_tone: 0.0,
        ..BlendWeights::default()
    };
    assert_close(
        env.anonymous_compute(&EnvironmentQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            w_atm,
        ))
        .unwrap(),
        0.5,
    );
    let w_tone = BlendWeights {
        sit_env_atmosphere: 0.0,
        sit_env_tone: 1.0,
        ..BlendWeights::default()
    };
    assert_close(
        env.anonymous_compute(&EnvironmentQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            w_tone,
        ))
        .unwrap(),
        0.8,
    );

    // —— Semantic：sem_concept / sem_description ——
    let sem = SemanticEmbedding::new(unit(), unit(), unit());
    let w_concept = BlendWeights {
        sem_concept: 1.0,
        sem_description: 0.0,
        ..BlendWeights::default()
    };
    assert_close(
        sem.anonymous_compute(&SemanticQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            w_concept,
        ))
        .unwrap(),
        0.5,
    );
    let w_desc = BlendWeights {
        sem_concept: 0.0,
        sem_description: 1.0,
        ..BlendWeights::default()
    };
    assert_close(
        sem.anonymous_compute(&SemanticQueryUnitEmbedding::test_new(
            Some(at(0.5)),
            Some(at(0.8)),
            w_desc,
        ))
        .unwrap(),
        0.8,
    );

    // —— Event 退化路径：sit_event_target_only_action ——
    // 权重全给 action ⇒ 仅 target 的查询得分恰好等于 action 分（0.9）
    let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
    let w_target_only = BlendWeights {
        sit_event_target_only_action: 1.0,
        ..BlendWeights::default()
    };
    assert_close(
        event
            .anonymous_compute(&EventQueryUnitEmbedding::test_new(
                at(0.9),
                None,
                Some(at(0.8)),
                w_target_only,
            ))
            .unwrap(),
        0.9,
    );
}

#[test]
fn test_location_anonymous_compute_with_coordinates() {
    let loc = LocationEmbedding::test_new(unit(), unit());
    let query =
        LocationQueryUnitEmbedding::test_new(at(0.5), Some(at(0.8)), BlendWeights::default());
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
    let query =
        ParticipantQueryUnitEmbedding::test_new(Some(at(0.5)), None, BlendWeights::default());
    let score = participant.anonymous_compute(&query).unwrap();
    assert_close(score, 0.5);
}

#[test]
fn test_participant_anonymous_compute_role_only() {
    let participant = ParticipantEmbedding::test_new(unit(), unit(), unit());
    let query =
        ParticipantQueryUnitEmbedding::test_new(None, Some(at(0.8)), BlendWeights::default());
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
    let query =
        EventQueryUnitEmbedding::test_new(at(0.9), Some(at(0.5)), None, BlendWeights::default());
    let score = event.anonymous_compute(&query).unwrap();
    // a_w = 0.6, i_w = 0.4: 0.4*0.5 + 0.6*0.9 = 0.74
    assert_close(score, 0.74);
}

#[test]
fn test_event_anonymous_compute_target_only() {
    let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
    let query =
        EventQueryUnitEmbedding::test_new(at(0.9), None, Some(at(0.8)), BlendWeights::default());
    let score = event.anonymous_compute(&query).unwrap();
    // a_w = 0.6, t_w = 0.4: 0.4*0.8 + 0.6*0.9 = 0.86
    assert_close(score, 0.86);
}

#[test]
fn test_event_anonymous_compute_action_only() {
    let event = EventEmbedding::test_new(unit(), unit(), unit(), 0.5);
    let query = EventQueryUnitEmbedding::test_new(at(0.9), None, None, BlendWeights::default());
    let score = event.anonymous_compute(&query).unwrap();
    assert_close(score, 0.9);
}

#[test]
fn test_semantic_anonymous_compute_with_description() {
    let sem = SemanticEmbedding::new(unit(), unit(), unit());
    let query =
        SemanticQueryUnitEmbedding::test_new(Some(at(0.5)), Some(at(0.8)), BlendWeights::default());
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
    let abstract_emb =
        AbstractSituationEmbedding::Location(LocationEmbedding::test_new(unit(), unit()));
    let query = SituationQueryUnitEmbedding::test_new(
        None,
        None,
        None,
        None,
        None,
        BlendWeights::default(),
    );
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
    let abstract_emb =
        AbstractSituationEmbedding::Location(LocationEmbedding::test_new(unit(), unit()));
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
    let query_sem =
        SemanticQueryUnitEmbedding::test_new(Some(unit()), None, BlendWeights::default());
    let variant = MemoryEmbeddingVariant::Semantic(sem);
    let query_variant = MemoryRetrieveQueryVariantEmbedding::Semantic(vec![query_sem]);
    let score = variant.anonymous_compute(&query_variant).unwrap();
    assert_close(score, 1.0);
}

#[test]
fn test_memory_variant_semantic_average_multiple_units() {
    // 多单元语义查询：按单元数归一化取平均（/len 而非 *len）
    let sem = SemanticEmbedding::new(unit(), unit(), unit());
    let query_sem_1 =
        SemanticQueryUnitEmbedding::test_new(Some(at(0.5)), None, BlendWeights::default());
    let query_sem_2 =
        SemanticQueryUnitEmbedding::test_new(Some(at(0.9)), None, BlendWeights::default());
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
    let bw = BlendWeights {
        tag: 0.4,
        variant: 0.6,
        ..Default::default()
    };
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
    let q_sem = SemanticQueryUnitEmbedding::test_new(None, Some(unit()), BlendWeights::default());
    let q_variant = MemoryRetrieveQueryVariantEmbedding::Semantic(vec![q_sem]);

    // query 无 tag（零向量）→ 纯 variant 分，不再被 0.4 压缩
    let query_no_tag =
        MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(2)).with_variant(q_variant.clone());
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
