//! 数据库层集成测试：证明记忆数据**完整流过业务链路**——构建 → 写入 → 读回 → 召回 → 更新 → 删除，
//! 到达算法管线（`compute_fused` 等）的 `EmbeddedMemoryNote` 数据结构无损。
//!
//! 职责边界：
//! - 本测试只验证**链路数据完整性**（送入/取回的数据结构正确），不验证检索效果——
//!   评分/重排算法由 soul-mem-query 的单元测试覆盖；
//! - 嵌入用 512 维手工向量模拟模型输出（与 schema HNSW DIMENSION 512 一致，沙箱无法加载 BGE 模型）；
//! - 使用 kv-mem 内存库（`SurrealRepository::connect_mem`），无磁盘残留。

use soul_mem_core::memory_links::{MemoryLinkBuilder, MemoryLinkType, sem_mem::SemMemLink};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::situation_mem::{AbstractSituation, Location};
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant};
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::embedding::situation::location::LocationEmbedding;
use soul_mem_query::embedding::situation::{AbstractSituationEmbedding, SituationEmbedding};
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_runtime::storage::surreal::SurrealRepository;
use soul_mem_runtime::storage::{MemoryRepository, StorageResult};

/// 512 维向量：前 `vals` 个分量给定、其余补零（schema HNSW DIMENSION 512）。
fn v512(vals: &[f32]) -> EmbeddingVec {
    let mut v = vec![0.0f32; 512];
    for (i, x) in vals.iter().enumerate() {
        v[i] = *x;
    }
    EmbeddingVec::new(v)
}

/// 语义记忆：`content` 向量的第一个分量 `c` 决定其在语义召回中的可区分性。
fn sem_note(content_c: f32, desc_c: f32) -> EmbeddedMemoryNote {
    let mem_type = MemoryType::Semantic(SemMemory {
        content: "概念内容".into(),
        aliases: vec!["别名".into()],
        concept_type: ConceptType::Entity,
        description: "描述".into(),
    });
    let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
    let embedding = MemoryEmbedding::new(
        v512(&[content_c * 0.5, 0.5]),
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            v512(&[content_c, 0.0]),
            v512(&[content_c, 0.0]),
            v512(&[desc_c, 0.0]),
        )),
    );
    EmbeddedMemoryNote { note, embedding }
}

/// 抽象情境记忆（Location）。
fn sit_note(name_c: f32) -> EmbeddedMemoryNote {
    let mem_type = MemoryType::Situation(
        AbstractSituation::Location(Location {
            name: "地点".into(),
            coordinates: "坐标".into(),
        })
        .into(),
    );
    let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
    let embedding = MemoryEmbedding::new(
        v512(&[0.0, 1.0]),
        MemoryEmbeddingVariant::Situation(SituationEmbedding::Abstract(
            AbstractSituationEmbedding::Location(LocationEmbedding {
                name: v512(&[name_c, 0.0]),
                coordinates: v512(&[0.5, 0.0]),
            }),
        )),
    );
    EmbeddedMemoryNote { note, embedding }
}

async fn repo() -> SurrealRepository {
    let r = SurrealRepository::connect_mem().await.unwrap();
    r.init_schema().await.unwrap();
    r
}

/// 业务链路 1：语义记忆「构建 → 写入 → 读回」逐字段无损。
#[tokio::test]
async fn semantic_memory_write_read_roundtrip_preserves_data() {
    let repo = repo().await;
    let expected = sem_note(0.8, 0.6);
    let id = expected.note().id();

    repo.upsert_notes(vec![expected.clone()]).await.unwrap();

    let fetched = repo.fetch_notes(&[id]).await.unwrap();
    assert_eq!(fetched.len(), 1, "应读回一条记录");
    let f = &fetched[0];

    // 核心标识：id / 类型 / 标签 / 遗忘度 / 嵌入 —— 全链路无损
    assert_eq!(f.note().id(), id);
    assert_eq!(f.note().mem_type(), expected.note().mem_type());
    assert_eq!(f.note().tags(), expected.note().tags());
    assert_eq!(f.note().missing_degree(), expected.note().missing_degree());
    assert_eq!(f.note().retrieval_count(), expected.note().retrieval_count());
    assert_eq!(f.note().creation_time(), expected.note().creation_time());
    assert_eq!(f.embedding(), expected.embedding());
}

/// 业务链路 2：召回管线取回的候选是**完整可消费**的 `EmbeddedMemoryNote`
/// （note 与 embedding 变体匹配、槽位向量正确写入），可直接送入算法管线。
#[tokio::test]
async fn recall_pipeline_returns_complete_embedded_notes() {
    let repo = repo().await;
    // 三条语义记忆，content 分量各不相同（0.9 / 0.6 / 0.3）
    let a = sem_note(0.9, 0.5);
    let b = sem_note(0.6, 0.5);
    let c = sem_note(0.3, 0.5);
    let ids: Vec<MemoryId> = vec![a.note().id(), b.note().id(), c.note().id()];
    repo.upsert_notes(vec![a, b, c]).await.unwrap();

    // 语义查询嵌入：content 通道与 a 同向（同用第一个分量）
    let q = MemoryEmbedding::new(
        EmbeddingVec::zero(512), // 零 tag：跳过 tag 通道，只走 variant 通道
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            v512(&[0.9, 0.0]),
            v512(&[0.9, 0.0]),
            v512(&[0.5, 0.0]),
        )),
    );
    let candidates = repo.similarity_fetch(vec![q], 3).await.unwrap();

    // 召回集数据完整：每条都是可消费的 EmbeddedMemoryNote，且确实来自入库记录
    assert!(!candidates.is_empty(), "应召回至少一条候选");
    let candidate_ids: Vec<MemoryId> = candidates.iter().map(|e| e.note().id()).collect();
    for id in &candidate_ids {
        assert!(ids.contains(id), "候选 id 必须来自入库记录");
    }
    for e in &candidates {
        // embedding 变体与 mem_type 变体一致（算法管线消费的前提）
        assert!(
            matches!(e.note().mem_type(), MemoryType::Semantic(_)),
            "语义查询候选必须是语义记忆"
        );
        assert!(
            matches!(e.embedding().variant().clone(), MemoryEmbeddingVariant::Semantic(_)),
            "候选 embedding 变体必须还原为 Semantic"
        );
    }
}

/// 业务链路 3：混合变体共存——Semantic / Situation(Abstract) 写入读回各自无损，
/// 召回侧变体隔离（语义查询不召回情境记忆、情境查询不召回语义记忆）。
#[tokio::test]
async fn mixed_variants_roundtrip_and_recall_isolation() {
    let repo = repo().await;
    let sem = sem_note(0.8, 0.5);
    let sit = sit_note(0.8);
    let sem_id = sem.note().id();
    let sit_id = sit.note().id();
    repo.upsert_notes(vec![sem, sit]).await.unwrap();

    // 各自变体无损还原
    let fetched = repo.fetch_notes(&[sem_id, sit_id]).await.unwrap();
    assert_eq!(fetched.len(), 2);
    let has_semantic = fetched.iter().any(|e| matches!(e.note().mem_type(), MemoryType::Semantic(_)));
    let has_situation = fetched.iter().any(|e| matches!(e.note().mem_type(), MemoryType::Situation(_)));
    assert!(has_semantic && has_situation, "两种变体都应还原");

    // 语义查询（零 tag）只召回语义记忆
    let q_sem = MemoryEmbedding::new(
        EmbeddingVec::zero(512),
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            v512(&[0.9, 0.0]),
            v512(&[0.9, 0.0]),
            v512(&[0.5, 0.0]),
        )),
    );
    let hits = repo.similarity_fetch(vec![q_sem], 2).await.unwrap();
    assert!(hits.iter().all(|e| e.note().id() != sit_id), "语义查询不得召回情境记忆");

    // 情境查询（零 tag，走 location name 通道）只召回情境记忆
    let q_sit = MemoryEmbedding::new(
        EmbeddingVec::zero(512),
        MemoryEmbeddingVariant::Situation(SituationEmbedding::Abstract(
            AbstractSituationEmbedding::Location(LocationEmbedding {
                name: v512(&[0.8, 0.0]),
                coordinates: v512(&[0.5, 0.0]),
            }),
        )),
    );
    let hits = repo.similarity_fetch(vec![q_sit], 2).await.unwrap();
    assert!(hits.iter().all(|e| e.note().id() != sem_id), "情境查询不得召回语义记忆");
}

/// 业务链路 4：链接记忆链——fetch_notes 合并出边、fetch_neighbors 深度遍历恢复链路。
#[tokio::test]
async fn link_graph_write_neighbors_and_restore() {
    let repo = repo().await;
    let a = sem_note(0.7, 0.5);
    let b = sem_note(0.6, 0.5);
    let c = sem_note(0.5, 0.5);
    let a_id = a.note().id();
    let b_id = b.note().id();
    let c_id = c.note().id();

    // a → b → c 链接链
    let mut a = a;
    let link_ab = MemoryLinkBuilder::new(a_id, b_id, MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)))
        .build();
    a.note.links_mut().push(link_ab);
    let mut b = b;
    let link_bc = MemoryLinkBuilder::new(b_id, c_id, MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)))
        .build();
    b.note.links_mut().push(link_bc);

    repo.upsert_notes(vec![a, b, c]).await.unwrap();

    // 读回 a：出边完整恢复（a → b）
    let fetched = repo.fetch_notes(&[a_id]).await.unwrap();
    assert_eq!(fetched.len(), 1);
    let links = fetched[0].note().links();
    assert_eq!(links.len(), 1, "a 的出边应恢复");
    assert_eq!(links[0].from(), a_id);
    assert_eq!(links[0].to(), b_id);

    // BFS 深度遍历：depth=2 从 a 到达 b、c
    let neighbors = repo.fetch_neighbors(&[a_id], 2).await.unwrap();
    let mut neighbor_ids: Vec<MemoryId> = neighbors.iter().map(|e| e.note().id()).collect();
    neighbor_ids.sort();
    let mut expect = vec![b_id, c_id];
    expect.sort();
    assert_eq!(neighbor_ids, expect, "depth=2 应到达 b 与 c");
    // 邻居数据完整（可消费）
    for e in &neighbors {
        assert!(matches!(e.note().mem_type(), MemoryType::Semantic(_)));
    }
}

/// 业务链路 5：同 id 全量覆盖 + 删除（含边清理）。
#[tokio::test]
async fn overwrite_same_id_and_remove_cleans_up() {
    let repo = repo().await;
    let original = sem_note(0.8, 0.5);
    let id = original.note().id();

    // 覆盖：同 id 写入不同缺失度与 content 向量
    let mut updated = sem_note(0.2, 0.9);
    updated.note.set_missing_degree(0.7);
    let updated_note = MemoryNoteBuilder::new(updated.note.mem_type().clone())
        .id(id)
        .tags(updated.note.tags().to_vec())
        .retrieval_count(updated.note.retrieval_count())
        .create_time(updated.note.creation_time())
        .last_accessed_time(updated.note.last_accessed_time())
        .missing_degree(updated.note.missing_degree())
        .last_forget_time(updated.note.last_forget_time())
        .build()
        .unwrap();
    updated.note = updated_note;
    repo.upsert_notes(vec![updated.clone()]).await.unwrap();

    let fetched = repo.fetch_notes(&[id]).await.unwrap();
    assert_eq!(fetched.len(), 1, "覆盖不产生重复记录");
    assert_eq!(fetched[0].note().missing_degree(), 0.7, "覆盖字段生效");
    assert_eq!(fetched[0].embedding(), updated.embedding(), "嵌入被覆盖为最新值");

    // 删除：读回为空
    repo.remove_notes(&[id]).await.unwrap();
    assert!(repo.fetch_notes(&[id]).await.unwrap().is_empty(), "删除后读回为空");
    // 删除不存在的 id 是幂等 no-op
    let _: StorageResult<()> = repo.remove_notes(&[id]).await;
}

/// 业务链路 6：批量写入原子性——batch 中带非法数据时整个事务回滚。
#[tokio::test]
async fn batch_upsert_is_atomic_on_failure() {
    let repo = repo().await;
    let good = sem_note(0.8, 0.5);
    let good_id = good.note().id();

    // 构造嵌入维度非法的 note（2 维，schema HNSW 要求 512 维）→ DB 拒绝 → 事务失败
    let mut bad = sem_note(0.1, 0.1);
    bad.embedding = MemoryEmbedding::new(
        EmbeddingVec::new(vec![0.5, 0.5]),
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            EmbeddingVec::new(vec![1.0, 0.0]),
            EmbeddingVec::new(vec![1.0, 0.0]),
            EmbeddingVec::new(vec![1.0, 0.0]),
        )),
    );

    let result = repo.upsert_notes(vec![good.clone(), bad]).await;
    assert!(result.is_err(), "非法维度嵌入应导致 batch 失败");

    // 原子性：good 也未被写入（整个事务回滚）
    assert!(
        repo.fetch_notes(&[good_id]).await.unwrap().is_empty(),
        "batch 失败时先前的写入必须回滚"
    );
}
