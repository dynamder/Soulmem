//! `prefetch_db` 集成测试：验证「数据库预取 → 工作记忆」链路——
//! 相似度召回命中 + `neighbor_depth` 跳邻居扩展，两次查询的结果都以
//! `EmbeddedMemoryNote` 形态写入工作记忆（可被后续检索算法直接消费），
//! 并由返回值 [`PrefetchOutcome`] 如实报告两侧召回。
//!
//! - 嵌入用 512 维手工向量模拟模型输出（与 schema HNSW DIMENSION 512 一致）；
//! - 使用 kv-mem 内存库（`SurrealRepository::connect_mem`），无磁盘残留；
//! - 链接链 a → b → c：查询只与 a 相似，b 是一跳邻居，c 是两跳。
//!   因此 depth 语义可直接读出：0 → {a}，1 → {a, b}，2 → {a, b, c}。

use soul_mem_algo::algo::retrieve::{DbPrefetchConfig, PrefetchOutcome, prefetch_db};
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_links::{MemoryLinkBuilder, MemoryLinkType};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_query::embedding::note::{
    EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
};
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding, MemoryRetrieveQueryVariantEmbedding,
};
use soul_mem_query::embedding::query::sem::SemanticQueryUnitEmbedding;
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
use soul_mem_runtime::storage::MemoryRepository;
use soul_mem_runtime::storage::surreal::SurrealRepository;
use soul_mem_runtime::working_memory::WorkingMemory;

/// 512 维向量：前 `vals` 个分量给定、其余补零（schema HNSW DIMENSION 512）。
fn v512(vals: &[f32]) -> EmbeddingVec {
    let mut v = vec![0.0f32; 512];
    for (i, x) in vals.iter().enumerate() {
        v[i] = *x;
    }
    EmbeddingVec::new(v)
}

/// 语义记忆：`content` 向量的第一个分量 `c` 决定其在语义召回中的可区分性。
fn sem_note(content_c: f32) -> EmbeddedMemoryNote {
    let mem_type = MemoryType::Semantic(SemMemory {
        content: "概念内容".into(),
        aliases: vec!["别名".into()],
        concept_type: ConceptType::Entity,
        description: "描述".into(),
    });
    let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
    let embedding = MemoryEmbedding::new(
        EmbeddingVec::zero(512),
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            v512(&[content_c, 0.0]),
            v512(&[content_c, 0.0]),
            v512(&[content_c, 0.0]),
        )),
    );
    EmbeddedMemoryNote { note, embedding }
}

/// 链接链 `a → b → c`：只有 `a` 与查询同向（0.9），`b`/`c` 仅靠链接可达。
struct Chain {
    repo: SurrealRepository,
    a_id: MemoryId,
    b_id: MemoryId,
    c_id: MemoryId,
}

async fn seed_chain() -> Chain {
    let repo = SurrealRepository::connect_mem().await.unwrap();
    repo.init_schema().await.unwrap();

    let mut a = sem_note(0.9);
    let mut b = sem_note(0.0);
    let c = sem_note(0.0);
    let a_id = a.note().id();
    let b_id = b.note().id();
    let c_id = c.note().id();

    a.note.links_mut().push(
        MemoryLinkBuilder::new(
            a_id,
            b_id,
            MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)),
        )
        .build(),
    );
    b.note.links_mut().push(
        MemoryLinkBuilder::new(
            b_id,
            c_id,
            MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)),
        )
        .build(),
    );

    repo.upsert_notes(vec![a, b, c]).await.unwrap();

    Chain {
        repo,
        a_id,
        b_id,
        c_id,
    }
}

/// 查询嵌入与 `a` 同向（content/aliases/description 三个槽位均为 0.9）。
fn query_like_a() -> EmbeddedMemoryRetrieveQuery {
    EmbeddedMemoryRetrieveQuery {
        embedding: MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(512)).with_variant(
            MemoryRetrieveQueryVariantEmbedding::Semantic(vec![SemanticQueryUnitEmbedding::new(
                Some(v512(&[0.9, 0.0])),
                Some(v512(&[0.9, 0.0])),
            )]),
        ),
        query: MemoryRetrieveQuery::new(vec![], MemoryRetrieveQueryVariant::Semantic(vec![])),
    }
}

/// 工作记忆里当前实际存在的节点（子图成员以工作记忆为准，而非以返回值推断）。
fn wm_ids(wm: &WorkingMemory) -> Vec<MemoryId> {
    wm.memory_cluster()
        .read_or_compute(|c| c.graph().node_weights().map(|n| n.note().id()).collect())
}

/// 跑一次预取并断言「写入工作记忆的节点集合」与「返回值报告的候选/邻居」一致。
async fn prefetch_depth(chain: &Chain, depth: usize) -> (PrefetchOutcome, Vec<MemoryId>) {
    let wm = WorkingMemory::new(10);
    let outcome = prefetch_db(
        &chain.repo,
        vec![query_like_a()],
        DbPrefetchConfig::new(1, depth),
        &wm,
    )
    .await
    .expect("prefetch_db should succeed");

    let ids = wm_ids(&wm);

    // 返回值是观测的唯一来源：它与实际写入集合必须同源一致
    let mut expected: Vec<MemoryId> = outcome.candidates.clone();
    expected.extend(outcome.neighbors.iter().copied());
    for id in &expected {
        assert!(
            ids.contains(id),
            "depth {depth}: 返回值报告的节点 {id:?} 必须在工作记忆中"
        );
    }
    assert_eq!(
        ids.len(),
        expected.len(),
        "depth {depth}: 工作记忆节点数应与 候选+邻居 一致（两侧无重叠、无重复）"
    );

    // 写入的是完整可消费的 `EmbeddedMemoryNote`（embedding 变体与 mem_type 一致）
    let all_ok = wm.memory_cluster().read_or_compute(|c| {
        c.graph().node_weights().all(|n| {
            matches!(n.note().mem_type(), MemoryType::Semantic(_))
                && matches!(n.embedding().variant(), MemoryEmbeddingVariant::Semantic(_))
        })
    });
    assert!(
        all_ok,
        "depth {depth}: 预取的 note 必须是完整 EmbeddedMemoryNote"
    );

    (outcome, ids)
}

#[tokio::test]
async fn prefetch_db_writes_similarity_hits_and_one_hop_neighbors() {
    let chain = seed_chain().await;
    let (outcome, ids) = prefetch_depth(&chain, 1).await;

    // 相似命中只有 a；一跳邻居只有 b
    assert_eq!(outcome.candidates, vec![chain.a_id]);
    assert_eq!(outcome.neighbors, vec![chain.b_id]);
    assert!(ids.contains(&chain.a_id), "相似命中 a 应写入工作记忆");
    assert!(ids.contains(&chain.b_id), "一跳邻居 b 应写入工作记忆");
    assert!(
        !ids.contains(&chain.c_id),
        "两跳节点 c 不应写入（depth = 1）"
    );
}

#[tokio::test]
async fn prefetch_db_depth_zero_writes_only_similarity_hits() {
    let chain = seed_chain().await;
    let (outcome, ids) = prefetch_depth(&chain, 0).await;

    // depth 0：不做邻居扩展，源集即全部命中
    assert_eq!(outcome.candidates, vec![chain.a_id]);
    assert!(
        outcome.neighbors.is_empty(),
        "depth 0 不应有邻居扩展结果，实际 {:?}",
        outcome.neighbors
    );
    assert!(ids.contains(&chain.a_id));
    assert!(!ids.contains(&chain.b_id), "depth 0 不应带入一跳邻居 b");
    assert!(!ids.contains(&chain.c_id));
}

#[tokio::test]
async fn prefetch_db_depth_two_reaches_two_hop_node() {
    let chain = seed_chain().await;
    let (outcome, ids) = prefetch_depth(&chain, 2).await;

    assert_eq!(outcome.candidates, vec![chain.a_id]);
    assert!(
        outcome.neighbors.contains(&chain.b_id) && outcome.neighbors.contains(&chain.c_id),
        "depth 2 的邻居应同时含一跳 b 与两跳 c，实际 {:?}",
        outcome.neighbors
    );
    assert!(
        ids.contains(&chain.c_id),
        "两跳节点 c 在 depth = 2 时应写入"
    );
}
