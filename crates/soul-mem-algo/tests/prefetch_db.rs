//! `prefetch_db` 集成测试：验证「数据库预取 → 工作记忆」链路——
//! 相似度召回命中 + 深度 1 邻居扩展，两次查询的结果都以 `EmbeddedMemoryNote`
//! 形态写入工作记忆（可被后续检索算法直接消费）。
//!
//! - 嵌入用 512 维手工向量模拟模型输出（与 schema HNSW DIMENSION 512 一致）；
//! - 使用 kv-mem 内存库（`SurrealRepository::connect_mem`），无磁盘残留；
//! - 链接链 a → b → c：查询只与 a 相似，b 是一跳邻居，c 是两跳（深度 1 不可达）。

use soul_mem_algo::algo::retrieve::prefetch_db;
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_links::{MemoryLinkBuilder, MemoryLinkType};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant};
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding, MemoryRetrieveQueryVariantEmbedding,
};
use soul_mem_query::embedding::query::sem::SemanticQueryUnitEmbedding;
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
use soul_mem_runtime::storage::surreal::SurrealRepository;
use soul_mem_runtime::storage::MemoryRepository;
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

#[tokio::test]
async fn prefetch_db_writes_similarity_hits_and_one_hop_neighbors() {
    let repo = SurrealRepository::connect_mem().await.unwrap();
    repo.init_schema().await.unwrap();

    // a 与查询同向（0.9）；b/c 与查询无关，只通过链接可达
    let mut a = sem_note(0.9);
    let mut b = sem_note(0.0);
    let c = sem_note(0.0);
    let a_id = a.note().id();
    let b_id = b.note().id();
    let c_id = c.note().id();

    // 链接链 a → b → c
    let link_ab = MemoryLinkBuilder::new(
        a_id,
        b_id,
        MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)),
    )
    .build();
    a.note.links_mut().push(link_ab);
    let link_bc = MemoryLinkBuilder::new(
        b_id,
        c_id,
        MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)),
    )
    .build();
    b.note.links_mut().push(link_bc);

    repo.upsert_notes(vec![a, b, c]).await.unwrap();

    // 查询嵌入与 a 同向（content/aliases/description 通道均为 0.9）
    let query_embedding = MemoryRetrieveQueryEmbedding::new(EmbeddingVec::zero(512)).with_variant(
        MemoryRetrieveQueryVariantEmbedding::Semantic(vec![SemanticQueryUnitEmbedding::new(
            Some(v512(&[0.9, 0.0])),
            Some(v512(&[0.9, 0.0])),
        )]),
    );
    let embedded_query = EmbeddedMemoryRetrieveQuery {
        embedding: query_embedding,
        query: MemoryRetrieveQuery::new(vec![], MemoryRetrieveQueryVariant::Semantic(vec![])),
    };

    let wm = WorkingMemory::new(10);
    prefetch_db(&repo, vec![embedded_query], 1, &wm)
        .await
        .expect("prefetch_db should succeed");

    // 相似命中 a + 一跳邻居 b 写入工作记忆；两跳的 c 不在深度 1 范围内
    let ids: Vec<MemoryId> = wm
        .memory_cluster()
        .read_or_compute(|c| c.graph().node_weights().map(|n| n.note().id()).collect());
    assert!(
        ids.contains(&a_id),
        "similarity hit a should be written into working memory"
    );
    assert!(
        ids.contains(&b_id),
        "one-hop neighbor b should be written into working memory"
    );
    assert!(
        !ids.contains(&c_id),
        "two-hop node c should NOT be written (depth = 1)"
    );

    // 写入的是完整可消费的 EmbeddedMemoryNote（embedding 变体与 mem_type 一致）
    let all_ok = wm.memory_cluster().read_or_compute(|c| {
        c.graph().node_weights().all(|n| {
            matches!(n.note().mem_type(), MemoryType::Semantic(_))
                && matches!(n.embedding().variant(), MemoryEmbeddingVariant::Semantic(_))
        })
    });
    assert!(all_ok, "prefetched notes must be complete EmbeddedMemoryNote");
}
