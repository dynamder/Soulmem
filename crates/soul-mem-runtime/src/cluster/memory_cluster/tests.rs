use super::*;
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_query::embedding::note::{MemoryEmbedding, MemoryEmbeddingVariant};
use soul_mem_query::embedding::sem::SemanticEmbedding;

fn mock_node(id: MemoryId) -> EmbeddedMemoryNote {
    let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
        content: "node".to_string(),
        aliases: vec![],
        concept_type: ConceptType::Entity,
        description: String::new(),
    }))
    .id(id)
    .build()
    .unwrap();
    let embedding = MemoryEmbedding::new(
        EmbeddingVec::zero(4),
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            EmbeddingVec::zero(4),
            EmbeddingVec::zero(4),
            EmbeddingVec::zero(4),
        )),
    );
    EmbeddedMemoryNote { note, embedding }
}

#[test]
fn test_refresh_node_merges_new_links() {
    let handle = MemoryCluster::new().into_handle();
    let a = MemoryId::new();
    let b = MemoryId::new();
    let c = MemoryId::new();
    let link_ab = MemoryLink::new(
        a,
        b,
        MemoryLinkType::Sem(SemMemLink::new("relates".to_string(), 1.0)),
    );
    let link_ac = MemoryLink::new(
        a,
        c,
        MemoryLinkType::Sem(SemMemLink::new("relates".to_string(), 1.0)),
    );

    handle.write(|cluster| {
        let mut node_a = mock_node(a);
        node_a.note.links_mut().push(link_ab.clone());
        cluster.add_single_node(node_a);
        cluster.add_single_node(mock_node(b));
        cluster.add_single_node(mock_node(c));
    });
    assert!(handle.read_or_compute(|cluster| cluster.has_edge(link_ab.id())));

    // 修改 A 的链接后必须调用 refresh_node 才会合并新边
    handle.write(|cluster| {
        cluster
            .get_node_mut(a)
            .unwrap()
            .note
            .links_mut()
            .push(link_ac.clone());
        cluster.refresh_node(&a);
    });
    assert!(handle.read_or_compute(|cluster| cluster.has_edge(link_ac.id())));
}

/// 回归：删除节点时必须同步清理 `link_id_to_index`，否则关联边**永远无法重建**。
///
/// `graph.remove_node` 会一并删除所有关联边（petgraph 保证），但索引表不会自动更新。
/// 残留条目让 `has_edge()` 永久返回 true，于是 `merge_edge` 里的
/// `if !self.has_edge(edge.id())` 守卫永久跳过这条边：
/// 结果是"图里已经不存在这条边、`note.links()` 里却仍列着它"，且重新加入节点也恢复不了。
#[test]
fn test_remove_node_prunes_link_index_so_edge_can_be_rebuilt() {
    let mut cluster = MemoryCluster::new();
    let a = MemoryId::new();
    let b = MemoryId::new();
    let link_ab = MemoryLink::new(
        a,
        b,
        MemoryLinkType::Sem(SemMemLink::new("relates".to_string(), 1.0)),
    );

    let mut node_a = mock_node(a);
    node_a.note.links_mut().push(link_ab.clone());
    cluster.add_single_node(node_a);
    cluster.add_single_node(mock_node(b));
    assert!(cluster.has_edge(link_ab.id()), "前置条件：A→B 边应已建立");

    // 删除 B：关联边随节点被删除，索引表必须同步
    cluster.remove_single_node(b).expect("B 应存在");
    assert!(
        !cluster.has_edge(link_ab.id()),
        "删除节点后 link_id_to_index 仍残留该边 id，has_edge() 会永久返回 true"
    );

    // 重新加入 B：pending 边应被重放，A→B 必须恢复
    cluster.add_single_node(mock_node(b));
    assert!(
        cluster.has_edge(link_ab.id()),
        "重新加入节点后 A→B 边未能重建（被 has_edge 守卫永久跳过）"
    );
    let edge_index = cluster.get_link_index(link_ab.id()).expect("索引应已重建");
    assert!(
        cluster.graph().edge_weight(edge_index).is_some(),
        "重建的 EdgeIndex 必须指向真实存在的边"
    );
}
