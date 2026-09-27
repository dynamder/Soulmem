use super::*;
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_query::embedding::note::MemoryEmbeddingVariant;
use soul_mem_query::embedding::sem::SemanticEmbedding;

fn sem_note(content: &str) -> EmbeddedMemoryNote {
    let mem_type = MemoryType::Semantic(SemMemory {
        content: content.to_string(),
        aliases: vec![],
        concept_type: ConceptType::Entity,
        description: String::new(),
    });
    let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
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

fn sem_link(from: MemoryId, to: MemoryId) -> MemoryLink {
    MemoryLink::from_tuple(
        from,
        to,
        MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 0.9)),
        0.9,
    )
}

fn note_with_links(id: MemoryId, content: &str, links: Vec<MemoryLink>) -> EmbeddedMemoryNote {
    let mem_type = MemoryType::Semantic(SemMemory {
        content: content.to_string(),
        aliases: vec![],
        concept_type: ConceptType::Entity,
        description: String::new(),
    });
    let note = MemoryNoteBuilder::new(mem_type)
        .id(id)
        .mem_links(links)
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
fn test_add_single_node_and_lookup() {
    let mut cluster = MemoryCluster::new();
    let node = sem_note("A");
    let id = node.note().id();
    cluster.add_single_node(node);
    assert!(cluster.contains_node(id));
    assert!(cluster.get_node(id).is_some());
    assert!(cluster.get_embedding(id).is_some());
    assert_eq!(cluster.graph().node_count(), 1);
}

#[test]
fn test_add_single_node_deduplicates() {
    let mut cluster = MemoryCluster::new();
    let node1 = sem_note("A");
    let id = node1.note().id();
    cluster.add_single_node(node1);
    let duplicate = note_with_links(id, "A", vec![]);
    cluster.add_single_node(duplicate);
    assert_eq!(cluster.graph().node_count(), 1);
    assert!(cluster.contains_node(id));
}

#[test]
fn test_merge_creates_edges() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    let node_a_with_link = note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]);
    cluster.add_single_node(node_a_with_link);
    cluster.add_single_node(node_b);
    assert_eq!(cluster.graph().node_count(), 2);
    // A->B 一条边
    assert_eq!(cluster.graph().edge_count(), 1);
    let linked_edges = cluster
        .get_all_linked_edges(id_a)
        .expect("linked edges")
        .collect::<Vec<_>>();
    assert_eq!(linked_edges.len(), 1);
    // 返回的必须是真实存在的 link id
    assert!(cluster.has_edge(linked_edges[0]));
}

#[test]
fn test_merge_handles_pending_edges() {
    // B 尚未加入时，A->B 的边进入 pending；随后加入 B 应补建边
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();

    let node_a_with_link = note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]);
    cluster.add_single_node(node_a_with_link);
    // B 未加入 → 边进入 incompletely_linked_note
    assert_eq!(cluster.graph().edge_count(), 0);

    cluster.add_single_node(node_b);
    assert_eq!(cluster.graph().edge_count(), 1);
}

#[test]
fn test_merge_does_not_duplicate_edges() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    let link = sem_link(id_a, id_b);
    let link_id = link.id();

    let a1 = note_with_links(id_a, "A", vec![link.clone()]);
    cluster.add_single_node(a1);
    // 同一 link_id 再次 merge 不应产生第二条边
    cluster.add_single_node(node_b);
    assert_eq!(cluster.graph().edge_count(), 1);
    cluster.merge_edge(cluster.get_mem_index(id_a).unwrap(), link.clone());
    assert_eq!(cluster.graph().edge_count(), 1);
    assert!(cluster.has_edge(link_id));
}

#[test]
fn test_merge_batch() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    let node_c = sem_note("C");
    let id_c = node_c.note().id();

    let a = note_with_links(id_a, "A", vec![sem_link(id_a, id_b), sem_link(id_a, id_c)]);
    let b = note_with_links(id_b, "B", vec![sem_link(id_b, id_c)]);
    cluster.merge(vec![a, b, node_c]);
    assert_eq!(cluster.graph().node_count(), 3);
    assert_eq!(cluster.graph().edge_count(), 3);
}

#[test]
fn test_remove_single_node() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
    cluster.add_single_node(node_b);

    let removed = cluster.remove_single_node(id_b);
    assert!(removed.is_some());
    assert!(!cluster.contains_node(id_b));
    assert_eq!(cluster.graph().node_count(), 1);
    // 删除后入边应转为 pending，不再存在于图中
    assert_eq!(cluster.graph().edge_count(), 0);
    assert!(cluster.incompletely_linked_note.contains_key(&id_b));
}

#[test]
fn test_remove_nonexistent_node() {
    let mut cluster = MemoryCluster::new();
    let result = cluster.remove_single_node(MemoryId::new());
    assert!(result.is_none());
}

#[test]
fn test_has_edge_and_link_index() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    let link = sem_link(id_a, id_b);
    let link_id = link.id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![link]));
    cluster.add_single_node(node_b);
    assert!(cluster.has_edge(link_id));
    assert!(cluster.get_link_index(link_id).is_some());
}

#[test]
fn test_sub_cluster_add_node() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
    cluster.add_single_node(node_b);

    let mut sub = cluster.sub_cluster(HashSet::from([id_a]), HashSet::new());
    assert!(sub.add_node(id_a).is_ok());
    assert!(sub.add_node(MemoryId::new()).is_err());
}

#[test]
fn test_sub_cluster_add_nodes() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    cluster.add_single_node(node_a);

    let mut sub = cluster.sub_cluster(HashSet::new(), HashSet::new());
    assert!(sub.add_nodes(&[id_a]).is_ok());
    let missing = MemoryId::new();
    assert!(sub.add_nodes(&[id_a, missing]).is_err());
    assert_eq!(sub.super_cluster().graph().node_count(), 1);
}

#[test]
fn test_refresh_node() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
    cluster.add_single_node(node_b);
    cluster.refresh_node(&id_a);
    assert_eq!(cluster.graph().edge_count(), 1);
}

#[test]
fn test_refresh_node_adds_missing_edges() {
    // 直接修改图节点上的 links，再调用 refresh_node 应补建缺失的边
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    let node_c = sem_note("C");
    let id_c = node_c.note().id();
    // A 初始无 links
    cluster.add_single_node(note_with_links(id_a, "A", vec![]));
    cluster.add_single_node(node_b);
    cluster.add_single_node(node_c);
    assert_eq!(cluster.graph().edge_count(), 0);

    // 直接通过 get_node_mut 在图中给 A 增加一条 link（绕过 add 接口）
    let link_b = sem_link(id_a, id_b);
    let link_c = sem_link(id_a, id_c);
    if let Some(node) = cluster.get_node_mut(id_a) {
        // 构造新的 MemoryNote（含 links）替换节点
        *node = note_with_links(id_a, "A", vec![link_b, link_c]);
    }

    cluster.refresh_node(&id_a);
    assert_eq!(cluster.graph().edge_count(), 2);
}

#[test]
fn test_get_node_mut() {
    let mut cluster = MemoryCluster::new();
    let node = sem_note("A");
    let id = node.note().id();
    cluster.add_single_node(node);
    let node_mut = cluster.get_node_mut(id).expect("node exists");
    assert_eq!(node_mut.note().id(), id);
    assert!(cluster.get_node_mut(MemoryId::new()).is_none());
}

#[test]
fn test_remove_single_node_cleans_pending_edges_from_source() {
    // A 指向 B，但 B 尚未加入 → (A→B) 进入 pending（origin=A）
    // 删除 A 后，该 pending 边应被清除
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
    assert!(cluster.incompletely_linked_note.contains_key(&id_b));

    cluster.remove_single_node(id_a);
    // pending 列表中 origin 为 A 的边已被 retain 清除
    let pending = cluster
        .incompletely_linked_note
        .get(&id_b)
        .cloned()
        .unwrap_or_default();
    assert!(
        pending.iter().all(|(origin, _)| *origin != id_a),
        "pending edges from removed source should be cleaned: {pending:?}"
    );
}

#[test]
fn test_get_directed_linked_edges() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![sem_link(id_a, id_b)]));
    cluster.add_single_node(node_b);

    let outgoing = cluster.get_directed_linked_edges(id_a, petgraph::Direction::Outgoing);
    assert!(outgoing.is_some());
    assert_eq!(outgoing.unwrap().count(), 1);
    assert!(
        cluster
            .get_directed_linked_edges(MemoryId::new(), petgraph::Direction::Outgoing)
            .is_none()
    );
}

#[test]
fn test_graph_mut_roundtrip() {
    let mut cluster = MemoryCluster::new();
    let node = sem_note("A");
    let id = node.note().id();
    cluster.add_single_node(node);
    {
        let graph = cluster.graph_mut();
        assert_eq!(graph.node_count(), 1);
    }
    assert_eq!(cluster.graph().node_count(), 1);
    assert!(cluster.get_mem_index(id).is_some());
}

#[test]
fn test_graph_memory_link_intensity_roundtrip() {
    let mut cluster = MemoryCluster::new();
    let node_a = sem_note("A");
    let id_a = node_a.note().id();
    let node_b = sem_note("B");
    let id_b = node_b.note().id();
    let link = sem_link(id_a, id_b);
    let link_id = link.id();
    cluster.add_single_node(note_with_links(id_a, "A", vec![link]));
    cluster.add_single_node(node_b);

    let edge_index = cluster.get_link_index(link_id).expect("edge index exists");
    let graph_link = cluster
        .graph()
        .edge_weight(edge_index)
        .expect("edge weight");
    assert_eq!(graph_link.intensity(), 0.9);
    assert_eq!(graph_link.id(), link_id);
    assert!(matches!(graph_link.link_type(), MemoryLinkType::Sem(_)));
}

#[test]
fn test_get_indexes_none_for_missing() {
    let cluster = MemoryCluster::new();
    assert!(cluster.get_mem_index(MemoryId::new()).is_none());
    assert!(cluster.get_link_index(LinkId::new()).is_none());
    assert!(!cluster.has_edge(LinkId::new()));
}

#[test]
fn test_cluster_error_not_implemented() {
    let mut cluster = MemoryCluster::new();
    let other = MemoryCluster::new();
    let result = cluster.merge_cluster(other);
    assert!(matches!(result, Err(ClusterError::NotImplemented(_))));
}

#[test]
fn test_cluster_debug_format() {
    let mut cluster = MemoryCluster::new();
    let node = sem_note("A");
    cluster.add_single_node(node);
    let debug = format!("{:?}", cluster);
    assert!(debug.contains("MemoryCluster"), "debug was: {debug}");
    assert!(!debug.is_empty());
}
