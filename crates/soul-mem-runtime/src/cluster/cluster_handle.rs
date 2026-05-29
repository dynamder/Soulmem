use std::sync::Arc;

use crate::cluster::memory_cluster::MemoryCluster;

use parking_lot::RwLock;

#[derive(Debug, Clone)]
pub struct MemoryClusterHandle {
    pub(super) cluster: Arc<RwLock<MemoryCluster>>,
}
impl MemoryClusterHandle {
    pub fn read_or_compute<R>(&self, read_comp_closure: impl FnOnce(&MemoryCluster) -> R) -> R {
        let cluster = self.cluster.read();
        read_comp_closure(&cluster)
    }

    pub fn write<R>(&self, write_closure: impl FnOnce(&mut MemoryCluster) -> R) -> R {
        let mut cluster = self.cluster.write();
        write_closure(&mut cluster)
    }
}

#[cfg(test)]
mod tests {
    use petgraph::algo::UnitMeasure;
    use soul_mem_algo::common::ord_float::OrdFloat;
    use soul_mem_algo::common::ppr::weighted_ppr_fp;
    use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };

    use super::*;
    use parking_lot::Mutex;
    use petgraph::visit::IntoNodeIdentifiers;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CONCURRENT_READ_COUNT: AtomicUsize = AtomicUsize::new(0);
    static CONCURRENT_WRITE_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn test_concurrent_reads_via_lock() {
        let cluster = MemoryCluster::new();
        let handle = MemoryClusterHandle {
            cluster: Arc::new(RwLock::new(cluster)),
        };

        let id = MemoryId::new();
        {
            let mut guard = handle.cluster.write();
            let mem_type = MemoryType::Semantic(SemMemory {
                content: "test".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            });
            let note = MemoryNoteBuilder::new(mem_type).id(id).build().unwrap();
            let embedding = MemoryEmbedding::new(
                soul_mem_query::embedding::EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(
                    soul_mem_query::embedding::sem::SemanticEmbedding::new(
                        soul_mem_query::embedding::EmbeddingVec::zero(128),
                        soul_mem_query::embedding::EmbeddingVec::zero(128),
                        soul_mem_query::embedding::EmbeddingVec::zero(128),
                    ),
                ),
            );
            guard.add_single_node(EmbeddedMemoryNote { note, embedding });
        }

        let handles: Vec<_> = (0..100)
            .map(|_| {
                let handle = handle.clone();
                let id = id;
                std::thread::spawn(move || {
                    let guard = handle.cluster.read();
                    CONCURRENT_READ_COUNT.fetch_add(1, Ordering::Relaxed);
                    guard.get_node(id).cloned()
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(results.len(), 100);
        assert_eq!(CONCURRENT_READ_COUNT.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_concurrent_writes_via_lock() {
        let cluster = MemoryCluster::new();
        let handle = MemoryClusterHandle {
            cluster: Arc::new(RwLock::new(cluster)),
        };

        let ids: Vec<_> = (0..100).map(|_| MemoryId::new()).collect();

        let handles: Vec<_> = ids
            .iter()
            .map(|id| {
                let handle = handle.clone();
                let id = *id;
                std::thread::spawn(move || {
                    let mut guard = handle.cluster.write();
                    CONCURRENT_WRITE_COUNT.fetch_add(1, Ordering::Relaxed);
                    let mem_type = MemoryType::Semantic(SemMemory {
                        content: "test".to_string(),
                        aliases: vec![],
                        concept_type: ConceptType::Entity,
                        description: String::new(),
                    });
                    let note = MemoryNoteBuilder::new(mem_type).id(id).build().unwrap();
                    let embedding = MemoryEmbedding::new(
                        soul_mem_query::embedding::EmbeddingVec::zero(128),
                        MemoryEmbeddingVariant::Semantic(
                            soul_mem_query::embedding::sem::SemanticEmbedding::new(
                                soul_mem_query::embedding::EmbeddingVec::zero(128),
                                soul_mem_query::embedding::EmbeddingVec::zero(128),
                                soul_mem_query::embedding::EmbeddingVec::zero(128),
                            ),
                        ),
                    );
                    guard.add_single_node(EmbeddedMemoryNote { note, embedding });
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(CONCURRENT_WRITE_COUNT.load(Ordering::Relaxed), 100);
        let count = handle.cluster.read().graph().node_count();
        assert_eq!(count, 100);
    }

    #[test]
    fn test_read_write_exclusion() {
        let counter = StdArc::new(Mutex::new(0usize));
        let write_counter = StdArc::new(Mutex::new(0usize));

        let cluster = MemoryCluster::new();
        let handle = MemoryClusterHandle {
            cluster: Arc::new(RwLock::new(cluster)),
        };

        let mut threads = vec![];

        for _ in 0..10 {
            let handle = handle.clone();
            let counter = StdArc::clone(&counter);
            threads.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let _guard = handle.cluster.read();
                    let mut count = counter.lock();
                    *count += 1;
                }
            }));
        }

        for _ in 0..5 {
            let handle = handle.clone();
            let write_counter = StdArc::clone(&write_counter);
            threads.push(std::thread::spawn(move || {
                for i in 0..100 {
                    let _guard = handle.cluster.write();
                    let mut count = write_counter.lock();
                    *count = i;
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(*counter.lock(), 1000);
        assert_eq!(*write_counter.lock(), 99);
    }

    #[test]
    fn test_arc_handle_clone_is_cheap() {
        let cluster = MemoryCluster::new();
        let handle = MemoryClusterHandle {
            cluster: Arc::new(RwLock::new(cluster)),
        };

        let handles: Vec<_> = (0..100).map(|_| handle.clone()).collect();
        assert_eq!(handles.len(), 100);
    }

    #[test]
    fn test_ppr_with_read_or_compute() {
        let cluster = MemoryCluster::new();
        let handle = MemoryClusterHandle {
            cluster: Arc::new(RwLock::new(cluster)),
        };

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();

        handle.write(|cluster| {
            let make_note = |id: MemoryId, content: &str| {
                let mem_type = MemoryType::Semantic(SemMemory {
                    content: content.to_string(),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                });
                let note = MemoryNoteBuilder::new(mem_type).id(id).build().unwrap();
                let embedding = MemoryEmbedding::new(
                    soul_mem_query::embedding::EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Semantic(
                        soul_mem_query::embedding::sem::SemanticEmbedding::new(
                            soul_mem_query::embedding::EmbeddingVec::zero(128),
                            soul_mem_query::embedding::EmbeddingVec::zero(128),
                            soul_mem_query::embedding::EmbeddingVec::zero(128),
                        ),
                    ),
                );
                EmbeddedMemoryNote { note, embedding }
            };

            cluster.add_single_node(make_note(id1, "A"));
            cluster.add_single_node(make_note(id2, "B"));
            cluster.add_single_node(make_note(id3, "C"));

            let sem_link = soul_mem_core::memory_links::sem_mem::SemMemLink::new(
                "related".to_string(),
                1.0,
                1.0,
            );
            let link_type = MemoryLinkType::Sem(sem_link);
            let _link1 = MemoryLink::new(id1, id2, link_type.clone());
            let _link2 = MemoryLink::new(id2, id3, link_type.clone());
            cluster.refresh_node(&id1);
            cluster.refresh_node(&id2);
        });

        let ppr_result = handle.read_or_compute(|cluster| {
            let graph = cluster.graph();
            let mut source_bias: std::collections::HashMap<_, OrdFloat<f64>> =
                std::collections::HashMap::new();

            for node_id in graph.node_identifiers() {
                let note = graph.node_weight(node_id).unwrap();
                if note.note().id() == id1 {
                    source_bias.insert(node_id, OrdFloat::from_f64(1.0));
                    break;
                }
            }

            weighted_ppr_fp(
                graph,
                OrdFloat::from_f64(0.85),
                source_bias,
                OrdFloat::from_f64(0.001),
                |_, _| OrdFloat::from_f64(1.0),
                &"query",
            )
        });

        let sum: f64 = ppr_result.values().copied().map(|v| v.into_inner()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "PPR values should sum to 1, got {}",
            sum
        );
    }
}
