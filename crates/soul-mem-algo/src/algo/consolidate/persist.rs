//! 巩固持久化：把本次巩固产生的新节点与新边、或工作记忆的当前快照写入长期存储。
//!
//! 长期存储里一个节点的 `links` 是它**完整出边集**（`upsert_notes` 会据此重建出边），
//! 因此任何新增边都必须先把源节点的既有出边一并带上再写回。这里从工作记忆取源节点的
//! 完整快照、把新边挂上去，整批一次事务写入；写入成功后才改动工作记忆，失败时工作
//! 记忆保持原样。

use std::collections::{HashMap, HashSet};

use soul_mem_core::memory_links::MemoryLink;
use soul_mem_core::memory_note::{MemoryId, MemoryNote};
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::{Embeddable, EmbeddingGenError, EmbeddingModel};
use soul_mem_runtime::storage::{MemoryRepository, StorageError};
use soul_mem_runtime::working_memory::WorkingMemory;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PersistError {
    #[error("embedding consolidated memory failed: {0}")]
    Embedding(#[from] EmbeddingGenError),
    #[error("writing consolidated memory to long-term storage failed: {0}")]
    Storage(#[from] StorageError),
}

/// 一次巩固持久化的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PersistReport {
    /// 本次写入长期存储的节点数。
    pub notes_written: usize,
    /// 新加入工作记忆的节点数。
    pub new_notes: usize,
    /// 成功挂到源节点上的新边数。
    pub links_attached: usize,
    /// 因端点不存在而被丢弃的边数。
    pub links_dropped: usize,
}

/// 把本次巩固的新节点与新边写入长期存储，并同步进工作记忆。
///
/// - `new_notes` 由 [`super::generate`] 从摘要拆出，尚无嵌入；本函数负责嵌入。
/// - `links` 由 [`super::link`] 生成，两端必须属于工作记忆或 `new_notes`；
///   端点未知的边会被丢弃并记一条日志。
/// - 新边挂到源节点后，源节点的全部既有出边会随快照一起写回，不会被覆盖丢失。
///
/// 写入成功后新节点才进入工作记忆；嵌入或存储失败时工作记忆不变。
pub async fn persist_consolidation(
    repo: &dyn MemoryRepository,
    working_mem: &WorkingMemory,
    model: &dyn EmbeddingModel,
    new_notes: Vec<MemoryNote>,
    links: Vec<MemoryLink>,
) -> Result<PersistReport, PersistError> {
    let new_ids: HashSet<MemoryId> = new_notes.iter().map(MemoryNote::id).collect();
    let known = working_mem.memory_cluster().read_or_compute(|cluster| {
        cluster
            .graph()
            .node_weights()
            .map(|note| note.note().id())
            .collect::<HashSet<_>>()
    });

    let mut links_dropped = 0;
    let links: Vec<MemoryLink> = links
        .into_iter()
        .filter(|link| {
            let known_endpoint = |id: MemoryId| known.contains(&id) || new_ids.contains(&id);
            let valid = known_endpoint(link.from()) && known_endpoint(link.to());
            if !valid {
                links_dropped += 1;
                log::warn!(
                    "巩固持久化丢弃端点未知的边 {} -> {}",
                    link.from(),
                    link.to()
                );
            }
            valid
        })
        .collect();

    let source_ids: HashSet<MemoryId> = links
        .iter()
        .map(|link| link.from())
        .filter(|id| !new_ids.contains(id))
        .collect();

    let mut embedded_new: Vec<EmbeddedMemoryNote> = Vec::with_capacity(new_notes.len());
    for note in new_notes {
        embedded_new.push(note.embed_and_fuse(model)?);
    }

    let mut existing_sources = working_mem.memory_cluster().read_or_compute(|cluster| {
        source_ids
            .iter()
            .filter_map(|id| cluster.get_node(*id).cloned().map(|note| (*id, note)))
            .collect::<HashMap<_, _>>()
    });

    let mut attached: HashMap<MemoryId, Vec<MemoryLink>> = HashMap::new();
    let mut links_attached = 0;
    for link in links {
        let source = if new_ids.contains(&link.from()) {
            embedded_new
                .iter_mut()
                .find(|note| note.note().id() == link.from())
        } else {
            existing_sources.get_mut(&link.from())
        };
        let Some(source) = source else {
            links_dropped += 1;
            log::warn!(
                "巩固持久化丢弃源节点已不在工作记忆中的边 {} -> {}",
                link.from(),
                link.to()
            );
            continue;
        };
        if attach_link(source, &link) {
            links_attached += 1;
            attached.entry(link.from()).or_default().push(link);
        }
    }

    let existing_sources: Vec<EmbeddedMemoryNote> = existing_sources
        .into_iter()
        .filter(|(id, _)| attached.contains_key(id))
        .map(|(_, note)| note)
        .collect();
    let notes_written = embedded_new.len() + existing_sources.len();
    if notes_written > 0 {
        let batch = embedded_new
            .iter()
            .cloned()
            .chain(existing_sources)
            .collect();
        repo.upsert_notes(batch).await?;
    }

    for note in embedded_new {
        working_mem.add_node(note);
    }

    if !attached.is_empty() {
        let cluster = working_mem.memory_cluster();
        cluster.write(|cluster| {
            for (from, links) in &attached {
                {
                    let Some(source) = cluster.get_node_mut(*from) else {
                        continue;
                    };
                    for link in links {
                        if !source
                            .note()
                            .links()
                            .iter()
                            .any(|edge| edge.id() == link.id())
                        {
                            source.note.links_mut().push(link.clone());
                        }
                    }
                }
                cluster.refresh_node(from);
            }
        });
    }

    Ok(PersistReport {
        notes_written,
        new_notes: new_ids.len(),
        links_attached,
        links_dropped,
    })
}

/// 把当前工作记忆的全部节点写回长期存储（定时快照或优雅退出时使用）。
///
/// 节点已带嵌入，这里不再调用嵌入模型；空工作记忆不产生写事务。
pub async fn persist_working_memory(
    repo: &dyn MemoryRepository,
    working_mem: &WorkingMemory,
) -> Result<PersistReport, PersistError> {
    let notes: Vec<EmbeddedMemoryNote> = working_mem
        .memory_cluster()
        .read_or_compute(|cluster| cluster.graph().node_weights().cloned().collect());
    let notes_written = notes.len();
    if notes_written > 0 {
        repo.upsert_notes(notes).await?;
    }
    Ok(PersistReport {
        notes_written,
        ..PersistReport::default()
    })
}

fn attach_link(note: &mut EmbeddedMemoryNote, link: &MemoryLink) -> bool {
    if note
        .note()
        .links()
        .iter()
        .any(|edge| edge.id() == link.id())
    {
        return false;
    }
    note.note.links_mut().push(link.clone());
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_links::{LinkId, MemoryLinkType};
    use soul_mem_core::memory_note::MemoryNoteBuilder;
    use soul_mem_core::memory_note::MemoryType;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
    use soul_mem_query::embedding::{EmbeddingGenResult, EmbeddingVec};
    use soul_mem_runtime::storage::StorageResult;
    use soul_mem_runtime::storage::surreal::SurrealRepository;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    const DIM: usize = 512;

    struct ZeroModel;

    impl EmbeddingModel for ZeroModel {
        fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
            Ok(input.iter().map(|_| EmbeddingVec::zero(DIM)).collect())
        }
        fn infer_with_chunk(&self, _input: &str) -> EmbeddingGenResult<EmbeddingVec> {
            Ok(EmbeddingVec::zero(DIM))
        }
        fn infer_and_fuse(&self, _input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
            Ok(EmbeddingVec::zero(DIM))
        }
        fn max_input_token(&self) -> usize {
            DIM
        }
        fn dim(&self) -> usize {
            DIM
        }
    }

    struct FailingModel;

    impl EmbeddingModel for FailingModel {
        fn infer_batch(&self, _input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
            Err(EmbeddingGenError::InvalidInput)
        }
        fn infer_with_chunk(&self, _input: &str) -> EmbeddingGenResult<EmbeddingVec> {
            Err(EmbeddingGenError::InvalidInput)
        }
        fn infer_and_fuse(&self, _input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
            Err(EmbeddingGenError::InvalidInput)
        }
        fn max_input_token(&self) -> usize {
            DIM
        }
        fn dim(&self) -> usize {
            DIM
        }
    }

    struct RemovingModel {
        working_mem: Arc<WorkingMemory>,
        victim: MemoryId,
        removed: AtomicBool,
    }

    impl RemovingModel {
        fn remove_once(&self) {
            if !self.removed.swap(true, Ordering::SeqCst) {
                self.working_mem.remove_node(self.victim);
            }
        }
    }

    impl EmbeddingModel for RemovingModel {
        fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
            self.remove_once();
            Ok(input.iter().map(|_| EmbeddingVec::zero(DIM)).collect())
        }
        fn infer_with_chunk(&self, _input: &str) -> EmbeddingGenResult<EmbeddingVec> {
            self.remove_once();
            Ok(EmbeddingVec::zero(DIM))
        }
        fn infer_and_fuse(&self, _input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
            self.remove_once();
            Ok(EmbeddingVec::zero(DIM))
        }
        fn max_input_token(&self) -> usize {
            DIM
        }
        fn dim(&self) -> usize {
            DIM
        }
    }

    struct FailingRepo;

    #[async_trait]
    impl MemoryRepository for FailingRepo {
        async fn upsert_notes(&self, _mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()> {
            Err(StorageError::InvalidArgument("boom".into()))
        }
        async fn fetch_neighbors(
            &self,
            _source_ids: &[MemoryId],
            _depth: usize,
        ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
            unimplemented!()
        }
        async fn fetch_notes(
            &self,
            _mem_ids: &[MemoryId],
        ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
            unimplemented!()
        }
        async fn similarity_fetch(
            &self,
            _queries: Vec<MemoryRetrieveQueryEmbedding>,
            _candidate_k: usize,
        ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
            unimplemented!()
        }
        async fn remove_notes(&self, _mem_ids: &[MemoryId]) -> StorageResult<()> {
            unimplemented!()
        }
        async fn remove_links(&self, _link_ids: &[LinkId]) -> StorageResult<()> {
            unimplemented!()
        }
    }

    fn sem_note(content: &str) -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            content.to_string(),
            ConceptType::Entity,
            String::new(),
        )))
        .build()
        .unwrap()
    }

    fn sem_link(from: MemoryId, to: MemoryId) -> MemoryLink {
        MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)),
        )
    }

    fn embed(note: MemoryNote) -> EmbeddedMemoryNote {
        note.embed_and_fuse(&ZeroModel).unwrap()
    }

    async fn repo() -> SurrealRepository {
        let repo = SurrealRepository::connect_mem().await.unwrap();
        repo.init_schema().await.unwrap();
        repo
    }

    fn links_of(note: &EmbeddedMemoryNote) -> &[MemoryLink] {
        note.note().links()
    }

    #[tokio::test]
    async fn persists_new_notes_and_links_into_long_term_storage() {
        let repo = repo().await;
        let wm = WorkingMemory::new(10);
        let a = sem_note("A");
        let b = sem_note("B");
        let (a_id, b_id) = (a.id(), b.id());
        let link = sem_link(a_id, b_id);

        let report = persist_consolidation(&repo, &wm, &ZeroModel, vec![a, b], vec![link.clone()])
            .await
            .unwrap();

        assert_eq!(
            report,
            PersistReport {
                notes_written: 2,
                new_notes: 2,
                links_attached: 1,
                links_dropped: 0,
            }
        );

        let fetched = repo.fetch_notes(&[a_id, b_id]).await.unwrap();
        assert_eq!(fetched.len(), 2);
        let a_fetched = fetched.iter().find(|n| n.note().id() == a_id).unwrap();
        assert_eq!(links_of(a_fetched), std::slice::from_ref(&link));

        assert!(wm.memory_cluster().read_or_compute(|cluster| {
            cluster.contains_node(a_id)
                && cluster.contains_node(b_id)
                && cluster.has_edge(link.id())
        }));
    }

    #[tokio::test]
    async fn keeps_existing_out_edges_when_attaching_new_links() {
        let repo = repo().await;
        let wm = WorkingMemory::new(10);

        let old_target = embed(sem_note("old target"));
        let old_target_id = old_target.note().id();
        let mut source = sem_note("active source");
        let source_id = source.id();
        let old_link_type = MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0));
        source
            .links_mut()
            .push(MemoryLink::new(source_id, old_target_id, old_link_type));
        let old_link_id = source.links()[0].id();
        let source = embed(source);

        repo.upsert_notes(vec![source.clone(), old_target])
            .await
            .unwrap();
        wm.add_node(source);

        let new_target = sem_note("new target");
        let new_target_id = new_target.id();
        let new_link = sem_link(source_id, new_target_id);
        let report = persist_consolidation(
            &repo,
            &wm,
            &ZeroModel,
            vec![new_target],
            vec![new_link.clone()],
        )
        .await
        .unwrap();

        assert_eq!(report.new_notes, 1);
        assert_eq!(report.notes_written, 2);
        assert_eq!(report.links_attached, 1);

        let fetched = repo.fetch_notes(&[source_id]).await.unwrap();
        let links = links_of(&fetched[0]);
        assert_eq!(links.len(), 2, "既有出边不得被新边覆盖");
        assert!(links.iter().any(|link| link.id() == old_link_id));
        assert!(links.iter().any(|link| link.id() == new_link.id()));

        // 工作记忆里的源节点同样保留旧边并挂上新边
        let source_in_wm = wm
            .memory_cluster()
            .read_or_compute(|cluster| cluster.get_node(source_id).cloned())
            .unwrap();
        assert_eq!(links_of(&source_in_wm).len(), 2);
    }

    #[tokio::test]
    async fn drops_links_with_unknown_endpoints() {
        let repo = repo().await;
        let wm = WorkingMemory::new(10);
        let a = sem_note("A");
        let a_id = a.id();
        let dangling = sem_link(a_id, MemoryId::new());

        let report = persist_consolidation(&repo, &wm, &ZeroModel, vec![a], vec![dangling])
            .await
            .unwrap();

        assert_eq!(report.notes_written, 1);
        assert_eq!(report.links_attached, 0);
        assert_eq!(report.links_dropped, 1);

        let fetched = repo.fetch_notes(&[a_id]).await.unwrap();
        assert!(links_of(&fetched[0]).is_empty());
    }

    #[tokio::test]
    async fn empty_batch_does_not_touch_storage() {
        let wm = WorkingMemory::new(10);

        let report = persist_consolidation(&FailingRepo, &wm, &ZeroModel, Vec::new(), Vec::new())
            .await
            .unwrap();

        assert_eq!(report, PersistReport::default());
    }

    #[tokio::test]
    async fn storage_failure_leaves_working_memory_unchanged() {
        let wm = WorkingMemory::new(10);
        let note = sem_note("A");
        let note_id = note.id();

        let err = persist_consolidation(&FailingRepo, &wm, &ZeroModel, vec![note], Vec::new())
            .await
            .unwrap_err();

        assert!(matches!(err, PersistError::Storage(_)));
        assert!(
            !wm.memory_cluster()
                .read_or_compute(|cluster| cluster.contains_node(note_id))
        );
    }

    #[tokio::test]
    async fn embedding_failure_leaves_working_memory_unchanged() {
        let wm = WorkingMemory::new(10);
        let note = sem_note("A");
        let note_id = note.id();

        let err = persist_consolidation(&FailingRepo, &wm, &FailingModel, vec![note], Vec::new())
            .await
            .unwrap_err();

        assert!(matches!(err, PersistError::Embedding(_)));
        assert!(
            !wm.memory_cluster()
                .read_or_compute(|cluster| cluster.contains_node(note_id))
        );
    }

    #[tokio::test]
    async fn repeated_persistence_is_idempotent() {
        let repo = repo().await;
        let wm = WorkingMemory::new(10);
        let a = sem_note("A");
        let b = sem_note("B");
        let (a_id, b_id) = (a.id(), b.id());
        let link = sem_link(a_id, b_id);

        for _ in 0..2 {
            let report = persist_consolidation(
                &repo,
                &wm,
                &ZeroModel,
                vec![a.clone(), b.clone()],
                vec![link.clone()],
            )
            .await
            .unwrap();
            assert_eq!(report.links_attached, 1);
        }

        let fetched = repo.fetch_notes(&[a_id]).await.unwrap();
        assert_eq!(links_of(&fetched[0]).len(), 1);

        let a_in_wm = wm
            .memory_cluster()
            .read_or_compute(|cluster| cluster.get_node(a_id).cloned())
            .unwrap();
        assert_eq!(links_of(&a_in_wm).len(), 1);
        assert_eq!(
            wm.memory_cluster()
                .read_or_compute(|c| c.graph().node_count()),
            2
        );
    }

    #[tokio::test]
    async fn snapshots_entire_working_memory_with_links() {
        let repo = repo().await;
        let wm = WorkingMemory::new(10);
        let b = embed(sem_note("B"));
        let b_id = b.note().id();
        let mut a = embed(sem_note("A"));
        let a_id = a.note().id();
        let link = sem_link(a_id, b_id);
        a.note.links_mut().push(link.clone());
        wm.add_node(a);
        wm.add_node(b);

        let report = persist_working_memory(&repo, &wm).await.unwrap();

        assert_eq!(report.notes_written, 2);
        let fetched = repo.fetch_notes(&[a_id, b_id]).await.unwrap();
        assert_eq!(fetched.len(), 2);
        let a_fetched = fetched.iter().find(|n| n.note().id() == a_id).unwrap();
        assert_eq!(links_of(a_fetched), std::slice::from_ref(&link));
    }

    #[tokio::test]
    async fn empty_working_memory_snapshot_skips_storage() {
        let wm = WorkingMemory::new(10);

        let report = persist_working_memory(&FailingRepo, &wm).await.unwrap();

        assert_eq!(report, PersistReport::default());
    }

    #[tokio::test]
    async fn counts_link_as_dropped_when_source_disappears_during_embedding() {
        let repo = repo().await;
        let wm = Arc::new(WorkingMemory::new(10));
        let active = embed(sem_note("active"));
        let active_id = active.note().id();
        wm.add_node(active);

        let new_note = sem_note("new");
        let new_id = new_note.id();
        let link = sem_link(active_id, new_id);
        let model = RemovingModel {
            working_mem: wm.clone(),
            victim: active_id,
            removed: AtomicBool::new(false),
        };

        let report = persist_consolidation(&repo, wm.as_ref(), &model, vec![new_note], vec![link])
            .await
            .unwrap();

        assert_eq!(report.links_dropped, 1);
        assert_eq!(report.links_attached, 0);
        assert_eq!(report.notes_written, 1);
        assert_eq!(repo.fetch_notes(&[new_id]).await.unwrap().len(), 1);
    }
}
