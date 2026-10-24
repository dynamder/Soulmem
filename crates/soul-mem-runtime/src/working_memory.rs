//! 工作记忆：滑动窗口 + 记忆簇 + 活跃记录。
//!
//! 四块状态各自内部可变，因此公开方法一律只取 `&self`；`Arc<WorkingMemory>` 可跨线程共享，
//! 调用方不需要（也无法）拿到独占引用。

// 为什么所有变更方法都只取 `&self`：检索管线按契约接收 `Arc<WorkingMemory>`
// （见 `DefaultPipelineConfig::into_request`），编排层并行处理一批 query 时同时存在多份 `Arc`。
// 若变更要求 `&mut self`，调用方只能退化成 `Arc::get_mut`，而共享计数 > 1 时它返回 `None`
// ——变更被静默跳过，既无法察觉也无法恢复。
//
// 四块状态的可变性载体：`WorkingState` 是 `parking_lot::RwLock`；`SlidingWindow` 自身即内部可变
// （`Arc<RwLock<..>>` + 原子计数）；记忆簇是 `MemoryClusterHandle`；`records` 是
// `parking_lot::RwLock<HashMap<..>>`。
//
// 锁的纪律：`records` 只经 `read_records` / `write_records` 的闭包访问，锁守卫一律不外泄
// （`parking_lot` 守卫默认 `!Send`，闭包形式从结构上排除"守卫跨越 await 存活"）。
//
// 这些锁只保护各自字段，跨字段的复合不变量（例如"簇里新增节点的同时登记活跃记录"）不由它们
// 保证：`add_node` 先写簇、再写 `records`，两把锁不同时持有；更强的原子性要由调用方定义顺序契约。

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod record;
pub mod sliding_window;

use self::record::{Record, UserFeedback};
use self::sliding_window::SlidingWindow;
use crate::cluster::cluster_handle::MemoryClusterHandle;
use crate::cluster::memory_cluster::MemoryCluster;
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;

/// 工作记忆状态。
///
/// 实现 `Copy`：它只经 [`WorkingMemory::state`] 按值读出，取值不越过任何锁边界。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WorkingState {
    /// 空闲：当前没有进行中的交互。
    Idle,
    /// 工作：正在处理一次交互。
    Working,
}

/// 工作记忆结构体(SlidingWindow & MemoryCluster & records)。
///
/// 全部字段内部可变，因此公开方法一律只取 `&self`（见模块注释）。
#[derive(Debug)]
pub struct WorkingMemory {
    state: RwLock<WorkingState>,
    sliding_window: SlidingWindow,
    memory_cluster: MemoryClusterHandle,
    records: RwLock<HashMap<MemoryId, Record>>,
}

impl WorkingMemory {
    /// 新建工作记忆。
    ///
    /// `window_capacity` 是滑动窗口的容量：`push` 报告超容后，由调用方 `pop` 队首并
    /// 决定是否摘要（见 [`sliding_window`] 的模块注释）。
    pub fn new(window_capacity: usize) -> Self {
        Self {
            state: RwLock::new(WorkingState::Idle),
            sliding_window: SlidingWindow::new(window_capacity),
            memory_cluster: MemoryCluster::new().into_handle(),
            records: RwLock::new(HashMap::new()),
        }
    }

    // 状态机

    /// 当前状态（按值读出）。
    pub fn state(&self) -> WorkingState {
        *self.state.read()
    }

    /// 转入 `Working`。
    pub fn transition_to_working(&self) {
        *self.state.write() = WorkingState::Working;
    }

    /// 转入 `Idle`。
    pub fn transition_to_idle(&self) {
        *self.state.write() = WorkingState::Idle;
    }

    /// 是否处于 `Working`。
    pub fn is_working(&self) -> bool {
        self.state() == WorkingState::Working
    }

    // 滑动窗口

    /// 滑动窗口引用。
    ///
    /// 返回共享引用即可：`SlidingWindow` 自身内部可变。
    pub fn sliding_window(&self) -> &SlidingWindow {
        &self.sliding_window
    }

    // 记忆簇

    /// 新增节点，并为它登记活跃记录（记录已存在时保留原记录，不重置计数）。
    pub fn add_node(&self, node: EmbeddedMemoryNote) {
        let node_id = node.note().id();
        self.memory_cluster
            .write(|cluster| cluster.add_single_node(node));

        self.records
            .write()
            .entry(node_id)
            .or_insert_with(|| Record::new(node_id));
    }

    /// 移除节点，同时移除对应的记录。
    pub fn remove_node(&self, node_id: MemoryId) -> Option<EmbeddedMemoryNote> {
        self.records.write().remove(&node_id);
        self.memory_cluster
            .write(|cluster| cluster.remove_single_node(node_id))
    }

    /// 记忆簇句柄（克隆即共享同一份簇，加锁粒度由句柄的 `read_or_compute`/`write` 决定）。
    pub fn memory_cluster(&self) -> MemoryClusterHandle {
        self.memory_cluster.clone()
    }

    // 活跃记录

    /// 记录一次检索命中；节点尚无记录时按需创建。
    pub fn record_retrieval(&self, node_id: MemoryId) {
        // "查-改-插"整体持一次写锁：拆成两次取锁的话，并发的两次命中会各自判定为
        // "无记录" 并各自插入，后插入的那次把先前的计数覆盖掉。
        let mut records = self.records.write();
        match records.get_mut(&node_id) {
            Some(record) => record.record_retrieval(),
            None => {
                let mut record = Record::new(node_id);
                record.record_retrieval();
                records.insert(node_id, record);
            }
        }
    }

    /// 追加一条用户反馈；节点尚无记录时按需创建，避免反馈被静默丢弃。
    pub fn add_feedback(&self, node_id: MemoryId, feedback: UserFeedback) {
        // 与 record_retrieval 同理：同一个复合操作只取一次写锁。
        let mut records = self.records.write();
        match records.get_mut(&node_id) {
            Some(record) => record.add_feedback(feedback),
            None => {
                let mut record = Record::new(node_id);
                record.add_feedback(feedback);
                records.insert(node_id, record);
            }
        }
    }

    /// 只读访问活跃记录。守卫不外泄（见文件头的锁纪律）。
    pub fn read_records<R>(&self, read: impl FnOnce(&HashMap<MemoryId, Record>) -> R) -> R {
        read(&self.records.read())
    }

    /// 可变访问活跃记录。守卫不外泄（见文件头的锁纪律）。
    pub fn write_records<R>(&self, write: impl FnOnce(&mut HashMap<MemoryId, Record>) -> R) -> R {
        write(&mut self.records.write())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::working_memory::sliding_window::Information;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{MemoryEmbedding, MemoryEmbeddingVariant};
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use std::sync::Arc;

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
    fn test_working_state_transitions() {
        let wm = WorkingMemory::new(10);
        assert!(!wm.is_working());
        assert_eq!(wm.state(), WorkingState::Idle);
        wm.transition_to_working();
        assert!(wm.is_working());
        assert_eq!(wm.state(), WorkingState::Working);
        wm.transition_to_idle();
        assert!(!wm.is_working());
        assert_eq!(wm.state(), WorkingState::Idle);
    }

    #[test]
    fn test_sliding_window_accessors() {
        let wm = WorkingMemory::new(10);
        assert!(wm.sliding_window().is_empty());
        {
            let window = wm.sliding_window().window();
            let mut guard = window.write();
            guard.push_back(Information::new("hello", "user"));
        }
        assert_eq!(wm.sliding_window().len(), 1);
        assert_eq!(wm.sliding_window().get_windows()[0].get_str(), "hello");
    }

    #[test]
    fn test_add_node_creates_record() {
        let wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        assert!(wm.read_records(|records| records.contains_key(&id)));
        // 重复添加不覆盖记录
        let node2 = sem_note("B");
        let id2 = node2.note().id();
        wm.add_node(node2);
        assert!(wm.read_records(|records| records.contains_key(&id)));
        assert!(wm.read_records(|records| records.contains_key(&id2)));
    }

    #[test]
    fn test_remove_node_removes_record() {
        let wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        assert!(wm.read_records(|records| records.contains_key(&id)));
        let removed = wm.remove_node(id);
        assert!(removed.is_some());
        assert!(!wm.read_records(|records| records.contains_key(&id)));
        assert!(wm.remove_node(id).is_none());
    }

    #[test]
    fn test_record_retrieval_and_feedback() {
        let wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.record_retrieval(id);
        assert_eq!(wm.read_records(|records| records[&id].retrieval_count()), 1);
        wm.record_retrieval(id);
        assert_eq!(wm.read_records(|records| records[&id].retrieval_count()), 2);

        wm.add_feedback(id, UserFeedback::Positive);
        assert_eq!(wm.read_records(|records| records[&id].feedback_score()), 1);

        // 无 record 时按需创建
        let new_id = MemoryId::new();
        wm.add_feedback(new_id, UserFeedback::Negative);
        assert_eq!(
            wm.read_records(|records| records[&new_id].feedback_score()),
            -1
        );
    }

    #[test]
    fn test_memory_cluster_handle() {
        let wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        let cluster = wm.memory_cluster();
        let contains = cluster.read_or_compute(|c| c.contains_node(id));
        assert!(contains);
    }

    #[test]
    fn test_write_records() {
        let wm = WorkingMemory::new(10);
        assert!(wm.read_records(|records| records.is_empty()));
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        wm.write_records(|records| {
            records.get_mut(&id).expect("record").record_retrieval();
        });
        assert_eq!(wm.read_records(|records| records[&id].retrieval_count()), 1);
    }

    #[test]
    fn test_add_node_registers_record_and_cluster_node() {
        let wm = WorkingMemory::new(10);
        let id = MemoryId::new();
        wm.add_node(mock_node(id));

        assert!(wm.read_records(|records| records.contains_key(&id)));
        assert!(wm.memory_cluster().read_or_compute(|c| c.contains_node(id)));
    }

    // 回归：存在多份 `Arc` 时状态变更不得被静默跳过。
    #[test]
    fn test_second_arc_clone_can_mutate() {
        let wm = Arc::new(WorkingMemory::new(10));
        let second = Arc::clone(&wm);
        // 共享计数为 2 ⇒ 旧实现的 `Arc::get_mut` 在此必然返回 `None`
        assert_eq!(Arc::strong_count(&second), 2);

        second.transition_to_working();
        assert!(wm.is_working(), "经第二个 Arc 的状态变更必须对第一个可见");

        let id = MemoryId::new();
        second.record_retrieval(id);
        assert_eq!(wm.read_records(|records| records[&id].retrieval_count()), 1);

        second.add_feedback(id, UserFeedback::Positive);
        assert_eq!(wm.read_records(|records| records[&id].feedback_score()), 1);
    }

    // 回归：`records` 的查-改-插必须整体持锁，否则并发命中会互相覆盖。
    #[test]
    fn test_concurrent_record_retrieval_accumulates() {
        const THREADS: usize = 4;
        const PER_THREAD: usize = 200;

        let wm = Arc::new(WorkingMemory::new(10));
        let id = MemoryId::new();
        wm.add_node(mock_node(id));

        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let wm = Arc::clone(&wm);
                std::thread::spawn(move || {
                    for _ in 0..PER_THREAD {
                        wm.record_retrieval(id);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("worker thread");
        }

        assert_eq!(
            wm.read_records(|records| records[&id].retrieval_count()),
            THREADS * PER_THREAD
        );
    }

    /// 共享引用下 `add_node` 必须同时落记录与簇节点（两处都要被第二个 `Arc` 观察到）。
    #[test]
    fn test_shared_arc_add_node_reaches_cluster_and_records() {
        let wm = Arc::new(WorkingMemory::new(10));
        let second = Arc::clone(&wm);
        let id = MemoryId::new();

        second.add_node(mock_node(id));

        assert!(wm.read_records(|records| records.contains_key(&id)));
        assert!(wm.memory_cluster().read_or_compute(|c| c.contains_node(id)));
        assert_eq!(Arc::strong_count(&second), 2);
    }

    /// `WorkingMemory` 必须能跨线程共享：检索管线在 rayon 池上并行使用 `Arc<WorkingMemory>`。
    ///
    /// 这是编译期不变量——若日后有字段把 `!Send` 的守卫类型存进来，本用例直接编译失败。
    #[test]
    fn test_working_memory_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WorkingMemory>();
    }
}
