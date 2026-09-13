use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod llm;
pub mod record;
pub mod sliding_window;

use self::record::{Record, UserFeedback};
use self::sliding_window::SlidingWindow;
use crate::cluster::cluster_handle::MemoryClusterHandle;
use crate::cluster::memory_cluster::MemoryCluster;
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;

// 工作记忆状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkingState {
    Idle,
    Working,
}

// 工作记忆结构体(SlidingWindow & MemoryCluster & records)
#[derive(Debug)]
pub struct WorkingMemory {
    state: WorkingState,
    sliding_window: SlidingWindow,
    memory_cluster: MemoryClusterHandle,
    records: HashMap<MemoryId, Record>,
}

impl WorkingMemory {
    pub fn new(window_capacity: usize) -> Self {
        Self {
            state: WorkingState::Idle,
            sliding_window: SlidingWindow::new(window_capacity),
            memory_cluster: MemoryCluster::new().into_handle(),
            records: HashMap::new(),
        }
    }

    // 状态机
    pub fn state(&self) -> &WorkingState {
        &self.state
    }

    pub fn transition_to_working(&mut self) {
        self.state = WorkingState::Working;
    }

    pub fn transition_to_idle(&mut self) {
        self.state = WorkingState::Idle;
    }

    pub fn is_working(&self) -> bool {
        self.state == WorkingState::Working
    }

    // 滑动窗口引用
    pub fn sliding_window(&self) -> &SlidingWindow {
        &self.sliding_window
    }

    pub fn sliding_window_mut(&mut self) -> &mut SlidingWindow {
        &mut self.sliding_window
    }

    // Cluster
    pub fn add_node(&mut self, node: EmbeddedMemoryNote) {
        let node_id = node.note().id();
        self.memory_cluster
            .write(|cluster| cluster.add_single_node(node));

        self.records
            .entry(node_id)
            .or_insert_with(|| Record::new(node_id));
    }

    /// 移除节点，同时移除对应的记录
    pub fn remove_node(&mut self, node_id: MemoryId) -> Option<EmbeddedMemoryNote> {
        self.records.remove(&node_id);
        self.memory_cluster
            .write(|cluster| cluster.remove_single_node(node_id))
    }

    pub fn memory_cluster(&self) -> MemoryClusterHandle {
        self.memory_cluster.clone()
    }

    // Record
    pub fn record_retrieval(&mut self, node_id: MemoryId) {
        if let Some(record) = self.records.get_mut(&node_id) {
            record.record_retrieval();
        } else {
            let mut record = Record::new(node_id);
            record.record_retrieval();
            self.records.insert(node_id, record);
        }
    }

    pub fn add_feedback(&mut self, node_id: MemoryId, feedback: UserFeedback) {
        //与record_retrieval一致：节点无record时按需创建，避免反馈被静默丢弃
        match self.records.get_mut(&node_id) {
            Some(record) => record.add_feedback(feedback),
            None => {
                let mut record = Record::new(node_id);
                record.add_feedback(feedback);
                self.records.insert(node_id, record);
            }
        }
    }

    pub fn records(&self) -> &HashMap<MemoryId, Record> {
        &self.records
    }

    pub fn records_mut(&mut self) -> &mut HashMap<MemoryId, Record> {
        &mut self.records
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
        let mut wm = WorkingMemory::new(10);
        assert!(!wm.is_working());
        assert_eq!(wm.state(), &WorkingState::Idle);
        wm.transition_to_working();
        assert!(wm.is_working());
        assert_eq!(wm.state(), &WorkingState::Working);
        wm.transition_to_idle();
        assert!(!wm.is_working());
        assert_eq!(wm.state(), &WorkingState::Idle);
    }

    #[test]
    fn test_sliding_window_accessors() {
        let mut wm = WorkingMemory::new(10);
        assert!(wm.sliding_window().is_empty());
        {
            let sw = wm.sliding_window_mut();
            let mut w = sw.window().write();
            w.push_back(Information::new("hello", "user"));
        }
        assert_eq!(wm.sliding_window().len(), 1);
        assert_eq!(wm.sliding_window().get_windows()[0].get_str(), "hello");
    }

    #[test]
    fn test_add_node_creates_record() {
        let mut wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        assert!(wm.records().contains_key(&id));
        // 重复添加不覆盖记录
        let node2 = sem_note("B");
        let id2 = node2.note().id();
        wm.add_node(node2);
        assert!(wm.records().contains_key(&id));
        assert!(wm.records().contains_key(&id2));
    }

    #[test]
    fn test_remove_node_removes_record() {
        let mut wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        assert!(wm.records().contains_key(&id));
        let removed = wm.remove_node(id);
        assert!(removed.is_some());
        assert!(!wm.records().contains_key(&id));
        assert!(wm.remove_node(id).is_none());
    }

    #[test]
    fn test_record_retrieval_and_feedback() {
        let mut wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.record_retrieval(id);
        assert_eq!(wm.records()[&id].retrieval_count(), 1);
        wm.record_retrieval(id);
        assert_eq!(wm.records()[&id].retrieval_count(), 2);

        wm.add_feedback(id, UserFeedback::Positive);
        assert_eq!(wm.records()[&id].feedback_score(), 1);

        // 无 record 时按需创建
        let new_id = MemoryId::new();
        wm.add_feedback(new_id, UserFeedback::Negative);
        assert_eq!(wm.records()[&new_id].feedback_score(), -1);
    }

    #[test]
    fn test_memory_cluster_handle() {
        let mut wm = WorkingMemory::new(10);
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        let cluster = wm.memory_cluster();
        let contains = cluster.read_or_compute(|c| c.contains_node(id));
        assert!(contains);
    }

    #[test]
    fn test_records_mut() {
        let mut wm = WorkingMemory::new(10);
        assert!(wm.records_mut().is_empty());
        let node = sem_note("A");
        let id = node.note().id();
        wm.add_node(node);
        wm.records_mut()
            .get_mut(&id)
            .expect("record")
            .record_retrieval();
        assert_eq!(wm.records()[&id].retrieval_count(), 1);
    }

    #[test]
    fn test_add_node_registers_record_and_cluster_node() {
        let mut wm = WorkingMemory::new(10);
        let id = MemoryId::new();
        wm.add_node(mock_node(id));

        assert!(wm.records().contains_key(&id));
        assert!(wm.memory_cluster().read_or_compute(|c| c.contains_node(id)));
    }
}
