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
        if let Some(record) = self.records.get_mut(&node_id) {
            record.add_feedback(feedback);
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
    fn test_add_node_registers_record_and_cluster_node() {
        let mut wm = WorkingMemory::new(10);
        let id = MemoryId::new();
        wm.add_node(mock_node(id));

        assert!(wm.records().contains_key(&id));
        assert!(wm.memory_cluster().read_or_compute(|c| c.contains_node(id)));
    }
}
