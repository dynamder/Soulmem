use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub mod llm;
pub mod record;
pub mod sliding_window;

use self::sliding_window::SlidingWindow;
use self::record::{Record, UserFeedback};
use crate::memory::memory_cluster::MemoryCluster;
use crate::memory::embedding::note::EmbeddedMemoryNote;
use crate::memory::memory_note::MemoryId;

// 工作记忆状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkingState {
    Idle,
    Working,
}


// 工作记忆结构体(SlidingWindow & MemoryCluster & records)
pub struct WorkingMemory {
    state: WorkingState,
    sliding_window: SlidingWindow,
    memory_cluster: MemoryCluster,
    records: HashMap<MemoryId, Record>,
}

impl WorkingMemory {
    pub fn new(window_capacity: usize) -> Self {
        Self {
            state: WorkingState::Idle,
            sliding_window: SlidingWindow::new(window_capacity),
            memory_cluster: MemoryCluster::new(),
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
        self.memory_cluster.add_single_node(node);
        
        if !self.records.contains_key(&node_id) {
            self.records.insert(node_id, Record::new(node_id));
        }
    }

    /// 移除节点，同时移除对应的记录
    pub fn remove_node(&mut self, node_id: MemoryId) -> Option<crate::memory::memory_note::MemoryNote> {
        self.records.remove(&node_id);
        self.memory_cluster.remove_single_node(node_id)
    }

    pub fn memory_cluster(&self) -> &MemoryCluster {
        &self.memory_cluster
    }

    pub fn memory_cluster_mut(&mut self) -> &mut MemoryCluster {
        &mut self.memory_cluster
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
