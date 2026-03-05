use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use fastembed::TextEmbedding;

pub mod llm;
pub mod record;
pub mod sliding_window;

use self::sliding_window::{Information, SlidingWindow};
use self::record::{Record, UserFeedback};
use crate::memory::memory_cluster::MemoryCluster;
use crate::memory::memory_note::{EmbedMemoryNote, MemoryId};

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
    pub fn new(window_capacity: usize, embedding_model: TextEmbedding) -> Self {
        Self {
            state: WorkingState::Idle,
            sliding_window: SlidingWindow::new(window_capacity),
            memory_cluster: MemoryCluster::new(embedding_model),
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


    // 滑动窗口
    pub async fn push_message(&mut self, message: Information) -> anyhow::Result<()> {
        self.sliding_window.push(message).await
    }

    pub fn window_len(&self) -> usize {
        self.sliding_window.len()
    }

    pub fn window_capacity(&self) -> usize {
        self.sliding_window.get_capacity()
    }

    pub fn get_window_message(&self, index: usize) -> Option<&Information> {
        self.sliding_window.get(index)
    }

    pub fn clear_window(&mut self) {
        self.sliding_window.clear();
    }

    pub fn is_window_empty(&self) -> bool {
        self.sliding_window.is_empty()
    }

    pub fn tag_window_message(&mut self, index: usize) {
        self.sliding_window.tag_information(index);
    }

    pub fn untag_window_message(&mut self, index: usize) {
        self.sliding_window.untag_information(index);
    }


    // Cluster
    pub fn add_node(&mut self, node: EmbedMemoryNote) {
        let node_id = node.note().id();
        self.memory_cluster.add_single_node(node);
        
        if !self.records.contains_key(&node_id) {
            self.records.insert(node_id, Record::new(node_id));
        }
    }

    pub fn get_node(&self, node_id: MemoryId) -> Option<&crate::memory::memory_note::MemoryNote> {
        self.memory_cluster.get_node(node_id)
    }

    pub fn contains_node(&self, node_id: MemoryId) -> bool {
        self.memory_cluster.contains_node(node_id)
    }

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

    pub fn get_record(&self, node_id: MemoryId) -> Option<&Record> {
        self.records.get(&node_id)
    }

    pub fn retrieval_count(&self, node_id: MemoryId) -> usize {
        self.records
            .get(&node_id)
            .map(|r| r.retrieval_count())
            .unwrap_or(0)
    }

    pub fn access_time_span(&self, node_id: MemoryId) -> Option<i64> {
        self.records.get(&node_id).map(|r| r.access_time_span())
    }

    pub fn first_access_time(&self, node_id: MemoryId) -> Option<chrono::DateTime<chrono::Utc>> {
        self.records.get(&node_id).map(|r| r.first_access_time())
    }

    pub fn last_access_time(&self, node_id: MemoryId) -> Option<chrono::DateTime<chrono::Utc>> {
        self.records.get(&node_id).map(|r| r.last_access_time())
    }

    pub fn feedback_score(&self, node_id: MemoryId) -> i32 {
        self.records
            .get(&node_id)
            .map(|r| r.feedback_score())
            .unwrap_or(0)
    }

    pub fn feedback_history_in_range(
        &self,
        node_id: MemoryId,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Option<Vec<(chrono::DateTime<chrono::Utc>, UserFeedback)>> {
        self.records
            .get(&node_id)
            .map(|r| r.feedback_history_in_range(start, end))
    }

    pub fn feedback_history_after(
        &self,
        node_id: MemoryId,
        start: chrono::DateTime<chrono::Utc>,
    ) -> Option<Vec<(chrono::DateTime<chrono::Utc>, UserFeedback)>> {
        self.records
            .get(&node_id)
            .map(|r| r.feedback_history_after(start))
    }

    pub fn feedback_history_before(
        &self,
        node_id: MemoryId,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Option<Vec<(chrono::DateTime<chrono::Utc>, UserFeedback)>> {
        self.records
            .get(&node_id)
            .map(|r| r.feedback_history_before(end))
    }

    pub fn retrieval_frequency(&self, node_id: MemoryId) -> Option<f64> {
        let record = self.records.get(&node_id)?;
        let time_span = record.access_time_span();
        if time_span == 0 {
            return None;
        }
        Some(record.retrieval_count() as f64 / time_span as f64)
    }

    pub fn records(&self) -> &HashMap<MemoryId, Record> {
        &self.records
    }

    pub fn iter_records(&self) -> impl Iterator<Item = (&MemoryId, &Record)> {
        self.records.iter()
    }
}
