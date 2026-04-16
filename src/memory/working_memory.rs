use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod llm;
pub mod record;
pub mod sliding_window;

use self::record::{Record, UserFeedback};
use self::sliding_window::SlidingWindow;
use crate::memory::cluster::memory_cluster::MemoryCluster;
use crate::memory::consolidation::service::ConsolidationService;
use crate::memory::embedding::{Embeddable, EmbeddingModel};
use crate::memory::embedding::note::EmbeddedMemoryNote;
use crate::memory::memory_note::{MemoryId, MemoryNote};

const DEFAULT_TOP_K: usize = 8;

// 工作记忆状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkingState {
    Idle,
    Working,
}

// 记录一次整合
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidationRunResult {
    pub summary_was_empty: bool,
    pub hot_memories_used: usize,
    pub notes_added: usize,
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

    // Working -> Idle 后触发巩固
    pub async fn transition_to_idle_and_consolidate(
        &mut self,
        llm: &llm::client::LlmClient,
        embedding_model: &dyn EmbeddingModel,
        hot_top_k: usize,
    ) -> Result<ConsolidationRunResult> {
        self.transition_to_idle();
        self.run_consolidation(llm, embedding_model, hot_top_k).await
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
    pub fn remove_node(
        &mut self,
        node_id: MemoryId,
    ) -> Option<crate::memory::memory_note::MemoryNote> {
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

    // 基于“摘要记忆 + 热点记忆”执行一次短期->工作记忆巩固
    pub async fn run_consolidation(
        &mut self,
        llm: &llm::client::LlmClient,
        embedding_model: &dyn EmbeddingModel,
        hot_top_k: usize,
    ) -> Result<ConsolidationRunResult> {
        let summary_text = self.sliding_window.get_summary_text().await;
        if summary_text.trim().is_empty() {
            return Ok(ConsolidationRunResult {
                summary_was_empty: true,
                hot_memories_used: 0,
                notes_added: 0,
            });
        }

        let top_k = if hot_top_k == 0 {
            DEFAULT_TOP_K
        } else {
            hot_top_k
        };
        let hot_memories = self.collect_hot_memories(top_k);

        let service = ConsolidationService::new();
        let mapped = service
            .split_summary_and_map(llm, &summary_text, &hot_memories)
            .await?;

        // TODO整合
        // 基于热点记忆对旧节点做内容更新/补边，目前先完成“摘要拆分并入图”主路径。这里重复入图的逻辑还没处理
        let notes_added = mapped.notes.len();
        for note in mapped.notes {
            let embedded = note.embed_and_fuse(embedding_model)?;
            self.add_node(embedded);
        }

        self.apply_ltp_ltd_updates();
        self.sliding_window.clear_summary().await;

        Ok(ConsolidationRunResult {
            summary_was_empty: false,
            hot_memories_used: hot_memories.len(),
            notes_added,
        })
    }

    // 使用默认 top-k 触发巩固
    pub async fn run_consolidation_with_default_top_k(
        &mut self,
        llm: &llm::client::LlmClient,
        embedding_model: &dyn EmbeddingModel,
    ) -> Result<ConsolidationRunResult> {
        self.run_consolidation(llm, embedding_model, DEFAULT_TOP_K)
            .await
    }

    fn collect_hot_memories(&self, top_k: usize) -> Vec<String> {
        if top_k == 0 {
            return Vec::new();
        }

        let mut ranked = self
            .records
            .values()
            .filter_map(|record| {
                if record.retrieval_count() == 0 {
                    return None;
                }

                let frequency = Self::compute_retrieval_frequency(
                    record.retrieval_count(),
                    record.access_time_span(),
                );

                Some((record.memory_id(), record.retrieval_count(), frequency))
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.1.cmp(&a.1)));

        ranked
            .into_iter()
            .take(top_k)
            .filter_map(|(memory_id, retrieval_count, frequency)| {
                self.memory_cluster
                    .get_node(memory_id)
                    .map(|note| Self::format_hot_memory(note, retrieval_count, frequency))
            })
            .collect()
    }

    fn compute_retrieval_frequency(retrieval_count: usize, span_seconds: i64) -> f64 {
        if retrieval_count == 0 {
            return 0.0;
        }

        let denominator = if span_seconds <= 0 {
            1.0
        } else {
            span_seconds as f64
        };

        retrieval_count as f64 / denominator
    }

    fn format_hot_memory(note: &MemoryNote, retrieval_count: usize, frequency: f64) -> String {
        format!(
            "memory_id={}, retrieval_count={}, frequency={:.6}, tags={:?}, content={:?}",
            note.id(),
            retrieval_count,
            frequency,
            note.tags(),
            note.mem_type()
        )
    }

    fn apply_ltp_ltd_updates(&mut self) {
        // TODO(beta_ver): 基于共激活频率更新连接强度（LTP/LTD），并处理低于阈值的断边。
        // 当前阶段仅完成“摘要拆分 -> 节点入图”的基础巩固流程。
    }
}

