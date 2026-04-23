use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub mod llm;
pub mod record;
pub mod sliding_window;

use self::record::{Record, UserFeedback};
use self::sliding_window::SlidingWindow;
use crate::memory::cluster::cluster_handle::MemoryClusterHandle;
use crate::memory::cluster::memory_cluster::MemoryCluster;
use crate::memory::consolidation::service::ConsolidationService;
use crate::memory::embedding::note::EmbeddedMemoryNote;
use crate::memory::embedding::{Embeddable, EmbeddingModel};
use crate::memory::memory_links::{
    MemoryLink, MemoryLinkType,
    proc_mem::ProcMemLink,
};
use crate::memory::memory_note::{MemoryId, MemoryNote};

// 默认检索的热点记忆数量
const DEFAULT_TOP_K: usize = 8;
// 共激活关系词
const ACTIVATION_RELATION: &str = "co_activated";
// 共激活链接的基础强度
const ACTIVATION_BASE_INTENSITY: f32 = 0.35;
// LTP/LTD 参数
const LTP_NORM_THRESHOLD: f64 = 0.50;
const LTP_STEP_MAX: f64 = 0.08;
const LTD_STEP_MAX: f64 = 0.05;
const LINK_INTENSITY_MIN: f64 = 0.0;
const LINK_INTENSITY_MAX: f64 = 1.0;

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

    // Working -> Idle 后触发巩固
    pub async fn transition_to_idle_and_consolidate(
        &mut self,
        llm: &llm::client::LlmClient,
        embedding_model: &dyn EmbeddingModel,
        hot_top_k: usize,
    ) -> Result<ConsolidationRunResult> {
        self.transition_to_idle();
        self.run_consolidation(llm, embedding_model, hot_top_k)
            .await
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

        if !self.records.contains_key(&node_id) {
            self.records.insert(node_id, Record::new(node_id));
        }
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
        let (hot_memories, activation_candidates) =
            self.collect_hot_memories_and_activation_candidates(top_k);

        let service = ConsolidationService::new();
        let mapped = service
            .split_summary_and_map(llm, &summary_text, &hot_memories)
            .await?;

        // TODO整合
        // 基于热点记忆对旧节点做内容更新/补边，目前先完成“摘要拆分并入图”主路径。这里重复入图的逻辑还没处理
        let notes_added = mapped.notes.len();
        for mut note in mapped.notes {
            // 当前阶段不做语义去重：仅当节点 ID 不存在时，按“新节点”对其边赋基础强度后再入图。
            let note_id = note.id();
            let exists = self
                .memory_cluster
                .read_or_compute(|cluster| cluster.contains_node(note_id));
            if !exists {
                Self::apply_base_intensity_to_note_links(&mut note);
            }

            let embedded = note.embed_and_fuse(embedding_model)?;
            self.add_node(embedded);
        }

        // 热点只更新已有共激活边强度，不在此阶段补新边。
        self.apply_ltp_ltd_updates(&activation_candidates);
        self.sliding_window.clear_summary().await;

        Ok(ConsolidationRunResult {
            summary_was_empty: false,
            hot_memories_used: hot_memories.len(),
            notes_added,
        })
    }

    fn collect_hot_memories_and_activation_candidates(
        &self,
        top_k: usize,
    ) -> (Vec<String>, Vec<MemoryId>) {
        let ranked = self.collect_top_memory_stats_by_frequency(top_k);
        if ranked.is_empty() {
            return (Vec::new(), Vec::new());
        }

        self.memory_cluster.read_or_compute(move |cluster| {
            let mut hot_memories = Vec::with_capacity(ranked.len());
            let mut activation_candidates = Vec::with_capacity(ranked.len());

            for (memory_id, retrieval_count, frequency) in ranked {
                if let Some(embedded) = cluster.get_node(memory_id) {
                    hot_memories.push(Self::format_hot_memory(
                        embedded.note(),
                        retrieval_count,
                        frequency,
                    ));
                    activation_candidates.push(memory_id);
                }
            }

            (hot_memories, activation_candidates)
        })
    }

    fn collect_top_memory_stats_by_frequency(&self, top_k: usize) -> Vec<(MemoryId, usize, f64)> {
        if top_k == 0 {
            return Vec::new();
        }

        let mut ranked = self
            .records
            .values()
            .map(|record| {
                let retrieval_count = record.retrieval_count();
                let frequency =
                    Self::compute_retrieval_frequency(retrieval_count, record.access_time_span());
                (record.memory_id(), retrieval_count, frequency)
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.1.cmp(&a.1)));
        ranked.into_iter().take(top_k).collect()
    }

    fn compute_retrieval_frequency(retrieval_count: usize, span_seconds: i64) -> f64 {
        let denominator = span_seconds.max(1) as f64;
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

    fn apply_base_intensity_to_note_links(note: &mut MemoryNote) {
        for link in note.links_mut() {
            link.intensity = ACTIVATION_BASE_INTENSITY as f64;
            match link.link_type_mut() {
                MemoryLinkType::Sem(sem) => {
                    sem.intensity = ACTIVATION_BASE_INTENSITY;
                }
                MemoryLinkType::Proc(ProcMemLink::TrigToAction(trig)) => {
                    trig.set_prob(ACTIVATION_BASE_INTENSITY);
                }
                MemoryLinkType::Sit(_) => {}
            }
        }
    }
    //公式待定
    fn apply_ltp_ltd_updates(&mut self, hot_ids: &[MemoryId]) {
        if hot_ids.len() < 2 {
            return;
        }

        let hot_set = hot_ids.iter().copied().collect::<HashSet<_>>();
        let mut freq_by_id = HashMap::with_capacity(hot_ids.len());
        let mut max_freq = 0.0_f64;

        for memory_id in hot_ids {
            let freq = self
                .records
                .get(memory_id)
                .map(|record| {
                    Self::compute_retrieval_frequency(
                        record.retrieval_count(),
                        record.access_time_span(),
                    )
                })
                .unwrap_or(0.0);
            if freq > max_freq {
                max_freq = freq;
            }
            freq_by_id.insert(*memory_id, freq);
        }

        if max_freq <= f64::EPSILON {
            return;
        }

        self.memory_cluster.write(|cluster| {
            let mut touched_sources = Vec::new();

            for from in hot_ids {
                let Some(source_note) = cluster.get_node_mut(*from) else {
                    continue;
                };

                let mut changed = false;
                let from_freq = *freq_by_id.get(from).unwrap_or(&0.0);

                for link in source_note.note.links_mut().iter_mut() {
                    let to = link.to();
                    if !hot_set.contains(&to) {
                        continue;
                    }
                    if !Self::is_coactivated_sem_link(link) {
                        continue;
                    }

                    let to_freq = *freq_by_id.get(&to).unwrap_or(&0.0);
                    let norm = (from_freq.min(to_freq) / max_freq).clamp(0.0, 1.0);
                    if norm >= LTP_NORM_THRESHOLD {
                        Self::shift_link_intensity(link, LTP_STEP_MAX * norm);
                    } else {
                        Self::shift_link_intensity(link, -LTD_STEP_MAX * (1.0 - norm));
                    }
                    changed = true;
                }

                if changed {
                    touched_sources.push(*from);
                }
            }

            for source_id in touched_sources {
                cluster.refresh_node(&source_id);
            }
        });
    }

    fn is_coactivated_sem_link(link: &MemoryLink) -> bool {
        matches!(
            link.link_type(),
            MemoryLinkType::Sem(sem) if sem.verb == ACTIVATION_RELATION
        )
    }

    fn shift_link_intensity(link: &mut MemoryLink, delta: f64) {
        link.intensity = (link.intensity + delta).clamp(LINK_INTENSITY_MIN, LINK_INTENSITY_MAX);
        match link.link_type_mut() {
            MemoryLinkType::Sem(sem) => {
                sem.intensity =
                    (sem.intensity as f64 + delta).clamp(LINK_INTENSITY_MIN, LINK_INTENSITY_MAX)
                        as f32;
            }
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(trig)) => {
                let next_prob = (trig.get_prob() as f64 + delta)
                    .clamp(LINK_INTENSITY_MIN, LINK_INTENSITY_MAX)
                    as f32;
                trig.set_prob(next_prob);
            }
            MemoryLinkType::Sit(_) => {}
        }
    }
}
