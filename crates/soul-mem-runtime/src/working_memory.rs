use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::{sync::Arc, time::Instant};

pub mod llm;
pub mod record;
pub mod sliding_window;

use self::record::{Record, UserFeedback};
use self::sliding_window::SlidingWindow;
use crate::cluster::cluster_handle::MemoryClusterHandle;
use crate::cluster::memory_cluster::MemoryCluster;
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType, proc_mem::ProcMemLink};
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder, MemoryType};
use soul_mem_query::consolidation::service::{ConsolidationLlm, ConsolidationService};
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::{Embeddable, EmbeddingModel};
use soul_mem_query::storage::{
    ConsolidationBatchResult, MemoryLinkRecord, MemoryNoteRecord, MemoryRepository,
};

// 默认检索的热点记忆数量
const DEFAULT_TOP_K: usize = 8;
// 共激活链接的基础强度
const ACTIVATION_BASE_INTENSITY: f64 = 0.35;
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
    pub timing: ConsolidationTiming,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConsolidationTiming {
    pub total_ms: u128,
    pub llm_and_mapping_ms: u128,
    pub embedding_ms: u128,
    pub database_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HotMemoryStat {
    pub memory_id: String,
    pub retrieval_count: usize,
    pub frequency: f64,
}

// 工作记忆结构体(SlidingWindow & MemoryCluster & records)
pub struct WorkingMemory {
    state: WorkingState,
    sliding_window: SlidingWindow,
    memory_cluster: MemoryClusterHandle,
    records: HashMap<MemoryId, Record>,
    repository: Option<Arc<dyn MemoryRepository>>,
    retrieval_revision: u64,
    last_plasticity_revision: u64,
}

impl WorkingMemory {
    pub fn new(window_capacity: usize) -> Self {
        Self {
            state: WorkingState::Idle,
            sliding_window: SlidingWindow::new(window_capacity),
            memory_cluster: MemoryCluster::new().into_handle(),
            records: HashMap::new(),
            repository: None,
            retrieval_revision: 0,
            last_plasticity_revision: 0,
        }
    }

    // 为巩固流程配置数据库仓储。
    pub fn with_repository(mut self, repository: Arc<dyn MemoryRepository>) -> Self {
        self.repository = Some(repository);
        self
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
        llm: &dyn ConsolidationLlm,
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
        self.retrieval_revision = self.retrieval_revision.saturating_add(1);
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

    // 返回当前按检索频率排序的热点记忆 Top-K。
    pub fn hot_memory_stats(&self, top_k: usize) -> Vec<HotMemoryStat> {
        let top_k = if top_k == 0 { DEFAULT_TOP_K } else { top_k };
        self.collect_top_memory_stats_by_frequency(top_k)
            .into_iter()
            .map(|(memory_id, retrieval_count, frequency)| HotMemoryStat {
                memory_id: memory_id.to_string(),
                retrieval_count,
                frequency,
            })
            .collect()
    }

    // 只执行一次 LTP/LTD，不调用 LLM、摘要或 embedding。
    pub async fn run_plasticity_update(&mut self, hot_top_k: usize) -> Result<Vec<MemoryLink>> {
        let top_k = if hot_top_k == 0 {
            DEFAULT_TOP_K
        } else {
            hot_top_k
        };
        let hot_ids = self
            .collect_top_memory_stats_by_frequency(top_k)
            .into_iter()
            .map(|(memory_id, _, _)| memory_id)
            .collect::<Vec<_>>();
        let changed_links = if self.repository.is_some() {
            self.plan_repository_ltp_ltd_updates(&hot_ids).await?
        } else {
            self.plan_ltp_ltd_updates(&hot_ids)
        };

        if changed_links.is_empty() {
            self.last_plasticity_revision = self.retrieval_revision;
            return Ok(Vec::new());
        }

        let changed_links = if let Some(repository) = self.repository.as_ref() {
            let batch = repository
                .save_consolidation_batch(&[], &changed_links)
                .await?;
            Self::align_link_ids(&changed_links, &batch.links)?
        } else {
            changed_links
        };

        self.apply_link_updates(&changed_links);
        self.last_plasticity_revision = self.retrieval_revision;
        Ok(changed_links)
    }

    // 基于“摘要记忆 + 热点记忆”执行一次短期->工作记忆巩固
    pub async fn run_consolidation(
        &mut self,
        llm: &dyn ConsolidationLlm,
        embedding_model: &dyn EmbeddingModel,
        hot_top_k: usize,
    ) -> Result<ConsolidationRunResult> {
        let started_at = Instant::now();
        let summary_text = self.sliding_window.get_summary_text().await;
        if summary_text.trim().is_empty() {
            return Ok(ConsolidationRunResult {
                summary_was_empty: true,
                hot_memories_used: 0,
                notes_added: 0,
                timing: ConsolidationTiming {
                    total_ms: started_at.elapsed().as_millis(),
                    ..Default::default()
                },
            });
        }

        let top_k = if hot_top_k == 0 {
            DEFAULT_TOP_K
        } else {
            hot_top_k
        };
        let (hot_memories, activation_candidates) =
            self.collect_hot_memories_and_activation_candidates(top_k);

        let llm_started_at = Instant::now();
        let service = ConsolidationService::new();
        let mapped = service
            .split_summary_and_map(llm, &summary_text, &hot_memories)
            .await?;
        let llm_and_mapping_ms = llm_started_at.elapsed().as_millis();

        // 合并同批重复节点，并复用数据库中已有节点的 ID。
        let notes = Self::deduplicate_notes(mapped.notes)?;
        let notes = self.reuse_existing_note_ids(notes).await?;
        let notes_added = notes.len();
        let mut embedded_notes = Vec::with_capacity(notes_added);
        let mut storage_bundles = Vec::with_capacity(notes_added);
        let embedding_started_at = Instant::now();

        for mut note in notes {
            // 新关系使用基础强度，数据库或内存中已有的关系保留原有强度。
            self.prepare_note_links(&mut note).await?;

            let storage_embedding = if self.repository.is_some() {
                Some(Self::build_storage_embedding(&note, embedding_model)?)
            } else {
                None
            };
            let embedded = note.embed_and_fuse(embedding_model)?;

            if let Some(storage_embedding) = storage_embedding {
                storage_bundles.push((embedded.note().clone(), storage_embedding));
            }
            embedded_notes.push(embedded);
        }
        let embedding_ms = embedding_started_at.elapsed().as_millis();

        // 先计算待更新的边，数据库成功后再修改内存图。
        let changed_links = self.plan_ltp_ltd_updates(&activation_candidates);

        let database_started_at = Instant::now();
        let database_batch = if let Some(repository) = self.repository.as_ref() {
            Some(
                repository
                    .save_consolidation_batch(&storage_bundles, &changed_links)
                    .await?,
            )
        } else {
            None
        };
        // 整个巩固批次成功写入数据库后，再加入内存集群并更新边。
        let (embedded_notes, changed_links) = if let Some(batch) = database_batch.as_ref() {
            (
                Self::align_embedded_notes(embedded_notes, batch)?,
                Self::align_link_ids(&changed_links, &batch.links)?,
            )
        } else {
            (embedded_notes, changed_links)
        };
        for embedded in embedded_notes {
            self.add_node(embedded);
        }
        self.apply_link_updates(&changed_links);
        self.last_plasticity_revision = self.retrieval_revision;
        self.sliding_window.clear_summary().await;
        let database_ms = database_started_at.elapsed().as_millis();

        Ok(ConsolidationRunResult {
            summary_was_empty: false,
            hot_memories_used: hot_memories.len(),
            notes_added,
            timing: ConsolidationTiming {
                total_ms: started_at.elapsed().as_millis(),
                llm_and_mapping_ms,
                embedding_ms,
                database_ms,
            },
        })
    }

    fn deduplicate_notes(notes: Vec<MemoryNote>) -> Result<Vec<MemoryNote>> {
        let mut canonical_notes: Vec<MemoryNote> = Vec::new();
        let mut canonical_by_identity: HashMap<(String, String), usize> = HashMap::new();
        let mut id_map: HashMap<MemoryId, MemoryId> = HashMap::new();

        for note in notes {
            let record = MemoryNoteRecord::from_note(&note)?;
            let identity = (record.kind.as_str().to_string(), record.identity_content);
            if let Some(&canonical_index) = canonical_by_identity.get(&identity) {
                let canonical_id = canonical_notes[canonical_index].id();
                id_map.insert(note.id(), canonical_id);
                let canonical_note = &mut canonical_notes[canonical_index];
                canonical_note.merge_tags(note.tags().iter().cloned());
                if let MemoryType::Semantic(semantic) = note.mem_type() {
                    canonical_note.merge_aliases(semantic.aliases.iter().cloned());
                }
                for link in note.links().iter().cloned() {
                    canonical_note.add_link(link);
                }
            } else {
                canonical_by_identity.insert(identity, canonical_notes.len());
                canonical_notes.push(note);
            }
        }

        canonical_notes
            .into_iter()
            .map(|note| Self::remap_note_ids(note, &id_map, None))
            .collect()
    }

    fn align_embedded_notes(
        embedded_notes: Vec<EmbeddedMemoryNote>,
        batch: &ConsolidationBatchResult,
    ) -> Result<Vec<EmbeddedMemoryNote>> {
        if embedded_notes.len() != batch.notes.len() {
            return Err(anyhow::anyhow!(
                "database returned {} note records for {} embedded notes",
                batch.notes.len(),
                embedded_notes.len()
            ));
        }

        let id_map = embedded_notes
            .iter()
            .zip(&batch.notes)
            .map(|(embedded, record)| Ok((embedded.note().id(), record.parse_memory_id()?)))
            .collect::<Result<HashMap<_, _>>>()?;

        embedded_notes
            .into_iter()
            .map(|embedded| {
                let mut note = Self::remap_note_ids(embedded.note, &id_map, None)?;
                for link in note.links_mut() {
                    *link = Self::align_link_id(link, &batch.links)?;
                }
                Ok(EmbeddedMemoryNote {
                    embedding: embedded.embedding,
                    note,
                })
            })
            .collect()
    }

    fn align_link_ids(
        links: &[MemoryLink],
        records: &[MemoryLinkRecord],
    ) -> Result<Vec<MemoryLink>> {
        links
            .iter()
            .map(|link| Self::align_link_id(link, records))
            .collect()
    }

    fn align_link_id(link: &MemoryLink, records: &[MemoryLinkRecord]) -> Result<MemoryLink> {
        let candidate = MemoryLinkRecord::from_link(link)?;
        let record = records
            .iter()
            .find(|record| record.has_same_identity_as(&candidate))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "database did not return memory link {} -> {}",
                    candidate.from,
                    candidate.to
                )
            })?;
        Ok(record.to_link()?)
    }

    async fn reuse_existing_note_ids(&self, notes: Vec<MemoryNote>) -> Result<Vec<MemoryNote>> {
        let Some(repository) = self.repository.as_ref() else {
            return Ok(notes);
        };

        let mut existing_records = HashMap::new();
        for note in &notes {
            if let Some(record) = repository.find_note_by_content(note).await? {
                existing_records.insert(note.id(), record);
            }
        }
        let id_map = existing_records
            .iter()
            .map(|(new_id, record)| Ok((*new_id, record.parse_memory_id()?)))
            .collect::<Result<HashMap<_, _>>>()?;

        notes
            .into_iter()
            .map(|note| {
                let note_id = note.id();
                Self::remap_note_ids(note, &id_map, existing_records.get(&note_id))
            })
            .collect()
    }

    fn remap_note_ids(
        note: MemoryNote,
        id_map: &HashMap<MemoryId, MemoryId>,
        existing_record: Option<&MemoryNoteRecord>,
    ) -> Result<MemoryNote> {
        let (from_id, note_id) = (note.id(), *id_map.get(&note.id()).unwrap_or(&note.id()));
        let links = note
            .links()
            .iter()
            .cloned()
            .map(|link| {
                let link_id = link.id();
                let (from, to, link_type, intensity) = link.into_tuple();
                MemoryLink::from_tuple_with_id(
                    link_id,
                    *id_map.get(&from).unwrap_or(&from),
                    *id_map.get(&to).unwrap_or(&to),
                    link_type,
                    intensity,
                )
            })
            .collect::<Vec<_>>();

        if let Some(existing_record) = existing_record {
            let mut existing_note = existing_record.to_note(links)?;
            existing_note.merge_tags(note.tags().iter().cloned());
            if let MemoryType::Semantic(semantic) = note.mem_type() {
                existing_note.merge_aliases(semantic.aliases.iter().cloned());
            }
            return Ok(existing_note);
        }

        MemoryNoteBuilder::new(note.mem_type().clone())
            .id(note_id)
            .tags(note.tags().to_vec())
            .retrieval_count(note.retrieval_count())
            .create_time(note.creation_time())
            .last_accessed_time(note.last_accessed_time())
            .mem_links(links)
            .build()
            .map_err(|err| anyhow::anyhow!("failed to remap memory note {from_id}: {err}"))
    }

    fn build_storage_embedding(
        note: &MemoryNote,
        embedding_model: &dyn EmbeddingModel,
    ) -> Result<Vec<f32>> {
        let input = serde_json::to_string(&(note.tags(), note.mem_type()))?;
        Ok(embedding_model
            .infer_with_chunk(&input)?
            .into_iter()
            .collect())
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

    async fn prepare_note_links(&self, note: &mut MemoryNote) -> Result<()> {
        let existing_links = if let Some(repository) = self.repository.as_ref() {
            repository.list_outbound_links(note.id()).await?
        } else {
            self.memory_cluster.read_or_compute(|cluster| {
                cluster
                    .get_node(note.id())
                    .map(|embedded| {
                        embedded
                            .note()
                            .links()
                            .iter()
                            .map(MemoryLinkRecord::from_link)
                            .collect::<soul_mem_query::storage::StorageResult<Vec<_>>>()
                    })
                    .transpose()
            })?
            .unwrap_or_default()
        };

        for link in note.links_mut() {
            let candidate = MemoryLinkRecord::from_link(link)?;
            if let Some(existing) = existing_links
                .iter()
                .find(|existing| existing.has_same_identity_as(&candidate))
            {
                *link = existing.to_link()?;
            } else {
                Self::apply_base_intensity(link);
            }
        }
        Ok(())
    }

    fn apply_base_intensity(link: &mut MemoryLink) {
        link.intensity = ACTIVATION_BASE_INTENSITY;
        match link.link_type_mut() {
            MemoryLinkType::Sem(sem) => {
                sem.confidence = ACTIVATION_BASE_INTENSITY as f32;
            }
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(trig)) => {
                trig.set_prob(ACTIVATION_BASE_INTENSITY);
            }
            MemoryLinkType::Sit(_) => {}
        }
    }
    // 计算热点节点之间已有边的 LTP/LTD 变化，不修改内存图。
    fn plan_ltp_ltd_updates(&self, hot_ids: &[MemoryId]) -> Vec<MemoryLink> {
        if hot_ids.len() < 2 || self.retrieval_revision == self.last_plasticity_revision {
            return Vec::new();
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
            return Vec::new();
        }

        self.memory_cluster.read_or_compute(|cluster| {
            let mut changed_links = Vec::new();
            for from in hot_ids {
                let Some(source_note) = cluster.get_node(*from) else {
                    continue;
                };

                let from_freq = *freq_by_id.get(from).unwrap_or(&0.0);

                for link in source_note.note.links() {
                    let to = link.to();
                    if !hot_set.contains(&to) {
                        continue;
                    }

                    let to_freq = *freq_by_id.get(&to).unwrap_or(&0.0);
                    let norm = (from_freq.min(to_freq) / max_freq).clamp(0.0, 1.0);
                    let mut updated_link = link.clone();
                    Self::shift_link_intensity(&mut updated_link, Self::plasticity_delta(norm));
                    changed_links.push(updated_link);
                }
            }
            changed_links
        })
    }

    // 从数据库读取已有边并计算 LTP/LTD，支持进程重启后继续更新旧数据。
    async fn plan_repository_ltp_ltd_updates(
        &self,
        hot_ids: &[MemoryId],
    ) -> Result<Vec<MemoryLink>> {
        if hot_ids.len() < 2 || self.retrieval_revision == self.last_plasticity_revision {
            return Ok(Vec::new());
        }
        let Some(repository) = self.repository.as_ref() else {
            return Ok(self.plan_ltp_ltd_updates(hot_ids));
        };

        let hot_set = hot_ids.iter().copied().collect::<HashSet<_>>();
        let mut freq_by_id = HashMap::with_capacity(hot_ids.len());
        let mut max_freq = 0.0_f64;
        for memory_id in hot_ids {
            let frequency = self
                .records
                .get(memory_id)
                .map(|record| {
                    Self::compute_retrieval_frequency(
                        record.retrieval_count(),
                        record.access_time_span(),
                    )
                })
                .unwrap_or(0.0);
            max_freq = max_freq.max(frequency);
            freq_by_id.insert(*memory_id, frequency);
        }
        if max_freq <= f64::EPSILON {
            return Ok(Vec::new());
        }

        let mut changed_links = Vec::new();
        for from in hot_ids {
            let from_frequency = *freq_by_id.get(from).unwrap_or(&0.0);
            for record in repository.list_outbound_links(*from).await? {
                let mut link = record.to_link()?;
                let to = link.to();
                if !hot_set.contains(&to) {
                    continue;
                }
                let to_frequency = *freq_by_id.get(&to).unwrap_or(&0.0);
                let norm = (from_frequency.min(to_frequency) / max_freq).clamp(0.0, 1.0);
                Self::shift_link_intensity(&mut link, Self::plasticity_delta(norm));
                changed_links.push(link);
            }
        }
        Ok(changed_links)
    }

    fn plasticity_delta(norm: f64) -> f64 {
        let threshold = LTP_NORM_THRESHOLD;
        if norm >= threshold {
            LTP_STEP_MAX * (norm - threshold) / (1.0 - threshold)
        } else {
            -LTD_STEP_MAX * (threshold - norm) / threshold
        }
    }

    // 将数据库已提交的 LTP/LTD 结果应用到内存图。
    fn apply_link_updates(&mut self, links: &[MemoryLink]) {
        let source_ids = links.iter().map(|link| link.from()).collect::<HashSet<_>>();

        self.memory_cluster.write(|cluster| {
            for updated_link in links {
                let Some(source_note) = cluster.get_node_mut(updated_link.from()) else {
                    continue;
                };
                if let Some(link) = source_note
                    .note
                    .links_mut()
                    .iter_mut()
                    .find(|link| link.id() == updated_link.id())
                {
                    *link = updated_link.clone();
                }
            }

            for source_id in source_ids {
                cluster.refresh_node(&source_id);
            }
        });
    }

    fn shift_link_intensity(link: &mut MemoryLink, delta: f64) {
        link.intensity = (link.intensity + delta).clamp(LINK_INTENSITY_MIN, LINK_INTENSITY_MAX);
        match link.link_type_mut() {
            MemoryLinkType::Sem(sem) => {
                sem.confidence = (sem.confidence as f64 + delta)
                    .clamp(LINK_INTENSITY_MIN, LINK_INTENSITY_MAX)
                    as f32;
            }
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(trig)) => {
                let next_prob =
                    (trig.get_prob() + delta).clamp(LINK_INTENSITY_MIN, LINK_INTENSITY_MAX);
                trig.set_prob(next_prob);
            }
            MemoryLinkType::Sit(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_note::{
        MemoryType,
        sem_mem::{ConceptType, SemMemory},
    };
    use soul_mem_query::embedding::{
        EmbeddingVec,
        note::{MemoryEmbedding, MemoryEmbeddingVariant},
        sem::SemanticEmbedding,
    };

    fn embedded_semantic_note(note: MemoryNote) -> EmbeddedMemoryNote {
        let embedding = MemoryEmbedding::new(
            EmbeddingVec::zero(128),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(128),
                EmbeddingVec::zero(128),
                EmbeddingVec::zero(128),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    #[test]
    fn deduplicate_notes_merges_same_kind_and_content() {
        let first = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "原神".to_string(),
            ConceptType::Entity,
            "第一条描述".to_string(),
        )))
        .tags(vec!["游戏".to_string()])
        .build()
        .expect("build first note");
        let mut second_memory = SemMemory::new(
            "原神".to_string(),
            ConceptType::Entity,
            "第二条描述".to_string(),
        );
        second_memory.aliases.push("Genshin Impact".to_string());
        let second = MemoryNoteBuilder::new(MemoryType::Semantic(second_memory))
            .tags(vec!["开放世界".to_string()])
            .build()
            .expect("build second note");

        let notes =
            WorkingMemory::deduplicate_notes(vec![first, second]).expect("deduplicate notes");

        assert_eq!(notes.len(), 1);
        assert_eq!(
            notes[0].tags(),
            &["游戏".to_string(), "开放世界".to_string()]
        );
        match notes[0].mem_type() {
            MemoryType::Semantic(semantic) => {
                assert_eq!(semantic.aliases, vec!["Genshin Impact".to_string()]);
            }
            _ => panic!("expected semantic note"),
        }
    }

    #[tokio::test]
    async fn existing_node_keeps_existing_link_and_initializes_new_link() {
        let source_id = MemoryId::new();
        let existing_target_id = MemoryId::new();
        let new_target_id = MemoryId::new();
        let mut existing_source = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "原神".to_string(),
            ConceptType::Entity,
            "一款游戏".to_string(),
        )))
        .id(source_id)
        .build()
        .expect("build existing source");
        let mut existing_link = MemoryLink::new(
            source_id,
            existing_target_id,
            MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 0.8)),
        );
        existing_link.intensity = 0.8;
        let existing_link_id = existing_link.id();
        existing_source.add_link(existing_link);

        let mut candidate = MemoryNoteBuilder::new(existing_source.mem_type().clone())
            .id(source_id)
            .build()
            .expect("build candidate source");
        candidate.add_link(MemoryLink::new(
            source_id,
            existing_target_id,
            MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 1.0)),
        ));
        candidate.add_link(MemoryLink::new(
            source_id,
            new_target_id,
            MemoryLinkType::Sem(SemMemLink::new("具有特征".to_string(), 1.0)),
        ));

        let mut working_memory = WorkingMemory::new(4);
        working_memory.add_node(embedded_semantic_note(existing_source));
        working_memory
            .prepare_note_links(&mut candidate)
            .await
            .expect("prepare candidate links");

        let preserved = candidate
            .links()
            .iter()
            .find(|link| link.to() == existing_target_id)
            .expect("find existing link");
        assert_eq!(preserved.id(), existing_link_id);
        assert!((preserved.intensity - 0.8).abs() < f64::EPSILON);

        let initialized = candidate
            .links()
            .iter()
            .find(|link| link.to() == new_target_id)
            .expect("find new link");
        assert!((initialized.intensity - ACTIVATION_BASE_INTENSITY).abs() < f64::EPSILON);
        match initialized.link_type() {
            MemoryLinkType::Sem(semantic) => {
                assert!(
                    (semantic.confidence - ACTIVATION_BASE_INTENSITY as f32).abs()
                        < f32::EPSILON
                );
            }
            _ => panic!("expected semantic link"),
        }
    }

    #[test]
    fn ltp_updates_existing_edge_with_chinese_relation() {
        let source_id = MemoryId::new();
        let target_id = MemoryId::new();
        let mut source = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "原神".to_string(),
            ConceptType::Entity,
            "一款游戏".to_string(),
        )))
        .id(source_id)
        .build()
        .expect("build source note");
        let target = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "游戏".to_string(),
            ConceptType::Abstract,
            "娱乐产品类型".to_string(),
        )))
        .id(target_id)
        .build()
        .expect("build target note");
        let mut link = MemoryLink::new(
            source_id,
            target_id,
            MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 0.4)),
        );
        link.intensity = 0.4;
        source.add_link(link);

        let mut working_memory = WorkingMemory::new(4);
        working_memory.add_node(embedded_semantic_note(source));
        working_memory.add_node(embedded_semantic_note(target));
        working_memory.record_retrieval(source_id);
        working_memory.record_retrieval(target_id);

        let updates = working_memory.plan_ltp_ltd_updates(&[source_id, target_id]);

        assert_eq!(updates.len(), 1);
        assert!((updates[0].intensity - 0.48).abs() < f64::EPSILON);
        match updates[0].link_type() {
            MemoryLinkType::Sem(semantic) => {
                assert_eq!(semantic.verb, "属于类型");
                assert!((semantic.confidence - 0.48).abs() < f32::EPSILON);
            }
            _ => panic!("expected semantic link"),
        }
    }

    #[test]
    fn plasticity_delta_is_continuous_and_bounded() {
        assert!((WorkingMemory::plasticity_delta(0.0) + 0.05).abs() < f64::EPSILON);
        assert!(WorkingMemory::plasticity_delta(0.5).abs() < f64::EPSILON);
        assert!((WorkingMemory::plasticity_delta(1.0) - 0.08).abs() < f64::EPSILON);
        assert!(WorkingMemory::plasticity_delta(0.49) < 0.0);
        assert!(WorkingMemory::plasticity_delta(0.51) > 0.0);
    }

    #[test]
    fn plasticity_skips_when_no_new_retrieval_exists() {
        let source_id = MemoryId::new();
        let target_id = MemoryId::new();
        let mut source = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "原神".to_string(),
            ConceptType::Entity,
            "一款游戏".to_string(),
        )))
        .id(source_id)
        .build()
        .expect("build source note");
        let target = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "游戏".to_string(),
            ConceptType::Abstract,
            "娱乐产品类型".to_string(),
        )))
        .id(target_id)
        .build()
        .expect("build target note");
        source.add_link(MemoryLink::new(
            source_id,
            target_id,
            MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 0.4)),
        ));

        let mut working_memory = WorkingMemory::new(4);
        working_memory.add_node(embedded_semantic_note(source));
        working_memory.add_node(embedded_semantic_note(target));
        working_memory.record_retrieval(source_id);
        working_memory.record_retrieval(target_id);

        assert_eq!(
            working_memory
                .plan_ltp_ltd_updates(&[source_id, target_id])
                .len(),
            1
        );
        working_memory.last_plasticity_revision = working_memory.retrieval_revision;
        assert!(
            working_memory
                .plan_ltp_ltd_updates(&[source_id, target_id])
                .is_empty()
        );
    }

    #[tokio::test]
    async fn plasticity_update_changes_existing_link_once() {
        let source_id = MemoryId::new();
        let target_id = MemoryId::new();
        let mut source = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "原神".to_string(),
            ConceptType::Entity,
            "一款游戏".to_string(),
        )))
        .id(source_id)
        .build()
        .expect("build source note");
        let target = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "游戏".to_string(),
            ConceptType::Abstract,
            "娱乐产品类型".to_string(),
        )))
        .id(target_id)
        .build()
        .expect("build target note");
        let mut link = MemoryLink::new(
            source_id,
            target_id,
            MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 0.4)),
        );
        link.intensity = 0.4;
        source.add_link(link);

        let mut working_memory = WorkingMemory::new(4);
        working_memory.add_node(embedded_semantic_note(source));
        working_memory.add_node(embedded_semantic_note(target));
        working_memory.record_retrieval(source_id);
        working_memory.record_retrieval(target_id);

        let changes = working_memory
            .run_plasticity_update(8)
            .await
            .expect("run plasticity update");
        assert_eq!(changes.len(), 1);
        assert!((changes[0].intensity - 0.48).abs() < f64::EPSILON);
        assert!(
            working_memory
                .run_plasticity_update(8)
                .await
                .expect("run second plasticity update")
                .is_empty()
        );
    }

    #[test]
    fn repeated_consolidation_uses_database_link_id_in_memory() {
        let source_id = MemoryId::new();
        let target_id = MemoryId::new();
        let make_source = || {
            let mut source = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
                "原神".to_string(),
                ConceptType::Entity,
                "一款游戏".to_string(),
            )))
            .id(source_id)
            .build()
            .expect("build source note");
            let mut link = MemoryLink::new(
                source_id,
                target_id,
                MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 0.35)),
            );
            link.intensity = 0.35;
            source.add_link(link);
            source
        };
        let target = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "游戏".to_string(),
            ConceptType::Abstract,
            "娱乐产品类型".to_string(),
        )))
        .id(target_id)
        .build()
        .expect("build target note");
        let first_source = make_source();
        let second_source = make_source();
        let first_temporary_id = first_source.links()[0].id();
        let second_temporary_id = second_source.links()[0].id();
        let mut database_link = MemoryLink::new(
            source_id,
            target_id,
            MemoryLinkType::Sem(SemMemLink::new("属于类型".to_string(), 0.35)),
        );
        database_link.intensity = 0.35;
        let database_link_id = database_link.id();
        let batch = ConsolidationBatchResult {
            notes: vec![
                MemoryNoteRecord::from_note(&first_source).expect("convert source record"),
                MemoryNoteRecord::from_note(&target).expect("convert target record"),
            ],
            links: vec![
                MemoryLinkRecord::from_link(&database_link).expect("convert database link"),
            ],
        };

        let first = WorkingMemory::align_embedded_notes(
            vec![
                embedded_semantic_note(first_source),
                embedded_semantic_note(target.clone()),
            ],
            &batch,
        )
        .expect("align first consolidation");
        let second = WorkingMemory::align_embedded_notes(
            vec![
                embedded_semantic_note(second_source),
                embedded_semantic_note(target),
            ],
            &batch,
        )
        .expect("align second consolidation");

        let mut working_memory = WorkingMemory::new(4);
        for embedded in first.into_iter().chain(second) {
            working_memory.add_node(embedded);
        }

        working_memory.memory_cluster.read_or_compute(|cluster| {
            assert!(cluster.has_edge(database_link_id));
            assert!(!cluster.has_edge(first_temporary_id));
            assert!(!cluster.has_edge(second_temporary_id));
            let source = cluster.get_node(source_id).expect("source exists");
            assert_eq!(source.note().links().len(), 1);
            assert_eq!(source.note().links()[0].id(), database_link_id);
        });
    }
}
