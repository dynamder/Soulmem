//! 服务核心：串行命令循环与各用例实现。
//!
//! 服务持有唯一的 `WorkingMemory`，所有请求经命令通道**串行**处理，因此工作记忆
//! 永不被并发改写。一次请求按 `docs/architecture/orchestration.md` 处理：
//!
//! 1. 信息增量：逐条压入滑动窗口（可能触发摘要）；
//! 2. 检索：query 集合逐条跑 `DefaultPipeline`，按 priority 加权合并，取 top-k
//!    并渲染成一段自然语言（命中节点渲染后从工作记忆移出，DB 中仍有权威副本）；
//! 3. 控制信号：巩固 / 持久化 / 快照 / 暂停 / 恢复 / 维护。
//!
//! 三部分可同时出现在一条请求里；任一步失败都只记录首个错误，其余步骤继续执行，
//! 响应里带回已经发生的变化（`accepted` / `state`）。
//!
//! ## 所有权：`WorkingMemory` 归服务独占
//!
//! 检索管线要求 `Arc<WorkingMemory>`（见 `DefaultPipelineConfig::into_request`），所以
//! 服务以 `Arc` 持有它，但**全进程只有服务持有这份 Arc**：每次检索只在调用点临时
//! `Arc::clone`，管线同步返回后引用即释放。因此需要可变访问时（状态迁移、活跃记录
//! 更新、入簇），可在这些边界上用 [`Arc::get_mut`] 取到唯一可变引用。`get_mut` 返回
//! `None` 属不可达情形，此时只记一条 `tracing::error!` 并跳过该次状态变更，不 panic
//! （见 `AGENTS.md` §4）。

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::{broadcast, mpsc, oneshot};

use soul_mem_algo::algo::consolidate::generate::generate_memories_from_summary;
use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_algo::algo::retrieve::association::AssociationConfig;
use soul_mem_algo::algo::retrieve::complex::default_pipeline::DefaultPipelineResult;
use soul_mem_algo::algo::retrieve::complex::{
    AssociateWithActionConfig, DefaultPipelineConfig, RetrDefaultPipeline,
};
use soul_mem_algo::algo::retrieve::prefetch_db;
use soul_mem_algo::algo::retrieve::short_only::ShortOnlyConfig;
use soul_mem_algo::algo::retrieve::similarity::SimilarityConfig;
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::{MemoryId, MemoryNote};
use soul_mem_llm::LlmEngine;
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::embedding::EmbeddingModel;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
use soul_mem_query::query::retrieve::PrioritizedMemoryRetrieveQuery;
use soul_mem_runtime::storage::MemoryRepository;
use soul_mem_runtime::working_memory::WorkingMemory;
use soul_mem_runtime::working_memory::sliding_window::{Information, Summary};

use super::types::{
    ControlAction, Delta, ServiceEvent, ServiceRequest, ServiceResponse, ServiceState,
};
use super::{merge, render};
use crate::config::Config;
use crate::error::{ServiceError, ServiceResult};

/// 默认相似度兜底分。
const SIMILARITY_THRESHOLD: f32 = 0.35;
/// 管线内 Bayes 动作推理保留的 top-k（动作结果当前不参与输出渲染）。
const ACTION_TOP_K: usize = 3;
/// 抽象情境源在 Bayes 动作提取中的权重倍率。
const ABSTRACT_SOURCE_PRIORITY: f64 = 2.0;
/// 巩固时新节点链接到活跃节点的数量上限。
const CONSOLIDATION_LINK_TOP_K: usize = 3;
/// 巩固生成链接的默认谓词。
const CONSOLIDATION_LINK_VERB: &str = "related";
/// 巩固生成链接的默认置信度。
const CONSOLIDATION_LINK_CONFIDENCE: f32 = 1.0;

/// 交给核心处理的命令。
pub enum Command {
    /// 组合请求（检索 / 增量 / 控制）。
    Exchange {
        /// 请求内容。
        request: ServiceRequest,
        /// 响应回传通道。
        reply: oneshot::Sender<ServiceResult<ServiceResponse>>,
    },
}

/// 服务核心状态。
pub struct MemoryService {
    config: Arc<Config>,
    wm: Arc<WorkingMemory>,
    repo: Arc<dyn MemoryRepository>,
    llm: Arc<LlmEngine>,
    model: Arc<dyn EmbeddingModel + Send + Sync>,
    pipeline: DefaultPipelineConfig,
    paused: bool,
    /// 最近一次有增量/检索输入的时刻，用于 Idle 判定。
    last_activity: Instant,
    last_consolidation_at: Option<DateTime<Utc>>,
    events: broadcast::Sender<ServiceEvent>,
    llm_available: bool,
}

impl MemoryService {
    /// 组装服务核心。各依赖由引导层注入，便于替换与测试。
    ///
    /// `config` 以 `Arc` 注入：服务与引导层共享同一份设置，运行期不再重建。
    pub fn new(
        config: Arc<Config>,
        wm: Arc<WorkingMemory>,
        repo: Arc<dyn MemoryRepository>,
        llm: Arc<LlmEngine>,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
        events: broadcast::Sender<ServiceEvent>,
        llm_available: bool,
    ) -> Self {
        let pipeline = DefaultPipelineConfig {
            short_mem_with_history: ShortOnlyConfig {
                clipping_length: None,
                include_summary: true,
            },
            similarity: SimilarityConfig {
                similarity_threshold: SIMILARITY_THRESHOLD,
                max_results: config.top_k,
            },
            assoc_with_action: AssociateWithActionConfig {
                association: AssociationConfig::default(),
                action_top_k: ACTION_TOP_K,
                abstract_source_priority: ABSTRACT_SOURCE_PRIORITY,
            },
        };
        Self {
            config,
            wm,
            repo,
            llm,
            model,
            pipeline,
            paused: false,
            last_activity: Instant::now(),
            last_consolidation_at: None,
            events,
            llm_available,
        }
    }

    /// 处理一次组合请求。
    ///
    /// 不在失败时丢弃整条请求：首个错误记录进 `error`，其余步骤尽力执行。
    async fn handle_exchange(&mut self, request: ServiceRequest) -> ServiceResponse {
        let has_activity = !request.deltas.is_empty() || !request.queries.is_empty();
        if has_activity {
            self.last_activity = Instant::now();
            self.with_wm(|wm| wm.transition_to_working());
        }

        let mut first_error: Option<ServiceError> = None;

        // 1) 信息增量：先压入滑窗（摘要可能随之更新）。
        let mut accepted: u32 = 0;
        for delta in &request.deltas {
            match self.ingest_one(delta).await {
                Ok(()) => accepted += 1,
                Err(error) => {
                    first_error = Some(error);
                    break;
                }
            }
        }

        // 2) 检索。
        let output = if request.queries.is_empty() {
            None
        } else {
            match self.retrieve(&request.queries).await {
                Ok(text) => Some(text),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                    None
                }
            }
        };

        // 3) 控制信号：即使前面出错也继续执行。
        for action in &request.controls {
            if let Err(error) = self.apply_control(*action).await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }

        ServiceResponse {
            output,
            state: self.snapshot(),
            accepted,
            error: first_error.map(|error| error.to_string()),
        }
    }

    /// 增量用例：把一条语句压入滑动窗口。
    async fn ingest_one(&mut self, delta: &Delta) -> ServiceResult<()> {
        if self.paused {
            return Err(ServiceError::Internal("服务已暂停，忽略增量".into()));
        }
        // proto 约定：空串或其他值都按 user 处理。
        let role = if delta.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        // push 仅在超容量且队首被标记时才触发摘要 LLM；未超容量时直接 Ok。
        // 因此 `Ok` 不蕴含"调用过 LLM"，不能据此点亮可用性标志——只在真正失败时置 false，
        // 置 true 交给启动探活与巩固（它们确定会调用 LLM）。
        if let Err(error) = self
            .wm
            .sliding_window()
            .push(&delta.statement, role, &self.llm)
            .await
        {
            self.llm_available = false;
            return Err(error.into());
        }
        Ok(())
    }

    /// 检索用例：query 集合 -> 一段自然语言。
    async fn retrieve(
        &mut self,
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> ServiceResult<String> {
        // 1) 嵌入全部 query（阻塞线程池）。
        let mut embedded: Vec<(u32, EmbeddedMemoryRetrieveQuery)> =
            Vec::with_capacity(queries.len());
        for query in queries {
            let fused = self.embed_query(query).await?;
            embedded.push((query.priority(), fused));
        }

        // 2) 一次数据库预取（覆盖所有 query 的 embedding）。
        let prefetch: Vec<EmbeddedMemoryRetrieveQuery> =
            embedded.iter().map(|(_, fused)| fused.clone()).collect();
        prefetch_db(
            self.repo.as_ref(),
            prefetch,
            self.config.candidate_k,
            &self.wm,
        )
        .await?;

        // 3) 逐 query 跑管线（仅在此处临时共享 Arc）。，
        // 放进阻塞线程池执行，避免占住 tokio worker（也避免 rayon 与异步运行时互等）。
        let mut results: Vec<DefaultPipelineResult> = Vec::with_capacity(embedded.len());
        for (priority, fused) in embedded {
            let request = self
                .pipeline
                .clone()
                .into_request(Arc::clone(&self.wm), fused, priority);
            let result = tokio::task::spawn_blocking(move || RetrDefaultPipeline.retrieve(request))
                .await
                .map_err(|error| ServiceError::Internal(format!("检索管线任务失败: {error}")))?;
            results.push(result);
        }

        // 4) 按 priority 加权合并记忆。
        let merged_memories = merge::merge_scored(
            results
                .iter()
                .map(|result| (result.priority, result.association.as_slice())),
            self.config.top_k,
        );

        // 5) 记录命中（供巩固建链与后续频次统计）。
        self.with_wm(|wm| {
            for (id, _) in &merged_memories {
                wm.record_retrieval(*id);
            }
        });

        // 6) 取短记忆（同一工作记忆，任取非空）。
        let short_mem = results
            .iter()
            .map(|result| result.short_mem.as_ref())
            .find(|text| !text.trim().is_empty())
            .unwrap_or("")
            .to_string();
        let short_history: Arc<[Information]> = results
            .first()
            .map(|result| Arc::clone(&result.short_history))
            .unwrap_or_else(|| Arc::from(Vec::<Information>::new()));

        // 7) 从簇取内容并渲染：命中节点直接移出交给 `note_text` 消费（零 clone）。
        //
        // 移出是安全的：节点在 DB 里都有权威副本（预取来自 DB；巩固新建的节点先落库再入簇），
        // 下次相似召回会重新加入，`incompletely_linked_note` 会恢复其入射边。
        // 这里保留 `records`（不调用 `WorkingMemory::remove_node`），因此累计检索次数不丢：
        // 节点若被再次召回，仍可参与巩固建链（`active_node_ids` 只在缺席期间把它过滤掉）。
        let cluster = self.wm.memory_cluster();
        let memory_texts = cluster.write(|c| {
            merged_memories
                .iter()
                .filter_map(|(id, _)| c.remove_single_node(*id))
                .filter_map(render::note_text)
                .collect::<Vec<_>>()
        });

        Ok(render::render_output(
            &short_mem,
            &memory_texts,
            short_history.as_ref(),
        ))
    }

    /// 在阻塞线程池里生成查询嵌入。
    async fn embed_query(
        &self,
        query: &PrioritizedMemoryRetrieveQuery,
    ) -> ServiceResult<EmbeddedMemoryRetrieveQuery> {
        let model = Arc::clone(&self.model);
        let owned = query.query().clone();
        tokio::task::spawn_blocking(move || owned.embed_and_fuse(model.as_ref()))
            .await
            .map_err(|error| ServiceError::Internal(format!("查询嵌入任务失败: {error}")))?
            .map_err(|error| ServiceError::Embedding(error.to_string()))
    }

    /// 在阻塞线程池里生成记忆节点嵌入。
    async fn embed_notes(&self, notes: Vec<MemoryNote>) -> ServiceResult<Vec<EmbeddedMemoryNote>> {
        let model = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || {
            notes
                .into_iter()
                .map(|note| note.embed_and_fuse(model.as_ref()))
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|error| ServiceError::Internal(format!("记忆嵌入任务失败: {error}")))?
        .map_err(|error| ServiceError::Embedding(error.to_string()))
    }

    /// 执行单个控制信号。
    async fn apply_control(&mut self, action: ControlAction) -> ServiceResult<()> {
        match action {
            ControlAction::Consolidate => {
                let created = self.consolidate().await?;
                tracing::info!(created, "巩固完成");
            }
            ControlAction::Persist => {
                self.persist().await?;
                tracing::info!("工作记忆已持久化");
            }
            ControlAction::Snapshot => {}
            ControlAction::Forget => {
                tracing::warn!("遗忘流程尚未实现（占位）");
            }
            ControlAction::Pause => self.paused = true,
            ControlAction::Resume => self.paused = false,
            ControlAction::Maintenance => {
                self.maintenance().await?;
            }
        }
        Ok(())
    }

    /// 内部维护：静默超过阈值后转入 Idle；摘要非空则巩固，并在簇过大时告警。
    async fn maintenance(&mut self) -> ServiceResult<()> {
        if self.last_activity.elapsed() < Duration::from_secs(self.config.idle_quiet_secs) {
            return Ok(());
        }
        self.with_wm(|wm| wm.transition_to_idle());
        if !self.wm.sliding_window().get_summary().trim().is_empty() {
            let created = self.consolidate().await?;
            tracing::info!(created, "定时巩固完成");
        }
        let node_count = self.snapshot().node_count;
        if node_count as usize > self.config.max_cluster_nodes {
            tracing::warn!(
                node_count,
                max = self.config.max_cluster_nodes,
                "工作记忆簇节点数超过软上限；本服务不自动淘汰，根因见 soul-mem-algo 的 prefetch_db"
            );
        }
        Ok(())
    }

    /// 巩固：摘要 -> 新记忆节点 -> 持久化 -> 入簇（并建链）。
    ///
    /// 顺序刻意是**先持久化、成功后才清摘要、最后入簇**：任一步失败都不会消耗摘要，
    /// 重试安全，也不会出现"摘要没了但节点没落库"的静默丢失。
    async fn consolidate(&mut self) -> ServiceResult<u32> {
        let summary_text = self.wm.sliding_window().get_summary();
        if summary_text.trim().is_empty() {
            return Ok(0);
        }
        // 复制为独立 Summary，避免跨 await 持有摘要锁。
        let mut summary = Summary::new();
        summary.update(summary_text.to_string());

        let notes = match generate_memories_from_summary(&summary, None, &self.llm).await {
            Ok(notes) => {
                self.llm_available = true;
                notes
            }
            Err(error) => {
                self.llm_available = false;
                return Err(error.into());
            }
        };
        if notes.is_empty() {
            // 摘要已被本次巩固消费：清空，避免下一轮重复触发。
            self.clear_summary();
            return Ok(0);
        }

        let notes = self.attach_active_links(notes);
        let embedded_notes = self.embed_notes(notes).await?;
        let count = embedded_notes.len() as u32;

        // 先落库；失败则摘要与工作记忆都不动，可安全重试。
        self.repo.upsert_notes(embedded_notes.clone()).await?;

        // 落库成功：消费摘要。
        self.clear_summary();

        // 入簇（同时注册活跃记录）。
        self.with_wm(|wm| {
            for node in embedded_notes {
                wm.add_node(node);
            }
        });

        self.last_consolidation_at = Some(Utc::now());
        if count > 0 {
            let _ = self.events.send(ServiceEvent::ConsolidationDone {
                created_notes: count,
                state: self.snapshot(),
            });
        }
        Ok(count)
    }

    /// 清空滑动窗口摘要（内部可变，无需 `&mut WorkingMemory`）。
    fn clear_summary(&self) {
        self.wm
            .sliding_window()
            .summary()
            .write()
            .update(String::new());
    }

    /// 给新生成的记忆节点附加到"活跃节点"（检索次数最高者）的出边。
    ///
    /// 这是对 `记忆算法概述` 中"由 LLM 建立新节点与提取频率 top-k 节点联系"的
    /// 无 LLM 最小实现：谓词与强度固定，只保证新节点不孤立、可被 PPR 触达。
    /// 后续接入 LLM 建链时替换本函数即可。
    fn attach_active_links(&self, mut notes: Vec<MemoryNote>) -> Vec<MemoryNote> {
        let active = self.active_node_ids();
        if active.is_empty() {
            return notes;
        }
        for note in &mut notes {
            let from = note.id();
            for &to in &active {
                if to == from {
                    continue;
                }
                note.links_mut().push(MemoryLink::new(
                    from,
                    to,
                    MemoryLinkType::Sem(SemMemLink::new(
                        CONSOLIDATION_LINK_VERB.to_string(),
                        CONSOLIDATION_LINK_CONFIDENCE,
                    )),
                ));
            }
        }
        notes
    }

    /// 活跃节点：检索次数 > 0 且仍存在于工作记忆簇中的 top-k。
    ///
    /// 簇的存在性用一次读锁一次性取全量 id 集合，避免对每条记录反复加锁。
    fn active_node_ids(&self) -> Vec<MemoryId> {
        let cluster = self.wm.memory_cluster();
        let present: HashSet<MemoryId> = cluster.read_or_compute(|c| {
            c.graph()
                .node_weights()
                .map(|node| node.note().id())
                .collect()
        });
        let mut entries: Vec<(MemoryId, usize)> = self
            .wm
            .records()
            .iter()
            .filter(|(id, record)| record.retrieval_count() > 0 && present.contains(id))
            .map(|(id, record)| (*id, record.retrieval_count()))
            .collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        entries
            .into_iter()
            .take(CONSOLIDATION_LINK_TOP_K)
            .map(|(id, _)| id)
            .collect()
    }

    /// 持久化：工作记忆全量写入数据库。
    async fn persist(&mut self) -> ServiceResult<()> {
        let notes: Vec<EmbeddedMemoryNote> = self
            .wm
            .memory_cluster()
            .read_or_compute(|c| c.graph().node_weights().cloned().collect());
        if notes.is_empty() {
            return Ok(());
        }
        self.repo.upsert_notes(notes).await?;
        Ok(())
    }

    /// 采集状态快照。
    fn snapshot(&self) -> ServiceState {
        let (node_count, edge_count) = self
            .wm
            .memory_cluster()
            .read_or_compute(|c| (c.graph().node_count() as u64, c.graph().edge_count() as u64));
        ServiceState {
            working: self.wm.is_working(),
            node_count,
            edge_count,
            llm_available: self.llm_available,
            last_consolidation_at: self.last_consolidation_at,
        }
    }

    /// 取 `WorkingMemory` 的唯一可变引用并执行闭包。
    ///
    /// 只有在存在未释放的共享引用时才会失败——按本文档开头的分析，那属不可达情形。
    fn with_wm<R>(&mut self, f: impl FnOnce(&mut WorkingMemory) -> R) -> Option<R> {
        match Arc::get_mut(&mut self.wm) {
            Some(wm) => Some(f(wm)),
            None => {
                tracing::error!("工作记忆存在未释放的共享引用，跳过本次状态变更");
                None
            }
        }
    }
}

/// 面向适配器的服务句柄：投递命令并等待结果。
#[derive(Clone)]
pub struct ServiceHandle {
    tx: mpsc::Sender<Command>,
}

impl ServiceHandle {
    /// 提交一次组合请求。
    pub async fn exchange(&self, request: ServiceRequest) -> ServiceResult<ServiceResponse> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Exchange { request, reply })
            .await
            .map_err(|_| ServiceError::Internal("服务循环已停止".into()))?;
        rx.await
            .map_err(|_| ServiceError::Internal("响应通道关闭".into()))?
    }

    /// 仅取状态快照。
    pub async fn snapshot(&self) -> ServiceResult<ServiceState> {
        Ok(self.exchange(ServiceRequest::default()).await?.state)
    }

    /// 仅发送控制信号。
    pub async fn control(&self, controls: Vec<ControlAction>) -> ServiceResult<ServiceState> {
        let request = ServiceRequest {
            controls,
            ..ServiceRequest::default()
        };
        Ok(self.exchange(request).await?.state)
    }
}

/// 启动串行命令循环；返回句柄与后台任务。
pub fn spawn(
    service: MemoryService,
    buffer: usize,
) -> (ServiceHandle, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<Command>(buffer);
    let join = tokio::spawn(async move {
        let mut service = service;
        while let Some(command) = rx.recv().await {
            match command {
                Command::Exchange { request, reply } => {
                    let response = service.handle_exchange(request).await;
                    let _ = reply.send(Ok(response));
                }
            }
        }
    });
    (ServiceHandle { tx }, join)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use async_trait::async_trait;
    use soul_mem_core::memory_links::LinkId;
    use soul_mem_llm::{BackendInfo, ChatBackend, Completion, EventStream, LlmError, Task};
    use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
    use soul_mem_query::embedding::{EmbeddingGenResult, EmbeddingModel, EmbeddingVec};
    use soul_mem_runtime::storage::{MemoryRepository, StorageError, StorageResult};

    use super::*;

    /// 内存假仓储：可注入"写入必失败"以验证巩固的原子性。
    #[derive(Default)]
    struct FakeRepo {
        notes: Mutex<HashMap<MemoryId, EmbeddedMemoryNote>>,
        fail_upsert: AtomicBool,
        upsert_calls: AtomicUsize,
    }

    #[async_trait]
    impl MemoryRepository for FakeRepo {
        async fn upsert_notes(&self, mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()> {
            self.upsert_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_upsert.load(Ordering::SeqCst) {
                return Err(StorageError::InvalidArgument("模拟写入失败".into()));
            }
            let mut notes = self.notes.lock().expect("notes lock");
            for note in mem_notes {
                notes.insert(note.note().id(), note);
            }
            Ok(())
        }

        async fn fetch_neighbors(
            &self,
            _source_ids: &[MemoryId],
            _depth: usize,
        ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
            Ok(Vec::new())
        }

        async fn fetch_notes(&self, ids: &[MemoryId]) -> StorageResult<Vec<EmbeddedMemoryNote>> {
            let notes = self.notes.lock().expect("notes lock");
            Ok(ids.iter().filter_map(|id| notes.get(id).cloned()).collect())
        }

        async fn similarity_fetch(
            &self,
            _queries: Vec<MemoryRetrieveQueryEmbedding>,
            _candidate_k: usize,
        ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
            Ok(Vec::new())
        }

        async fn remove_notes(&self, _mem_ids: &[MemoryId]) -> StorageResult<()> {
            Ok(())
        }

        async fn remove_links(&self, _link_ids: &[LinkId]) -> StorageResult<()> {
            Ok(())
        }
    }

    /// 恒返回零向量的假嵌入模型（不加载 BGE）。
    struct FakeModel;

    impl EmbeddingModel for FakeModel {
        fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
            Ok(input.iter().map(|_| EmbeddingVec::zero(4)).collect())
        }

        fn infer_with_chunk(&self, _input: &str) -> EmbeddingGenResult<EmbeddingVec> {
            Ok(EmbeddingVec::zero(4))
        }

        fn infer_and_fuse(&self, _input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
            Ok(EmbeddingVec::zero(4))
        }

        fn max_input_token(&self) -> usize {
            512
        }

        fn dim(&self) -> usize {
            4
        }
    }

    /// 恒返回一条语义记忆的假 LLM。
    struct FakeLlm;

    #[async_trait]
    impl ChatBackend for FakeLlm {
        fn info(&self) -> BackendInfo {
            BackendInfo::new("fake")
        }

        async fn complete(&self, _task: Task) -> Result<Completion, LlmError> {
            Ok(Completion::new(
                r#"[{"type":"semantic","content":"测试记忆","description":"描述"}]"#,
            ))
        }

        async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
            Err(LlmError::unsupported("fake 后端不支持流式"))
        }
    }

    fn test_config() -> Config {
        Config {
            character: "test".into(),
            db_path: std::path::PathBuf::from("test.db"),
            key_prefix: "test".into(),
            enable_zenoh: false,
            enable_grpc: false,
            grpc_addr: "127.0.0.1:0".parse().expect("addr"),
            window_capacity: 20,
            consolidate_interval_secs: 300,
            idle_quiet_secs: 1,
            heartbeat_interval_secs: 30,
            candidate_k: 8,
            top_k: 5,
            max_cluster_nodes: 2048,
        }
    }

    fn build_service(repo: Arc<FakeRepo>) -> MemoryService {
        let events = broadcast::channel(8).0;
        MemoryService::new(
            Arc::new(test_config()),
            Arc::new(WorkingMemory::new(20)),
            repo,
            Arc::new(LlmEngine::new(Arc::new(FakeLlm))),
            Arc::new(FakeModel),
            events,
            true,
        )
    }

    fn delta(text: &str) -> Delta {
        Delta {
            statement: text.to_string(),
            role: "user".to_string(),
        }
    }

    #[tokio::test]
    async fn ingest_accepts_delta_and_marks_working() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        let request = ServiceRequest {
            deltas: vec![delta("你好")],
            ..ServiceRequest::default()
        };
        let response = service.handle_exchange(request).await;
        assert_eq!(response.accepted, 1);
        assert!(response.error.is_none());
        assert!(service.wm.is_working());
    }

    /// 回归：未超容量的 ingest 不触发摘要，`push` 的 `Ok` 不得点亮 llm_available。
    #[tokio::test]
    async fn ingest_below_capacity_does_not_mark_llm_available() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.llm_available = false;
        let request = ServiceRequest {
            deltas: vec![delta("你好")],
            ..ServiceRequest::default()
        };
        let response = service.handle_exchange(request).await;
        assert!(response.error.is_none());
        assert!(!service.llm_available, "未调用 LLM 不应把可用性置真");
    }

    #[tokio::test]
    async fn paused_reports_error_and_zero_accepted() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.paused = true;
        let request = ServiceRequest {
            deltas: vec![delta("你好")],
            ..ServiceRequest::default()
        };
        let response = service.handle_exchange(request).await;
        assert_eq!(response.accepted, 0);
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn consolidate_persists_then_clears_summary() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo.clone());
        service
            .wm
            .sliding_window()
            .summary()
            .write()
            .update("测试摘要".to_string());

        let created = service.consolidate().await.expect("巩固应成功");
        assert_eq!(created, 1);
        assert!(service.wm.sliding_window().get_summary().trim().is_empty());
        assert_eq!(service.snapshot().node_count, 1);
        assert_eq!(repo.upsert_calls.load(Ordering::SeqCst), 1);
        assert_eq!(repo.notes.lock().expect("notes lock").len(), 1);
    }

    #[tokio::test]
    async fn consolidate_keeps_summary_when_persist_fails() {
        let repo = Arc::new(FakeRepo::default());
        repo.fail_upsert.store(true, Ordering::SeqCst);
        let mut service = build_service(repo);
        service
            .wm
            .sliding_window()
            .summary()
            .write()
            .update("测试摘要".to_string());

        let result = service.consolidate().await;
        assert!(result.is_err());
        assert_eq!(
            service.wm.sliding_window().get_summary().as_ref(),
            "测试摘要"
        );
        assert_eq!(service.snapshot().node_count, 0);
    }

    #[tokio::test]
    async fn maintenance_goes_idle_after_quiet_period() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.with_wm(|wm| wm.transition_to_working());
        service.last_activity = Instant::now() - Duration::from_secs(10);
        service.maintenance().await.expect("维护应成功");
        assert!(!service.wm.is_working());
    }

    #[tokio::test]
    async fn maintenance_skips_when_recent_activity() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.with_wm(|wm| wm.transition_to_working());
        service.last_activity = Instant::now();
        service.maintenance().await.expect("维护应成功");
        assert!(service.wm.is_working());
    }
}
