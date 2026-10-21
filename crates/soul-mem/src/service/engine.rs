//! 服务核心：串行命令循环与各用例实现。
//!
//! 服务持有唯一的 `WorkingMemory`，所有请求经命令通道**串行**处理；检索批次内部
//! 并行只读，写回仍串行，因此工作记忆永不被并发改写。请求是原子的：
//!
//! 1. 信息增量：压入滑动窗口（可能触发摘要）；
//! 2. 检索：query 集合并行跑 `DefaultPipeline`，按 priority 加权合并，取 top-k
//!    并渲染成一段自然语言（命中节点保留在簇内，检索计数与惰性遗忘结果写回）；
//!    命中节点执行**惰性遗忘**（刷新缺失度，必要时遮罩 + LLM 补全）；
//! 3. 控制信号：巩固 / 持久化 / 快照 / 暂停 / 恢复 / 探活 / 维护 / 遗忘衰减。
//!
//! 一次请求只做其中一件；失败以 `Err` 返回给调用方。
//!
//! ## 并发与所有权
//!
//! `WorkingMemory` 内部可变（见 `soul-mem-runtime`），因此检索批次的多个 query 可以
//! 同时持有 `Arc<WorkingMemory>` 并行**只读**。并行批次 join 之后，才在命令循环线程上
//! 串行完成写回（检索计数、惰性遗忘、写回簇）；跨请求的串行保证不变。

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use jieba_rs::Jieba;
use rayon::ThreadPool;
use rayon::prelude::*;
use tokio::sync::{broadcast, mpsc, oneshot};

use soul_mem_algo::algo::consolidate::generate::generate_memories_from_summary;
use soul_mem_algo::algo::forget::decay_revise::{compute_and_update_missing_degree, lazy_forget};
use soul_mem_algo::algo::retrieve::RetrStrategy;
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
use soul_mem_core::render::{Render, RenderedMemoryNote};
use soul_mem_llm::{LlmEngine, Sampling, Task};
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::embedding::EmbeddingGenResult;
use soul_mem_query::embedding::EmbeddingModel;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
use soul_mem_query::query::retrieve::PrioritizedMemoryRetrieveQuery;
use soul_mem_runtime::storage::MemoryRepository;
use soul_mem_runtime::working_memory::WorkingMemory;
use soul_mem_runtime::working_memory::sliding_window::{Information, PushOutcome, Summary};

use super::types::{
    ControlAction, Delta, Retrieval, ServiceEvent, ServiceRequest, ServiceResponse, ServiceState,
};
use super::{merge, render};
use crate::config::Config;
use crate::error::{ServiceError, ServiceResult};

/// 检索用例的产物：自然语言段落 + 结构化记忆集合（二者并列）。
struct RetrieveOutcome {
    /// 一段自然语言（摘要 + 相关记忆 + 最近对话）。
    output: String,
    /// 命中的记忆节点（结构化，含 kind/score/content）。
    memories: Vec<RenderedMemoryNote>,
}

/// 交给核心处理的命令。
pub enum Command {
    /// 一次原子请求。
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
    /// 检索批次的并行只读池（引导时构建一次）。
    pool: Arc<ThreadPool>,
    /// 待摘要的淘汰消息；摘要失败时保留，按淘汰顺序重试。
    pending_evicted: VecDeque<Information>,
    /// 分词器：惰性遗忘的遮罩（`mask_text`）需要。
    jieba: Jieba,
    paused: bool,
    /// 最近一次有增量/检索输入的时刻，用于 Idle 判定。
    last_activity: Instant,
    last_consolidation_at: Option<DateTime<Utc>>,
    events: broadcast::Sender<ServiceEvent>,
    llm_available: bool,
}

#[derive(Default)]
pub struct MemoryServiceBuilder {
    config: Option<Arc<Config>>,
    wm: Option<Arc<WorkingMemory>>,
    repo: Option<Arc<dyn MemoryRepository>>,
    llm: Option<Arc<LlmEngine>>,
    model: Option<Arc<dyn EmbeddingModel + Send + Sync>>,
    jieba: Option<Jieba>,
    events: Option<broadcast::Sender<ServiceEvent>>,
    llm_available: Option<bool>,
    pool: Option<Arc<ThreadPool>>,
}

impl MemoryServiceBuilder {
    /// 共享配置（服务与引导层共用同一份）。
    pub fn config(mut self, config: Arc<Config>) -> Self {
        self.config = Some(config);
        self
    }

    pub fn working_memory(mut self, wm: Arc<WorkingMemory>) -> Self {
        self.wm = Some(wm);
        self
    }

    pub fn repository(mut self, repo: Arc<dyn MemoryRepository>) -> Self {
        self.repo = Some(repo);
        self
    }

    pub fn llm(mut self, llm: Arc<LlmEngine>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn embedding_model(mut self, model: Arc<dyn EmbeddingModel + Send + Sync>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn jieba(mut self, jieba: Jieba) -> Self {
        self.jieba = Some(jieba);
        self
    }

    pub fn events(mut self, events: broadcast::Sender<ServiceEvent>) -> Self {
        self.events = Some(events);
        self
    }

    pub fn llm_available(mut self, llm_available: bool) -> Self {
        self.llm_available = Some(llm_available);
        self
    }

    /// 检索批次的并行只读池（引导时构建一次）。
    pub fn retrieval_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// 组装服务核心；缺少任一必需依赖时返回错误。
    pub fn build(self) -> ServiceResult<MemoryService> {
        let config = self.config.ok_or_else(|| missing_dependency("config"))?;
        let wm = self
            .wm
            .ok_or_else(|| missing_dependency("working_memory"))?;
        let repo = self.repo.ok_or_else(|| missing_dependency("repository"))?;
        let llm = self.llm.ok_or_else(|| missing_dependency("llm"))?;
        let model = self
            .model
            .ok_or_else(|| missing_dependency("embedding_model"))?;
        let jieba = self.jieba.ok_or_else(|| missing_dependency("jieba"))?;
        let events = self.events.ok_or_else(|| missing_dependency("events"))?;
        let llm_available = self
            .llm_available
            .ok_or_else(|| missing_dependency("llm_available"))?;
        let pool = self
            .pool
            .ok_or_else(|| missing_dependency("retrieval_pool"))?;

        Ok(MemoryService {
            config,
            wm,
            repo,
            llm,
            model,
            pool,
            pending_evicted: VecDeque::new(),
            jieba,
            paused: false,
            last_activity: Instant::now(),
            last_consolidation_at: None,
            events,
            llm_available,
        })
    }
}

fn missing_dependency(field: &str) -> ServiceError {
    ServiceError::Internal(format!("MemoryService 构造缺少依赖: {field}"))
}

impl MemoryService {
    /// 开始构造服务核心；依赖通过 [`MemoryServiceBuilder`] 逐个注入。
    pub fn builder() -> MemoryServiceBuilder {
        MemoryServiceBuilder::default()
    }

    /// 处理一次原子请求。
    async fn handle_request(&mut self, request: ServiceRequest) -> ServiceResult<ServiceResponse> {
        match request {
            ServiceRequest::Ingest(window_delta) => {
                self.last_activity = Instant::now();
                self.wm.transition_to_working();
                self.ingest_one(&window_delta).await?;
                Ok(ServiceResponse::Ingested(self.snapshot()))
            }
            ServiceRequest::Retrieve(queries) => {
                self.last_activity = Instant::now();
                self.wm.transition_to_working();
                let RetrieveOutcome { output, memories } = self.retrieve(&queries).await?;
                Ok(ServiceResponse::Retrieved(Retrieval {
                    output,
                    memories,
                    state: self.snapshot(),
                }))
            }
            ServiceRequest::Control(action) => {
                self.apply_control(action).await?;
                Ok(ServiceResponse::Controlled(self.snapshot()))
            }
        }
    }

    /// 增量用例：把一条语句压入滑动窗口。
    async fn ingest_one(&mut self, window_delta: &Delta) -> ServiceResult<()> {
        if self.paused {
            return Err(ServiceError::Internal("服务已暂停，忽略增量".into()));
        }
        // proto 约定：空串或其他值都按 user 处理。
        let role = if window_delta.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        // push 只入队；超过容量后由调用方 pop 队首，带标记的进入待摘要队列。
        // `Ok` 不蕴含"调用过 LLM"；失败只在 flush 里置 false，置 true 交给探活与巩固。
        match self.wm.sliding_window().push(&window_delta.statement, role) {
            PushOutcome::Enqueued => {}
            PushOutcome::NeedsEviction => {
                if let Some(evicted) = self.wm.sliding_window().pop()
                    && evicted.is_tagged()
                {
                    self.pending_evicted.push_back(evicted);
                }
            }
        }
        self.flush_pending_summaries().await
    }

    /// 按淘汰顺序重试待摘要消息；失败时把队首放回，下次 ingest / 维护继续。
    async fn flush_pending_summaries(&mut self) -> ServiceResult<()> {
        while let Some(evicted) = self.pending_evicted.pop_front() {
            match self
                .wm
                .sliding_window()
                .summarize(&self.llm, &evicted)
                .await
            {
                Ok(()) => {}
                Err(error) => {
                    self.pending_evicted.push_front(evicted);
                    self.llm_available = false;
                    return Err(error.into());
                }
            }
        }
        Ok(())
    }

    /// 每次检索重建管线配置：`into_request` 会消费所有权，配置不长期持有。
    fn pipeline_config(&self) -> DefaultPipelineConfig {
        let retrieval = &self.config.retrieval;
        DefaultPipelineConfig {
            short_mem_with_history: ShortOnlyConfig {
                clipping_length: retrieval.short_clipping_length,
                include_summary: retrieval.include_short_summary,
            },
            similarity: SimilarityConfig {
                similarity_threshold: retrieval.similarity_threshold,
                max_results: retrieval.top_k,
            },
            assoc_with_action: AssociateWithActionConfig {
                association: retrieval.association.clone(),
                abstract_source_priority: retrieval.abstract_source_priority,
            },
        }
    }

    /// 检索用例：query 集合 -> 一段自然语言。
    async fn retrieve(
        &mut self,
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> ServiceResult<RetrieveOutcome> {
        // 1) 整批嵌入 query：一次派发到阻塞线程池，在专用 rayon 池上并行。
        let embedded = self.embed_queries(queries).await?;

        // 2) 一次数据库预取（覆盖所有 query 的 embedding）。
        let prefetch: Vec<EmbeddedMemoryRetrieveQuery> =
            embedded.iter().map(|(_, fused)| fused.clone()).collect();
        prefetch_db(
            self.repo.as_ref(),
            prefetch,
            self.config.retrieval.candidate_k,
            &self.wm,
        )
        .await?;

        // 3) 整批跑管线：并行只读工作记忆；join 后再串行落写。
        let results = self.run_pipelines(self.pipeline_config(), embedded).await?;

        // 4) 按 priority 加权合并记忆。
        let merged_memories = merge::merge_scored(
            results
                .iter()
                .map(|result| (result.priority, result.association.as_slice())),
            self.config.retrieval.top_k,
        );

        // 5) 取短记忆（同一工作记忆，任取非空）。
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

        // 6) 命中节点保留在簇里：递增检索计数，克隆出来做惰性遗忘/渲染，再写回簇。
        let cluster = self.wm.memory_cluster();
        let hits: Vec<(f64, EmbeddedMemoryNote)> = cluster.write(|c| {
            merged_memories
                .iter()
                .filter_map(|(id, score)| {
                    let node = c.get_node_mut(*id)?;
                    node.note.retrieval_increment();
                    Some((*score, node.clone()))
                })
                .collect()
        });

        // 惰性遗忘：命中的记忆作为最终结果时刷新缺失度，必要时遮罩并用 LLM 补全；
        // 仅在 LLM 可用时传入引擎（否则退化为仅遮罩）。
        let now = Utc::now();
        let engine = self.llm_available.then_some(self.llm.as_ref());
        let mut memories: Vec<RenderedMemoryNote> = Vec::with_capacity(hits.len());
        let mut updated: Vec<EmbeddedMemoryNote> = Vec::with_capacity(hits.len());
        for (score, mut embedded) in hits {
            lazy_forget(&mut embedded.note, now, &self.jieba, None, engine).await;
            memories.push(embedded.note().render(score));
            updated.push(embedded);
        }
        cluster.write(|c| {
            updated.iter().for_each(|node| {
                if let Some(slot) = c.get_node_mut(node.note().id()) {
                    *slot = node.clone();
                }
            });
        });
        if let Err(error) = self.repo.upsert_notes(updated).await {
            tracing::warn!(%error, "检索命中持久化失败");
        }
        let memory_texts: Vec<String> = memories
            .iter()
            .filter_map(|memory| {
                let trimmed = memory.content.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
            .collect();

        let output = render::render_output(&short_mem, &memory_texts, short_history.as_ref());
        Ok(RetrieveOutcome { output, memories })
    }

    /// 整批生成查询嵌入：一次 `spawn_blocking`，在专用 rayon 池上并行。
    ///
    /// 任一条查询嵌入失败即整批失败——与逐条串行版的行为一致。失败时按下标 +
    /// priority 记录日志，并用 `Debug` 保留 `EmbeddingGenError` 的 variant 与 cause。
    async fn embed_queries(
        &self,
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> ServiceResult<Vec<(u32, EmbeddedMemoryRetrieveQuery)>> {
        let model = Arc::clone(&self.model);
        let pool = Arc::clone(&self.pool);
        let queries = queries.to_vec();
        let embedded = tokio::task::spawn_blocking(move || {
            pool.install(|| {
                queries
                    .into_par_iter()
                    .map(|query| {
                        let priority = query.priority();
                        query
                            .downgrade()
                            .embed_and_fuse(model.as_ref())
                            .map(|fused| (priority, fused))
                            .map_err(|error| (priority, error))
                    })
                    .collect::<Vec<_>>()
            })
        })
        .await
        .map_err(|error| ServiceError::Internal(format!("查询嵌入任务失败: {error}")))?;
        embedded
            .into_iter()
            .enumerate()
            .map(|(index, result)| result.map_err(|(priority, error)| (index, priority, error)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|(index, priority, error)| {
                tracing::warn!(index, priority, error = ?error, "查询嵌入失败");
                ServiceError::Embedding(error.to_string())
            })
    }

    /// 整批执行检索管线：一次 `spawn_blocking`，在专用 rayon 池上并行。
    ///
    /// 管线只读工作记忆；检索计数、惰性遗忘与写回簇由调用方在 join 后串行完成。
    async fn run_pipelines(
        &self,
        pipeline: DefaultPipelineConfig,
        embedded: Vec<(u32, EmbeddedMemoryRetrieveQuery)>,
    ) -> ServiceResult<Vec<DefaultPipelineResult>> {
        let wm = Arc::clone(&self.wm);
        let pool = Arc::clone(&self.pool);
        tokio::task::spawn_blocking(move || {
            pool.install(|| {
                embedded
                    .into_par_iter()
                    .map(|(priority, fused)| {
                        let request =
                            pipeline
                                .clone()
                                .into_request(Arc::clone(&wm), fused, priority);
                        RetrDefaultPipeline.retrieve(request)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .await
        .map_err(|error| ServiceError::Internal(format!("检索管线任务失败: {error}")))
    }

    /// 在阻塞线程池里生成记忆节点嵌入。
    async fn embed_notes(&self, notes: Vec<MemoryNote>) -> ServiceResult<Vec<EmbeddedMemoryNote>> {
        self.spawn_embed("记忆嵌入", move |model| {
            notes
                .into_iter()
                .map(|note| note.embed_and_fuse(model.as_ref()))
                .collect::<Result<Vec<_>, _>>()
        })
        .await
    }

    /// 在阻塞线程池里执行一次嵌入，统一「任务失败 / 嵌入失败」两层错误映射。
    ///
    /// 放到阻塞线程池执行，避免占住 tokio worker（也避免 rayon 与异步运行时互等）。
    async fn spawn_embed<T>(
        &self,
        label: &str,
        f: impl FnOnce(Arc<dyn EmbeddingModel + Send + Sync>) -> EmbeddingGenResult<T> + Send + 'static,
    ) -> ServiceResult<T>
    where
        T: Send + 'static,
    {
        let model = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || f(model))
            .await
            .map_err(|error| ServiceError::Internal(format!("{label}任务失败: {error}")))?
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
                self.forget_pass().await?;
            }
            ControlAction::Pause => self.paused = true,
            ControlAction::Resume => self.paused = false,
            ControlAction::ProbeLlm => {
                self.llm_available = self.probe_llm().await;
                tracing::info!(available = self.llm_available, "LLM 探活完成");
            }
            ControlAction::Maintenance => {
                self.maintenance().await?;
            }
        }
        Ok(())
    }

    /// 探活：发一条极小补全，更新 `llm_available`。
    ///
    /// 由 `ControlAction::ProbeLlm` 触发；启动时不再自动探测。
    async fn probe_llm(&self) -> bool {
        let task = Task::system_user("ping", "pong")
            .with_sampling(Sampling::default().with_max_output_tokens(1));
        let timeout = Duration::from_secs(self.config.llm.probe_timeout_secs);
        match tokio::time::timeout(timeout, self.llm.complete(task)).await {
            Ok(Ok(_)) => true,
            Ok(Err(error)) => {
                tracing::warn!(%error, "LLM 探活失败，记为不可用");
                false
            }
            Err(_) => {
                tracing::warn!("LLM 探活超时，记为不可用");
                false
            }
        }
    }

    /// 遗忘衰减：刷新工作记忆全部节点的缺失度并落库。
    ///
    /// 这里只做缺失度衰减（无 LLM、无遮罩）；遮罩与补全是**惰性**的，发生在节点
    /// 作为检索结果被取出时（见 [`Self::retrieve`]）。受限于仓储无"全量读取"接口，
    /// 本操作只覆盖工作记忆中的节点，DB 中未加载的节点不会被触碰。
    async fn forget_pass(&mut self) -> ServiceResult<()> {
        let now = Utc::now();
        let updated: Vec<EmbeddedMemoryNote> = {
            let cluster = self.wm.memory_cluster();
            cluster.write(|c| {
                let ids: Vec<MemoryId> = c
                    .graph()
                    .node_weights()
                    .map(|node| node.note().id())
                    .collect();
                ids.iter().for_each(|id| {
                    if let Some(node) = c.get_node_mut(*id) {
                        compute_and_update_missing_degree(&mut node.note, now);
                    }
                });
                ids.iter()
                    .filter_map(|id| c.get_node(*id).cloned())
                    .collect()
            })
        };
        if !updated.is_empty() {
            self.repo.upsert_notes(updated).await?;
        }
        Ok(())
    }

    /// 内部维护：静默超过阈值后转入 Idle；摘要非空则巩固，并在簇过大时告警。
    async fn maintenance(&mut self) -> ServiceResult<()> {
        if self.last_activity.elapsed()
            < Duration::from_secs(self.config.consolidation.idle_quiet_secs)
        {
            return Ok(());
        }
        self.wm.transition_to_idle();
        self.flush_pending_summaries().await?;
        if !self.wm.sliding_window().get_summary().trim().is_empty() {
            let created = self.consolidate().await?;
            tracing::info!(created, "定时巩固完成");
        }
        let node_count = self.snapshot().node_count;
        let max_cluster_nodes = self.config.working_memory.max_cluster_nodes;
        if node_count as usize > max_cluster_nodes {
            tracing::warn!(
                node_count,
                max = max_cluster_nodes,
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
        self.flush_pending_summaries().await?;
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

        let (notes, active_ids) = self.attach_active_links(notes);
        let embedded_notes = self.embed_notes(notes).await?;
        let new_ids: Vec<MemoryId> = embedded_notes.iter().map(|node| node.note().id()).collect();
        let count = embedded_notes.len() as u32;

        // 既有活跃节点补 `活跃 -> 新` 出边。PPR 只沿出边传播，只建有 `新 -> 活跃`
        // 会让新记忆永远无法被既有记忆联想到。LinkId 只生成一次，落库与入簇共用，
        // 避免内存与数据库对同一条边持有不同身份。
        let mut updated_active: Vec<EmbeddedMemoryNote> = Vec::new();
        let mut active_links: Vec<(MemoryId, MemoryLink)> = Vec::new();
        {
            let cluster = self.wm.memory_cluster();
            cluster.read_or_compute(|c| {
                active_ids.iter().for_each(|id| {
                    let Some(active_node) = c.get_node(*id) else {
                        return;
                    };
                    let from = active_node.note().id();
                    let mut clone = active_node.clone();
                    new_ids
                        .iter()
                        .filter(|&&new_id| new_id != from)
                        .for_each(|&new_id| {
                            let link = self.consolidation_link(from, new_id);
                            clone.note.links_mut().push(link.clone());
                            active_links.push((from, link));
                        });
                    updated_active.push(clone);
                });
            });
        }

        // 先落库（新节点 + 被更新的活跃节点）；失败则摘要与工作记忆都不动，可安全重试。
        let mut batch = embedded_notes.clone();
        batch.extend(updated_active);
        self.repo.upsert_notes(batch).await?;

        // 落库成功：消费摘要。
        self.clear_summary();

        // 入簇（同时注册活跃记录）。
        embedded_notes
            .into_iter()
            .for_each(|node| self.wm.add_node(node));

        // 让簇内的活跃节点也带上 `活跃 -> 新` 出边（复用同一批 LinkId）。
        let cluster = self.wm.memory_cluster();
        cluster.write(|c| {
            active_links.into_iter().for_each(|(from, link)| {
                if let Some(node) = c.get_node_mut(from) {
                    node.note.links_mut().push(link);
                }
            });
            active_ids.iter().for_each(|active| c.refresh_node(active));
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

    /// 给新生成的记忆节点附加指向活跃节点的出边（新 -> 活跃），并返回活跃节点集合。
    ///
    /// 这是对 `记忆算法概述` 中"由 LLM 建立新节点与提取频率 top-k 节点联系"的
    /// 无 LLM 最小实现：谓词与强度固定。反向的 `活跃 -> 新` 出边由
    /// [`Self::consolidate`] 补齐——PPR 只沿出边传播，只建单向会让新记忆孤立。
    /// 后续接入 LLM 建链时替换本函数即可。
    fn attach_active_links(&self, mut notes: Vec<MemoryNote>) -> (Vec<MemoryNote>, Vec<MemoryId>) {
        let active = self.active_node_ids();
        if active.is_empty() {
            return (notes, active);
        }
        notes.iter_mut().for_each(|note| {
            let from = note.id();
            active.iter().filter(|&&to| to != from).for_each(|&to| {
                note.links_mut().push(self.consolidation_link(from, to));
            });
        });
        (notes, active)
    }

    /// 构造一条巩固联系边（Sem 关联）。
    fn consolidation_link(&self, from: MemoryId, to: MemoryId) -> MemoryLink {
        MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new(
                self.config.consolidation.link_verb.clone(),
                self.config.consolidation.link_confidence,
            )),
        )
    }

    /// 活跃节点：簇内检索次数 > 0 的 top-k（计数取自持久化的 note 字段）。
    fn active_node_ids(&self) -> Vec<MemoryId> {
        let cluster = self.wm.memory_cluster();
        let mut entries: Vec<(MemoryId, usize)> = cluster.read_or_compute(|c| {
            c.graph()
                .node_weights()
                .filter_map(|node| {
                    let count = node.note().retrieval_count();
                    (count > 0).then(|| (node.note().id(), count))
                })
                .collect()
        });
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries
            .into_iter()
            .take(self.config.consolidation.link_top_k)
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
}

/// 面向适配器的服务句柄：投递命令并等待结果。
#[derive(Clone)]
pub struct ServiceHandle {
    tx: mpsc::Sender<Command>,
}

impl ServiceHandle {
    /// 投递一次原子请求。
    pub(crate) async fn send(&self, request: ServiceRequest) -> ServiceResult<ServiceResponse> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Command::Exchange { request, reply })
            .await
            .map_err(|_| ServiceError::Internal("服务循环已停止".into()))?;
        rx.await
            .map_err(|_| ServiceError::Internal("响应通道关闭".into()))?
    }

    /// 压入一条信息增量。
    pub async fn ingest(&self, window_delta: Delta) -> ServiceResult<ServiceState> {
        match self.send(ServiceRequest::Ingest(window_delta)).await? {
            ServiceResponse::Ingested(state) => Ok(state),
            other => Err(mismatch(&other)),
        }
    }

    /// 执行一次检索。
    pub async fn retrieve(
        &self,
        queries: Vec<PrioritizedMemoryRetrieveQuery>,
    ) -> ServiceResult<Retrieval> {
        match self.send(ServiceRequest::Retrieve(queries)).await? {
            ServiceResponse::Retrieved(retrieval) => Ok(retrieval),
            other => Err(mismatch(&other)),
        }
    }

    /// 执行一个控制信号。
    pub async fn control(&self, action: ControlAction) -> ServiceResult<ServiceState> {
        match self.send(ServiceRequest::Control(action)).await? {
            ServiceResponse::Controlled(state) => Ok(state),
            other => Err(mismatch(&other)),
        }
    }

    /// 仅取状态快照。
    pub async fn snapshot(&self) -> ServiceResult<ServiceState> {
        self.control(ControlAction::Snapshot).await
    }
}

fn mismatch(response: &ServiceResponse) -> ServiceError {
    ServiceError::Internal(format!("响应与请求不匹配: {response:?}"))
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
                    let _ = reply.send(service.handle_request(request).await);
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
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_llm::{BackendInfo, ChatBackend, Completion, EventStream, LlmError, Task};
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::{
        EmbeddingGenError, EmbeddingGenResult, EmbeddingModel, EmbeddingVec,
    };
    use soul_mem_query::query::retrieve::{
        MemoryRetrieveQuery, MemoryRetrieveQueryVariant, SemanticQueryUnit,
    };
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
            mem_notes.into_iter().for_each(|note| {
                notes.insert(note.note().id(), note);
            });
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

    /// 按文本产出确定性单位向量的模型：查询侧 tag / variant 两条路径都覆写，
    /// passage 侧返回哨兵向量，用来证明查询走的是 query 接口而非 passage 接口。
    struct ScriptedModel {
        /// 命中该文本时返回错误，用于验证批量中单条失败会让整批失败。
        fail_on: Option<&'static str>,
    }

    impl ScriptedModel {
        fn new() -> Self {
            Self { fail_on: None }
        }

        fn failing_on(text: &'static str) -> Self {
            Self {
                fail_on: Some(text),
            }
        }

        fn encode(&self, text: &str) -> EmbeddingGenResult<EmbeddingVec> {
            if self.fail_on == Some(text) {
                return Err(EmbeddingGenError::InvalidInput);
            }
            let index = match text {
                "b" => 1,
                "c" => 2,
                _ => 0,
            };
            Ok(unit(index))
        }
    }

    impl EmbeddingModel for ScriptedModel {
        fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
            input.iter().map(|text| self.encode(text)).collect()
        }

        fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
            self.encode(input)
        }

        fn infer_and_fuse(&self, _input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
            // passage 侧哨兵：查询路径若误用本接口，断言会立即失败。
            Ok(unit(3))
        }

        fn infer_query_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
            self.encode(input.first().copied().unwrap_or(""))
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

    /// 首次调用失败、之后成功的假 LLM，用于验证摘要失败后的重试。
    #[derive(Default)]
    struct FailOnceLlm {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ChatBackend for FailOnceLlm {
        fn info(&self) -> BackendInfo {
            BackendInfo::new("fail-once")
        }

        async fn complete(&self, _task: Task) -> Result<Completion, LlmError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(LlmError::unsupported("boom"));
            }
            Ok(Completion::new("摘要"))
        }

        async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
            Err(LlmError::unsupported("不支持流式"))
        }
    }

    fn test_config() -> Config {
        use crate::config::{
            ConsolidationConfig, GrpcConfig, LlmConfig, RetrievalConfig, ServiceConfig,
            WorkingMemoryConfig, ZenohConfig,
        };
        Config {
            character: "test".into(),
            db_path: std::path::PathBuf::from("test.db"),
            zenoh: ZenohConfig {
                enable: false,
                key_prefix: Some("test".into()),
            },
            grpc: GrpcConfig {
                enable: false,
                addr: "127.0.0.1:0".parse().expect("addr"),
            },
            working_memory: WorkingMemoryConfig {
                window_capacity: 20,
                max_cluster_nodes: 2048,
            },
            retrieval: RetrievalConfig {
                candidate_k: 8,
                top_k: 5,
                similarity_threshold: 0.35,
                abstract_source_priority: 2.0,
                threads: 0,
                include_short_summary: true,
                short_clipping_length: None,
                association: Default::default(),
            },
            consolidation: ConsolidationConfig {
                interval_secs: 300,
                idle_quiet_secs: 1,
                link_top_k: 3,
                link_verb: "related".into(),
                link_confidence: 1.0,
            },
            service: ServiceConfig {
                heartbeat_interval_secs: 30,
                event_channel_capacity: 8,
                command_buffer: 32,
                shutdown_grace_secs: 5,
            },
            llm: LlmConfig {
                base_url: String::new(),
                model: String::new(),
                api_key: None,
                probe_timeout_secs: 5,
            },
        }
    }

    fn build_service(repo: Arc<FakeRepo>) -> MemoryService {
        build_service_with_model(repo, Arc::new(FakeModel))
    }

    fn build_service_with_model(
        repo: Arc<FakeRepo>,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
    ) -> MemoryService {
        build_service_with(repo, model, Arc::new(LlmEngine::new(Arc::new(FakeLlm))), 20)
    }

    fn build_service_with(
        repo: Arc<FakeRepo>,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
        llm: Arc<LlmEngine>,
        capacity: usize,
    ) -> MemoryService {
        let events = broadcast::channel(8).0;
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("测试线程池"),
        );
        MemoryService::builder()
            .config(Arc::new(test_config()))
            .working_memory(Arc::new(WorkingMemory::new(capacity)))
            .repository(repo)
            .llm(llm)
            .embedding_model(model)
            .jieba(jieba_rs::Jieba::new())
            .events(events)
            .llm_available(true)
            .retrieval_pool(pool)
            .build()
            .expect("构造测试服务")
    }

    /// 4 维单位向量：第 `index` 个分量为 1，其余为 0。
    fn unit(index: usize) -> EmbeddingVec {
        let mut values = vec![0.0f32; 4];
        values[index] = 1.0;
        EmbeddingVec::new(values)
    }

    /// 往工作记忆簇里放一条语义记忆，tag 与 concept 向量分开给出。
    fn seed_note(
        service: &MemoryService,
        id: MemoryId,
        content: &str,
        tag: EmbeddingVec,
        concept: EmbeddingVec,
    ) {
        let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: content.to_string(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: String::new(),
        }))
        .id(id)
        .build()
        .expect("构造记忆节点");
        let embedding = MemoryEmbedding::new(
            tag,
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                concept,
                EmbeddingVec::zero(4),
                EmbeddingVec::zero(4),
            )),
        );
        service.wm.memory_cluster().write(|cluster| {
            cluster.add_single_node(EmbeddedMemoryNote { note, embedding });
        });
    }

    /// 构造一条带 tag 与 concept 的语义 query。
    fn prioritized_query(
        priority: u32,
        tag: &str,
        concept: &str,
    ) -> PrioritizedMemoryRetrieveQuery {
        MemoryRetrieveQuery::new(
            vec![tag.to_string()],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier(concept.to_string()),
            ]),
        )
        .with_priority(priority)
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
        let response = service
            .handle_request(ServiceRequest::Ingest(delta("你好")))
            .await;
        assert!(matches!(response, Ok(ServiceResponse::Ingested(_))));
        assert!(service.wm.is_working());
    }

    /// 回归：未超容量的 ingest 不触发摘要，`push` 的 `Ok` 不得点亮 llm_available。
    #[tokio::test]
    async fn ingest_below_capacity_does_not_mark_llm_available() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.llm_available = false;
        let response = service
            .handle_request(ServiceRequest::Ingest(delta("你好")))
            .await;
        assert!(response.is_ok());
        assert!(!service.llm_available, "未调用 LLM 不应把可用性置真");
    }

    #[tokio::test]
    async fn paused_rejects_ingest() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.paused = true;
        let result = service
            .handle_request(ServiceRequest::Ingest(delta("你好")))
            .await;
        assert!(result.is_err());
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
        service.wm.transition_to_working();
        service.last_activity = Instant::now() - Duration::from_secs(10);
        service.maintenance().await.expect("维护应成功");
        assert!(!service.wm.is_working());
    }

    #[tokio::test]
    async fn maintenance_skips_when_recent_activity() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service(repo);
        service.wm.transition_to_working();
        service.last_activity = Instant::now();
        service.maintenance().await.expect("维护应成功");
        assert!(service.wm.is_working());
    }

    /// 并行批次必须保序，且 tag 走 `infer_query_and_fuse`（而非 passage 的 `infer_and_fuse`）。
    #[tokio::test]
    async fn embed_queries_uses_query_side_and_preserves_order() {
        let repo = Arc::new(FakeRepo::default());
        let service = build_service_with_model(repo, Arc::new(ScriptedModel::new()));
        let queries = vec![
            prioritized_query(7, "c", "c"),
            prioritized_query(1, "a", "a"),
            prioritized_query(3, "b", "b"),
        ];

        let embedded = service.embed_queries(&queries).await.expect("嵌入应成功");
        let priorities: Vec<u32> = embedded.iter().map(|(priority, _)| *priority).collect();
        assert_eq!(priorities, vec![7, 1, 3], "并行嵌入必须保持输入顺序");
        let tags: Vec<EmbeddingVec> = embedded
            .iter()
            .map(|(_, fused)| fused.embedding.tag().clone())
            .collect();
        assert_eq!(tags, vec![unit(2), unit(0), unit(1)]);

        let results = service
            .run_pipelines(service.pipeline_config(), embedded)
            .await
            .expect("管线应成功");
        let priorities: Vec<u32> = results.iter().map(|result| result.priority).collect();
        assert_eq!(priorities, vec![7, 1, 3], "并行管线必须保持输入顺序");
    }

    #[tokio::test]
    async fn retrieve_matches_seeded_note_end_to_end() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service_with_model(repo, Arc::new(ScriptedModel::new()));
        let id = MemoryId::new();
        seed_note(&service, id, "a", unit(0), unit(0));

        let response = service
            .handle_request(ServiceRequest::Retrieve(vec![prioritized_query(
                1, "a", "a",
            )]))
            .await
            .expect("检索应成功");
        let ServiceResponse::Retrieved(retrieval) = response else {
            panic!("期望 Retrieved 响应");
        };

        assert_eq!(retrieval.memories.len(), 1, "应命中预置节点");
        assert_eq!(retrieval.memories[0].content, "a");
        assert!(!retrieval.output.trim().is_empty(), "应渲染出非空输出");
        // 命中节点保留在簇里，并递增自身检索计数。
        let (node_count, count) = service.wm.memory_cluster().read_or_compute(|cluster| {
            let count = cluster
                .get_node(id)
                .map(|node| node.note().retrieval_count());
            (cluster.graph().node_count(), count)
        });
        assert_eq!(node_count, 1);
        assert_eq!(count, Some(1));
    }

    #[tokio::test]
    async fn retrieved_node_becomes_active() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service_with_model(repo, Arc::new(ScriptedModel::new()));
        let id = MemoryId::new();
        seed_note(&service, id, "a", unit(0), unit(0));

        service
            .handle_request(ServiceRequest::Retrieve(vec![prioritized_query(
                1, "a", "a",
            )]))
            .await
            .expect("检索应成功");

        assert_eq!(service.active_node_ids(), vec![id]);
    }

    #[tokio::test]
    async fn retrieve_batch_fails_when_any_query_embedding_fails() {
        let repo = Arc::new(FakeRepo::default());
        let mut service =
            build_service_with_model(repo, Arc::new(ScriptedModel::failing_on("boom")));

        let result = service
            .handle_request(ServiceRequest::Retrieve(vec![
                prioritized_query(1, "a", "a"),
                prioritized_query(1, "boom", "boom"),
            ]))
            .await;

        assert!(result.is_err(), "任一条嵌入失败应使整批失败");
    }

    #[tokio::test]
    async fn retrieve_equal_scores_break_ties_by_memory_id() {
        let repo = Arc::new(FakeRepo::default());
        let mut service = build_service_with_model(repo, Arc::new(ScriptedModel::new()));

        let first = MemoryId::new();
        let second = MemoryId::new();
        // 两个节点 embedding 完全相同 → 同分；按 id 升序兜底。
        let (low, high, low_content, high_content) = if first < second {
            (first, second, "first", "second")
        } else {
            (second, first, "second", "first")
        };
        seed_note(&service, low, low_content, unit(0), unit(0));
        seed_note(&service, high, high_content, unit(0), unit(0));

        let response = service
            .handle_request(ServiceRequest::Retrieve(vec![prioritized_query(
                1, "a", "a",
            )]))
            .await
            .expect("检索应成功");
        let ServiceResponse::Retrieved(retrieval) = response else {
            panic!("期望 Retrieved 响应");
        };
        let contents: Vec<&str> = retrieval
            .memories
            .iter()
            .map(|memory| memory.content.as_str())
            .collect();
        assert_eq!(
            contents,
            vec![low_content, high_content],
            "同分应按 MemoryId 升序"
        );
    }

    #[tokio::test]
    async fn summarize_failure_keeps_evicted_message_for_retry() {
        let repo = Arc::new(FakeRepo::default());
        let llm = Arc::new(LlmEngine::new(Arc::new(FailOnceLlm::default())));
        let mut service = build_service_with(repo, Arc::new(FakeModel), llm, 1);

        service
            .handle_request(ServiceRequest::Ingest(delta("a")))
            .await
            .expect("首次入队应成功");

        let result = service
            .handle_request(ServiceRequest::Ingest(delta("b")))
            .await;
        assert!(result.is_err(), "首次摘要失败应返回错误");
        assert_eq!(service.pending_evicted.len(), 1, "被弹出的消息必须保留");
        assert!(!service.llm_available);

        service
            .handle_request(ServiceRequest::Ingest(delta("c")))
            .await
            .expect("重试摘要应成功");
        assert!(
            service.pending_evicted.is_empty(),
            "重试成功后应清空待摘要队列"
        );
        assert!(!service.wm.sliding_window().get_summary().trim().is_empty());
    }
}
