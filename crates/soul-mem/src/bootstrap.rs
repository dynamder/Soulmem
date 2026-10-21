//! 引导：按配置构造仓储、LLM、嵌入模型与服务核心。
//!
//! 模型供给策略：LLM 走远程 OpenAI 兼容端点（`[llm]` 段的 `base_url` / `model` /
//! `api_key`），嵌入模型用进程内 BGE（首次会下载权重）。密钥只进入 `OaiCompatConfig`，
//! **不进日志**（`LlmConfig` 的 `Debug` 已脱敏）。
//!
//! LLM **可选**：缺少 `[llm].base_url` / `model`（或后端构造失败）时注入一个恒错的
//! [`UnavailableBackend`]，服务照常启动——检索不依赖 LLM，只有摘要与巩固会失败。
//! 这符合"能不用 LLM 就不用"的原则，也让纯检索部署无需 LLM。
//!
//! 启动时**不**做探活：`llm_available` 初始为 `false`，由控制信号 `ProbeLlm` 触发
//! （见 `MemoryService::apply_control`）。

use std::sync::Arc;

use jieba_rs::Jieba;
use secrecy::ExposeSecret;
use soul_mem_llm::{
    BackendInfo, ChatBackend, Completion, EventStream, LlmEngine, LlmError, OaiCompatBackend,
    OaiCompatConfig, Task,
};
use soul_mem_query::embedding::EmbeddingModel;
use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;
use soul_mem_runtime::storage::surreal::SurrealRepository;
use soul_mem_runtime::storage::{MemoryRepository, StorageError};
use soul_mem_runtime::working_memory::WorkingMemory;
use tokio::sync::broadcast;

use crate::config::{Config, LlmConfig};
use crate::error::{ServiceError, ServiceResult};
use crate::service::{MemoryService, ServiceEvent};

/// 引导产物：服务核心与事件广播发送端。
pub struct Bootstrap {
    /// 服务核心。
    pub service: MemoryService,
    /// 事件广播发送端（心跳/巩固完成），供各适配器订阅。
    pub events: broadcast::Sender<ServiceEvent>,
}

/// 按配置完成全部依赖的构造。
///
/// `config` 以 `Arc` 注入服务核心，进程内只保留这一份设置。
pub async fn bootstrap(config: Arc<Config>) -> ServiceResult<Bootstrap> {
    let repo = SurrealRepository::connect(&config.db_path, &config.character)
        .await
        .map_err(StorageError::from)?;
    repo.init_schema().await?;
    let repo: Arc<dyn MemoryRepository> = Arc::new(repo);

    let llm = build_llm_engine(&config.llm);
    let model: Arc<dyn EmbeddingModel + Send + Sync> = Arc::new(build_embedding_model().await?);
    let jieba = Jieba::new();
    let working_memory = Arc::new(WorkingMemory::new(config.working_memory.window_capacity));
    let retrieval_pool = Arc::new(build_retrieval_pool(config.retrieval.threads)?);

    let (events, _receiver) = broadcast::channel(config.service.event_channel_capacity);
    let service = MemoryService::builder()
        .config(config)
        .working_memory(working_memory)
        .repository(repo)
        .llm(Arc::new(llm))
        .embedding_model(model)
        .jieba(jieba)
        .events(events.clone())
        .llm_available(false)
        .retrieval_pool(retrieval_pool)
        .build()?;

    Ok(Bootstrap { service, events })
}

/// 检索并行池：`threads` 为 `0` 时取 `核数 - 1`（至少 1），否则用配置值。
///
/// 池在引导时构建一次并注入服务核心；一次检索批次内的并行只读都落在这个池上。
fn build_retrieval_pool(threads: usize) -> ServiceResult<rayon::ThreadPool> {
    let workers = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .saturating_sub(1)
            .max(1)
    } else {
        threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("soulmem-retrieve-{index}"))
        .build()
        .map_err(|error| ServiceError::Internal(format!("构造检索线程池失败: {error}")))
}

/// 按 `[llm]` 配置构造远程 LLM 引擎；缺失配置时返回恒错后端。
fn build_llm_engine(cfg: &LlmConfig) -> LlmEngine {
    let base_url = cfg.base_url.trim();
    let model = cfg.model.trim();
    if base_url.is_empty() || model.is_empty() {
        tracing::warn!("未配置 [llm].base_url/model，LLM 摘要与巩固不可用（检索仍可用）");
        return unavailable_engine();
    }

    let mut oai = OaiCompatConfig::new("soulmem", base_url, model);
    if let Some(key) = cfg.api_key.as_ref() {
        let key = key.expose_secret();
        if !key.is_empty() {
            oai = oai.with_api_key(key);
        }
    }
    if let Err(error) = oai.validate() {
        tracing::warn!(%error, "LLM 配置校验失败，降级为不可用");
        return unavailable_engine();
    }
    match OaiCompatBackend::new(oai) {
        Ok(backend) => LlmEngine::new(Arc::new(backend)),
        Err(error) => {
            tracing::warn!(%error, "LLM 后端构造失败，降级为不可用");
            unavailable_engine()
        }
    }
}

/// 构造一个对任何调用都返回 `BackendUnavailable` 的引擎。
fn unavailable_engine() -> LlmEngine {
    LlmEngine::new(Arc::new(UnavailableBackend))
}

/// LLM 未配置时的占位后端：让服务在无 LLM 时仍能启动与检索。
struct UnavailableBackend;

#[async_trait::async_trait]
impl ChatBackend for UnavailableBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo::new("unavailable")
    }

    async fn complete(&self, _task: Task) -> Result<Completion, LlmError> {
        Err(LlmError::unavailable(
            "LLM 未配置（[llm] 缺少 base_url / model）",
        ))
    }

    async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
        Err(LlmError::unavailable(
            "LLM 未配置（[llm] 缺少 base_url / model）",
        ))
    }
}

/// 构造进程内 BGE 嵌入模型（阻塞、可能下载权重，放到阻塞线程池）。
async fn build_embedding_model() -> ServiceResult<BgeSmallZh> {
    tokio::task::spawn_blocking(BgeSmallZh::default_cpu)
        .await
        .map_err(|e| ServiceError::Internal(format!("嵌入模型构造任务失败: {e}")))?
        .map_err(|e| ServiceError::Embedding(format!("嵌入模型构造失败: {e}")))
}
