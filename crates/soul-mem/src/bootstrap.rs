//! 引导：按配置构造仓储、LLM、嵌入模型与服务核心。
//!
//! 模型供给策略：LLM 走远程 OpenAI 兼容端点（`.env` 的 `API_BASE` / `API_KEY` /
//! `MODEL`），嵌入模型用进程内 BGE（首次会下载权重）。密钥只进入 `OaiCompatConfig`，
//! **不落盘、不进日志**。
//!
//! LLM **可选**：缺少 `API_BASE` / `MODEL`（或后端构造失败）时注入一个恒错的
//! [`UnavailableBackend`]，服务照常启动——检索不依赖 LLM，只有摘要与巩固会失败。
//! 这符合"能不用 LLM 就不用"的原则，也让纯检索部署无需 LLM。

use std::sync::Arc;

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

use crate::config::Config;
use crate::error::{ServiceError, ServiceResult};
use crate::service::{MemoryService, ServiceEvent};

/// 事件广播通道容量。
const EVENT_CHANNEL_CAPACITY: usize = 64;

/// 引导产物：服务核心与事件广播发送端。
pub struct Bootstrap {
    /// 服务核心。
    pub service: MemoryService,
    /// 事件广播发送端（心跳/巩固完成），供各适配器订阅。
    pub events: broadcast::Sender<ServiceEvent>,
}

/// 按配置完成全部依赖的构造。
pub async fn bootstrap(config: &Config) -> ServiceResult<Bootstrap> {
    let repo = SurrealRepository::connect(&config.db_path, &config.character)
        .await
        .map_err(StorageError::from)?;
    repo.init_schema().await?;
    let repo: Arc<dyn MemoryRepository> = Arc::new(repo);

    let (llm, llm_available) = build_llm_engine();
    let model: Arc<dyn EmbeddingModel + Send + Sync> = Arc::new(build_embedding_model().await?);
    let working_memory = Arc::new(WorkingMemory::new(config.window_capacity));

    let (events, _receiver) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
    let service = MemoryService::new(
        config.clone(),
        working_memory,
        repo,
        Arc::new(llm),
        model,
        events.clone(),
        llm_available,
    );

    Ok(Bootstrap { service, events })
}

/// 从环境变量构造远程 LLM 引擎；缺失配置时返回恒错后端并标记不可用。
fn build_llm_engine() -> (LlmEngine, bool) {
    let base_url = std::env::var("API_BASE")
        .ok()
        .filter(|value| !value.is_empty());
    let model = std::env::var("MODEL")
        .ok()
        .filter(|value| !value.is_empty());

    let (Some(base_url), Some(model)) = (base_url, model) else {
        tracing::warn!("未配置 API_BASE/MODEL，LLM 摘要与巩固不可用（检索仍可用）");
        return (unavailable_engine(), false);
    };

    let mut config = OaiCompatConfig::new("soulmem", base_url, model);
    if let Ok(key) = std::env::var("API_KEY")
        && !key.is_empty()
    {
        config = config.with_api_key(key);
    }
    if let Err(error) = config.validate() {
        tracing::warn!(%error, "LLM 配置校验失败，降级为不可用");
        return (unavailable_engine(), false);
    }
    match OaiCompatBackend::new(config) {
        Ok(backend) => (LlmEngine::new(Arc::new(backend)), true),
        Err(error) => {
            tracing::warn!(%error, "LLM 后端构造失败，降级为不可用");
            (unavailable_engine(), false)
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
        Err(LlmError::unavailable("LLM 未配置（缺少 API_BASE / MODEL）"))
    }

    async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
        Err(LlmError::unavailable("LLM 未配置（缺少 API_BASE / MODEL）"))
    }
}

/// 构造进程内 BGE 嵌入模型（阻塞、可能下载权重，放到阻塞线程池）。
async fn build_embedding_model() -> ServiceResult<BgeSmallZh> {
    tokio::task::spawn_blocking(BgeSmallZh::default_cpu)
        .await
        .map_err(|e| ServiceError::Internal(format!("嵌入模型构造任务失败: {e}")))?
        .map_err(|e| ServiceError::Embedding(format!("嵌入模型构造失败: {e}")))
}
