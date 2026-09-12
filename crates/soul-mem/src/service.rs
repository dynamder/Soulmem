//! Service 编排层：`SoulMemService` 门面 + 共享状态 + 编排原语。
//!
//! 对应 orchestration.md 的“Service 编排层”：zenoh 通道与 mock-device 都只与本模块交互，
//! 业务收敛于此。并发模型见 `WorkingMemorySlot`（Arc 共享 + 独占可迁移写）。
//!
//! 对外方法使用 `wire::pb`（protobuf 生成的消息类型）；内部领域转换见 `wire::convert`。

mod control;
mod ingest;
mod note_ops;
mod retrieve;

use crate::config::{Config, EmbeddingMode};
use crate::error::{Error, Result};
use crate::store::{Snapshot, Store, WindowEntryDto};
use crate::wire::pb;
use soul_mem_algo::algo::retrieve::complex::assoc_with_action::AssociateWithActionConfig;
use soul_mem_algo::algo::retrieve::complex::default_pipeline::DefaultPipelineConfig;
use soul_mem_algo::algo::retrieve::short_only::ShortOnlyConfig;
use soul_mem_algo::algo::retrieve::similarity::SimilarityConfig;
use soul_mem_query::embedding::{EmbeddingGenResult, EmbeddingModel, EmbeddingVec};
use soul_mem_runtime::working_memory::llm::client::LlmClient;
use soul_mem_runtime::working_memory::llm::config::LLMConfig;
use soul_mem_runtime::working_memory::sliding_window::Information;
use soul_mem_runtime::working_memory::{WorkingMemory, WorkingState};
use std::sync::Arc;

/// WorkingMemory 共享槽：对外只暴露两种访问方式。
///
/// - `wm_arc()`：只读共享（供检索管线拿 `Arc<WorkingMemory>`）。
/// - `with_mut()`：独占可变（内部用 `Arc::get_mut` 迁移，非唯一时让出 CPU 重试），
///   用于 add_node / record_retrieval / feedback / 状态迁移等。
///
/// 依赖：调用方在“持有一份 Arc 克隆做长 await”期间，`with_mut` 会自旋等待克隆释放；
/// Demo 并发度低，可接受；将来若需严格并发可换成按字段拆分锁。
pub(crate) struct WorkingMemorySlot {
    inner: tokio::sync::Mutex<Option<Arc<WorkingMemory>>>,
}

impl WorkingMemorySlot {
    fn new(wm: WorkingMemory) -> Self {
        WorkingMemorySlot {
            inner: tokio::sync::Mutex::new(Some(Arc::new(wm))),
        }
    }

    /// 取一份只读 Arc 克隆（不持有互斥锁返回）。
    pub(crate) async fn arc(&self) -> Arc<WorkingMemory> {
        let guard = self.inner.lock().await;
        guard
            .as_ref()
            .expect("slot always occupied between operations")
            .clone()
    }

    /// 独占可变访问；若 Arc 非唯一（有并发只读克隆）则让出并重试。
    pub(crate) async fn with_mut<R>(&self, f: impl FnOnce(&mut WorkingMemory) -> R) -> R {
        loop {
            let mut guard = self.inner.lock().await;
            let mut owned = match guard.take() {
                Some(owned) => owned,
                None => {
                    drop(guard);
                    tokio::task::yield_now().await;
                    continue;
                }
            };
            if let Some(wm) = Arc::get_mut(&mut owned) {
                let out = f(wm);
                *guard = Some(owned);
                return out;
            }
            *guard = Some(owned);
            drop(guard);
            tokio::task::yield_now().await;
        }
    }
}

/// 服务运行时上下文（共享于所有通道任务）。
pub(crate) struct ServiceCore {
    pub(crate) device_id: String,
    pub(crate) slot: WorkingMemorySlot,
    pub(crate) model: Arc<dyn EmbeddingModel + Send + Sync>,
    pub(crate) llm: Arc<LlmClient>,
    pub(crate) pipeline: DefaultPipelineConfig,
    pub(crate) store: Store,
}

/// 对外门面。
#[derive(Clone)]
pub struct SoulMemService {
    pub(crate) core: Arc<ServiceCore>,
}

impl SoulMemService {
    /// 依据配置与（可选）已加载快照构造服务。
    ///
    /// `loaded` 来自 `store.load()`；传入即做启动恢复（新增节点/回填窗口与摘要）。
    pub async fn from_config(
        config: &Config,
        store: Store,
        loaded: Option<Snapshot>,
    ) -> Result<SoulMemService> {
        config.validate()?;

        let mut wm = WorkingMemory::new(config.window_capacity);
        if let Some(snapshot) = loaded {
            restore_snapshot(&mut wm, &snapshot)?;
        }

        let model = build_model(config)?;
        let llm = Arc::new(LlmClient::new(LLMConfig::new(
            &config.llm_api_key,
            &config.llm_base_url,
            &config.llm_model,
        )));

        let pipeline = DefaultPipelineConfig {
            short_mem_with_history: ShortOnlyConfig {
                clipping_length: None,
                include_summary: true,
            },
            similarity: SimilarityConfig {
                similarity_threshold: config.similarity_threshold,
                max_results: config.similarity_max_results,
            },
            assoc_with_action: AssociateWithActionConfig {
                action_top_k: 3,
                ..Default::default()
            },
        };

        Ok(SoulMemService {
            core: Arc::new(ServiceCore {
                device_id: config.device_id.clone(),
                slot: WorkingMemorySlot::new(wm),
                model,
                llm,
                pipeline,
                store,
            }),
        })
    }

    /// 只读共享的工作记忆（检索管线使用）。
    pub(crate) async fn wm_arc(&self) -> Arc<WorkingMemory> {
        self.core.slot.arc().await
    }

    /// 独占可变访问工作记忆。
    pub(crate) async fn with_wm<R>(&self, f: impl FnOnce(&mut WorkingMemory) -> R) -> Result<R> {
        Ok(self.core.slot.with_mut(f).await)
    }

    /// 生成当前内存状态的快照（共享读，不阻塞写入）。
    pub(crate) async fn snapshot(&self) -> Result<Snapshot> {
        let wm = self.wm_arc().await;
        let nodes = {
            let handle = wm.memory_cluster();
            handle.read_or_compute(|cluster| {
                use petgraph::visit::IntoNodeIdentifiers;
                cluster
                    .graph()
                    .node_identifiers()
                    .filter_map(|ix| cluster.graph().node_weight(ix).cloned())
                    .collect::<Vec<_>>()
            })
        };
        let window: Vec<WindowEntryDto> = {
            let sw = wm.sliding_window();
            sw.get_windows()
                .iter()
                .map(|info| WindowEntryDto {
                    role: information_role(info).to_string(),
                    content: info.get_str().to_string(),
                    tagged: info.is_tagged(),
                })
                .collect()
        };
        let summary = wm.sliding_window().get_summary().to_string();
        Ok(Snapshot::new(summary, window, nodes))
    }

    /// 立即持久化（定时/退出/控制信号共用）。
    pub async fn persist(&self) -> Result<()> {
        let snapshot = self.snapshot().await?;
        self.core.store.save(&snapshot).await
    }

    pub fn device_id(&self) -> &str {
        &self.core.device_id
    }

    /// 存活探测：返回本服务 device_id。
    pub async fn ping(&self) -> Result<pb::PingResponse> {
        Ok(pb::PingResponse {
            device_id: self.core.device_id.clone(),
        })
    }

    /// 工作记忆当前是否空闲（后台巩固/遗忘任务的门控依据）。
    pub(crate) async fn is_idle(&self) -> bool {
        let wm = self.wm_arc().await;
        matches!(wm.state(), WorkingState::Idle)
    }
}

/// 把快照恢复到（新建的）工作记忆上。
fn restore_snapshot(wm: &mut WorkingMemory, snapshot: &Snapshot) -> Result<()> {
    snapshot.validate()?;
    for node in &snapshot.nodes {
        wm.add_node(node.clone());
    }
    {
        let sw = wm.sliding_window_mut();
        {
            let mut queue = sw.window().write();
            for entry in &snapshot.window {
                queue.push_back(Information::new(&entry.content, &entry.role));
            }
        }
        for (i, entry) in snapshot.window.iter().enumerate() {
            if entry.tagged {
                sw.tag_information(i);
            }
        }
        if !snapshot.summary.is_empty() {
            sw.summary().write().update(snapshot.summary.clone());
        }
    }
    Ok(())
}

/// Information → 角色字符串。
pub(crate) fn information_role(info: &Information) -> &'static str {
    match info {
        Information::User(_) => "user",
        Information::Assistant(_) => "assistant",
    }
}

/// 依据配置构造 embedding 模型。
fn build_model(config: &Config) -> Result<Arc<dyn EmbeddingModel + Send + Sync>> {
    match config.embedding_mode {
        EmbeddingMode::Hash => Ok(Arc::new(HashEmbeddingModel::new(64))),
        EmbeddingMode::Bge => Err(Error::Unimplemented(
            "embedding mode \"bge\" is not wired yet; use \"hash\" for this demo".into(),
        )),
    }
}

/// 确定性哈希 embedding 模型（离线、可复现）。
///
/// 用途：单机 Demo / 测试在**不下载真实模型**的前提下走通完整检索链路。
/// 相同/相似文本得到高相似向量，便于端到端断言；生产环境应替换为 BGE/Qwen3 等。
struct HashEmbeddingModel {
    dim: usize,
}

impl HashEmbeddingModel {
    fn new(dim: usize) -> Self {
        HashEmbeddingModel { dim }
    }

    fn raw_vec(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let mut sum = 0.0f32;
        for ch in text.chars() {
            let code = ch as u32 as usize;
            let idx = code.wrapping_mul(2_654_435_761) % self.dim;
            let w = (code % 997) as f32 / 997.0 + 1.0;
            v[idx] += w;
            sum += w;
        }
        if sum > 0.0 {
            for x in &mut v {
                *x /= sum;
            }
        }
        v
    }
}

impl EmbeddingModel for HashEmbeddingModel {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        Ok(input
            .iter()
            .map(|s| EmbeddingVec::new(self.raw_vec(s)))
            .collect())
    }

    fn infer_with_chunk(&self, input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        Ok(EmbeddingVec::new(self.raw_vec(input)))
    }

    fn infer_and_fuse(&self, input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        let mut fused = vec![0.0f32; self.dim];
        for s in input {
            let v = self.raw_vec(s);
            for (acc, x) in fused.iter_mut().zip(v.iter()) {
                *acc += x;
            }
        }
        if !input.is_empty() {
            for x in &mut fused {
                *x /= input.len() as f32;
            }
        }
        Ok(EmbeddingVec::new(fused))
    }

    fn max_input_token(&self) -> usize {
        128
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
