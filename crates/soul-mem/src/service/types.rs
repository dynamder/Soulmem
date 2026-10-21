//! 传输无关的领域 DTO。
//!
//! 适配器把 proto 消息转换成这里的类型再投递给核心，核心实现不依赖 `proto`。

use chrono::{DateTime, Utc};
use soul_mem_core::render::RenderedMemoryNote;
use soul_mem_query::query::retrieve::PrioritizedMemoryRetrieveQuery;

/// 一次原子请求：同一请求只做一件事。
#[derive(Debug, Clone)]
pub enum ServiceRequest {
    /// 压入一条信息增量。
    Ingest(Delta),
    /// 执行一次检索（query 集合整体合并）。
    Retrieve(Vec<PrioritizedMemoryRetrieveQuery>),
    /// 执行一个控制信号。
    Control(ControlAction),
}

/// 一条信息增量：语句 + 角色。
#[derive(Debug, Clone)]
pub struct Delta {
    /// 语句内容。
    pub statement: String,
    /// 角色：`"user"` / `"assistant"`（其他值按 user 处理）。
    pub role: String,
}

/// 检索产物：自然语言 + 结构化记忆 + 状态快照。
#[derive(Debug, Clone)]
pub struct Retrieval {
    /// 一段自然语言。
    pub output: String,
    /// 命中的记忆节点。
    pub memories: Vec<RenderedMemoryNote>,
    /// 检索后的状态快照。
    pub state: ServiceState,
}

/// 一次原子请求的成功响应。
#[derive(Debug, Clone)]
pub enum ServiceResponse {
    Ingested(ServiceState),
    Retrieved(Retrieval),
    Controlled(ServiceState),
}

impl ServiceResponse {
    /// 响应携带的状态快照。
    pub fn state(&self) -> &ServiceState {
        match self {
            Self::Ingested(state) | Self::Controlled(state) => state,
            Self::Retrieved(retrieval) => &retrieval.state,
        }
    }
}

/// 控制信号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlAction {
    /// 强制巩固（摘要 -> 新记忆节点）。
    Consolidate,
    /// 工作记忆持久化到数据库。
    Persist,
    /// 导出状态快照。状态本就随每个 `Reply.state` 返回，因此这是显式的空操作。
    Snapshot,
    /// 触发遗忘衰减（刷新工作记忆缺失度）；遮罩与补全在检索时惰性发生。
    Forget,
    /// 暂停接收信息增量。
    Pause,
    /// 恢复接收信息增量。
    Resume,
    /// 触发一次 LLM 探活，更新可用性标志（启动时不再自动探测）。
    ProbeLlm,
    /// 内部维护：静默超过阈值后转入 Idle 并（摘要非空时）巩固。
    ///
    /// 通常由服务自身的定时任务发出；暴露在协议里是为了便于外部主动触发一次维护。
    Maintenance,
}

/// 服务主动推送的事件。
#[derive(Debug, Clone)]
pub enum ServiceEvent {
    /// 周期心跳。
    Heartbeat(ServiceState),
    /// 一次巩固完成。
    ConsolidationDone {
        /// 本次巩固新建的记忆节点数。
        created_notes: u32,
        /// 巩固后的状态快照。
        state: ServiceState,
    },
}

/// 服务状态快照。
#[derive(Debug, Clone)]
pub struct ServiceState {
    /// 是否处于 Working 状态。
    pub working: bool,
    /// 工作记忆节点数。
    pub node_count: u64,
    /// 工作记忆边数。
    pub edge_count: u64,
    /// LLM 是否可用。
    pub llm_available: bool,
    /// 上次巩固完成时间。
    pub last_consolidation_at: Option<DateTime<Utc>>,
}
