//! 传输无关的领域 DTO。
//!
//! 适配器把 proto 消息转换成这里的类型再投递给核心，核心实现不依赖 `proto`。

use chrono::{DateTime, Utc};
use soul_mem_core::render::RenderedMemoryNote;
use soul_mem_query::query::retrieve::PrioritizedMemoryRetrieveQuery;

/// 一次请求：query 集合 + 信息增量 + 控制信号（均可空，可同时）。
#[derive(Debug, Clone, Default)]
pub struct ServiceRequest {
    /// 检索 query 集合，每条带优先级。
    pub queries: Vec<PrioritizedMemoryRetrieveQuery>,
    /// 信息增量，按顺序压入滑动窗口。
    pub deltas: Vec<Delta>,
    /// 控制信号，按顺序执行。
    pub controls: Vec<ControlAction>,
}

/// 一条信息增量：语句 + 角色。
#[derive(Debug, Clone)]
pub struct Delta {
    /// 语句内容。
    pub statement: String,
    /// 角色：`"user"` / `"assistant"`（其他值按 user 处理）。
    pub role: String,
}

/// 一次响应。
///
/// 即使某一步失败也照常返回：`error` 承载首个失败原因，`state`/`accepted` 反映
/// 已经发生的状态变化，避免"失败即丢掉部分结果"。
#[derive(Debug, Clone)]
pub struct ServiceResponse {
    /// 检索结果（一段自然语言）；无 query 时为 `None`。
    pub output: Option<String>,
    /// 结构化检索结果（命中的记忆节点），与 `output` 并列；无 query 时为空。
    pub memories: Vec<RenderedMemoryNote>,
    /// 服务状态快照。
    pub state: ServiceState,
    /// 实际接受的信息增量条数。
    pub accepted: u32,
    /// 首个失败原因；`None` 表示本次请求全部成功。
    pub error: Option<String>,
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
    /// 强制遗忘（占位，未实现）。
    Forget,
    /// 暂停接收信息增量。
    Pause,
    /// 恢复接收信息增量。
    Resume,
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
