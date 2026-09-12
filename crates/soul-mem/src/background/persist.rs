//! 定时持久化任务（真实可用）：周期把工作记忆快照交给 MemoryStore。
//! 也用于优雅退出的“退出前落盘”。

use crate::error::Result;
use crate::service::SoulMemService;

/// 执行一次持久化（返回给调度器；错误由调度器记录并隔离）。
pub(crate) async fn run_once(service: &SoulMemService) -> Result<()> {
    service.persist().await
}
