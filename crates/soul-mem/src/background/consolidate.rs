//! 巩固任务挂接点（占位）。
//!
//! 巩固算法（摘要+窗口 → 新 MemoryNote 并入簇）由下层 crate 负责，尚未就绪；
//! 这里保留调度位与 Idle 门控，触发时返回明确的“未实现”，不伪造结果。

use crate::error::Result;
use crate::service::SoulMemService;
use crate::wire::pb;

pub(crate) async fn run_once(service: &SoulMemService) -> Result<()> {
    // Idle 门控：仅空闲时允许巩固。
    if !service.is_idle().await {
        return Ok(());
    }
    service
        .control(pb::Control {
            kind: pb::ControlKind::ControlConsolidate as i32,
        })
        .await
        .map(|_| ())
}
