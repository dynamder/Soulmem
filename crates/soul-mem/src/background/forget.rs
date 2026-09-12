//! 遗忘相关任务挂接点（占位：遮罩衰减/遗忘补全）。
//!
//! 同 consolidate：下层算法就绪前仅保留调度位与 Idle 门控。

use crate::error::Result;
use crate::service::SoulMemService;
use crate::wire::pb;

pub(crate) async fn run_once(service: &SoulMemService) -> Result<()> {
    if !service.is_idle().await {
        return Ok(());
    }
    service
        .control(pb::Control {
            kind: pb::ControlKind::ControlForget as i32,
        })
        .await
        .map(|_| ())
}
