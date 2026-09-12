//! control：控制信号 → 强制触发后台任务。
//!
//! 对应 orchestration「控制信号（强制触发定时任务）🔲」。
//! persist 已有真实实现；consolidate / forget 所依赖的下层算法尚未就绪，
//! 这里以明确的 `Unimplemented` 返回（不伪造结果，见 plan.md §2.5）。

use super::SoulMemService;
use crate::error::{Error, Result};
use crate::wire::pb;

impl SoulMemService {
    /// 强制触发指定后台任务。
    pub async fn control(&self, req: pb::Control) -> Result<pb::ControlResponse> {
        let kind =
            pb::ControlKind::try_from(req.kind).unwrap_or(pb::ControlKind::ControlUnspecified);
        match kind {
            pb::ControlKind::ControlPersist => {
                self.persist().await?;
                Ok(pb::ControlResponse {
                    signal: "persist".to_string(),
                    ok: true,
                    message: "snapshot persisted".to_string(),
                })
            }
            pb::ControlKind::ControlConsolidate => Err(Error::Unimplemented(
                "consolidation task is a placeholder: lower-crate algorithm not ready yet".into(),
            )),
            pb::ControlKind::ControlForget => Err(Error::Unimplemented(
                "forget task is a placeholder: lower-crate algorithm not ready yet".into(),
            )),
            pb::ControlKind::ControlUnspecified => {
                Err(Error::InvalidArgument("control kind is unspecified".into()))
            }
        }
    }
}
