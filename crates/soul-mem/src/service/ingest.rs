//! ingest：信息增量 → 压入滑动窗口（必要时触发摘要）。
//!
//! 对应 orchestration「行为流程」：信息增量压入窗口，触发 auto_tag / 可能的摘要生成。

use super::SoulMemService;
use crate::error::{Error, Result};
use crate::wire::convert::{MAX_DELTA_CHARS, role_to_str};
use crate::wire::pb;

impl SoulMemService {
    /// 压入一条或多条信息增量（protobuf 请求 → ack）。
    pub async fn ingest(&self, req: pb::IngestRequest) -> Result<pb::Ack> {
        if req.deltas.is_empty() {
            return Err(Error::InvalidArgument(
                "ingest requires at least one delta".into(),
            ));
        }
        for delta in &req.deltas {
            if delta.content.trim().is_empty() {
                return Err(Error::InvalidArgument(
                    "delta content must not be empty".into(),
                ));
            }
            if delta.content.chars().count() > MAX_DELTA_CHARS {
                return Err(Error::InvalidArgument(format!(
                    "delta content too long: exceeds {MAX_DELTA_CHARS} chars"
                )));
            }
            // 提前校验角色，避免把非法角色静默当成 user。
            let _ = role_to_str(delta.role)?;
        }

        // 持只读 Arc 跨 await：滑动窗口内部自带上锁，push 期间允许其它只读任务并行。
        let wm = self.wm_arc().await;
        for delta in &req.deltas {
            let role = role_to_str(delta.role)?;
            wm.sliding_window()
                .push(&delta.content, role, &self.core.llm)
                .await
                .map_err(|e| Error::internal(format!("push delta into sliding window: {e:#}")))?;
        }
        drop(wm);

        Ok(pb::Ack {
            message: format!("ingested {} delta(s)", req.deltas.len()),
        })
    }
}
