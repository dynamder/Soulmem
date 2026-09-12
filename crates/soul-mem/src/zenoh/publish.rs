//! publish：主动事件广播（记忆变更/巩固完成等）与检索结果回投。
//!
//! Demo 期由 service 侧可选调用：向 `events` 主题发布 `EventNotice`（protobuf），
//! 让关心“记忆变化”的订阅方实时感知而无需轮询。

use crate::error::{Error, Result};
use crate::wire::pb;
use prost::Message;
use zenoh::Session;

/// 向事件主题广播一条 `EventNotice`（尽力而为：失败仅记录）。
///
/// 预留：未来在 ingest/consolidate 完成后调用（当前无自动触发点）。
#[allow(dead_code)]
pub async fn publish_event(
    session: &Session,
    event_key: &str,
    event: &pb::EventNotice,
) -> Result<()> {
    session
        .put(event_key, event.encode_to_vec())
        .await
        .map_err(|e| Error::internal(format!("publish event: {e}")))?;
    Ok(())
}
