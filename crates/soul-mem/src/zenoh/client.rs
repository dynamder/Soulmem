//! client：基于 zenoh 订阅/发布的客户端（protobuf 载荷）。
//!
//! 请求-应答流程：订阅本请求的应答主题 → 发布 `RequestEnvelope` → 等待并匹配
//! `request_id` 的 `ReplyEnvelope`。供 mock-device 等联网演示/设备端复用。
//!
//! 注意：自动化测试不依赖网络，不使用本模块；网络行为由 `mock-device` 手动演示。

use super::keys::Keys;
use crate::error::{Error, Result};
use crate::wire::op;
use crate::wire::pb;
use prost::Message;
use std::time::Duration;

/// 默认等待应答的超时。
pub const DEFAULT_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// zenoh 客户端（设备端）。
pub struct ZenohClient {
    session: zenoh::Session,
    keys: Keys,
    device_id: String,
}

impl ZenohClient {
    /// 建立会话。`device_id` 仅作为客户端标识。
    pub async fn open(prefix: impl Into<String>, device_id: impl Into<String>) -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::internal(format!("open zenoh session: {e}")))?;
        Ok(ZenohClient {
            session,
            keys: Keys::new(prefix),
            device_id: device_id.into(),
        })
    }

    pub fn keys(&self) -> &Keys {
        &self.keys
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// 发一次请求-应答：`Req`/`Resp` 均为 protobuf 消息。
    pub async fn call<Req, Resp>(&self, op_name: &str, req: &Req) -> Result<Resp>
    where
        Req: Message,
        Resp: Message + Default,
    {
        let request_id = uuid::Uuid::new_v4().to_string();
        let reply_key = self.keys.reply_for(&request_id);
        let subscriber = self
            .session
            .declare_subscriber(reply_key)
            .await
            .map_err(|e| Error::Unavailable(format!("declare reply subscriber: {e}")))?;

        let envelope = pb::RequestEnvelope {
            request_id: request_id.clone(),
            op: op_name.to_string(),
            payload: req.encode_to_vec(),
        };
        self.session
            .put(self.keys.request(), envelope.encode_to_vec())
            .await
            .map_err(|e| Error::Unavailable(format!("publish request: {e}")))?;

        let deadline = tokio::time::Instant::now() + DEFAULT_REPLY_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(Error::Unavailable(format!(
                    "request {op_name} timed out waiting for reply"
                )));
            }
            match tokio::time::timeout(remaining, subscriber.recv_async()).await {
                Ok(Ok(sample)) => {
                    let bytes = sample.payload().to_bytes().to_vec();
                    let reply = pb::ReplyEnvelope::decode(&bytes[..])
                        .map_err(|e| Error::internal(format!("decode ReplyEnvelope: {e}")))?;
                    if reply.request_id != request_id {
                        continue;
                    }
                    if let Some(err) = reply.error {
                        return Err(Error::internal(format!("{}: {}", err.code, err.message)));
                    }
                    return Resp::decode(&reply.payload[..])
                        .map_err(|e| Error::internal(format!("decode response payload: {e}")));
                }
                Ok(Err(e)) => {
                    return Err(Error::Unavailable(format!("reply stream error: {e}")));
                }
                Err(_) => {
                    return Err(Error::Unavailable(format!(
                        "request {op_name} timed out waiting for reply"
                    )));
                }
            }
        }
    }

    pub async fn ping(&self) -> Result<pb::PingResponse> {
        self.call(op::PING, &pb::Empty {}).await
    }

    /// 通过请求-应答方式 ingest（可拿到 ack）。
    pub async fn ingest(&self, dto: &pb::IngestRequest) -> Result<pb::Ack> {
        self.call(op::INGEST, dto).await
    }

    /// 单向发布信息增量到 ingest 主题（无 ack，体现推式输入）。
    pub async fn publish_ingest(&self, dto: &pb::IngestRequest) -> Result<()> {
        self.session
            .put(self.keys.ingest(), dto.encode_to_vec())
            .await
            .map_err(|e| Error::Unavailable(format!("publish ingest: {e}")))
    }

    pub async fn retrieve(&self, dto: &pb::RetrieveRequest) -> Result<pb::RetrieveResponse> {
        self.call(op::RETRIEVE, dto).await
    }

    pub async fn read_note(&self, id: &str) -> Result<pb::ReadNoteResponse> {
        self.call(op::READ, &pb::ReadNoteRequest { id: id.to_string() })
            .await
    }

    pub async fn write_note(&self, dto: &pb::WriteNoteRequest) -> Result<pb::WriteNoteResponse> {
        self.call(op::WRITE, dto).await
    }

    pub async fn feedback(&self, dto: &pb::Feedback) -> Result<pb::Ack> {
        self.call(op::FEEDBACK, dto).await
    }

    pub async fn control(&self, kind: pb::ControlKind) -> Result<pb::ControlResponse> {
        self.call(op::CONTROL, &pb::Control { kind: kind as i32 })
            .await
    }

    /// 观察 liveliness：打印并计数指定 device_id 的 token 上线（超时自动退出）。
    ///
    /// 返回观察到的样本数（best-effort：若订阅晚于 token 声明，可能收不到历史样本）。
    pub async fn observe_liveliness(&self, target: &str, seconds: u64) -> Result<usize> {
        let subscriber = self
            .session
            .declare_subscriber(self.keys.liveliness_all())
            .await
            .map_err(|e| Error::Unavailable(format!("declare liveliness subscriber: {e}")))?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
        let mut seen = 0usize;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, subscriber.recv_async()).await {
                Ok(Ok(sample)) => {
                    if sample.key_expr().to_string().contains(target) {
                        seen += 1;
                        log::info!("liveliness observed: {} online", sample.key_expr());
                    }
                }
                _ => break,
            }
        }
        Ok(seen)
    }

    pub async fn close(self) {
        let _ = self.session.close().await;
    }
}
