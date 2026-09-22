//! request：用 zenoh 订阅/发布实现的请求-应答（protobuf 载荷）。
//!
//! 客户端发布 `RequestEnvelope`（protobuf）到 `<prefix>/request`；本模块订阅该主题，
//! 按 `op` 解码内层请求消息、调用 `SoulMemService`，再把 `ReplyEnvelope`（protobuf）
//! 发布到 `<prefix>/reply/<request_id>`。这是设备间通信的唯一入口。

use super::keys::Keys;
use crate::error::{Error, Result};
use crate::service::SoulMemService;
use crate::wire::op;
use crate::wire::pb;
use prost::Message;
use zenoh::Session;

/// 订阅请求主题并逐个处理。
pub(crate) fn spawn_request_subscriber(
    session: Session,
    service: SoulMemService,
    keys: Keys,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let subscriber = match session.declare_subscriber(keys.request()).await {
            Ok(sub) => sub,
            Err(e) => {
                log::error!("declare request subscriber: {e}");
                return;
            }
        };
        log::info!("zenoh request subscriber ready at {}", keys.request());
        while let Ok(sample) = subscriber.recv_async().await {
            let bytes = sample.payload().to_bytes().to_vec();
            let request = match pb::RequestEnvelope::decode(&bytes[..]) {
                Ok(req) => req,
                Err(e) => {
                    log::warn!("bad RequestEnvelope: {e}");
                    continue;
                }
            };
            let request_id = request.request_id.clone();
            let reply = match dispatch(&service, &request.op, &request.payload).await {
                Ok(payload) => pb::ReplyEnvelope {
                    request_id,
                    payload,
                    error: None,
                },
                Err(e) => pb::ReplyEnvelope {
                    request_id,
                    payload: Vec::new(),
                    error: Some(pb::ErrorPayload {
                        code: e.code().as_str().to_string(),
                        message: e.to_string(),
                    }),
                },
            };
            let reply_key = keys.reply_for(&reply.request_id);
            let body = reply.encode_to_vec();
            if let Err(e) = session.put(reply_key, body).await {
                log::warn!("publish reply failed: {e}");
            }
        }
    })
}

/// 按操作名解码请求、调用 service、编码响应。
async fn dispatch(service: &SoulMemService, op: &str, payload: &[u8]) -> Result<Vec<u8>> {
    match op {
        op::PING => {
            decode::<pb::Empty>(payload)?;
            encode(&service.ping().await?)
        }
        op::INGEST => {
            let req = decode::<pb::IngestRequest>(payload)?;
            encode(&service.ingest(req).await?)
        }
        op::RETRIEVE => {
            let req = decode::<pb::RetrieveRequest>(payload)?;
            encode(&service.retrieve(req).await?)
        }
        op::READ => {
            let req = decode::<pb::ReadNoteRequest>(payload)?;
            encode(&service.read_note(req).await?)
        }
        op::WRITE => {
            let req = decode::<pb::WriteNoteRequest>(payload)?;
            encode(&service.write_note(req).await?)
        }
        op::FEEDBACK => {
            let req = decode::<pb::Feedback>(payload)?;
            encode(&service.feedback(req).await?)
        }
        op::CONTROL => {
            let req = decode::<pb::Control>(payload)?;
            encode(&service.control(req).await?)
        }
        other => Err(Error::InvalidArgument(format!("unknown op: {other:?}"))),
    }
}

fn decode<M: Message + Default>(payload: &[u8]) -> Result<M> {
    M::decode(payload).map_err(|e| Error::InvalidArgument(format!("decode request payload: {e}")))
}

fn encode<M: Message>(message: &M) -> Result<Vec<u8>> {
    Ok(message.encode_to_vec())
}
