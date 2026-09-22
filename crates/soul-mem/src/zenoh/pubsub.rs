//! pubsub：订阅外部设备发布的信息增量（推式输入）。
//!
//! 外部设备把 `IngestRequest`（protobuf）发布到 `<prefix>/ingest`，本服务被动接收。

use crate::service::SoulMemService;
use crate::wire::pb;
use prost::Message;
use zenoh::Session;

/// 订阅 `<prefix>/ingest`：解码 `IngestRequest` 并转交 service。
pub(crate) fn spawn_ingest_subscriber(
    session: Session,
    service: SoulMemService,
    key: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let subscriber = match session.declare_subscriber(key).await {
            Ok(sub) => sub,
            Err(e) => {
                log::error!("declare ingest subscriber: {e}");
                return;
            }
        };
        log::info!("zenoh ingest subscriber ready");
        while let Ok(sample) = subscriber.recv_async().await {
            let bytes = sample.payload().to_bytes().to_vec();
            match pb::IngestRequest::decode(&bytes[..]) {
                Ok(req) => match service.ingest(req).await {
                    Ok(_) => {}
                    Err(e) => log::warn!("zenoh ingest failed: {e}"),
                },
                Err(e) => log::warn!("zenoh ingest bad payload: {e}"),
            }
        }
    })
}
