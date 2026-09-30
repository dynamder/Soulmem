//! zenoh pub/sub 适配器。
//!
//! 三条 keyexpr：
//! - `<prefix>/req`：订阅客户端请求。
//! - `<prefix>/resp`：发布响应（按 `request_id` 配对）。
//! - `<prefix>/event`：发布服务主动事件（心跳、巩固完成）。
//!
//! pub/sub 无应答语义，故请求与响应是两条独立的 publish；客户端订阅 `resp`
//! 并按 `request_id` 过滤。事件来源是服务内部广播通道。

use tokio::sync::broadcast;
use zenoh::bytes::ZBytes;

use super::{convert, dispatch};
use crate::service::{ServiceEvent, ServiceHandle};

/// 启动 zenoh 适配器；返回时表示会话/订阅已结束。
pub async fn run(
    handle: ServiceHandle,
    prefix: String,
    mut events: broadcast::Receiver<ServiceEvent>,
) -> anyhow::Result<()> {
    let session = zenoh::open(zenoh::Config::default())
        .await
        .map_err(to_anyhow)?;
    let req_key = format!("{prefix}/req");
    let resp_key = format!("{prefix}/resp");
    let event_key = format!("{prefix}/event");

    let subscriber = session
        .declare_subscriber(req_key.clone())
        .await
        .map_err(to_anyhow)?;
    let responder = session
        .declare_publisher(resp_key.clone())
        .await
        .map_err(to_anyhow)?;
    let event_publisher = session
        .declare_publisher(event_key.clone())
        .await
        .map_err(to_anyhow)?;

    let event_task = tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) => {
                    let bytes = dispatch::encode_event(&convert::event(event));
                    if let Err(error) = event_publisher.put(ZBytes::from(bytes)).await {
                        tracing::warn!(%error, "事件发布失败");
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!(skipped, "事件订阅落后，已跳过部分事件");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    tracing::info!(%req_key, %resp_key, %event_key, "zenoh 适配器已启动");

    loop {
        let sample = match subscriber.recv_async().await {
            Ok(sample) => sample,
            Err(error) => {
                tracing::warn!(%error, "zenoh 订阅结束");
                break;
            }
        };
        let bytes = sample.payload().to_bytes();
        let request = match dispatch::decode_request(&bytes) {
            Ok(request) => request,
            Err(error) => {
                tracing::warn!(%error, "请求解码失败，已丢弃");
                continue;
            }
        };
        let reply = dispatch::dispatch(&handle, request).await;
        let encoded = dispatch::encode_reply(&reply);
        if let Err(error) = responder.put(ZBytes::from(encoded)).await {
            tracing::warn!(%error, "响应发布失败");
        }
    }

    event_task.abort();
    Ok(())
}

/// zenoh 的错误是 `Box<dyn Error + Send + Sync>`，不能直接经 `?` 转成 `anyhow::Error`，
/// 统一折叠为其展示文本。
fn to_anyhow(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}
