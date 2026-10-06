//! zenoh pub/sub 适配器。
//!
//! 三条 keyexpr：
//! - `<prefix>/req`：订阅客户端请求。
//! - `<prefix>/resp`：发布响应（按 `request_id` 配对）。
//! - `<prefix>/event`：发布服务主动事件（心跳、巩固完成）。
//!
//! pub/sub 无应答语义，故请求与响应是两条独立的 publish；客户端订阅 `resp`
//! 并按 `request_id` 过滤。事件来源是服务内部广播通道。
//!
//! 结构：适配器的全部逻辑都收在 [`ZenohService`] 的 `impl` 块里——会话/端点声明
//! （[`ZenohService::declare`]）、事件转发（[`ZenohService::spawn_event_forwarder`]）、
//! 请求服务循环（[`ZenohService::serve_requests`]）与单条处理（[`ZenohService::handle_sample`]）；
//! [`ZenohService::run`] 只负责把它们串起来。

use prost::Message;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use zenoh::Session;
use zenoh::bytes::ZBytes;
use zenoh::handlers::FifoChannelHandler;
use zenoh::pubsub::{Publisher, Subscriber};
use zenoh::sample::Sample;

use super::{convert, dispatch};
use crate::proto::v1;
use crate::service::{ServiceEvent, ServiceHandle};

/// 请求订阅者类型（`declare_subscriber` 的默认 handler 是 FIFO 通道）。
type RequestSubscriber = Subscriber<FifoChannelHandler<Sample>>;
/// 发布者类型。keyexpr 由 owned `String` 构造，因此生命周期是 `'static`。
type PublisherHandle = Publisher<'static>;

/// zenoh 适配器：持有服务句柄、keyexpr 前缀与事件广播源。
pub struct ZenohService {
    handle: ServiceHandle,
    prefix: String,
    events: broadcast::Sender<ServiceEvent>,
}

impl ZenohService {
    /// 构造 zenoh 适配器。
    pub fn new(
        handle: ServiceHandle,
        prefix: String,
        events: broadcast::Sender<ServiceEvent>,
    ) -> Self {
        Self {
            handle,
            prefix,
            events,
        }
    }

    /// 启动适配器：开会话、声明端点、起事件转发、跑请求循环；返回时订阅已结束。
    pub async fn run(self) -> anyhow::Result<()> {
        let session = Self::open_session().await?;
        let (subscriber, responder, event_publisher) = self.declare(&session).await?;
        let event_task = self.spawn_event_forwarder(event_publisher);

        tracing::info!(
            req_key = %self.request_key(),
            resp_key = %self.response_key(),
            event_key = %self.event_key(),
            "zenoh 适配器已启动"
        );

        self.serve_requests(&subscriber, &responder).await;

        event_task.abort();
        Ok(())
    }

    /// `<prefix>/req`。
    fn request_key(&self) -> String {
        format!("{}/req", self.prefix)
    }

    /// `<prefix>/resp`。
    fn response_key(&self) -> String {
        format!("{}/resp", self.prefix)
    }

    /// `<prefix>/event`。
    fn event_key(&self) -> String {
        format!("{}/event", self.prefix)
    }

    /// 打开默认配置的 zenoh 会话。
    async fn open_session() -> anyhow::Result<Session> {
        zenoh::open(zenoh::Config::default())
            .await
            .map_err(to_anyhow)
    }

    /// 声明请求订阅者与两个发布者（响应、事件）。
    async fn declare(
        &self,
        session: &Session,
    ) -> anyhow::Result<(RequestSubscriber, PublisherHandle, PublisherHandle)> {
        let subscriber = session
            .declare_subscriber(self.request_key())
            .await
            .map_err(to_anyhow)?;
        let responder = session
            .declare_publisher(self.response_key())
            .await
            .map_err(to_anyhow)?;
        let event_publisher = session
            .declare_publisher(self.event_key())
            .await
            .map_err(to_anyhow)?;
        Ok((subscriber, responder, event_publisher))
    }

    /// 事件转发任务：把内部广播转成 zenoh 事件发布。
    ///
    /// 只在广播通道关闭（`Closed`）时退出；落后（`Lagged`）只跳过并让出一次调度，
    /// 避免生产快于消费时紧循环空转。
    fn spawn_event_forwarder(&self, publisher: PublisherHandle) -> JoinHandle<()> {
        let mut events = self.events.subscribe();
        tokio::spawn(async move {
            loop {
                match events.recv().await {
                    Ok(event) => {
                        let bytes = convert::event(event).encode_to_vec();
                        if let Err(error) = publisher.put(ZBytes::from(bytes)).await {
                            tracing::warn!(%error, "事件发布失败");
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "事件订阅落后，已跳过部分事件");
                        tokio::task::yield_now().await;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        })
    }

    /// 请求服务循环：订阅结束（`Err`）即退出，避免围绕失效会话空转。
    async fn serve_requests(&self, subscriber: &RequestSubscriber, responder: &PublisherHandle) {
        loop {
            let sample = match subscriber.recv_async().await {
                Ok(sample) => sample,
                Err(error) => {
                    tracing::warn!(%error, "zenoh 订阅结束");
                    break;
                }
            };
            if let Some(encoded) = self.handle_sample(&sample).await
                && let Err(error) = responder.put(ZBytes::from(encoded)).await
            {
                tracing::warn!(%error, "响应发布失败");
            }
        }
    }

    /// 处理一条请求样本：解码 -> 分派 -> 编码。
    ///
    /// 解码失败只丢弃该样本并返回 `None`，不影响循环继续。
    async fn handle_sample(&self, sample: &Sample) -> Option<Vec<u8>> {
        let payload = sample.payload().to_bytes();
        let bytes: &[u8] = &payload;
        let request = match v1::Request::decode(bytes) {
            Ok(request) => request,
            Err(error) => {
                tracing::warn!(%error, "请求解码失败，已丢弃");
                return None;
            }
        };
        Some(
            dispatch::dispatch(&self.handle, request)
                .await
                .encode_to_vec(),
        )
    }
}

/// zenoh 的错误是 `Box<dyn Error + Send + Sync>`，不能直接经 `?` 转成 `anyhow::Error`，
/// 统一折叠为其展示文本。
fn to_anyhow(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}
