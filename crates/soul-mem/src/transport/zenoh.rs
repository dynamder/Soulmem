//! zenoh 适配器：请求走 query/reply，事件走 pub/sub。
//!
//! 两条 keyexpr：
//! - `<prefix>/req`：声明 queryable，接收客户端 `get` 请求并直接 reply。
//! - `<prefix>/event`：发布服务主动事件（心跳、巩固完成）。
//!
//! query/reply 自带应答语义：响应只回到发起 query 的客户端，不再需要公共的
//! `<prefix>/resp`，也无需按 `request_id` 过滤。事件来源是服务内部广播通道。
//!
//! 结构：适配器的全部逻辑都收在 [`ZenohService`] 的 `impl` 块里——会话/端点声明
//! （[`ZenohService::declare`]）、事件转发（[`ZenohService::spawn_event_forwarder`]）、
//! 请求服务循环（[`ZenohService::serve_queries`]）与单条处理（[`ZenohService::handle_query`]）；
//! [`ZenohService::run`] 只负责把它们串起来。

use prost::Message;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use zenoh::Session;
use zenoh::bytes::ZBytes;
use zenoh::handlers::FifoChannelHandler;
use zenoh::pubsub::Publisher;
use zenoh::query::{Query, Queryable};

use super::{convert, dispatch};
use crate::proto::v1;
use crate::service::{ServiceEvent, ServiceHandle};

/// 请求 queryable 类型（`declare_queryable` 的默认 handler 是 FIFO 通道）。
type RequestQueryable = Queryable<FifoChannelHandler<Query>>;
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

    /// 启动适配器：开会话、声明端点、起事件转发、跑请求循环；返回时 queryable 已结束。
    pub async fn run(self) -> anyhow::Result<()> {
        let session = Self::open_session().await?;
        let (queryable, event_publisher) = self.declare(&session).await?;
        let event_task = self.spawn_event_forwarder(event_publisher);

        tracing::info!(
            req_key = %self.request_key(),
            event_key = %self.event_key(),
            "zenoh 适配器已启动"
        );

        self.serve_queries(&queryable).await;

        event_task.abort();
        Ok(())
    }

    /// `<prefix>/req`。
    fn request_key(&self) -> String {
        format!("{}/req", self.prefix)
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

    /// 声明请求 queryable 与事件发布者。
    async fn declare(
        &self,
        session: &Session,
    ) -> anyhow::Result<(RequestQueryable, PublisherHandle)> {
        let queryable = session
            .declare_queryable(self.request_key())
            .await
            .map_err(to_anyhow)?;
        let event_publisher = session
            .declare_publisher(self.event_key())
            .await
            .map_err(to_anyhow)?;
        Ok((queryable, event_publisher))
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

    /// 请求服务循环：queryable 结束（`Err`）即退出，避免围绕失效会话空转。
    async fn serve_queries(&self, queryable: &RequestQueryable) {
        loop {
            let query = match queryable.recv_async().await {
                Ok(query) => query,
                Err(error) => {
                    tracing::warn!(%error, "zenoh queryable 结束");
                    break;
                }
            };
            if let Some(encoded) = self.handle_query(&query).await
                && let Err(error) = query.reply(self.request_key(), ZBytes::from(encoded)).await
            {
                tracing::warn!(%error, "响应发送失败");
            }
        }
    }

    /// 处理一条 query：解码 payload -> 分派 -> 编码。
    ///
    /// 解码失败只丢弃该 query 并返回 `None`，不影响循环继续。
    async fn handle_query(&self, query: &Query) -> Option<Vec<u8>> {
        let Some(payload) = query.payload() else {
            tracing::warn!("请求缺少 payload，已丢弃");
            return None;
        };
        let payload = payload.to_bytes();
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
