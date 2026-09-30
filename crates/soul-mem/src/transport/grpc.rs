//! gRPC 适配器（tonic）。
//!
//! `Exchange` 与 zenoh 的 `<prefix>/req` + `<prefix>/resp` 等价；`Subscribe` 与
//! `<prefix>/event` 等价。消息均来自同一份 `proto/soulmem.proto`。

use std::pin::Pin;

use futures::Stream;
use tokio::sync::broadcast;
use tonic::{Request, Response, Status};

use super::{convert, dispatch};
use crate::proto::v1;
use crate::service::{ServiceEvent, ServiceHandle};

/// gRPC 服务实现，持有服务句柄与事件广播源。
pub struct GrpcService {
    handle: ServiceHandle,
    events: broadcast::Sender<ServiceEvent>,
}

impl GrpcService {
    /// 构造 gRPC 适配器。
    pub fn new(handle: ServiceHandle, events: broadcast::Sender<ServiceEvent>) -> Self {
        Self { handle, events }
    }
}

/// 事件流类型。
type SubscribeStream = Pin<Box<dyn Stream<Item = Result<v1::Event, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl v1::soul_mem_server::SoulMem for GrpcService {
    async fn exchange(&self, request: Request<v1::Request>) -> Result<Response<v1::Reply>, Status> {
        let reply = dispatch::dispatch(&self.handle, request.into_inner()).await;
        Ok(Response::new(reply))
    }

    type SubscribeStream = SubscribeStream;

    async fn subscribe(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let mut receiver = self.events.subscribe();
        let stream = async_stream::stream! {
            loop {
                match receiver.recv().await {
                    Ok(event) => yield Ok(convert::event(event)),
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "事件订阅落后，已跳过部分事件");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}
