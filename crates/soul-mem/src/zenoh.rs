//! zenoh 通道：**唯一**的对外通信方式（订阅/发布）。
//!
//! - `keys`：Key Expression 集中定义；
//! - `liveliness`：服务发现/心跳；
//! - `request`：订阅请求主题、按 op 调用 service、发布应答（请求-应答用 pub/sub 模拟）；
//! - `pubsub`：订阅 ingest 主题（单向推式输入）；
//! - `client`：可复用的设备端客户端（供 mock-device 等联网演示/设备端复用）。

mod client;
mod keys;
mod liveliness;
mod pubsub;
mod request;

pub use client::ZenohClient;
pub use keys::Keys;
pub use liveliness::announce as announce_liveliness;

use crate::error::{Error, Result};
use crate::service::SoulMemService;
use zenoh::liveliness::LivelinessToken;

/// zenoh 侧运行句柄：持有 session 与 liveliness token，运行期不得 drop token。
pub struct ZenohRuntime {
    session: zenoh::Session,
    _token: LivelinessToken,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl ZenohRuntime {
    /// 建立会话并注册全部对外接口。
    pub async fn start(service: SoulMemService, prefix: impl Into<String>) -> Result<Self> {
        let keys = Keys::new(prefix);
        let config = zenoh::Config::default();
        let session = zenoh::open(config)
            .await
            .map_err(|e| Error::internal(format!("open zenoh session: {e}")))?;

        let device_id = service.device_id().to_string();
        let token = liveliness::announce(&session, &keys.liveliness_token(&device_id)).await?;

        let tasks = vec![
            request::spawn_request_subscriber(session.clone(), service.clone(), keys.clone()),
            pubsub::spawn_ingest_subscriber(session.clone(), service, keys.ingest()),
        ];

        log::info!(
            "zenoh runtime ready (device_id={device_id}, prefix={}, pub/sub only)",
            keys.prefix()
        );
        Ok(ZenohRuntime {
            session,
            _token: token,
            tasks,
        })
    }

    /// 优雅退出：中止对外任务、下线 liveliness（token 随 runtime 释放）。
    pub async fn shutdown(self) {
        for task in self.tasks {
            task.abort();
            let _ = task.await;
        }
        drop(self._token);
        let _ = self.session.close().await;
    }
}
