//! 运行时装配与生命周期：把 Config 组装成可运行的服务并安排优雅退出顺序。
//!
//! 对外通信仅使用 zenoh 订阅/发布；启动顺序：加载快照 → 构建 ServiceCore →
//! 启动 zenoh 与后台任务 → 等待退出信号 →「持久化 → 停后台 → 下线 zenoh」优雅退出。
//! `main.rs` 只负责薄入口。

use crate::background::{BackgroundRuntime, run_background};
use crate::config::Config;
use crate::error::Result;
use crate::service::SoulMemService;
use crate::store::Store;

#[cfg(feature = "zenoh")]
use crate::zenoh::ZenohRuntime;

/// 运行中的服务句柄（供 main 使用；离线测试直接驱动 `SoulMemService`）。
pub struct RunningServer {
    pub service: SoulMemService,
    #[cfg(feature = "zenoh")]
    pub zenoh: Option<ZenohRuntime>,
    pub background: Option<BackgroundRuntime>,
}

impl RunningServer {
    /// 启动服务并阻塞等待 Ctrl-C，随后优雅退出。
    pub async fn run(config: Config) -> Result<()> {
        let server = Self::start(config).await?;
        wait_for_shutdown_signal().await;
        server.shutdown().await;
        Ok(())
    }

    /// 启动全部组件并返回运行句柄（不阻塞）。
    pub async fn start(config: Config) -> Result<Self> {
        config.validate()?;
        let device_id = config.device_id.clone();

        let store = Store::from_config(config.store_path.clone());
        let loaded = store.load().await?;
        let service = SoulMemService::from_config(&config, store, loaded).await?;

        let background = Some(run_background(service.clone(), &config));

        let mut server = RunningServer {
            service: service.clone(),
            background,
            #[cfg(feature = "zenoh")]
            zenoh: None,
        };

        #[cfg(feature = "zenoh")]
        {
            server.zenoh =
                Some(ZenohRuntime::start(service.clone(), config.zenoh_key_prefix.clone()).await?);
            log::info!("zenoh channel ready (device_id={device_id})");
        }

        Ok(server)
    }

    /// 优雅退出：先持久化，再停止后台任务与 zenoh。
    pub async fn shutdown(mut self) {
        // 1) 退出前兜底持久化。
        if let Err(e) = self.service.persist().await {
            log::error!("final persist failed: {e}");
        }

        // 2) 停后台任务。
        if let Some(bg) = self.background.take() {
            bg.shutdown().await;
        }

        // 3) 下线 zenoh（含 liveliness token）。
        #[cfg(feature = "zenoh")]
        if let Some(zenoh) = self.zenoh.take() {
            zenoh.shutdown().await;
        }

        log::info!("soul-mem server shutdown complete");
    }
}

/// 等待 Ctrl-C / SIGTERM 的跨平台实现。
async fn wait_for_shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
