//! 运行时装配与生命周期：把 Config 组装成可运行的服务并安排优雅退出顺序。
//!
//! 对外通信仅使用 zenoh 订阅/发布；启动顺序：加载快照 → 构建 ServiceCore →
//! 启动 zenoh 与后台任务 → 等待退出信号 → 优雅退出。
//!
//! 优雅退出顺序（修复"先取快照、后停入口"的丢数据窗口）：
//! 1) 停后台任务（不再有周期持久化，消除与退出兜底持久化的并发）；
//! 2) 下线 zenoh（静默入口，不再接受新的请求）；
//! 3) 退出兜底持久化；失败时返回 `Err`（由 main 转为非零退出码）。
//!
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
        server.shutdown().await
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

    /// 优雅退出：停后台 → 静默入口（下线 zenoh）→ 退出兜底持久化。
    ///
    /// 返回最终持久化的结果：失败会向上传播为非零退出码，避免"数据已丢却假装干净退出"。
    pub async fn shutdown(mut self) -> Result<()> {
        // 1) 停后台任务：此后不再有周期持久化，消除与退出兜底的并发。
        if let Some(bg) = self.background.take() {
            bg.shutdown().await;
        }

        // 2) 静默入口：下线 zenoh（中止订阅任务 + 关闭 session），不再接受新请求。
        #[cfg(feature = "zenoh")]
        if let Some(zenoh) = self.zenoh.take() {
            zenoh.shutdown().await;
        }

        // 3) 退出兜底持久化；失败上报。
        let persist_result = self.service.persist().await;
        match &persist_result {
            Ok(()) => log::info!("soul-mem server shutdown complete"),
            Err(e) => log::error!("final persist failed: {e}"),
        }
        persist_result
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
