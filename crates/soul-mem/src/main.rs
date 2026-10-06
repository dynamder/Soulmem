//! SoulMem 记忆服务入口：引导依赖、启动 zenoh / gRPC 适配器、处理优雅退出。

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use soul_mem::bootstrap::bootstrap;
use soul_mem::config::Config;
use soul_mem::proto::v1;
use soul_mem::service::{ControlAction, ServiceEvent, ServiceHandle, spawn};
use soul_mem::transport::{GrpcService, ZenohService};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    init_tracing();

    let config = Arc::new(Config::load()?);
    tracing::info!(
        character = %config.character,
        db = %config.db_path.display(),
        "启动 SoulMem 服务"
    );

    let boot = bootstrap(Arc::clone(&config)).await?;
    let (handle, service_loop) = spawn(boot.service, 32);

    let heartbeat = spawn_heartbeat(
        handle.clone(),
        boot.events.clone(),
        config.heartbeat_interval_secs,
    );
    let maintenance = spawn_maintenance(handle.clone(), config.consolidate_interval_secs);

    let mut adapters: Vec<JoinHandle<()>> = Vec::new();

    if config.enable_grpc {
        let service = GrpcService::new(handle.clone(), boot.events.clone());
        let addr = config.grpc_addr;
        adapters.push(tokio::spawn(async move {
            tracing::info!(%addr, "gRPC 适配器监听中");
            if let Err(error) = tonic::transport::Server::builder()
                .add_service(v1::soul_mem_server::SoulMemServer::new(service))
                .serve(addr)
                .await
            {
                tracing::error!(%error, "gRPC 适配器退出");
            }
        }));
    }

    if config.enable_zenoh {
        let service = ZenohService::new(
            handle.clone(),
            config.key_prefix.clone(),
            boot.events.clone(),
        );
        adapters.push(tokio::spawn(async move {
            if let Err(error) = service.run().await {
                tracing::error!(%error, "zenoh 适配器退出");
            }
        }));
    }

    tokio::signal::ctrl_c().await?;
    tracing::info!("收到退出信号，持久化工作记忆");
    let _ = handle.control(vec![ControlAction::Persist]).await;

    // 先停掉所有持有服务句柄的任务，再等待服务循环把在途命令排空。
    heartbeat.abort();
    maintenance.abort();
    for adapter in &adapters {
        adapter.abort();
    }
    for adapter in adapters {
        let _ = adapter.await;
    }
    let _ = heartbeat.await;
    let _ = maintenance.await;
    drop(handle);

    match tokio::time::timeout(Duration::from_secs(5), service_loop).await {
        Ok(_) => tracing::info!("服务循环已正常结束"),
        Err(_) => tracing::warn!("服务循环未在超时内结束，强制退出"),
    }
    Ok(())
}

/// 周期发布心跳事件。
fn spawn_heartbeat(
    handle: ServiceHandle,
    events: broadcast::Sender<ServiceEvent>,
    secs: u64,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let period = Duration::from_secs(secs.max(1));
        // 用 interval_at 跳过 interval 的"首 tick 立即触发"，避免启动即发一次心跳。
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        loop {
            ticker.tick().await;
            if let Ok(state) = handle.snapshot().await {
                let _ = events.send(ServiceEvent::Heartbeat(state));
            }
        }
    })
}

/// 周期触发内部维护（静默转 Idle + 巩固）。
fn spawn_maintenance(handle: ServiceHandle, secs: u64) -> JoinHandle<()> {
    tokio::spawn(async move {
        let period = Duration::from_secs(secs.max(1));
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        loop {
            ticker.tick().await;
            if let Err(error) = handle.control(vec![ControlAction::Maintenance]).await {
                tracing::warn!(%error, "定时维护失败");
            }
        }
    })
}

/// 初始化 `tracing`（`RUST_LOG` 可覆盖，缺省 `info`）。
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}
