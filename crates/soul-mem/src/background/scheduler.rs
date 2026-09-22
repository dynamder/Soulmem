//! 后台任务调度器：按配置间隔周期触发各任务，任务之间失败隔离；
//! 通过停止信号支持优雅退出（任务最多在一个周期内退出）。
//!
//! 任务实现直接内联于此（persist 真实可用；consolidate/forget 为占位，
//! 经 Idle 门控后调用 `service.control`，返回 `Unimplemented`）。

use crate::config::Config;
use crate::error::Result;
use crate::service::SoulMemService;
use crate::wire::pb;
use std::time::Duration;
use tokio::sync::watch;

/// 调度任务类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    Persist,
    Consolidate,
    Forget,
}

/// 后台运行句柄：停止发送 + 已生成的任务。
pub struct BackgroundRuntime {
    stop: watch::Sender<bool>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl BackgroundRuntime {
    /// 请求全部任务停止并等待其退出。
    pub async fn shutdown(self) {
        let _ = self.stop.send(true);
        for task in self.tasks {
            let _ = task.await;
        }
    }
}

/// 依据配置启动后台任务（interval 为 0 表示停用该任务）。
pub fn run_background(service: SoulMemService, config: &Config) -> BackgroundRuntime {
    let (stop, _) = watch::channel(false);
    let mut tasks = Vec::new();

    if let Some(handle) = spawn_task(
        service.clone(),
        config.persist_interval_secs,
        stop.subscribe(),
        TaskKind::Persist,
    ) {
        tasks.push(handle);
    }
    if let Some(handle) = spawn_task(
        service.clone(),
        config.consolidate_interval_secs,
        stop.subscribe(),
        TaskKind::Consolidate,
    ) {
        tasks.push(handle);
    }
    if let Some(handle) = spawn_task(
        service.clone(),
        config.forget_interval_secs,
        stop.subscribe(),
        TaskKind::Forget,
    ) {
        tasks.push(handle);
    }

    BackgroundRuntime { stop, tasks }
}

fn spawn_task(
    service: SoulMemService,
    interval_secs: u64,
    mut stop: watch::Receiver<bool>,
    kind: TaskKind,
) -> Option<tokio::task::JoinHandle<()>> {
    if interval_secs == 0 {
        return None;
    }
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    Some(tokio::spawn(async move {
        let _ = ticker.tick().await; // 消费“立即触发”的第一次，首个任务在完整周期后运行
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    match run_once(kind, &service).await {
                        Ok(()) => log::debug!("background task {:?} done", kind),
                        Err(e) => log::warn!("background task {:?} failed (isolated): {}", kind, e),
                    }
                }
                _ = stop.changed() => break,
            }
        }
    }))
}

/// 单次执行（错误隔离由调用方处理）。
async fn run_once(kind: TaskKind, service: &SoulMemService) -> Result<()> {
    match kind {
        TaskKind::Persist => service.persist().await,
        TaskKind::Consolidate => {
            // Idle 门控：仅空闲时允许巩固。
            if !service.is_idle().await {
                return Ok(());
            }
            service
                .control(pb::Control {
                    kind: pb::ControlKind::ControlConsolidate as i32,
                })
                .await
                .map(|_| ())
        }
        TaskKind::Forget => {
            if !service.is_idle().await {
                return Ok(());
            }
            service
                .control(pb::Control {
                    kind: pb::ControlKind::ControlForget as i32,
                })
                .await
                .map(|_| ())
        }
    }
}
