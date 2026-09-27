//! 一次语义调用的观测上下文，通过 tokio task-local 传播。
//!
//! 为什么需要它：[`crate::ChatBackend`] 的方法签名里没有、也不该有观测参数
//! （那会把观测细节塞进调用契约）。但传输层的 tower 重试策略必须能上报"发生了一次重试、
//! 等了多久、原因是什么"，并必须把 `Retry-After` 关联到**这一次**调用上。
//!
//! 评估过的替代方案：
//! - 把观测器注入传输层并让它发不带 `call_id` 的事件 —— 并发下无法与调用配对；
//! - 把重试全部挪到引擎层 —— 会丢掉 `Retry-After`（引擎拿不到响应头），
//!   也会把"读 body 之前就能完成的廉价重试"变成整次重发。
//!
//! task-local 是唯一能在**不污染 trait 签名**的前提下把上下文送到传输层的机制：
//! 重试循环运行在同一个 task 内，不跨越 spawn 边界。
//!
//! # 谁写、谁读
//!
//! | 角色 | 位置 | 动作 |
//! |---|---|---|
//! | 写（进入作用域） | `engine.rs` | [`scoped`]：把 `CallCtx` 挂到当前 task 上，覆盖整次调用 |
//! | 写（计数） | `oai_comp/transport.rs` 的重试策略 | [`CallCtx::note_inner_retry`]、[`CallCtx::note_retry_after`] |
//! | 写（计数） | `engine.rs` 的整调用循环 | [`CallCtx::note_whole_retry`] |
//! | 读 | `engine.rs`（收尾时补 `retry_after`、读重试次数） | [`CallCtx::last_retry_after`] 等 |
//!
//! **限制**：只在同一 task 内可见。流式场景下 async-openai 会把 SSE 读取放到 spawn
//! 出来的任务里，所以逐 chunk 的处理不依赖本模块（见 `oai_comp/backend.rs` 的 `ChunkState`）。

use crate::error::LlmErrorKind;
use crate::observer::{LlmObserver, RetryEvent, RetryLayer};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tokio::task_local;

task_local! {
    static CURRENT: Arc<CallCtx>;
}

pub(crate) struct CallCtx {
    pub call_id: String,
    pub backend: String,
    pub observer: Arc<dyn LlmObserver>,
    started: Instant,
    inner_retries: AtomicU32,
    whole_retries: AtomicU32,
    last_retry_after: parking_lot::Mutex<Option<Duration>>,
}

impl CallCtx {
    pub fn new(call_id: String, backend: String, observer: Arc<dyn LlmObserver>) -> Self {
        Self {
            call_id,
            backend,
            observer,
            started: Instant::now(),
            inner_retries: AtomicU32::new(0),
            whole_retries: AtomicU32::new(0),
            last_retry_after: parking_lot::Mutex::new(None),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// 传输层决定重试时调用：计数 + 上报（传输层重试的唯一上报点）。
    pub fn note_inner_retry(&self, attempt: u32, delay: Duration, kind: LlmErrorKind) {
        self.inner_retries.fetch_add(1, Ordering::Relaxed);
        self.observer.on_retry(&RetryEvent {
            call_id: &self.call_id,
            layer: RetryLayer::Transport,
            attempt,
            delay_ms: delay.as_millis() as u64,
            kind,
        });
    }

    /// 引擎决定重发整次调用时调用。
    pub fn note_whole_retry(&self, attempt: u32, delay: Duration, kind: LlmErrorKind) {
        self.whole_retries.fetch_add(1, Ordering::Relaxed);
        self.observer.on_retry(&RetryEvent {
            call_id: &self.call_id,
            layer: RetryLayer::WholeCall,
            attempt,
            delay_ms: delay.as_millis() as u64,
            kind,
        });
    }

    pub fn inner_retries(&self) -> u32 {
        self.inner_retries.load(Ordering::Relaxed)
    }

    pub fn whole_retries(&self) -> u32 {
        self.whole_retries.load(Ordering::Relaxed)
    }

    /// 记录服务端在响应头里给出的建议等待时长。
    ///
    /// 传输层读到它就马上用掉；但重试预算耗尽后向上抛出的错误已经拿不到响应头了
    /// （`ApiErrorResponse` 只带 status 与错误体），所以在耗尽时由引擎把它补回错误上。
    pub fn note_retry_after(&self, retry_after: Duration) {
        let mut guard = self.last_retry_after.lock();
        if guard.is_none() {
            *guard = Some(retry_after);
        }
    }

    pub fn last_retry_after(&self) -> Option<Duration> {
        *self.last_retry_after.lock()
    }
}

/// 取出当前 task 的调用上下文。传输层在 tower 服务里调用它。
pub(crate) fn current() -> Option<Arc<CallCtx>> {
    CURRENT.try_with(|ctx| ctx.clone()).ok()
}

/// 在给定上下文里执行 `future`。
pub(crate) async fn scoped<F: Future>(ctx: Arc<CallCtx>, future: F) -> F::Output {
    CURRENT.scope(ctx, future).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::NoopObserver;

    fn ctx() -> Arc<CallCtx> {
        Arc::new(CallCtx::new(
            "call-1".into(),
            "mock".into(),
            Arc::new(NoopObserver),
        ))
    }

    #[tokio::test]
    async fn context_is_visible_inside_scope_only() {
        assert!(current().is_none(), "scope 外不应有上下文");

        let c = ctx();
        let seen = scoped(c.clone(), async { current().map(|c| c.call_id.clone()) }).await;
        assert_eq!(seen.as_deref(), Some("call-1"));

        assert!(current().is_none(), "scope 结束后应恢复为空");
    }

    #[tokio::test]
    async fn retries_are_counted_per_layer() {
        let c = ctx();
        assert_eq!(c.inner_retries(), 0);
        c.note_inner_retry(1, Duration::from_millis(100), LlmErrorKind::RateLimited);
        c.note_inner_retry(2, Duration::from_millis(200), LlmErrorKind::ServerError);
        assert_eq!(c.inner_retries(), 2);
        assert_eq!(c.whole_retries(), 0);

        c.note_whole_retry(1, Duration::from_millis(50), LlmErrorKind::Timeout);
        assert_eq!(c.whole_retries(), 1);
    }

    #[tokio::test]
    async fn first_retry_after_wins() {
        let c = ctx();
        assert_eq!(c.last_retry_after(), None);
        c.note_retry_after(Duration::from_secs(2));
        c.note_retry_after(Duration::from_secs(9));
        assert_eq!(c.last_retry_after(), Some(Duration::from_secs(2)));
    }

    #[tokio::test]
    async fn elapsed_is_measured_from_construction() {
        let c = ctx();
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(c.elapsed_ms() >= 5);
    }
}
