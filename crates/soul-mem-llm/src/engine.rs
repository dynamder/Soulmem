//! 语义调用的门面：**调用方唯一需要认识的入口**。
//!
//! 职责只有两件：编排观测生命周期、重发整次调用。编码/解码/传输都不在这里。
//!
//! # 数据流
//!
//! ```text
//! LlmEngine::complete(&Task)
//!   ├─ ctx::CallCtx::new + 进入 task-local     → ctx.rs
//!   ├─ observer.on_start(CallStart)            → observer.rs
//!   ├─ with_whole_call_retry{ backend.complete(task) }   ← 可能重发整次
//!   └─ observer.on_end(CallEnd)                → observer.rs
//! ```
//!
//! `stream` 的区别只在最后一步：它返回私有的 `ObservedStream` 包装，把"结束上报"推迟到
//! 流终止（正常结束 / 出错 / 被消费方提前丢弃），保证**恰好一次**。
//!
//! # 为什么外层还需要一次重试
//!
//! 传输层（tower）只看得到"发出请求 → 拿到响应头"这一段。async-openai 把**响应体读取**
//! 放在那之上（`Client::execute_response` 拿到 `Response` 后由 `read_response` 读 body），
//! 因此下面这些失败传输层根本看不到：
//!
//! - 后端的整体超时与首字节超时（后端用 `tokio::time::timeout` 包在响应体读取之外）；
//! - 读到一半连接被重置；
//! - 流在产出任何内容之前就断了。
//!
//! 这些必须由本层重发整次调用。两层预算独立且有界：
//! 单次语义调用的最大请求数 = `(1 + 传输层重试) × (1 + 整调用重试)`。
//! 完整链路见 `docs/architecture/llm-layer.md`。

use crate::backend::{BackendInfo, ChatBackend, Completion, EventStream, StreamEvent, Task, Usage};
use crate::ctx::{self, CallCtx};
use crate::error::{LlmError, LlmErrorKind};
use crate::observer::{CallEnd, CallStart, LlmObserver, NoopObserver};
use futures::Stream;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

/// 外层重试的退避基值与上限。外层重发代价高于内层（要重新生成），
/// 因此预算默认只有 1 次，退避也不必可配。抖动在此不必要：重试次数少、退避起点已足够分散。
const WHOLE_CALL_BASE_DELAY: Duration = Duration::from_millis(500);
const WHOLE_CALL_MAX_DELAY: Duration = Duration::from_secs(8);

/// 进程内单调递增的调用序号，保证 trace 里的 `call_id` 唯一。
static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

/// 一次语义调用的对外入口。
pub struct LlmEngine {
    backend: Arc<dyn ChatBackend>,
    observer: Arc<dyn LlmObserver>,
    whole_call_retries: u32,
}

impl LlmEngine {
    /// 默认：不观测、整调用重试 1 次。
    pub fn new(backend: Arc<dyn ChatBackend>) -> Self {
        Self {
            backend,
            observer: Arc::new(NoopObserver),
            whole_call_retries: 1,
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn LlmObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// 整调用重试次数（不含首次）。0 表示"超时/流中断不重发"。
    pub fn with_whole_call_retries(mut self, retries: u32) -> Self {
        self.whole_call_retries = retries;
        self
    }

    /// 一次性完成。**这是调用方最常用的入口。**
    ///
    /// 流程：`begin`（建上下文 + `on_start`）→ 整调用重试循环里调 `backend.complete`
    /// → `emit_end`（`on_end`，含耗时/token/重试次数/错误类别）。
    ///
    /// 编码、传输、解码都在 `backend` 内部（如 `oai_comp::OaiCompatBackend`），本函数不碰。
    pub async fn complete(&self, task: Task) -> Result<Completion, LlmError> {
        let call = self.begin(&task, false);
        let backend = self.backend.clone();
        let outcome = {
            let task = task.clone();
            self.with_whole_call_retry(&call, || backend.complete(task.clone()))
                .await
        };
        match outcome {
            Ok(completion) => {
                self.emit_end(
                    &call,
                    EndParts {
                        ok: true,
                        kind: None,
                        usage: completion.usage,
                        stop: Some(completion.stop),
                        text_chars: completion.text.chars().count(),
                    },
                );
                Ok(completion)
            }
            Err(err) => {
                self.emit_end(&call, EndParts::failure(&err));
                Err(err)
            }
        }
    }

    /// 流式完成。
    ///
    /// 只有"开流"这一步会走整调用重试（此时还没有任何内容产出，重发是安全的）；
    /// 一旦拿到流，**流内错误不再重试**——已产出的文本重放会造成重复内容。
    /// 消费方需要自己检查 [`LlmError::partial`] 决定丢弃还是保留半截结果。
    ///
    /// 结束事件由 `ObservedStream`（本文件私有）在流终止时上报（含被提前丢弃的情形）。
    pub async fn stream(&self, task: Task) -> Result<EventStream, LlmError> {
        let call = self.begin(&task, true);
        let backend = self.backend.clone();
        let opened = {
            let task = task.clone();
            self.with_whole_call_retry(&call, || backend.stream(task.clone()))
                .await
        };
        match opened {
            Ok(stream) => Ok(Box::pin(ObservedStream::new(
                stream,
                call,
                self.observer.clone(),
            ))),
            Err(err) => {
                self.emit_end(&call, EndParts::failure(&err));
                Err(err)
            }
        }
    }

    fn begin(&self, task: &Task, streaming: bool) -> Arc<CallCtx> {
        let info: BackendInfo = self.backend.info();
        let seq = CALL_SEQ.fetch_add(1, Ordering::Relaxed);
        let call = Arc::new(CallCtx::new(
            format!("{}-{seq}", info.name),
            info.name.clone(),
            self.observer.clone(),
        ));
        self.observer.on_start(&CallStart {
            call_id: &call.call_id,
            backend: &info.name,
            model: info.model.as_deref(),
            streaming,
        });
        let _ = task;
        call
    }

    /// 整调用重试循环。只有 [`LlmError::is_retryable`] 为真才重发。
    ///
    /// 为什么需要它：传输层（tower）只看得到"发请求 → 拿到响应头"，而超时、读到一半断连、
    /// 流未产出即中断都发生在**响应体读取阶段**（在 tower 之上），传输层看不到。
    ///
    /// 重试次数与等待时长都会上报（`ctx::CallCtx::note_whole_retry`），因此 trace 里
    /// 能区分"内层重试"与"整调用重发"。
    async fn with_whole_call_retry<T, F, Fut>(
        &self,
        call: &Arc<CallCtx>,
        mut op: F,
    ) -> Result<T, LlmError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LlmError>>,
    {
        let mut used = 0u32;
        loop {
            match ctx::scoped(call.clone(), op()).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if used >= self.whole_call_retries || !err.is_retryable() {
                        return Err(self.enrich(err, call));
                    }
                    used += 1;
                    let delay = whole_call_delay(used);
                    call.note_whole_retry(used, delay, err.kind());
                    tracing::warn!(
                        call_id = %call.call_id,
                        attempt = used,
                        kind = err.kind().as_str(),
                        delay_ms = delay.as_millis() as u64,
                        "重发整次调用"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// 补齐错误上缺失的信息。
    ///
    /// 传输层读到 `Retry-After` 时记录了它，但重试预算耗尽后向上抛的错误已经拿不到响应头
    /// （`ApiErrorResponse` 只带 status 与错误体），这里把它补回去，让调用方能正确退避。
    fn enrich(&self, err: LlmError, call: &Arc<CallCtx>) -> LlmError {
        let mut err = err;
        if err.kind() == LlmErrorKind::RateLimited
            && err.retry_after().is_none()
            && let Some(retry_after) = call.last_retry_after()
        {
            err.set_retry_after(retry_after);
        }
        if err.backend().is_none() {
            err = err.with_backend(call.backend.clone());
        }
        err
    }

    fn emit_end(&self, call: &Arc<CallCtx>, parts: EndParts) {
        self.observer.on_end(&CallEnd {
            call_id: &call.call_id,
            ok: parts.ok,
            kind: parts.kind,
            latency_ms: call.elapsed_ms(),
            usage: parts.usage,
            stop: parts.stop,
            text_chars: parts.text_chars,
            inner_retries: call.inner_retries(),
            whole_call_retries: call.whole_retries(),
        });
    }
}

/// 外层第 `attempt` 次重试的退避时长（纯函数，便于断言）。
fn whole_call_delay(attempt: u32) -> Duration {
    let step = attempt.saturating_sub(1).min(16);
    WHOLE_CALL_BASE_DELAY
        .saturating_mul(1u32 << step)
        .min(WHOLE_CALL_MAX_DELAY)
}

struct EndParts {
    ok: bool,
    kind: Option<LlmErrorKind>,
    usage: Option<Usage>,
    stop: Option<crate::backend::StopReason>,
    text_chars: usize,
}

impl EndParts {
    fn failure(err: &LlmError) -> Self {
        Self {
            ok: false,
            kind: Some(err.kind()),
            usage: None,
            stop: None,
            text_chars: 0,
        }
    }
}

/// 包装事件流，并保证**恰好一次**结束上报。
///
/// 被消费方提前丢弃时也要上报（[`LlmErrorKind::Cancelled`]）——否则 trace 里会留下
/// 一堆"开始了但没有结束"的记录，无法区分"还在跑"与"被扔了"。
///
/// "恰好一次"由三处共同保证：`Done` 事件（语义终点）、流结束/出错、以及 `Drop`；
/// `reported` 标志让后到者变成空操作。
struct ObservedStream {
    inner: EventStream,
    call: Arc<CallCtx>,
    observer: Arc<dyn LlmObserver>,
    reported: bool,
    text_chars: usize,
    completion: Option<Completion>,
}

impl ObservedStream {
    fn new(inner: EventStream, call: Arc<CallCtx>, observer: Arc<dyn LlmObserver>) -> Self {
        Self {
            inner,
            call,
            observer,
            reported: false,
            text_chars: 0,
            completion: None,
        }
    }

    fn observe(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Delta(text) => {
                self.text_chars += text.chars().count();
            }
            StreamEvent::ReasoningDelta(_) => {}
            StreamEvent::Done(completion) => {
                self.completion = Some((**completion).clone());
                // `Done` 就是流的语义终点：在这里收尾，消费方随后即使只是 drop 掉流，
                // 也不会被误判成"提前取消"。
                self.report(None);
            }
        }
    }

    fn report(&mut self, error: Option<&LlmError>) {
        if self.reported {
            return;
        }
        self.reported = true;

        let (ok, kind, usage, stop) = match error {
            Some(err) => (false, Some(err.kind()), None, None),
            None => (
                true,
                None,
                self.completion.as_ref().and_then(|c| c.usage),
                self.completion.as_ref().map(|c| c.stop),
            ),
        };

        self.observer.on_end(&CallEnd {
            call_id: &self.call.call_id,
            ok,
            kind,
            latency_ms: self.call.elapsed_ms(),
            usage,
            stop,
            text_chars: self.text_chars,
            inner_retries: self.call.inner_retries(),
            whole_call_retries: self.call.whole_retries(),
        });
    }
}

impl Stream for ObservedStream {
    type Item = Result<StreamEvent, LlmError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                this.report(None);
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(event))) => {
                this.observe(&event);
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(error))) => {
                this.report(Some(&error));
                Poll::Ready(Some(Err(error)))
            }
        }
    }
}

impl Drop for ObservedStream {
    fn drop(&mut self) {
        if !self.reported {
            let error = LlmError::new(LlmErrorKind::Cancelled, "事件流未被消费完即被丢弃");
            self.report(Some(&error));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Hints, Sampling};
    use crate::mock::{MockBackend, MockOutcome, drain};
    use crate::observer::{RecordingObserver, RetryLayer};
    use async_trait::async_trait;
    use std::sync::atomic::AtomicU32;

    fn retryable() -> LlmError {
        LlmError::new(LlmErrorKind::Timeout, "整次调用超时")
    }

    fn build(
        outcomes: Vec<MockOutcome>,
        observer: Arc<RecordingObserver>,
        whole_call_retries: u32,
    ) -> (LlmEngine, Arc<MockBackend>) {
        let backend = Arc::new(MockBackend::new(
            BackendInfo::new("mock")
                .with_model("mock-model")
                .with_streaming(true),
            outcomes,
        ));
        let engine = LlmEngine::new(backend.clone())
            .with_observer(observer)
            .with_whole_call_retries(whole_call_retries);
        (engine, backend)
    }

    fn task() -> Task {
        Task::system_user("系统", "用户")
            .with_sampling(Sampling::default().with_max_output_tokens(64))
            .with_hints(Hints::default().with_suppress_reasoning(true))
    }

    #[test]
    fn whole_call_backoff_grows_then_caps() {
        assert_eq!(whole_call_delay(1), Duration::from_millis(500));
        assert_eq!(whole_call_delay(2), Duration::from_secs(1));
        assert_eq!(whole_call_delay(3), Duration::from_secs(2));
        assert_eq!(whole_call_delay(50), WHOLE_CALL_MAX_DELAY, "必须被上限夹住");
    }

    #[tokio::test]
    async fn complete_reports_start_and_end_once() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, backend) = build(vec![MockOutcome::Text("结果".into())], observer.clone(), 1);

        let completion = engine.complete(task()).await.expect("应成功");
        assert_eq!(completion.text, "结果");
        assert_eq!(backend.call_count(), 1);

        let starts = observer.starts();
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].backend, "mock");
        assert_eq!(starts[0].model.as_deref(), Some("mock-model"));
        assert!(!starts[0].streaming);

        let ends = observer.ends();
        assert_eq!(ends.len(), 1);
        assert!(ends[0].ok);
        assert_eq!(ends[0].kind, None);
        assert_eq!(ends[0].stop, Some(crate::backend::StopReason::Completed));
        assert_eq!(ends[0].text_chars, 2);
        assert_eq!(ends[0].inner_retries, 0);
        assert_eq!(ends[0].whole_call_retries, 0);
    }

    /// 超时、流中断这类失败传输层看不到，必须由整调用重试补上。
    #[tokio::test]
    async fn whole_call_retry_reissues_after_retryable_failure() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, backend) = build(
            vec![
                MockOutcome::Fail(retryable()),
                MockOutcome::Text("第二次成功".into()),
            ],
            observer.clone(),
            1,
        );

        let completion = engine.complete(task()).await.expect("第二次应成功");
        assert_eq!(completion.text, "第二次成功");
        assert_eq!(backend.call_count(), 2, "必须真的重发");

        let retries = observer.retries();
        assert_eq!(retries.len(), 1);
        assert_eq!(retries[0].layer, RetryLayer::WholeCall);
        assert_eq!(retries[0].attempt, 1);
        assert_eq!(retries[0].kind, LlmErrorKind::Timeout);
        assert_eq!(retries[0].delay_ms, 500);

        let ends = observer.ends();
        assert_eq!(ends.len(), 1, "只上报一次结束");
        assert!(ends[0].ok);
        assert_eq!(ends[0].whole_call_retries, 1);
    }

    #[tokio::test]
    async fn non_retryable_failure_is_not_retried() {
        for kind in [
            LlmErrorKind::Auth,
            LlmErrorKind::BadRequest,
            LlmErrorKind::Quota,
            LlmErrorKind::EmptyCompletion,
            LlmErrorKind::Decode,
        ] {
            let observer = Arc::new(RecordingObserver::new());
            let (engine, backend) = build(
                vec![
                    MockOutcome::Fail(LlmError::new(kind, "永久失败")),
                    MockOutcome::Text("不该被用到".into()),
                ],
                observer.clone(),
                3,
            );

            let error = engine.complete(task()).await.expect_err("应失败");
            assert_eq!(error.kind(), kind);
            assert_eq!(backend.call_count(), 1, "{kind:?} 不该重试");
            assert_eq!(observer.retry_count(), 0);

            let ends = observer.ends();
            assert_eq!(ends.len(), 1);
            assert!(!ends[0].ok);
            assert_eq!(ends[0].kind, Some(kind));
        }
    }

    #[tokio::test]
    async fn whole_call_retry_budget_is_bounded() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, backend) = build(vec![MockOutcome::Fail(retryable())], observer.clone(), 0);

        let error = engine
            .complete(task())
            .await
            .expect_err("预算为 0 时不重发");
        assert!(error.is_retryable());
        assert_eq!(backend.call_count(), 1);
        assert_eq!(observer.retry_count(), 0);
    }

    #[tokio::test]
    async fn failure_carries_backend_name_even_if_backend_forgot_it() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, _) = build(
            vec![MockOutcome::Fail(LlmError::new(
                LlmErrorKind::Auth,
                "没有 backend 字段",
            ))],
            observer,
            0,
        );

        let error = engine.complete(task()).await.expect_err("应失败");
        assert_eq!(error.backend(), Some("mock"));
    }

    /// 传输层读到的 `Retry-After` 在预算耗尽后要能带回给调用方。
    #[tokio::test]
    async fn retry_after_is_recovered_from_the_call_context() {
        struct RateLimitedBackend {
            calls: AtomicU32,
        }

        #[async_trait]
        impl ChatBackend for RateLimitedBackend {
            fn info(&self) -> BackendInfo {
                BackendInfo::new("limited")
            }

            async fn complete(&self, _task: Task) -> Result<Completion, LlmError> {
                // 模拟传输层：在重试过程中读到了 Retry-After 并记进上下文
                if let Some(ctx) = crate::ctx::current() {
                    ctx.note_retry_after(Duration::from_secs(11));
                }
                self.calls.fetch_add(1, Ordering::Relaxed);
                Err(LlmError::new(LlmErrorKind::RateLimited, "429"))
            }

            async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
                Err(LlmError::unsupported("不支持流式"))
            }
        }

        let backend = Arc::new(RateLimitedBackend {
            calls: AtomicU32::new(0),
        });
        let engine = LlmEngine::new(backend).with_whole_call_retries(0);

        let error = engine.complete(task()).await.expect_err("应失败");
        assert_eq!(error.kind(), LlmErrorKind::RateLimited);
        assert_eq!(
            error.retry_after(),
            Some(Duration::from_secs(11)),
            "服务端建议的等待时长必须在预算耗尽后仍可见"
        );
    }

    #[tokio::test]
    async fn existing_retry_after_is_preserved() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, _) = build(
            vec![MockOutcome::Fail(
                LlmError::new(LlmErrorKind::RateLimited, "429")
                    .with_retry_after(Duration::from_secs(3)),
            )],
            observer,
            0,
        );
        let error = engine.complete(task()).await.expect_err("应失败");
        assert_eq!(error.retry_after(), Some(Duration::from_secs(3)));
    }

    #[tokio::test]
    async fn stream_reports_end_exactly_once() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, backend) = build(
            vec![MockOutcome::Stream(vec![
                StreamEvent::Delta("你".into()),
                StreamEvent::Delta("好".into()),
                StreamEvent::Done(Box::new(Completion::new("你好"))),
            ])],
            observer.clone(),
            0,
        );

        let stream = engine.stream(task()).await.expect("应能开流");
        let (events, error) = drain(stream).await;
        assert!(error.is_none());
        assert_eq!(events.len(), 3);
        assert_eq!(backend.call_count(), 1);

        assert!(observer.starts()[0].streaming, "开始事件必须标明是流式");

        let ends = observer.ends();
        assert_eq!(ends.len(), 1, "Done 之后 drop 不得再报一次");
        assert!(ends[0].ok);
        assert_eq!(ends[0].stop, Some(crate::backend::StopReason::Completed));
        assert_eq!(ends[0].text_chars, 2);
    }

    #[tokio::test]
    async fn stream_error_is_surfaced_and_reported() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, _) = build(
            vec![MockOutcome::StreamThenFail {
                events: vec![StreamEvent::Delta("半".into())],
                error: LlmError::stream_interrupted("连接被重置", Some("半".into())),
            }],
            observer.clone(),
            3,
        );

        let stream = engine.stream(task()).await.expect("应能开流");
        let (events, error) = drain(stream).await;

        assert_eq!(events.len(), 1);
        let error = error.expect("必须把错误交给调用方");
        assert_eq!(error.kind(), LlmErrorKind::StreamInterrupted);
        assert_eq!(error.partial(), Some("半"));

        let ends = observer.ends();
        assert_eq!(ends.len(), 1);
        assert!(!ends[0].ok);
        assert_eq!(ends[0].kind, Some(LlmErrorKind::StreamInterrupted));
        assert_eq!(ends[0].text_chars, 1);
    }

    /// 未消费完就丢弃的流也要留痕，否则 trace 里分不清"还在跑"与"被扔了"。
    #[tokio::test]
    async fn dropping_a_stream_early_reports_cancelled() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, _) = build(
            vec![MockOutcome::Stream(vec![
                StreamEvent::Delta("一".into()),
                StreamEvent::Delta("二".into()),
            ])],
            observer.clone(),
            0,
        );

        let mut stream = engine.stream(task()).await.expect("应能开流");
        let first = futures::StreamExt::next(&mut stream).await;
        assert!(matches!(first, Some(Ok(StreamEvent::Delta(_)))));
        drop(stream);

        let ends = observer.ends();
        assert_eq!(ends.len(), 1);
        assert!(!ends[0].ok);
        assert_eq!(ends[0].kind, Some(LlmErrorKind::Cancelled));
        assert_eq!(ends[0].text_chars, 1, "已消费的部分应被记下");
    }

    #[tokio::test]
    async fn stream_open_failure_is_retried_by_the_whole_call_layer() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, backend) = build(
            vec![
                MockOutcome::Fail(LlmError::new(LlmErrorKind::Timeout, "首字节超时")),
                MockOutcome::Stream(vec![
                    StreamEvent::Delta("ok".into()),
                    StreamEvent::Done(Box::new(Completion::new("ok"))),
                ]),
            ],
            observer.clone(),
            1,
        );

        let stream = engine.stream(task()).await.expect("第二次应能开流");
        let (events, error) = drain(stream).await;
        assert!(error.is_none());
        assert_eq!(events.len(), 2);
        assert_eq!(backend.call_count(), 2, "开流失败应重发");
        assert_eq!(observer.retries()[0].layer, RetryLayer::WholeCall);
    }

    #[tokio::test]
    async fn stream_failure_that_is_not_retryable_stops_immediately() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, backend) = build(
            vec![MockOutcome::Fail(LlmError::new(
                LlmErrorKind::Unsupported,
                "该后端不支持流式",
            ))],
            observer.clone(),
            3,
        );

        // EventStream 不是 Debug，不能用 expect_err
        let error = match engine.stream(task()).await {
            Ok(_) => panic!("不支持流式的后端必须报错"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), LlmErrorKind::Unsupported);
        assert_eq!(backend.call_count(), 1);
        assert_eq!(observer.retry_count(), 0);
        assert_eq!(observer.ends().len(), 1);
    }

    #[tokio::test]
    async fn reasoning_is_carried_through_without_faking_it() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, _) = build(
            vec![MockOutcome::Reasoned {
                reasoning: "想一下".into(),
                text: "答案".into(),
            }],
            observer,
            0,
        );

        let completion = engine.complete(task()).await.expect("应成功");
        assert_eq!(completion.text, "答案");
        assert_eq!(completion.reasoning.as_deref(), Some("想一下"));
    }

    #[tokio::test]
    async fn call_ids_are_unique_and_prefixed_by_backend() {
        let observer = Arc::new(RecordingObserver::new());
        let (engine, _) = build(
            vec![MockOutcome::Text("a".into()), MockOutcome::Text("b".into())],
            observer.clone(),
            0,
        );

        engine.complete(task()).await.expect("应成功");
        engine.complete(task()).await.expect("应成功");

        let starts = observer.starts();
        assert!(
            starts[0].call_id.starts_with("mock-"),
            "{}",
            starts[0].call_id
        );
        assert_ne!(starts[0].call_id, starts[1].call_id, "call_id 必须唯一");
    }
}
