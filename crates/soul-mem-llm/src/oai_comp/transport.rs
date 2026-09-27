//! 传输层：tower 执行栈 + 重试策略。
//!
//! # 本模块在数据流中的位置
//!
//! ```text
//! OaiCompatBackend::new ──► build_http_client（构造 reqwest::Client：connect_timeout / no_proxy）
//!                       └─► build_client（装栈）──► Client::build(http, provider).with_http_service(stack)
//!                                                      │  这个 Client 交给 oai_comp/backend.rs 使用
//! async-openai 的 executor ──► service.oneshot(factory) ──► ★ 本模块的 RetryLayer ★
//!                                                              └─► ReqwestService::call
//!                                                                   ★ HTTP 请求在此发出 ★
//! ```
//!
//! 也就是说：**我们提供栈，async-openai 负责把请求推进它的最内层。**
//! 本模块自己不发请求，也不解析响应体——它只决定"要不要重试、等多久"。
//!
//! # 为什么自带策略而不是直接用 `OpenAIRetryLayer`
//!
//! async-openai 0.41.3 自带的 `OpenAIRetryLayer` 已覆盖 429 / 5xx / 连接失败并遵守
//! `Retry-After`，但缺**重试观测**（重试发生在 tower service 内部，调用方看不到次数，
//! 而 trace 必须记录重试）。`SimpleRetryPolicy` 是官方给出的"自备延迟策略、复用分类"
//! 的路径，本模块即照此实现：`Err` 分支直接调用上游的 `should_retry` 复用分类，
//! `Ok` 分支自己区分 429 与 5xx 以便给出准确的 kind。
//!
//! # 已知取舍
//!
//! `SimpleRetryPolicy` 式的策略看不到响应体，因此无法像 `OpenAIRetryLayer` 那样区分
//! "429 限流"与"429 配额耗尽"。代价是配额耗尽的 429 也会被重试到预算耗尽——这类响应
//! 失败极快且不产生 token 成本，换来的是准确的重试计数。若要精确区分，需要换回
//! `OpenAIRetryLayer` 并放弃重试观测，二者不可兼得。
//!
//! # 超时**不在**这里
//!
//! 本模块不装 tower 的 timeout 层：整体时限与流式首字节时限由后端用
//! `tokio::time::timeout` 包在响应体读取之外（因为传输层看不到 body 阶段）。
//! 详见 `engine.rs` 模块注释与 `docs/architecture/llm-layer.md`。

use super::config::OaiCompatConfig;
use crate::ctx;
use crate::error::{LlmError, LlmErrorKind};
use async_openai::Client;
use async_openai::config::Config;
use async_openai::error::OpenAIError;
use async_openai::middleware::retry::should_retry;
use async_openai::middleware::{HttpRequestFactory, ReqwestService};
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceBuilder;
use tower::retry::Policy;

/// 退避基值（第 1 次重试的等待）。
const RETRY_BASE_DELAY: Duration = Duration::from_millis(200);
/// 指数退避上限。
const RETRY_MAX_DELAY: Duration = Duration::from_secs(8);
/// 抖动比例：纯指数退避会让并发调用在同一时刻重试（thundering herd），把刚恢复的服务再打挂。
const RETRY_JITTER_RATIO: f64 = 0.5;
/// 尊重 `Retry-After` 时的等待上限，避免服务端给出离谱值把调用挂住。
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// 传输层重试的可重试原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetryCause {
    RateLimited,
    Server,
    Connect,
}

impl RetryCause {
    fn kind(self) -> LlmErrorKind {
        match self {
            Self::RateLimited => LlmErrorKind::RateLimited,
            Self::Server => LlmErrorKind::ServerError,
            Self::Connect => LlmErrorKind::Transport,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::RateLimited => "rate_limited",
            Self::Server => "server_error",
            Self::Connect => "connect",
        }
    }
}

/// 带抖动与观测的重试策略。tower 每次调用克隆一份，因此计数器天然是"每次调用"的。
#[derive(Clone)]
struct JitterRetryPolicy {
    max_retries: u32,
    used: u32,
}

impl JitterRetryPolicy {
    fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            used: 0,
        }
    }
}

impl Policy<HttpRequestFactory, reqwest::Response, OpenAIError> for JitterRetryPolicy {
    type Future = futures::future::BoxFuture<'static, ()>;

    /// tower 在每次尝试失败后调用本方法询问"要不要再试、等多久"。
    ///
    /// 返回 `Some(future)` 表示重试（future 完成即等待结束，tower 随后用 `clone_request`
    /// 重建请求再发）；返回 `None` 表示放弃，把最终结果交回上层。
    ///
    /// 副作用有两个，都在这里发生：读取并记录 `Retry-After`（`ctx::CallCtx::note_retry_after`）、
    /// 上报重试事件（`ctx::CallCtx::note_inner_retry` → observer）。
    fn retry(
        &mut self,
        _req: &mut HttpRequestFactory,
        result: &mut Result<reqwest::Response, OpenAIError>,
    ) -> Option<Self::Future> {
        let cause = retry_cause(result)?;
        if self.used >= self.max_retries {
            return None;
        }
        self.used += 1;

        let retry_after = response_retry_after(result);
        if let Some(ctx) = ctx::current()
            && let Some(retry_after) = retry_after
        {
            // 预算耗尽后向上抛的错误拿不到响应头了，先把服务端建议留在上下文里
            ctx.note_retry_after(retry_after);
        }

        let delay = next_delay(self.used, retry_after);

        if let Some(ctx) = ctx::current() {
            ctx.note_inner_retry(self.used, delay, cause.kind());
        }
        tracing::warn!(
            attempt = self.used,
            cause = cause.as_str(),
            delay_ms = delay.as_millis() as u64,
            "传输层重试"
        );

        Some(Box::pin(tokio::time::sleep(delay)))
    }

    fn clone_request(&mut self, req: &HttpRequestFactory) -> Option<HttpRequestFactory> {
        Some(req.clone())
    }
}

/// 第 `attempt` 次重试的等待时长（纯函数，便于精确断言）。
///
/// 服务端给了 `Retry-After` 就照它等（夹在上限内），否则指数退避 + 抖动。
fn next_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(retry_after) = retry_after {
        return retry_after.min(MAX_RETRY_AFTER);
    }
    let step = attempt.saturating_sub(1).min(16);
    let raw = RETRY_BASE_DELAY
        .saturating_mul(1u32 << step)
        .min(RETRY_MAX_DELAY);
    apply_jitter(raw, RETRY_JITTER_RATIO)
}

/// 抖动：在 `[1-ratio, 1+ratio)` 上均匀缩放时长。
fn apply_jitter(delay: Duration, ratio: f64) -> Duration {
    if !ratio.is_finite() || ratio <= 0.0 {
        return delay;
    }
    let factor = 1.0 + rand::rng().random_range(-ratio..ratio);
    if factor <= 0.0 {
        return Duration::ZERO;
    }
    delay.mul_f64(factor)
}

/// 判断本次结果是否值得重试，并给出原因。
fn retry_cause(result: &Result<reqwest::Response, OpenAIError>) -> Option<RetryCause> {
    match result {
        Ok(response) => {
            let status = response.status();
            if status.as_u16() == 429 {
                Some(RetryCause::RateLimited)
            } else if status.is_server_error() {
                Some(RetryCause::Server)
            } else {
                None
            }
        }
        // 复用上游分类（其 Err 分支目前只认 connect 类错误），上游将来放宽范围时我们自动跟随
        Err(_) if should_retry(result) => Some(RetryCause::Connect),
        Err(_) => None,
    }
}

/// 从响应头读取服务端建议的等待时长。
fn response_retry_after(result: &Result<reqwest::Response, OpenAIError>) -> Option<Duration> {
    let response = result.as_ref().ok()?;
    let raw = response.headers().get(reqwest::header::RETRY_AFTER)?;
    parse_retry_after(raw.to_str().ok()?)
}

/// 只解析 `Retry-After` 的"秒数"形式。
///
/// RFC 7231 也允许 HTTP-date 形式，本项目不处理：遇到时回退到指数退避（安全的一侧），
/// 而不是猜一个时间戳。
fn parse_retry_after(raw: &str) -> Option<Duration> {
    raw.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// 按配置创建 HTTP client。
pub(super) fn build_http_client(cfg: &OaiCompatConfig) -> Result<reqwest::Client, LlmError> {
    let mut builder = reqwest::Client::builder().connect_timeout(cfg.connect_timeout);
    if cfg.resolved_no_proxy() {
        builder = builder.no_proxy();
    }
    builder.build().map_err(|e| {
        LlmError::new(
            LlmErrorKind::Internal,
            format!("创建 HTTP client 失败: {e}"),
        )
        .with_backend(cfg.name.clone())
        .with_source(e)
    })
}

/// 装配 `Client` + tower 栈（重试 → 实际发送）。
///
/// 返回的 `Client` 交给 [`super::backend::OaiCompatBackend`] 持有；async-openai 之后会把
/// 每个请求推进这个栈的最内层（`ReqwestService`），**HTTP 请求在那里才真正发出**。
///
/// 注意**没有**在这里放超时层：整调用总时限与流式首字节时限由后端用
/// `tokio::time::timeout` 包在响应体读取之外，因为传输层看不到 body 阶段。
pub(super) fn build_client(
    cfg: &OaiCompatConfig,
    provider: Arc<dyn Config>,
    http: reqwest::Client,
) -> Client<Arc<dyn Config>> {
    let stack = ServiceBuilder::new()
        .layer(tower::retry::RetryLayer::new(JitterRetryPolicy::new(
            cfg.max_retries,
        )))
        .service(ReqwestService::new(http.clone()));
    Client::build(http, provider).with_http_service(stack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::RecordingObserver;

    fn factory() -> HttpRequestFactory {
        HttpRequestFactory::new(|| async {
            reqwest::Client::new()
                .get("http://127.0.0.1:1/")
                .build()
                .map_err(OpenAIError::Reqwest)
        })
    }

    fn response(status: u16, retry_after: Option<&str>) -> reqwest::Response {
        let mut builder = http::Response::builder().status(status);
        if let Some(value) = retry_after {
            builder = builder.header("retry-after", value);
        }
        reqwest::Response::from(builder.body(String::new()).expect("构造响应"))
    }

    #[test]
    fn parse_retry_after_only_accepts_seconds() {
        assert_eq!(parse_retry_after("2"), Some(Duration::from_secs(2)));
        assert_eq!(parse_retry_after(" 30 "), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("-1"), None);
    }

    #[test]
    fn retry_after_is_used_verbatim_and_capped() {
        assert_eq!(
            next_delay(1, Some(Duration::from_secs(2))),
            Duration::from_secs(2)
        );
        assert_eq!(
            next_delay(1, Some(Duration::from_secs(3600))),
            MAX_RETRY_AFTER,
            "离谱的 Retry-After 要被夹住"
        );
        assert_eq!(next_delay(3, Some(Duration::ZERO)), Duration::ZERO);
    }

    #[test]
    fn backoff_grows_exponentially_then_caps() {
        // 抖动按比例缩放，因此断言区间而不是精确值
        let bounds = |attempt: u32| {
            let nominal = RETRY_BASE_DELAY
                .saturating_mul(1u32 << (attempt - 1))
                .min(RETRY_MAX_DELAY);
            let low = nominal.mul_f64(1.0 - RETRY_JITTER_RATIO);
            let high = nominal.mul_f64(1.0 + RETRY_JITTER_RATIO);
            (low, high, nominal)
        };

        for attempt in [1u32, 2, 3] {
            let (low, high, _) = bounds(attempt);
            for _ in 0..32 {
                let delay = next_delay(attempt, None);
                assert!(
                    delay >= low && delay <= high,
                    "attempt {attempt} 的退避 {delay:?} 超出 [{low:?}, {high:?}]"
                );
            }
        }

        // 上限：即使步数很大也不超过 RETRY_MAX_DELAY × (1+ratio)
        for _ in 0..32 {
            let delay = next_delay(64, None);
            assert!(delay <= RETRY_MAX_DELAY.mul_f64(1.0 + RETRY_JITTER_RATIO));
        }
    }

    #[test]
    fn backoff_grows_on_average() {
        let mean = |attempt: u32| {
            let mut total = Duration::ZERO;
            for _ in 0..64 {
                total += next_delay(attempt, None);
            }
            total / 64
        };
        // 期望值应按 2 的幂增长（抖动把它打散，因此比较的是均值而非单个样本）
        assert!(mean(2) > mean(1));
        assert!(mean(3) > mean(2));
    }

    /// 我们的分类必须与上游 `should_retry` 一致，否则"复用分类"就名不副实。
    #[test]
    fn classification_agrees_with_upstream_should_retry() {
        for status in [200u16, 400, 401, 404, 422, 429, 500, 502, 503] {
            let result: Result<reqwest::Response, OpenAIError> = Ok(response(status, None));
            assert_eq!(
                retry_cause(&result).is_some(),
                should_retry(&result),
                "status {status} 的分类与上游不一致"
            );

            match status {
                429 => assert_eq!(retry_cause(&result), Some(RetryCause::RateLimited)),
                s if (500..600).contains(&s) => {
                    assert_eq!(retry_cause(&result), Some(RetryCause::Server));
                }
                _ => assert_eq!(retry_cause(&result), None),
            }
        }
    }

    #[test]
    fn non_retryable_statuses_yield_no_cause() {
        for status in [400u16, 401, 403, 404, 422] {
            let result: Result<reqwest::Response, OpenAIError> = Ok(response(status, None));
            assert_eq!(retry_cause(&result), None, "status {status} 不该重试");
        }
    }

    /// 需要 tokio 运行时：重试返回的是真实的 `tokio::time::sleep`。
    #[tokio::test]
    async fn retry_budget_is_enforced_by_the_policy() {
        let mut policy = JitterRetryPolicy::new(2);
        for expected in 1..=2 {
            let mut req = factory();
            let mut result: Result<reqwest::Response, OpenAIError> = Ok(response(500, None));
            let pending = policy
                .retry(&mut req, &mut result)
                .unwrap_or_else(|| panic!("第 {expected} 次应重试"));
            drop(pending); // 不等：这里只验证预算判定
        }
        let mut req = factory();
        let mut result: Result<reqwest::Response, OpenAIError> = Ok(response(500, None));
        assert!(
            policy.retry(&mut req, &mut result).is_none(),
            "预算耗尽后必须停止"
        );
    }

    #[test]
    fn policy_never_retries_a_client_error() {
        let mut policy = JitterRetryPolicy::new(3);
        let mut req = factory();
        let mut result: Result<reqwest::Response, OpenAIError> = Ok(response(400, None));
        assert!(policy.retry(&mut req, &mut result).is_none());
    }

    /// 重试必须上报到调用上下文，否则 trace 里永远看不到"发生过重试"。
    #[tokio::test]
    async fn retries_are_reported_to_the_call_context() {
        let observer = Arc::new(RecordingObserver::new());
        let ctx = Arc::new(crate::ctx::CallCtx::new(
            "call-x".into(),
            "oai_compat".into(),
            observer.clone(),
        ));

        let mut policy = JitterRetryPolicy::new(3);
        crate::ctx::scoped(ctx.clone(), async {
            let mut req = factory();
            let mut result: Result<reqwest::Response, OpenAIError> = Ok(response(429, Some("7")));
            // 等待是真的：这里只验证上报与计时，不等返回的 future 完成
            let pending = policy.retry(&mut req, &mut result).expect("429 应重试");
            drop(pending);
        })
        .await;

        assert_eq!(ctx.inner_retries(), 1);
        assert_eq!(ctx.last_retry_after(), Some(Duration::from_secs(7)));

        let retries = observer.retries();
        assert_eq!(retries.len(), 1);
        assert_eq!(retries[0].layer, crate::observer::RetryLayer::Transport);
        assert_eq!(retries[0].attempt, 1);
        assert_eq!(retries[0].delay_ms, 7000, "有 Retry-After 时应按它上报");
        assert_eq!(retries[0].kind, LlmErrorKind::RateLimited);
    }

    #[test]
    fn http_client_builds_and_respects_no_proxy_choice() {
        let local = OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1", "m");
        assert!(build_http_client(&local).is_ok());

        let remote =
            OaiCompatConfig::new("remote", "https://api.example.com/v1", "m").with_no_proxy(true);
        assert!(build_http_client(&remote).is_ok());
    }
}
