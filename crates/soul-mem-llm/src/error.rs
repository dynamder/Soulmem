//! 语义错误：调用方据此决定**重试**还是**降级**，而不是去解析错误字符串。
//!
//! # 谁产生、谁消费
//!
//! - **产生**：各后端把传输层错误映射成 [`LlmError`]（OpenAI-compatible 的映射在
//!   `oai_comp::classify_error`）。只有后端知道自己的 HTTP status 与错误体形状。
//! - **消费**：
//!   - [`LlmError::is_retryable`] —— 由 [`crate::LlmEngine`] 的整调用重试循环使用；
//!   - [`LlmError::is_unavailable`] —— 由业务侧决定降级（例如遗忘路径降级为仅遮罩）；
//!   - [`LlmError::retry_after`] —— 由调用方决定自己退避多久。
//!
//! # 两个谓词的边界
//!
//! [`LlmError::is_retryable`] 的例外是流式中断：只在**尚未产出任何内容**时可重试。
//! 一旦吐出了文本，重放会让消费方看到重复内容（摘要 / 记忆写入场景下是静默损坏）。

use std::fmt;
use std::time::Duration;

/// LLM 调用失败的语义分类。
///
/// **类别**由顶层定义（调用方需要一套稳定的判断口径），**映射**由各后端负责——
/// 只有后端知道自己的 HTTP status、错误体字段、SSE 中断方式与本地推理失败原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlmErrorKind {
    /// 连接层失败：拒绝连接、连接重置、DNS 失败。
    Transport,
    /// 超时：连接超时、整体时限、首字节时限。由后端在其传输边界上判定。
    Timeout,
    /// 429 限流（**不是**配额耗尽）。
    RateLimited,
    /// 5xx 服务端错误。
    ServerError,
    /// 配额 / 余额耗尽（例如 429 `insufficient_quota`、402）。
    Quota,
    /// 鉴权失败（401 / 403）。
    Auth,
    /// 请求被拒（400 / 404 / 405 / 422）：通常是配置或参数问题，重发无用。
    BadRequest,
    /// 成功响应但没有任何可用内容（内容过滤、仅工具调用、空 delta 等）。
    EmptyCompletion,
    /// 生成被输出上限截断。
    Truncated,
    /// 被内容策略拦截。
    ContentFiltered,
    /// 反序列化失败，或非 5xx 且错误体不是可识别形状（此时 HTTP status 已丢失）。
    Decode,
    /// 流在中途中断（已产出部分文本由 [`LlmError::partial`] 携带）。
    StreamInterrupted,
    /// 后端不可用 / 未配置。**不是**网络错误。
    BackendUnavailable,
    /// 后端不支持被要求的能力（如未启用流式却被要求 `stream`）。
    Unsupported,
    /// 调用被取消（含消费方提前丢弃流）。
    Cancelled,
    /// 我方内部错误（不应发生）。
    Internal,
}

impl LlmErrorKind {
    /// 稳定的机器可读名，用于 trace 与日志。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::Timeout => "timeout",
            Self::RateLimited => "rate_limited",
            Self::ServerError => "server_error",
            Self::Quota => "quota",
            Self::Auth => "auth",
            Self::BadRequest => "bad_request",
            Self::EmptyCompletion => "empty_completion",
            Self::Truncated => "truncated",
            Self::ContentFiltered => "content_filtered",
            Self::Decode => "decode",
            Self::StreamInterrupted => "stream_interrupted",
            Self::BackendUnavailable => "backend_unavailable",
            Self::Unsupported => "unsupported",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal",
        }
    }

    /// 重发**同一**请求是否可能成功。
    ///
    /// 注意 [`Self::StreamInterrupted`] 不在此列：流已经产出内容时重放会产生重复文本，
    /// 该情形由 [`LlmError::is_retryable`] 按"是否已产出"单独判定。
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::Transport | Self::Timeout | Self::RateLimited | Self::ServerError
        )
    }

    /// 后端此刻整体不可用 → 调用方应**降级**（如遗忘路径降级为仅遮罩）而不是把错误抛给用户。
    ///
    /// 把 [`Self::Auth`] 与 [`Self::Quota`] 也算进来是刻意的：对使用者而言
    /// "密钥错 / 余额尽"与"服务没起来"是同一类处境——能力缺失，而非这次调用运气不好。
    pub fn is_unavailable(self) -> bool {
        matches!(
            self,
            Self::Transport | Self::Timeout | Self::BackendUnavailable | Self::Auth | Self::Quota
        )
    }
}

/// 一次 LLM 调用失败的结构化描述。
///
/// 按 [`Self::retry_after`] 决定退避，按 [`Self::partial`] 决定已产出的内容是否可用。
#[derive(Debug)]
pub struct LlmError {
    kind: LlmErrorKind,
    message: String,
    backend: Option<String>,
    status: Option<u16>,
    retry_after: Option<Duration>,
    partial: Option<String>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl LlmError {
    pub fn new(kind: LlmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            backend: None,
            status: None,
            retry_after: None,
            partial: None,
            source: None,
        }
    }

    /// 后端不可用（未配置 / 未启动）。调用方通常据此降级。
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(LlmErrorKind::BackendUnavailable, message)
    }

    /// 后端不支持被要求的能力。
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(LlmErrorKind::Unsupported, message)
    }

    /// 已产出部分文本的流式中断。`partial` 为 `None` 表示尚未产出任何内容。
    pub fn stream_interrupted(message: impl Into<String>, partial: Option<String>) -> Self {
        Self::new(LlmErrorKind::StreamInterrupted, message).with_partial_opt(partial)
    }

    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }

    /// 附加已产出的文本（流式中断时使用）。
    pub fn with_partial(mut self, partial: impl Into<String>) -> Self {
        self.partial = Some(partial.into());
        self
    }

    fn with_partial_opt(mut self, partial: Option<String>) -> Self {
        self.partial = partial.filter(|p| !p.is_empty());
        self
    }

    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    pub fn kind(&self) -> LlmErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn backend(&self) -> Option<&str> {
        self.backend.as_deref()
    }

    pub fn status(&self) -> Option<u16> {
        self.status
    }

    /// 服务端建议的等待时长（由后端从自己的响应头解析而来）。
    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }

    /// 流式中断前已产出的文本。`None` 表示未产出。
    pub fn partial(&self) -> Option<&str> {
        self.partial.as_deref()
    }

    /// 重发同一请求是否可能成功。
    ///
    /// 流式中断只在**尚未产出任何内容**时可重试：一旦吐出了文本，重放会让消费方
    /// 看到重复内容（摘要/记忆写入场景下是不可接受的静默损坏）。
    pub fn is_retryable(&self) -> bool {
        match self.kind {
            LlmErrorKind::StreamInterrupted => self.partial.is_none(),
            k => k.is_retryable(),
        }
    }

    /// 后端整体不可用 → 调用方应降级。
    pub fn is_unavailable(&self) -> bool {
        self.kind.is_unavailable()
    }

    /// 补齐缺失的建议等待时长（供外层在耗尽重试后仍能把服务端建议带给调用方）。
    pub fn set_retry_after(&mut self, retry_after: Duration) {
        if self.retry_after.is_none() {
            self.retry_after = Some(retry_after);
        }
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind.as_str(), self.message)?;
        if let Some(backend) = &self.backend {
            write!(f, " (backend={backend})")?;
        }
        if let Some(status) = self.status {
            write!(f, " (status={status})")?;
        }
        if let Some(retry_after) = self.retry_after {
            write!(f, " (retry_after={}ms)", retry_after.as_millis())?;
        }
        Ok(())
    }
}

impl std::error::Error for LlmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|s| s.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_kinds_are_exactly_the_transient_ones() {
        for kind in [
            LlmErrorKind::Transport,
            LlmErrorKind::Timeout,
            LlmErrorKind::RateLimited,
            LlmErrorKind::ServerError,
        ] {
            assert!(kind.is_retryable(), "{kind:?} 应可重试");
        }
        for kind in [
            LlmErrorKind::Quota,
            LlmErrorKind::Auth,
            LlmErrorKind::BadRequest,
            LlmErrorKind::EmptyCompletion,
            LlmErrorKind::Truncated,
            LlmErrorKind::ContentFiltered,
            LlmErrorKind::Decode,
            LlmErrorKind::BackendUnavailable,
            LlmErrorKind::Unsupported,
            LlmErrorKind::Cancelled,
            LlmErrorKind::Internal,
        ] {
            assert!(!kind.is_retryable(), "{kind:?} 不应可重试");
        }
    }

    #[test]
    fn unavailable_kinds_drive_degradation() {
        for kind in [
            LlmErrorKind::Transport,
            LlmErrorKind::Timeout,
            LlmErrorKind::BackendUnavailable,
            LlmErrorKind::Auth,
            LlmErrorKind::Quota,
        ] {
            assert!(kind.is_unavailable(), "{kind:?} 应视为不可用");
        }
        // 参数写错不算"不可用"：它需要人改配置，不是降级能绕过的
        for kind in [LlmErrorKind::BadRequest, LlmErrorKind::Decode] {
            assert!(!kind.is_unavailable(), "{kind:?} 不应视为不可用");
        }
    }

    #[test]
    fn stream_interrupted_is_retryable_only_before_any_output() {
        let empty = LlmError::stream_interrupted("断流", None);
        assert!(empty.is_retryable(), "未产出内容的重放是安全的");

        let blank = LlmError::stream_interrupted("断流", Some(String::new()));
        assert!(blank.is_retryable(), "空串等同未产出");

        let partial = LlmError::stream_interrupted("断流", Some("半截".into()));
        assert!(!partial.is_retryable(), "已产出内容时重放会造成重复文本");
        assert_eq!(partial.partial(), Some("半截"));
    }

    #[test]
    fn set_retry_after_does_not_overwrite_backend_value() {
        let mut e = LlmError::new(LlmErrorKind::RateLimited, "429")
            .with_retry_after(Duration::from_secs(7));
        e.set_retry_after(Duration::from_secs(1));
        assert_eq!(e.retry_after(), Some(Duration::from_secs(7)));

        let mut none = LlmError::new(LlmErrorKind::RateLimited, "429");
        none.set_retry_after(Duration::from_secs(3));
        assert_eq!(none.retry_after(), Some(Duration::from_secs(3)));
    }

    #[test]
    fn display_carries_kind_backend_status_and_retry_after() {
        let e = LlmError::new(LlmErrorKind::RateLimited, "too many")
            .with_backend("local")
            .with_status(429)
            .with_retry_after(Duration::from_millis(1500));
        let text = e.to_string();
        assert!(text.contains("rate_limited"), "{text}");
        assert!(text.contains("backend=local"), "{text}");
        assert!(text.contains("status=429"), "{text}");
        assert!(text.contains("retry_after=1500ms"), "{text}");
    }

    #[test]
    fn kind_names_are_unique_and_snake_case() {
        let all = [
            LlmErrorKind::Transport,
            LlmErrorKind::Timeout,
            LlmErrorKind::RateLimited,
            LlmErrorKind::ServerError,
            LlmErrorKind::Quota,
            LlmErrorKind::Auth,
            LlmErrorKind::BadRequest,
            LlmErrorKind::EmptyCompletion,
            LlmErrorKind::Truncated,
            LlmErrorKind::ContentFiltered,
            LlmErrorKind::Decode,
            LlmErrorKind::StreamInterrupted,
            LlmErrorKind::BackendUnavailable,
            LlmErrorKind::Unsupported,
            LlmErrorKind::Cancelled,
            LlmErrorKind::Internal,
        ];
        let mut names: Vec<&str> = all.iter().map(|k| k.as_str()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "kind 名必须唯一（trace 靠它区分）");
        assert!(
            names
                .iter()
                .all(|n| n.chars().all(|c| c.is_ascii_lowercase() || c == '_')),
            "kind 名应为 snake_case：{names:?}"
        );
    }
}
