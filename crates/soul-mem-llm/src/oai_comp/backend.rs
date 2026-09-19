//! OpenAI-compatible 后端实现。
//!
//! 这里是"编码 / 解码 / 错误映射"三件事的唯一归属地：
//! `encode` 把语义 [`Task`] 变成 wire 请求，`decode` 把 wire 响应变成语义 [`Completion`]，
//! [`classify_error`] 把传输层错误变成语义 [`LlmError`]。
//! 顶层契约不认识其中任何一步的细节。
//!
//! # 本文件在链路中的位置
//!
//! ```text
//! crate::engine ──► ChatBackend::complete(task)            ← 本文件的入口
//!                     │
//!                     ├─ encode(&task, false) ──► wire::OaiChatRequest（唯一出现 provider 字段名处）
//!                     ├─ tokio::time::timeout(总时限, pending)   ← 包住整次调用（含 body 读取）
//!                     ├─ client.chat().create_byot::<OaiChatRequest, OaiResponse>(..)
//!                     │      └──► async-openai：Client::post → tower 栈（transport.rs）→ 发出
//!                     │            ★ 请求发出点在 async_openai::middleware::ReqwestService::call
//!                     ├─ .map_err(classify_error)          ← HTTP/传输错误 → LlmError
//!                     └─ decode(OaiResponse) ──► Completion（text / reasoning / stop / usage）
//! ```
//!
//! 流式把后三步换成 `create_stream_byot` + [`ChunkState`] + [`map_chunks`]；
//! 此时传输层只跑到"拿到响应头"（首字节时限的边界）。
//!
//! 子树的整体布局见 `oai_comp.rs` 的模块注释；跨 crate 的完整链路见
//! `docs/architecture/llm-layer.md`。

use super::config::{OaiCompatConfig, TokenFieldPolicy};
use super::provider::OaiCompatProvider;
use super::transport::{build_client, build_http_client};
use super::wire::{
    OaiChatRequest, OaiChunk, OaiMessage, OaiResponse, OaiUsage, TokenField, stop_reason_from_wire,
};
use crate::backend::{
    BackendInfo, ChatBackend, Completion, EventStream, StopReason, StreamEvent, Task,
};
use crate::error::{LlmError, LlmErrorKind};
use crate::json::preview_text;
use async_openai::Client;
use async_openai::config::Config;
use async_openai::error::OpenAIError;
use async_trait::async_trait;
use futures::StreamExt;
use std::pin::Pin;
use std::sync::Arc;

/// OpenAI-compatible 后端（远程 API 与本地 llama-server 共用同一实现）。
///
/// 构造时做三件事：校验配置 → 建 `Config` 实现（`provider.rs`）→ 建带 tower 栈的
/// `async_openai::Client`（`transport.rs`）。此后本结构体只负责编码/解码/错误映射，
/// 不直接接触任何 HTTP 细节。
pub struct OaiCompatBackend {
    cfg: OaiCompatConfig,
    client: Client<Arc<dyn Config>>,
    info: BackendInfo,
}

impl OaiCompatBackend {
    pub fn new(cfg: OaiCompatConfig) -> Result<Self, LlmError> {
        cfg.validate()?;

        let provider: Arc<dyn Config> = Arc::new(OaiCompatProvider::from_config(&cfg));
        let http = build_http_client(&cfg)?;
        let client = build_client(&cfg, provider, http);

        let info = BackendInfo::new(cfg.name.clone())
            .with_model(cfg.model.clone())
            .with_streaming(cfg.supports_streaming);

        Ok(Self { cfg, client, info })
    }

    /// 语义请求 → wire 请求。
    ///
    /// 这是 `Task` 里的语义被**唯一一次**翻译成 provider 词汇的地方：
    /// `hints.suppress_reasoning` → `reasoning_suppression_body`（如 `chat_template_kwargs`），
    /// `sampling.max_output_tokens` → `max_tokens` 或 `max_completion_tokens`（按 [`TokenFieldPolicy`]）。
    /// 缺省字段一律不出现在 JSON 里（部分服务对未知/空参数直接 400）。
    fn encode(&self, task: &Task, stream: bool) -> OaiChatRequest {
        let mut extra = self.cfg.extra_body.clone();
        if task.hints.suppress_reasoning {
            for (key, value) in &self.cfg.reasoning_suppression_body {
                extra.insert(key.clone(), value.clone());
            }
        }

        let token_limit = match (self.cfg.token_field, task.sampling.max_output_tokens) {
            (TokenFieldPolicy::MaxTokens, Some(max)) => TokenField::max_tokens(max),
            (TokenFieldPolicy::MaxCompletionTokens, Some(max)) => {
                TokenField::max_completion_tokens(max)
            }
            (_, None) => TokenField::default(),
        };

        OaiChatRequest {
            model: self.cfg.model.clone(),
            messages: task
                .messages
                .iter()
                .map(|message| OaiMessage {
                    role: message.role.as_str().to_string(),
                    content: message.text.clone(),
                })
                .collect(),
            temperature: task.sampling.temperature,
            stream: stream.then_some(true),
            token_limit,
            extra,
        }
    }

    /// wire 响应 → 语义结果。
    ///
    /// 三种结局：正常提取正文；正文空但 `reasoning_content` 非空 → **兜底使用并记 warn**
    /// （不丢内容，也不假装那是"正常正文"）；两者都空 → 按 `finish_reason` 归类成
    /// [`LlmErrorKind::ContentFiltered`] / [`LlmErrorKind::Truncated`] /
    /// [`LlmErrorKind::EmptyCompletion`]。
    ///
    /// 调用方：`ChatBackend::complete`（非流式）；流式的对应逻辑在 [`ChunkState::build_completion`]。
    fn decode(&self, response: OaiResponse) -> Result<Completion, LlmError> {
        let usage = response
            .usage
            .filter(|usage| !usage.is_empty())
            .map(|usage| usage.to_usage());

        let Some(choice) = response.choices.into_iter().next() else {
            return Err(self.err(LlmErrorKind::EmptyCompletion, "响应里没有任何 choice"));
        };

        let stop = stop_reason_from_wire(choice.finish_reason.as_deref());
        let message = choice.message.unwrap_or_default();
        let text = message.content.unwrap_or_default();
        let reasoning = message
            .reasoning_content
            .filter(|reasoning| !reasoning.trim().is_empty());

        if !text.trim().is_empty() {
            return Ok(Completion {
                text: text.trim().to_string(),
                reasoning,
                stop,
                usage,
            });
        }

        // 正文为空但思考内容非空：Qwen3 系在未禁用 thinking 时会把正文写进 reasoning_content。
        // 这里兜底而不是丢内容，但要留下日志——否则"模型只会想不会说"这件事没人知道。
        if let Some(reasoning) = reasoning {
            tracing::warn!(
                backend = %self.cfg.name,
                "正文为空，改用 reasoning_content 作为结果"
            );
            return Ok(Completion {
                text: reasoning.trim().to_string(),
                reasoning: None,
                stop,
                usage,
            });
        }

        let kind = match stop {
            StopReason::Filtered => LlmErrorKind::ContentFiltered,
            StopReason::LengthCapped => LlmErrorKind::Truncated,
            _ => LlmErrorKind::EmptyCompletion,
        };
        Err(self.err(kind, "响应正文与思考内容都为空"))
    }

    fn err(&self, kind: LlmErrorKind, message: impl Into<String>) -> LlmError {
        LlmError::new(kind, message).with_backend(self.cfg.name.clone())
    }

    fn classify(&self, error: OpenAIError) -> LlmError {
        classify_error(&self.cfg.name, error)
    }
}

/// 传输层错误 → 语义错误。
///
/// 映射必须在这里：只有本后端知道自己的 HTTP status 语义、错误体形状、
/// `insufficient_quota` 这类 provider 专有标记。
///
/// 几个刻意的判定：
/// - `401/403 → Auth`、`429 + insufficient_quota → Quota`、其余 `429 → RateLimited`；
/// - `400/404/405/409/422 → BadRequest`（重发无用，提示调用方改配置）；
/// - 5xx 与**无法从前缀判断**的状态码分开处理：前者是 `ServerError`（可重试），
///   后者归 `Decode`（不猜语义）；
/// - 非 OpenAI 形状的错误体（如反代返回 HTML）会落到 `JSONDeserialize`，
///   **HTTP status 在这一路径上已丢失**，因此只能归 `Decode`——错误信息里带上原始 body 片段。
///
/// 调用方：`ChatBackend::complete`/`stream` 的 `map_err`，以及 [`ChunkState::next_event`]。
pub(super) fn classify_error(backend: &str, error: OpenAIError) -> LlmError {
    match error {
        OpenAIError::Reqwest(error) => {
            let kind = if error.is_timeout() {
                LlmErrorKind::Timeout
            } else {
                LlmErrorKind::Transport
            };
            LlmError::new(kind, format!("HTTP 请求失败: {error}"))
                .with_backend(backend)
                .with_source(error)
        }
        OpenAIError::ApiError(response) => {
            let status = response.status_code.as_u16();
            let api = response.api_error;
            let kind = match status {
                401 | 403 => LlmErrorKind::Auth,
                402 => LlmErrorKind::Quota,
                429 => {
                    if api.r#type.as_deref() == Some("insufficient_quota") {
                        LlmErrorKind::Quota
                    } else {
                        LlmErrorKind::RateLimited
                    }
                }
                400 | 404 | 405 | 409 | 422 => LlmErrorKind::BadRequest,
                status if (500..600).contains(&status) => LlmErrorKind::ServerError,
                // 其余状态码的语义无法从前缀判断，如实归到"未知/解析"而不是瞎猜
                _ => LlmErrorKind::Decode,
            };
            let mut message = format!("API 返回错误: {}", api.message);
            if let Some(code) = api.code.as_deref() {
                message.push_str(&format!(" (code={code})"));
            }
            if let Some(param) = api.param.as_deref() {
                message.push_str(&format!(" (param={param})"));
            }
            LlmError::new(kind, message)
                .with_backend(backend)
                .with_status(status)
        }
        OpenAIError::JSONDeserialize(error, body) => LlmError::new(
            LlmErrorKind::Decode,
            format!(
                "响应反序列化失败（body 前 512 字符）: {}",
                preview_text(&body, 512)
            ),
        )
        .with_backend(backend)
        .with_source(error),
        OpenAIError::InvalidArgument(message) => {
            LlmError::new(LlmErrorKind::Internal, format!("请求参数被拒: {message}"))
                .with_backend(backend)
        }
        OpenAIError::StreamError(error) => {
            LlmError::stream_interrupted(format!("SSE 流错误: {error}"), None).with_backend(backend)
        }
        OpenAIError::Boxed(error) => {
            if error.is::<tokio::time::error::Elapsed>() {
                LlmError::new(LlmErrorKind::Timeout, "请求超过时限").with_backend(backend)
            } else {
                LlmError::new(LlmErrorKind::Internal, format!("中间件错误: {error}"))
                    .with_backend(backend)
            }
        }
        other => LlmError::new(LlmErrorKind::Internal, format!("未归类的错误: {other}"))
            .with_backend(backend),
    }
}

#[async_trait]
impl ChatBackend for OaiCompatBackend {
    fn info(&self) -> BackendInfo {
        self.info.clone()
    }

    /// 一次性完成。
    ///
    /// 三步：编码 → 带总时限发出 → 解码。
    /// `create_byot` 是 async-openai 的 `#[byot]` 宏生成的泛型方法——它接受**我们自己的**
    /// 请求类型并反序列化成**我们自己的**响应类型，而 URL 拼接、鉴权头、tower 重试栈仍由
    /// async-openai 负责。源码里 grep 不到这个函数，详见 `docs/architecture/llm-layer.md`。
    async fn complete(&self, task: Task) -> Result<Completion, LlmError> {
        let request = self.encode(&task, false);
        // `Chat` 借用 client，必须绑定到局部变量，否则临时值在 await 期间已被释放
        let chat = self.client.chat();
        let pending = chat.create_byot::<OaiChatRequest, OaiResponse>(request);

        // 总时限必须包在响应体读取之外：传输层只覆盖到"拿到响应头"，
        // 而一次长生成的时间几乎全花在读 body 上。
        let response = match self.cfg.total_timeout {
            Some(limit) => tokio::time::timeout(limit, pending).await.map_err(|_| {
                self.err(
                    LlmErrorKind::Timeout,
                    format!("整次调用超过 {}ms", limit.as_millis()),
                )
            })?,
            None => pending.await,
        };

        let response = response.map_err(|error| self.classify(error))?;
        self.decode(response)
    }

    async fn stream(&self, task: Task) -> Result<EventStream, LlmError> {
        if !self.info.supports_streaming {
            return Err(self.err(LlmErrorKind::Unsupported, "该 provider 配置为不支持流式"));
        }

        let request = self.encode(&task, true);
        let chat = self.client.chat();
        let pending = chat.create_stream_byot::<OaiChatRequest, OaiChunk>(request);

        // 只限制"首字节"：拿到响应头即算通过，流的总时长不受此限制。
        let opened = match self.cfg.first_byte_timeout {
            Some(limit) => tokio::time::timeout(limit, pending).await.map_err(|_| {
                self.err(
                    LlmErrorKind::Timeout,
                    format!("流式首字节超过 {}ms", limit.as_millis()),
                )
            })?,
            None => pending.await,
        };

        let inner = opened.map_err(|error| self.classify(error))?;
        Ok(map_chunks(inner, self.cfg.name.clone()))
    }
}

/// 流式 chunk 的解析状态机。
///
/// 输入是 async-openai 已经反序列化好的 `OaiChunk` 流，输出是语义 [`StreamEvent`]。
///
/// 两个刻意的判定（都直接关系到"内容会不会被静默损坏"）：
/// - **已产出内容后再出错** → 统一改写成 [`LlmError::stream_interrupted`] 并带上已产出文本，
///   从而 [`LlmError::is_retryable`] 为假，**不可能被静默重放**；
/// - **没有任何内容、也没有 `finish_reason` 就结束** → 判定为中断（可重试），
///   而不是伪造一个空结果。
///
/// `[DONE]` 与"连接正常关闭"在 async-openai 层不可区分（都表现为流结束）；
/// 有内容但缺 `finish_reason` 时保留内容并把 `stop` 标为 `Unknown`，同时记 warn。
struct ChunkState {
    inner: Pin<Box<dyn futures::Stream<Item = Result<OaiChunk, OpenAIError>> + Send>>,
    backend: String,
    text: String,
    reasoning: String,
    finish_reason: Option<String>,
    usage: Option<OaiUsage>,
    emitted: bool,
    finished: bool,
}

impl ChunkState {
    /// 取下一个**语义**事件。空 chunk（只带 usage、只带 finish_reason、空 delta）会被跳过。
    async fn next_event(&mut self) -> Option<Result<StreamEvent, LlmError>> {
        if self.finished {
            return None;
        }
        loop {
            match self.inner.next().await {
                None => {
                    self.finished = true;
                    if !self.emitted && self.finish_reason.is_none() {
                        // 没有任何内容、也没有停止原因就结束了：这更像中断而不是"空回答"。
                        // 如实报成可重试的中断，而不是伪造一个空结果。
                        return Some(Err(LlmError::stream_interrupted(
                            "流结束但既没有内容也没有 finish_reason",
                            None,
                        )
                        .with_backend(self.backend.clone())));
                    }
                    if self.finish_reason.is_none() {
                        // 有内容但没有停止原因：OpenAI-compatible 服务在正常收尾时会给
                        // finish_reason，缺失通常是服务端实现差异。保留结果，但把
                        // stop=unknown 记下来，让"可能被截断"这件事在 trace 里可见。
                        tracing::warn!(
                            backend = %self.backend,
                            "流结束但缺少 finish_reason，停止原因记为 unknown"
                        );
                    }
                    let completion = self.build_completion();
                    return Some(Ok(StreamEvent::Done(Box::new(completion))));
                }
                Some(Err(error)) => {
                    self.finished = true;
                    let classified = classify_error(&self.backend, error);
                    // 已经吐过内容：把底层错误统一改写成"流中断 + 已产出文本"，
                    // 从而不可能被静默重放（见 LlmError::is_retryable）。
                    return Some(Err(match self.emitted {
                        true => LlmError::stream_interrupted(
                            classified.message().to_string(),
                            Some(self.text.clone()),
                        )
                        .with_backend(self.backend.clone()),
                        false => classified,
                    }));
                }
                Some(Ok(chunk)) => {
                    if let Some(usage) = chunk.usage.filter(|usage| !usage.is_empty()) {
                        self.usage = Some(usage);
                    }
                    let Some(choice) = chunk.choices.into_iter().next() else {
                        continue;
                    };
                    if choice.finish_reason.is_some() {
                        self.finish_reason = choice.finish_reason;
                    }
                    let Some(delta) = choice.delta else {
                        continue;
                    };
                    if let Some(reasoning) = delta.reasoning_content.filter(|r| !r.is_empty()) {
                        self.reasoning.push_str(&reasoning);
                        self.emitted = true;
                        return Some(Ok(StreamEvent::ReasoningDelta(reasoning)));
                    }
                    if let Some(content) = delta.content.filter(|c| !c.is_empty()) {
                        self.text.push_str(&content);
                        self.emitted = true;
                        return Some(Ok(StreamEvent::Delta(content)));
                    }
                    continue;
                }
            }
        }
    }

    fn build_completion(&self) -> Completion {
        let has_text = !self.text.trim().is_empty();
        let reasoning = (!self.reasoning.trim().is_empty()).then(|| self.reasoning.clone());
        // 与 decode 一致：正文为空时用思考内容兜底，避免"只想了没说"导致内容全丢
        let (text, reasoning) = if has_text {
            (self.text.clone(), reasoning)
        } else {
            (self.reasoning.clone(), None)
        };
        Completion {
            text: text.trim().to_string(),
            reasoning,
            stop: stop_reason_from_wire(self.finish_reason.as_deref()),
            usage: self
                .usage
                .as_ref()
                .filter(|usage| !usage.is_empty())
                .map(|usage| usage.to_usage()),
        }
    }
}

/// wire chunk 流 → 语义事件流。
///
/// 这是流式路径上"provider 形状 → 语义"的唯一转换点，由 `ChatBackend::stream` 返回给
/// [`crate::LlmEngine`]，后者再用 `ObservedStream` 包一层做观测收尾。
fn map_chunks(
    inner: Pin<Box<dyn futures::Stream<Item = Result<OaiChunk, OpenAIError>> + Send>>,
    backend: String,
) -> EventStream {
    let state = ChunkState {
        inner,
        backend,
        text: String::new(),
        reasoning: String::new(),
        finish_reason: None,
        usage: None,
        emitted: false,
        finished: false,
    };
    Box::pin(futures::stream::unfold(state, |mut state| async move {
        state.next_event().await.map(|event| (event, state))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::error::{ApiError, ApiErrorResponse};
    use reqwest::StatusCode;
    use serde_json::json;

    fn api_error(
        status: u16,
        kind: Option<&str>,
        message: &str,
        param: Option<&str>,
    ) -> OpenAIError {
        OpenAIError::ApiError(ApiErrorResponse {
            status_code: StatusCode::from_u16(status).expect("合法状态码"),
            api_error: ApiError {
                message: message.to_string(),
                r#type: kind.map(str::to_string),
                param: param.map(str::to_string),
                code: None,
            },
        })
    }

    #[test]
    fn api_status_maps_to_the_right_kind() {
        let cases = [
            (401u16, None, LlmErrorKind::Auth),
            (403, None, LlmErrorKind::Auth),
            (402, None, LlmErrorKind::Quota),
            (429, Some("rate_limit_exceeded"), LlmErrorKind::RateLimited),
            (429, Some("insufficient_quota"), LlmErrorKind::Quota),
            (400, None, LlmErrorKind::BadRequest),
            (404, None, LlmErrorKind::BadRequest),
            (405, None, LlmErrorKind::BadRequest),
            (409, None, LlmErrorKind::BadRequest),
            (422, None, LlmErrorKind::BadRequest),
            (500, None, LlmErrorKind::ServerError),
            (503, None, LlmErrorKind::ServerError),
            // 无法从前缀判断语义的状态码：如实归到 Decode，不瞎猜
            (418, None, LlmErrorKind::Decode),
            (302, None, LlmErrorKind::Decode),
        ];

        for (status, api_type, expected) in cases {
            let error = classify_error("p", api_error(status, api_type, "boom", None));
            assert_eq!(
                error.kind(),
                expected,
                "status {status} / type {api_type:?} 的归类不符"
            );
            assert_eq!(error.status(), Some(status));
            assert_eq!(error.backend(), Some("p"));
            assert!(error.message().contains("boom"));
        }
    }

    #[test]
    fn api_error_message_carries_param_for_configuration_fixes() {
        let error = classify_error(
            "p",
            api_error(400, None, "unknown parameter", Some("max_tokens")),
        );
        assert!(
            error.message().contains("max_tokens"),
            "参数名要带出来，否则无法据此改配置：{}",
            error.message()
        );
    }

    #[test]
    fn non_ai_shaped_error_body_is_classified_as_decode_with_the_body() {
        let error = classify_error(
            "p",
            OpenAIError::JSONDeserialize(
                serde_json::from_str::<serde_json::Value>("{").expect_err("必然失败"),
                "<html>Bad Request</html>".into(),
            ),
        );
        assert_eq!(error.kind(), LlmErrorKind::Decode);
        assert!(error.message().contains("Bad Request"));
    }

    #[test]
    fn invalid_argument_is_an_internal_error() {
        let error = classify_error("p", OpenAIError::InvalidArgument("bad".into()));
        assert_eq!(error.kind(), LlmErrorKind::Internal);
    }

    fn backend() -> OaiCompatBackend {
        OaiCompatBackend::new(OaiCompatConfig::new(
            "test",
            "http://127.0.0.1:1/v1",
            "test-model",
        ))
        .expect("后端构建成功")
    }

    fn decode(value: serde_json::Value) -> Result<Completion, LlmError> {
        let response: OaiResponse = serde_json::from_value(value).expect("响应可解析");
        backend().decode(response)
    }

    #[test]
    fn decode_extracts_text_stop_reason_and_usage() {
        let completion = decode(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "  结果  "},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }))
        .expect("应成功");

        assert_eq!(completion.text, "结果", "首尾空白应被去掉");
        assert_eq!(completion.stop, StopReason::Completed);
        assert_eq!(completion.usage.unwrap().total_tokens, Some(8));
    }

    #[test]
    fn decode_reports_missing_choices_instead_of_empty_text() {
        let error = decode(json!({"choices": []})).expect_err("空 choices 必须报错");
        assert_eq!(error.kind(), LlmErrorKind::EmptyCompletion);
        assert!(error.message().contains("choice"));
    }

    #[test]
    fn decode_maps_content_filter_and_truncation_to_distinct_kinds() {
        let filtered = decode(json!({
            "choices": [{"message": {"role": "assistant", "content": null}, "finish_reason": "content_filter"}]
        }))
        .expect_err("应失败");
        assert_eq!(filtered.kind(), LlmErrorKind::ContentFiltered);

        let truncated = decode(json!({
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]
        }))
        .expect_err("应失败");
        assert_eq!(truncated.kind(), LlmErrorKind::Truncated);
    }

    #[test]
    fn decode_falls_back_to_reasoning_without_losing_content() {
        let completion = decode(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "", "reasoning_content": "只有思考内容"},
                "finish_reason": "stop"
            }]
        }))
        .expect("应成功");

        assert_eq!(completion.text, "只有思考内容");
        assert_eq!(completion.reasoning, None, "已当作正文使用，不应重复计一次");
    }

    #[test]
    fn decode_keeps_reasoning_when_text_is_present() {
        let completion = decode(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "答案", "reasoning_content": "推理"},
                "finish_reason": "stop"
            }]
        }))
        .expect("应成功");

        assert_eq!(completion.text, "答案");
        assert_eq!(completion.reasoning.as_deref(), Some("推理"));
    }

    #[test]
    fn decode_omits_empty_usage() {
        let completion = decode(json!({
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null}
        }))
        .expect("应成功");
        assert_eq!(
            completion.usage, None,
            "全空的 usage 不应写成 Some 的假数据"
        );
    }

    #[test]
    fn encode_maps_semantic_sampling_to_wire_fields() {
        let task = Task::system_user("系统", "用户").with_sampling(
            crate::backend::Sampling::default()
                .with_temperature(0.25)
                .with_max_output_tokens(64),
        );

        let req = backend().encode(&task, false);
        assert_eq!(req.model, "test-model");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.temperature, Some(0.25));
        assert_eq!(req.token_limit.max_tokens, Some(64));
        assert_eq!(req.stream, None, "非流式不应带 stream");
    }

    #[test]
    fn encode_injects_reasoning_suppression_only_when_asked() {
        let backend = OaiCompatBackend::new(
            OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1", "qwen3-4b")
                .with_reasoning_suppression(
                    "chat_template_kwargs",
                    json!({"enable_thinking": false}),
                ),
        )
        .expect("后端构建成功");

        let plain = backend.encode(&Task::system_user("s", "u"), false);
        assert!(plain.extra.is_empty(), "没要求时不得注入");

        let suppressing = backend.encode(
            &Task::system_user("s", "u")
                .with_hints(crate::backend::Hints::default().with_suppress_reasoning(true)),
            false,
        );
        assert_eq!(
            suppressing.extra["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
    }

    #[test]
    fn encode_marks_the_stream_request() {
        let req = backend().encode(&Task::system_user("s", "u"), true);
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn backend_info_reports_model_and_streaming() {
        let info = backend().info();
        assert_eq!(info.name, "test");
        assert_eq!(info.model.as_deref(), Some("test-model"));
        assert!(info.supports_streaming);
    }

    #[test]
    fn missing_base_url_is_reported_as_unavailable() {
        let error = OaiCompatBackend::new(OaiCompatConfig::new("p", "", "m"))
            .err()
            .expect("缺 base_url 应构建失败");
        assert_eq!(error.kind(), LlmErrorKind::BackendUnavailable);
    }
}
