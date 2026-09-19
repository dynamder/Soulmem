//! 端到端测试：真实 HTTP + 真实 tower 栈 + 真实 SSE 解析。
//!
//! 只放**必须**经过真实传输的用例：请求真的发到网线上的字节、重试是否真的发生、
//! 超时是否真的生效、SSE 是否真的被解析成语义事件。
//! 状态码映射与响应解码的用例在 `backend.rs` 的单元测试里，不必走 HTTP。

use super::OaiCompatBackend;
use super::config::{OaiCompatConfig, TokenFieldPolicy};
use crate::backend::{Hints, Sampling, StopReason, StreamEvent, Task};
use crate::engine::LlmEngine;
use crate::error::LlmErrorKind;
use crate::observer::{RecordingObserver, RetryLayer};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// 服务器按脚本作答的一条回复。
#[derive(Clone)]
enum Reply {
    /// 普通 JSON 响应。
    Json {
        status: u16,
        body: String,
        retry_after: Option<u64>,
    },
    /// 先延迟再回答（用于触发超时）。
    SlowJson {
        delay_ms: u64,
        status: u16,
        body: String,
    },
    /// SSE 流式响应；`done` 决定是否发 `data: [DONE]`。
    Sse { chunks: Vec<String>, done: bool },
    /// 声明一个比实际发送内容更长的 `Content-Length` 后断开：
    /// 制造"已经产出内容、随后读取失败"的真实中断（用于验证不可重放）。
    TruncatedSse {
        chunks: Vec<String>,
        declared_length: usize,
    },
}

/// 记录收到的原始请求的脚本化服务器。
struct ScriptedServer {
    addr: SocketAddr,
    seen: Arc<parking_lot::Mutex<Vec<String>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl ScriptedServer {
    async fn start(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("绑定回环端口");
        let addr = listener.local_addr().expect("取得端口");
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let seen_for_task = seen.clone();

        let handle = tokio::spawn(async move {
            let queue = Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::from(
                replies,
            )));
            while let Ok((mut stream, _)) = listener.accept().await {
                let queue = queue.clone();
                let seen = seen_for_task.clone();
                // 每个连接一个任务：若串行处理，一次"慢回复"会把后续请求也堵在后面，
                // 于是重试的第二次请求还没被受理就已经超时——测试会假阳性。
                tokio::spawn(async move {
                    let request = read_request(&mut stream).await;
                    seen.lock().push(request);
                    let reply = queue.lock().pop_front().unwrap_or(Reply::Json {
                        status: 500,
                        body: json!({"error": {"message": "server script exhausted"}}).to_string(),
                        retry_after: None,
                    });
                    reply.send(&mut stream).await;
                    let _ = stream.shutdown().await;
                });
            }
        });

        Self { addr, seen, handle }
    }

    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn requests(&self) -> Vec<String> {
        self.seen.lock().clone()
    }

    fn request_count(&self) -> usize {
        self.seen.lock().len()
    }

    fn json_bodies(&self) -> Vec<Value> {
        self.requests()
            .iter()
            .map(|raw| {
                let body = raw.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
                serde_json::from_str(body).unwrap_or_else(|e| panic!("请求体不是 JSON: {e}\n{raw}"))
            })
            .collect()
    }

    fn request_line(&self, index: usize) -> String {
        self.requests()
            .get(index)
            .and_then(|raw| raw.lines().next())
            .unwrap_or_default()
            .to_string()
    }

    fn shutdown(self) {
        self.handle.abort();
    }
}

impl Reply {
    /// 写回这条回复。延迟用异步睡眠——测试跑在 current_thread 运行时上，
    /// 用 `std::thread::sleep` 会把客户端一起冻住，超时永远不会触发。
    async fn send(self, stream: &mut TcpStream) {
        let raw = match self {
            Reply::Json {
                status,
                body,
                retry_after,
            } => json_response(status, &body, retry_after),
            Reply::SlowJson {
                delay_ms,
                status,
                body,
            } => {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                json_response(status, &body, None)
            }
            Reply::Sse { chunks, done } => sse_response(&chunks, done),
            Reply::TruncatedSse {
                chunks,
                declared_length,
            } => truncated_sse_response(&chunks, declared_length),
        };
        let _ = stream.write_all(raw.as_bytes()).await;
        let _ = stream.flush().await;
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

fn json_response(status: u16, body: &str, retry_after: Option<u64>) -> String {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n",
        reason(status),
        body.len()
    );
    if let Some(seconds) = retry_after {
        head.push_str(&format!("retry-after: {seconds}\r\n"));
    }
    format!("{head}\r\n{body}")
}

fn sse_body(chunks: &[String], done: bool) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    if done {
        body.push_str("data: [DONE]\n\n");
    }
    body
}

fn sse_response(chunks: &[String], done: bool) -> String {
    let body = sse_body(chunks, done);
    format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{body}")
}

/// 声明一个比实际内容更长的长度再断开，让客户端在读 body 时报错。
fn truncated_sse_response(chunks: &[String], declared_length: usize) -> String {
    let body = sse_body(chunks, false);
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {declared_length}\r\nconnection: close\r\n\r\n{body}"
    )
}

/// 读到完整的请求（头 + 按 Content-Length 读完 body）。
async fn read_request(stream: &mut TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        buffer.extend_from_slice(&chunk[..read]);

        if let Some(headers_end) = find_headers_end(&buffer) {
            let headers = String::from_utf8_lossy(&buffer[..headers_end]).to_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buffer.len() >= headers_end + 4 + content_length {
                break;
            }
        }
        if buffer.len() > 1 << 20 {
            break;
        }
    }
    String::from_utf8_lossy(&buffer).to_string()
}

fn find_headers_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|w| w == b"\r\n\r\n")
}

/// 一份合法的非流式响应体。
fn completion_body(text: &str) -> String {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
    .to_string()
}

fn chunk(delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
    })
    .to_string()
}

struct Harness {
    engine: LlmEngine,
    observer: Arc<RecordingObserver>,
    server: ScriptedServer,
}

impl Harness {
    fn new(server: ScriptedServer, cfg: OaiCompatConfig, whole_call_retries: u32) -> Self {
        let observer = Arc::new(RecordingObserver::new());
        let backend = OaiCompatBackend::new(cfg).expect("后端构建成功");
        let engine = LlmEngine::new(Arc::new(backend))
            .with_observer(observer.clone())
            .with_whole_call_retries(whole_call_retries);
        Self {
            engine,
            observer,
            server,
        }
    }

    fn config(
        server: &ScriptedServer,
        configure: impl FnOnce(OaiCompatConfig) -> OaiCompatConfig,
    ) -> OaiCompatConfig {
        let base = OaiCompatConfig::new("local-test", server.base_url(), "test-model")
            .with_auth_header(None, None)
            .with_retries(2)
            .with_total_timeout(Some(Duration::from_secs(5)))
            .with_first_byte_timeout(Some(Duration::from_secs(5)));
        configure(base)
    }

    fn task() -> Task {
        Task::system_user("系统提示", "用户输入")
            .with_sampling(Sampling::default().with_max_output_tokens(64))
            .with_hints(Hints::default().with_suppress_reasoning(true))
    }
}

#[tokio::test]
async fn complete_sends_the_expected_wire_request_and_parses_the_response() {
    let server = ScriptedServer::start(vec![Reply::Json {
        status: 200,
        body: completion_body("总结结果"),
        retry_after: None,
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| {
        cfg.with_token_field(TokenFieldPolicy::MaxTokens)
            .with_reasoning_suppression("chat_template_kwargs", json!({"enable_thinking": false}))
    });
    let harness = Harness::new(server, cfg, 0);

    let completion = harness
        .engine
        .complete(Harness::task())
        .await
        .expect("调用成功");
    assert_eq!(completion.text, "总结结果");
    assert_eq!(completion.stop, StopReason::Completed);
    assert_eq!(completion.usage.unwrap().total_tokens, Some(8));

    assert_eq!(
        harness.server.request_line(0),
        "POST /v1/chat/completions HTTP/1.1",
        "base 路径必须被拼上"
    );

    let body = &harness.server.json_bodies()[0];
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "系统提示");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["max_tokens"], 64, "策略为 MaxTokens 时只发 max_tokens");
    assert!(body.get("max_completion_tokens").is_none());
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": false}),
        "provider 扩展必须真的出现在网线上"
    );
    assert!(body.get("temperature").is_none(), "未指定温度时不应发送");
    assert!(body.get("stream").is_none(), "非流式请求不应带 stream");

    assert_eq!(harness.observer.ends().len(), 1);
    assert!(harness.observer.ends()[0].ok);
    harness.server.shutdown();
}

#[tokio::test]
async fn token_field_policy_switches_the_wire_field() {
    let server = ScriptedServer::start(vec![Reply::Json {
        status: 200,
        body: completion_body("ok"),
        retry_after: None,
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| {
        cfg.with_token_field(TokenFieldPolicy::MaxCompletionTokens)
    });
    let harness = Harness::new(server, cfg, 0);

    harness
        .engine
        .complete(Harness::task())
        .await
        .expect("调用成功");

    let body = &harness.server.json_bodies()[0];
    assert_eq!(body["max_completion_tokens"], 64);
    assert!(body.get("max_tokens").is_none(), "两个字段绝不能同时出现");
    harness.server.shutdown();
}

#[tokio::test]
async fn retries_a_429_and_returns_the_second_response() {
    let server = ScriptedServer::start(vec![
        Reply::Json {
            status: 429,
            body: json!({"error": {"message": "rate limited", "type": "rate_limit_exceeded"}})
                .to_string(),
            // 明确给出 0，测试不必真的等
            retry_after: Some(0),
        },
        Reply::Json {
            status: 200,
            body: completion_body("第二次成功"),
            retry_after: None,
        },
    ])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    let harness = Harness::new(server, cfg, 0);

    let completion = harness
        .engine
        .complete(Harness::task())
        .await
        .expect("重试后成功");
    assert_eq!(completion.text, "第二次成功");
    assert_eq!(harness.server.request_count(), 2, "必须真的重发");

    let retries = harness.observer.retries();
    assert_eq!(retries.len(), 1);
    assert_eq!(retries[0].layer, RetryLayer::Transport);
    assert_eq!(retries[0].delay_ms, 0, "应遵守 Retry-After: 0");
    assert_eq!(
        retries[0].kind,
        LlmErrorKind::RateLimited,
        "429 应归类为限流"
    );

    let ends = harness.observer.ends();
    assert_eq!(ends.len(), 1, "一次语义调用只上报一次结束");
    assert_eq!(ends[0].inner_retries, 1);
    assert_eq!(ends[0].whole_call_retries, 0);
    harness.server.shutdown();
}

#[tokio::test]
async fn retries_a_500_then_surfaces_exhaustion_with_status() {
    let server = ScriptedServer::start(vec![
        Reply::Json {
            status: 500,
            body: json!({"error": {"message": "boom"}}).to_string(),
            retry_after: None,
        },
        Reply::Json {
            status: 500,
            body: json!({"error": {"message": "boom again"}}).to_string(),
            retry_after: None,
        },
    ])
    .await;

    // 传输层预算 1 → 共 2 次请求（退避 200ms，可接受）；整调用预算 0 → 不再放大
    let cfg = Harness::config(&server, |cfg| cfg.with_retries(1));
    let harness = Harness::new(server, cfg, 0);

    let error = harness
        .engine
        .complete(Harness::task())
        .await
        .expect_err("应失败");
    assert_eq!(error.kind(), LlmErrorKind::ServerError);
    assert_eq!(error.status(), Some(500));
    assert_eq!(harness.server.request_count(), 2, "首次 + 1 次重试");
    assert_eq!(harness.observer.retries().len(), 1);
    harness.server.shutdown();
}

#[tokio::test]
async fn does_not_retry_a_400() {
    let server = ScriptedServer::start(vec![Reply::Json {
        status: 400,
        body: json!({"error": {"message": "unknown parameter", "type": "invalid_request_error", "param": "max_tokens"}})
            .to_string(),
        retry_after: None,
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    let harness = Harness::new(server, cfg, 2);

    let error = harness
        .engine
        .complete(Harness::task())
        .await
        .expect_err("应失败");
    assert_eq!(error.kind(), LlmErrorKind::BadRequest);
    assert!(error.message().contains("max_tokens"));
    assert_eq!(harness.server.request_count(), 1, "4xx 不该重发");
    assert!(harness.observer.retries().is_empty());
    harness.server.shutdown();
}

#[tokio::test]
async fn total_timeout_is_classified_as_timeout() {
    let server = ScriptedServer::start(vec![Reply::SlowJson {
        delay_ms: 400,
        status: 200,
        body: completion_body("太慢了"),
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| {
        cfg.with_total_timeout(Some(Duration::from_millis(50)))
    });
    let harness = Harness::new(server, cfg, 0);

    let started = std::time::Instant::now();
    let error = harness
        .engine
        .complete(Harness::task())
        .await
        .expect_err("应超时");
    assert_eq!(error.kind(), LlmErrorKind::Timeout);
    assert!(
        started.elapsed() < Duration::from_millis(350),
        "总时限必须真的生效，而不是等服务器慢慢回"
    );
    harness.server.shutdown();
}

/// 总时限超时是"传输层看不到的失败"，必须由整调用重试补上。
#[tokio::test]
async fn total_timeout_is_reissued_by_the_whole_call_layer() {
    let server = ScriptedServer::start(vec![
        Reply::SlowJson {
            delay_ms: 400,
            status: 200,
            body: completion_body("太慢了"),
        },
        Reply::Json {
            status: 200,
            body: completion_body("第二次及时"),
            retry_after: None,
        },
    ])
    .await;

    let cfg = Harness::config(&server, |cfg| {
        cfg.with_total_timeout(Some(Duration::from_millis(80)))
    });
    let harness = Harness::new(server, cfg, 1);

    let completion = harness
        .engine
        .complete(Harness::task())
        .await
        .expect("重发后成功");
    assert_eq!(completion.text, "第二次及时");
    assert_eq!(harness.server.request_count(), 2);

    let retries = harness.observer.retries();
    assert_eq!(retries.len(), 1);
    assert_eq!(retries[0].layer, RetryLayer::WholeCall);
    assert_eq!(retries[0].kind, LlmErrorKind::Timeout);
    harness.server.shutdown();
}

#[tokio::test]
async fn streaming_parses_sse_into_semantic_events() {
    let server = ScriptedServer::start(vec![Reply::Sse {
        chunks: vec![
            chunk(json!({"role": "assistant"}), None),
            chunk(json!({"reasoning_content": "先想"}), None),
            chunk(json!({"content": "你"}), None),
            chunk(json!({"content": "好"}), None),
            chunk(json!({}), Some("stop")),
            json!({
                "id": "x",
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            })
            .to_string(),
        ],
        done: true,
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    let harness = Harness::new(server, cfg, 0);

    let stream = harness
        .engine
        .stream(Harness::task())
        .await
        .expect("开流成功");
    let (events, error) = crate::mock::drain(stream).await;
    assert!(error.is_none(), "不该有错误：{error:?}");

    let mut deltas = Vec::new();
    let mut reasoning = Vec::new();
    let mut done = None;
    for event in &events {
        match event {
            StreamEvent::Delta(text) => deltas.push(text.clone()),
            StreamEvent::ReasoningDelta(text) => reasoning.push(text.clone()),
            StreamEvent::Done(completion) => done = Some((**completion).clone()),
        }
    }
    assert_eq!(deltas, vec!["你", "好"]);
    assert_eq!(reasoning, vec!["先想"]);
    let done = done.expect("必须有 Done");
    assert_eq!(done.text, "你好");
    assert_eq!(done.stop, StopReason::Completed);
    assert_eq!(
        done.usage.unwrap().total_tokens,
        Some(6),
        "usage 必须从流里捞到"
    );

    let body = &harness.server.json_bodies()[0];
    assert_eq!(body["stream"], true, "流式请求必须显式带 stream");

    let ends = harness.observer.ends();
    assert_eq!(ends.len(), 1);
    assert!(ends[0].ok);
    assert_eq!(ends[0].text_chars, 2);
    harness.server.shutdown();
}

/// 服务端省略 `[DONE]` 与 `finish_reason` 时，有内容就保留内容、把停止原因标为 unknown。
///
/// 这是 async-openai 解析层的固有歧义：`[DONE]` 与"连接正常关闭"都表现为流结束，
/// 无法区分。宁可如实标注 unknown，也不伪装成正常结束或丢掉整段生成。
#[tokio::test]
async fn streaming_with_content_but_no_done_keeps_content_with_unknown_stop() {
    let server = ScriptedServer::start(vec![
        Reply::Sse {
            chunks: vec![chunk(json!({"content": "只有内容"}), None)],
            done: false,
        },
        Reply::Json {
            status: 200,
            body: completion_body("不该被用到"),
            retry_after: None,
        },
    ])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    let harness = Harness::new(server, cfg, 3);

    let stream = harness
        .engine
        .stream(Harness::task())
        .await
        .expect("开流成功");
    let (events, error) = crate::mock::drain(stream).await;
    assert!(error.is_none(), "有内容时不得判为中断：{error:?}");

    let done = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Done(completion) => Some((**completion).clone()),
            _ => None,
        })
        .expect("必须有 Done");
    assert_eq!(done.text, "只有内容");
    assert_eq!(done.stop, StopReason::Unknown);
    assert_eq!(harness.server.request_count(), 1, "有内容时不得重放");
    harness.server.shutdown();
}

/// 一点内容都没产出就断掉：这更像中断而不是"空回答"，且应当可重试。
#[tokio::test]
async fn streaming_that_ends_with_nothing_is_reported_as_interrupted() {
    let server = ScriptedServer::start(vec![Reply::Sse {
        chunks: vec![],
        done: false,
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    let harness = Harness::new(server, cfg, 0);

    let stream = harness
        .engine
        .stream(Harness::task())
        .await
        .expect("开流成功");
    let (events, error) = crate::mock::drain(stream).await;
    assert!(events.is_empty());

    let error = error.expect("必须报错而不是给一个空结果");
    assert_eq!(error.kind(), LlmErrorKind::StreamInterrupted);
    assert!(error.is_retryable(), "未产出内容时重放是安全的");
    harness.server.shutdown();
}

/// 已经产出内容后断流：错误必须带上半截文本，且**不可**被重放。
#[tokio::test]
async fn streaming_interrupted_after_content_is_not_retryable() {
    let server = ScriptedServer::start(vec![
        Reply::TruncatedSse {
            chunks: vec![chunk(json!({"content": "已经说了一半"}), None)],
            declared_length: 4096,
        },
        Reply::Json {
            status: 200,
            body: completion_body("不该被用到"),
            retry_after: None,
        },
    ])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    // 整调用预算开着：即便如此也不允许重放（否则调用方会看到重复文本）
    let harness = Harness::new(server, cfg, 3);

    let stream = harness
        .engine
        .stream(Harness::task())
        .await
        .expect("开流成功");
    let (events, error) = crate::mock::drain(stream).await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Delta(t) if t == "已经说了一半")),
        "中断前产出的增量必须已经交给调用方"
    );

    let error = error.expect("读取失败必须报错");
    assert_eq!(error.kind(), LlmErrorKind::StreamInterrupted);
    assert_eq!(
        error.partial(),
        Some("已经说了一半"),
        "半截文本必须随错误带出，供调用方决定丢弃还是保留"
    );
    assert!(
        !error.is_retryable(),
        "已产出内容的中断绝不可重放——这正是原来会静默损坏记忆的路径"
    );
    assert_eq!(
        harness.server.request_count(),
        1,
        "整调用预算再大也不得重发已产出内容的流"
    );

    let ends = harness.observer.ends();
    assert_eq!(ends.len(), 1);
    assert!(!ends[0].ok);
    assert_eq!(ends[0].kind, Some(LlmErrorKind::StreamInterrupted));
    assert_eq!(ends[0].text_chars, 6, "\"已经说了一半\" 是 6 个字符");
    harness.server.shutdown();
}

#[tokio::test]
async fn invalid_json_error_body_still_yields_a_usable_error() {
    let server = ScriptedServer::start(vec![Reply::Json {
        status: 400,
        // 反代常见的 HTML 错误页：不是 OpenAI 形状
        body: "<html><body>Bad Request</body></html>".to_string(),
        retry_after: None,
    }])
    .await;

    let cfg = Harness::config(&server, |cfg| cfg);
    let harness = Harness::new(server, cfg, 0);

    let error = harness
        .engine
        .complete(Harness::task())
        .await
        .expect_err("应失败");
    assert_eq!(
        error.kind(),
        LlmErrorKind::Decode,
        "非 OpenAI 形状的错误体只能归到 Decode（status 在解析路径上已丢失）"
    );
    assert!(
        error.message().contains("Bad Request"),
        "原始错误体必须带出来供排障：{}",
        error.message()
    );
    harness.server.shutdown();
}
