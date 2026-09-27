//! 调用观测：三个语义事件（开始 / 重试 / 结束）。
//!
//! 事件里只放"模型 / 耗时 / token / 重试 / 错误"这类排障必需的信息，
//! 不放 wire 字段、也不放提示词正文——任何后端都能填满，且不会把用户数据写进日志。
//!
//! # 每个事件由谁产生
//!
//! | 事件 | 产生点 | 触发时机 |
//! |---|---|---|
//! | [`CallStart`] | `engine.rs::LlmEngine::begin` | 进入一次语义调用时 |
//! | [`RetryEvent`]（`Transport` 层） | `oai_comp::transport` 的重试策略，经 `ctx::CallCtx::note_inner_retry` | 传输层决定重试时 |
//! | [`RetryEvent`]（`WholeCall` 层） | `engine.rs::LlmEngine::with_whole_call_retry`，经 `ctx::CallCtx::note_whole_retry` | 引擎决定重发整次调用时 |
//! | [`CallEnd`] | `engine.rs`（非流式）/ `engine.rs::ObservedStream`（流式） | 调用结束、流出错、或流被提前丢弃 |
//!
//! 内层重试发生在 tower 服务内部，是**跨模块**上报：它通过 `ctx` 的 task-local 上下文
//! 找到当前调用的 `CallCtx`（见 `ctx.rs` 的模块注释）。
//!
//! # 用法
//!
//! ```no_run
//! # use soul_mem_llm::{JsonlObserver, LlmEngine};
//! # fn build(
//! #     backend: std::sync::Arc<dyn soul_mem_llm::ChatBackend>,
//! # ) -> Result<LlmEngine, std::io::Error> {
//! let observer = JsonlObserver::create("llm-trace.jsonl")?; // 一行一条 JSON，写完即 flush
//! Ok(LlmEngine::new(backend).with_observer(observer))
//! # }
//! ```

use crate::backend::{StopReason, Usage};
use crate::error::LlmErrorKind;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// 重试发生的位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryLayer {
    /// 传输层：状态码 / 连接失败，发生在"发出请求到拿到响应头"之间。
    Transport,
    /// 整调用：超时、响应体读取失败、流未产出即中断。
    WholeCall,
}

impl RetryLayer {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::WholeCall => "whole_call",
        }
    }
}

/// 一次调用的开始。
#[derive(Debug, Clone)]
pub struct CallStart<'a> {
    pub call_id: &'a str,
    pub backend: &'a str,
    pub model: Option<&'a str>,
    pub streaming: bool,
}

/// 一次重试。
#[derive(Debug, Clone)]
pub struct RetryEvent<'a> {
    pub call_id: &'a str,
    pub layer: RetryLayer,
    /// 该层第几次重试（从 1 开始）。
    pub attempt: u32,
    pub delay_ms: u64,
    pub kind: LlmErrorKind,
}

/// 一次调用的结束。流式调用在流终止（正常结束 / 出错 / 被消费方提前丢弃）时触发。
#[derive(Debug, Clone)]
pub struct CallEnd<'a> {
    pub call_id: &'a str,
    pub ok: bool,
    pub kind: Option<LlmErrorKind>,
    pub latency_ms: u64,
    pub usage: Option<Usage>,
    pub stop: Option<StopReason>,
    /// 产出的字符数（流式按增量累加）。
    pub text_chars: usize,
    pub inner_retries: u32,
    pub whole_call_retries: u32,
}

/// 观测回调。实现必须是廉价且不 panic 的：它在调用路径上被同步调用。
pub trait LlmObserver: Send + Sync {
    fn on_start(&self, _event: &CallStart<'_>) {}
    fn on_retry(&self, _event: &RetryEvent<'_>) {}
    fn on_end(&self, _event: &CallEnd<'_>) {}
}

/// 空实现：不观测。
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopObserver;

impl LlmObserver for NoopObserver {}

/// 一行一条 JSON 的落盘观测。
///
/// 每行写完立即 flush：进程崩溃时 trace 仍然完整（那正是最需要它的时刻）。
/// 事件里本来就没有密钥与正文，因此不会泄露。
#[derive(Debug)]
pub struct JsonlObserver {
    file: Mutex<BufWriter<File>>,
}

impl JsonlObserver {
    /// 追加模式打开（或创建）trace 文件。
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Arc<Self>> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Arc::new(Self {
            file: Mutex::new(BufWriter::new(file)),
        }))
    }

    fn write_line(&self, value: &Value) {
        let mut guard = self.file.lock();
        // 写失败不 panic：观测不能反过来把调用搞崩
        if serde_json::to_writer(&mut *guard, value).is_ok() {
            let _ = guard.write_all(b"\n");
            let _ = guard.flush();
        }
    }
}

impl LlmObserver for JsonlObserver {
    fn on_start(&self, event: &CallStart<'_>) {
        let mut obj = base("start");
        put(&mut obj, "call_id", event.call_id);
        put(&mut obj, "backend", event.backend);
        put_opt(&mut obj, "model", event.model);
        put(&mut obj, "streaming", event.streaming);
        self.write_line(&Value::Object(obj));
    }

    fn on_retry(&self, event: &RetryEvent<'_>) {
        let mut obj = base("retry");
        put(&mut obj, "call_id", event.call_id);
        put(&mut obj, "layer", event.layer.as_str());
        put(&mut obj, "attempt", event.attempt);
        put(&mut obj, "delay_ms", event.delay_ms);
        put(&mut obj, "kind", event.kind.as_str());
        self.write_line(&Value::Object(obj));
    }

    fn on_end(&self, event: &CallEnd<'_>) {
        let mut obj = base("end");
        put(&mut obj, "call_id", event.call_id);
        put(&mut obj, "ok", event.ok);
        put_opt(&mut obj, "kind", event.kind.map(|k| k.as_str()));
        put(&mut obj, "latency_ms", event.latency_ms);
        if let Some(usage) = event.usage {
            let mut u = Map::new();
            put_opt(&mut u, "prompt", usage.prompt_tokens);
            put_opt(&mut u, "completion", usage.completion_tokens);
            put_opt(&mut u, "total", usage.total_tokens);
            if !u.is_empty() {
                put(&mut obj, "usage", Value::Object(u));
            }
        }
        put_opt(&mut obj, "stop", event.stop.map(|s| s.as_str()));
        put(&mut obj, "text_chars", event.text_chars);
        put(&mut obj, "inner_retries", event.inner_retries);
        put(&mut obj, "whole_call_retries", event.whole_call_retries);
        self.write_line(&Value::Object(obj));
    }
}

fn base(event: &str) -> Map<String, Value> {
    let mut obj = Map::new();
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default();
    put(&mut obj, "ts_ms", ts_ms);
    put(&mut obj, "ev", event);
    obj
}

fn put<V: Into<Value>>(obj: &mut Map<String, Value>, key: &str, value: V) {
    obj.insert(key.to_string(), value.into());
}

fn put_opt<V: Into<Value>>(obj: &mut Map<String, Value>, key: &str, value: Option<V>) {
    if let Some(value) = value {
        put(obj, key, value);
    }
}

/// 记录型观测器：把事件留在内存里供断言。
#[cfg(test)]
#[derive(Debug, Default)]
pub struct RecordingObserver {
    starts: Mutex<Vec<StartRecord>>,
    retries: Mutex<Vec<RetryRecord>>,
    ends: Mutex<Vec<EndRecord>>,
}

/// 开始事件的记录。
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartRecord {
    pub call_id: String,
    pub backend: String,
    pub model: Option<String>,
    pub streaming: bool,
}

/// 重试事件的记录。
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryRecord {
    pub layer: RetryLayer,
    pub attempt: u32,
    pub delay_ms: u64,
    pub kind: LlmErrorKind,
}

/// 结束事件的记录。
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndRecord {
    pub ok: bool,
    pub kind: Option<LlmErrorKind>,
    pub latency_ms: u64,
    pub usage: Option<Usage>,
    pub stop: Option<StopReason>,
    pub text_chars: usize,
    pub inner_retries: u32,
    pub whole_call_retries: u32,
}

#[cfg(test)]
impl RecordingObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn starts(&self) -> Vec<StartRecord> {
        self.starts.lock().clone()
    }

    pub fn retries(&self) -> Vec<RetryRecord> {
        self.retries.lock().clone()
    }

    pub fn ends(&self) -> Vec<EndRecord> {
        self.ends.lock().clone()
    }

    pub fn retry_count(&self) -> usize {
        self.retries.lock().len()
    }

    pub fn end_count(&self) -> usize {
        self.ends.lock().len()
    }
}

#[cfg(test)]
impl LlmObserver for RecordingObserver {
    fn on_start(&self, event: &CallStart<'_>) {
        self.starts.lock().push(StartRecord {
            call_id: event.call_id.to_string(),
            backend: event.backend.to_string(),
            model: event.model.map(str::to_owned),
            streaming: event.streaming,
        });
    }

    fn on_retry(&self, event: &RetryEvent<'_>) {
        self.retries.lock().push(RetryRecord {
            layer: event.layer,
            attempt: event.attempt,
            delay_ms: event.delay_ms,
            kind: event.kind,
        });
    }

    fn on_end(&self, event: &CallEnd<'_>) {
        self.ends.lock().push(EndRecord {
            ok: event.ok,
            kind: event.kind,
            latency_ms: event.latency_ms,
            usage: event.usage,
            stop: event.stop,
            text_chars: event.text_chars,
            inner_retries: event.inner_retries,
            whole_call_retries: event.whole_call_retries,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("soulmem-llm-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("临时目录");
        let path = dir.join(format!("{name}.jsonl"));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn read_lines(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .expect("trace 文件可读")
            .lines()
            .map(|line| serde_json::from_str(line).expect("每行都是合法 JSON"))
            .collect()
    }

    #[test]
    fn jsonl_records_three_event_kinds_and_omits_absent_fields() {
        let path = trace_path("three");
        let observer = JsonlObserver::create(&path).expect("创建 trace");

        observer.on_start(&CallStart {
            call_id: "c1",
            backend: "oai_compat",
            model: Some("qwen3-4b"),
            streaming: false,
        });
        observer.on_retry(&RetryEvent {
            call_id: "c1",
            layer: RetryLayer::Transport,
            attempt: 1,
            delay_ms: 300,
            kind: LlmErrorKind::RateLimited,
        });
        observer.on_end(&CallEnd {
            call_id: "c1",
            ok: false,
            kind: Some(LlmErrorKind::RateLimited),
            latency_ms: 1234,
            usage: None,
            stop: None,
            text_chars: 0,
            inner_retries: 1,
            whole_call_retries: 0,
        });

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["ev"], "start");
        assert_eq!(lines[0]["backend"], "oai_compat");
        assert_eq!(lines[0]["model"], "qwen3-4b");
        assert_eq!(lines[1]["ev"], "retry");
        assert_eq!(lines[1]["layer"], "transport");
        assert_eq!(lines[1]["kind"], "rate_limited");
        assert_eq!(lines[1]["delay_ms"], 300);
        assert_eq!(lines[2]["ev"], "end");
        assert_eq!(lines[2]["ok"], false);
        assert_eq!(lines[2]["inner_retries"], 1);
        assert!(lines[2].get("usage").is_none(), "没用量时不写空对象");
        assert!(lines[2].get("stop").is_none());
        assert!(lines.iter().all(|line| line.get("ts_ms").is_some()));
    }

    #[test]
    fn jsonl_records_usage_and_stop_when_present() {
        let path = trace_path("usage");
        let observer = JsonlObserver::create(&path).expect("创建 trace");

        observer.on_end(&CallEnd {
            call_id: "c2",
            ok: true,
            kind: None,
            latency_ms: 42,
            usage: Some(Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(3),
                total_tokens: Some(13),
            }),
            stop: Some(StopReason::Completed),
            text_chars: 7,
            inner_retries: 0,
            whole_call_retries: 1,
        });

        let lines = read_lines(&path);
        assert_eq!(lines[0]["usage"]["total"], 13);
        assert_eq!(lines[0]["usage"]["prompt"], 10);
        assert_eq!(lines[0]["stop"], "completed");
        assert_eq!(lines[0]["text_chars"], 7);
        assert_eq!(lines[0]["whole_call_retries"], 1);
        assert!(lines[0].get("kind").is_none(), "成功时没有错误类别");
    }

    #[test]
    fn jsonl_appends_rather_than_truncating() {
        let path = trace_path("append");
        for _ in 0..2 {
            let observer = JsonlObserver::create(&path).expect("创建 trace");
            observer.on_start(&CallStart {
                call_id: "c",
                backend: "b",
                model: None,
                streaming: true,
            });
        }
        assert_eq!(read_lines(&path).len(), 2, "重复创建必须追加而不是覆盖");
    }

    #[test]
    fn recording_observer_captures_all_three_event_kinds() {
        let observer = RecordingObserver::new();
        observer.on_start(&CallStart {
            call_id: "c1",
            backend: "b",
            model: None,
            streaming: false,
        });
        observer.on_retry(&RetryEvent {
            call_id: "c1",
            layer: RetryLayer::WholeCall,
            attempt: 2,
            delay_ms: 500,
            kind: LlmErrorKind::Timeout,
        });
        observer.on_end(&CallEnd {
            call_id: "c1",
            ok: true,
            kind: None,
            latency_ms: 9,
            usage: None,
            stop: Some(StopReason::Completed),
            text_chars: 1,
            inner_retries: 0,
            whole_call_retries: 2,
        });

        assert_eq!(observer.starts().len(), 1);
        assert_eq!(observer.retry_count(), 1);
        assert_eq!(observer.ends().len(), 1);
        assert_eq!(observer.retries()[0].layer, RetryLayer::WholeCall);
        assert_eq!(observer.ends()[0].whole_call_retries, 2);
        assert_eq!(observer.end_count(), 1);
    }
}
