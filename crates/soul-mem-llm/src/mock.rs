//! 脚本化的测试后端。
//!
//! 让"重试、分类、流中断、观测"这些最该被验证的逻辑**不需要真机**：
//! 每次调用按脚本弹出下一个结果，因此可以精确编排"第一次失败、第二次成功"。
//! 脚本耗尽时报错而不是循环复用最后一个结果——静默复用会让测试看起来通过而实际行为错乱。

use crate::backend::{
    BackendInfo, ChatBackend, Completion, EventStream, StopReason, StreamEvent, Task,
};
use crate::error::{LlmError, LlmErrorKind};
use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use std::collections::VecDeque;

/// 脚本里的一步。
#[derive(Debug)]
pub enum MockOutcome {
    /// 返回这段文本。
    Text(String),
    /// 返回思考内容 + 正文。
    Reasoned { reasoning: String, text: String },
    /// 返回一个失败（用于编排重试与降级）。
    Fail(LlmError),
    /// 返回这些流式事件后正常结束（**不带 `Done`**，用于验证后端契约）。
    Stream(Vec<StreamEvent>),
    /// 先产出这些事件，再以错误终止（用于验证流中断语义）。
    StreamThenFail {
        events: Vec<StreamEvent>,
        error: LlmError,
    },
}

/// 记录收到的任务并按脚本作答的后端。
#[derive(Debug)]
pub struct MockBackend {
    info: BackendInfo,
    script: Mutex<VecDeque<MockOutcome>>,
    seen: Mutex<Vec<Task>>,
}

impl MockBackend {
    pub fn new(info: BackendInfo, script: Vec<MockOutcome>) -> Self {
        Self {
            info,
            script: Mutex::new(script.into()),
            seen: Mutex::new(Vec::new()),
        }
    }

    /// 至今收到的全部任务（按顺序）。
    pub fn seen(&self) -> Vec<Task> {
        self.seen.lock().clone()
    }

    pub fn call_count(&self) -> usize {
        self.seen.lock().len()
    }

    fn take(&self) -> Result<MockOutcome, LlmError> {
        self.script.lock().pop_front().ok_or_else(|| {
            LlmError::new(
                LlmErrorKind::Internal,
                "mock 脚本已耗尽：请为每次预期调用准备一个结果",
            )
        })
    }
}

fn text_stream(events: Vec<StreamEvent>) -> EventStream {
    Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
}

fn failing_stream(events: Vec<StreamEvent>, error: LlmError) -> EventStream {
    Box::pin(
        futures::stream::iter(events.into_iter().map(Ok))
            .chain(futures::stream::once(async move { Err(error) })),
    )
}

#[async_trait]
impl ChatBackend for MockBackend {
    fn info(&self) -> BackendInfo {
        self.info.clone()
    }

    async fn complete(&self, task: Task) -> Result<Completion, LlmError> {
        self.seen.lock().push(task);
        match self.take()? {
            MockOutcome::Text(text) => Ok(Completion::new(text)),
            MockOutcome::Reasoned { reasoning, text } => Ok(Completion {
                text,
                reasoning: Some(reasoning),
                stop: StopReason::Completed,
                usage: None,
            }),
            MockOutcome::Fail(error) => Err(error),
            MockOutcome::Stream(_) | MockOutcome::StreamThenFail { .. } => Err(LlmError::new(
                LlmErrorKind::Internal,
                "mock 脚本给的是流式结果，却调用了 complete",
            )),
        }
    }

    async fn stream(&self, task: Task) -> Result<EventStream, LlmError> {
        self.seen.lock().push(task);
        if !self.info.supports_streaming {
            return Err(LlmError::unsupported("该后端未启用流式"));
        }
        match self.take()? {
            MockOutcome::Stream(events) => Ok(text_stream(events)),
            MockOutcome::StreamThenFail { events, error } => Ok(failing_stream(events, error)),
            MockOutcome::Fail(error) => Err(error),
            MockOutcome::Text(text) => {
                let completion = Completion::new(text.clone());
                Ok(text_stream(vec![
                    StreamEvent::Delta(text),
                    StreamEvent::Done(Box::new(completion)),
                ]))
            }
            MockOutcome::Reasoned { reasoning, text } => {
                let completion = Completion::new(text.clone());
                Ok(text_stream(vec![
                    StreamEvent::ReasoningDelta(reasoning),
                    StreamEvent::Delta(text),
                    StreamEvent::Done(Box::new(completion)),
                ]))
            }
        }
    }
}

/// 便于测试：把流消费成 `(事件序列, 结果)`。
pub async fn drain(mut stream: EventStream) -> (Vec<StreamEvent>, Option<LlmError>) {
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(error) => return (events, Some(error)),
        }
    }
    (events, None)
}
