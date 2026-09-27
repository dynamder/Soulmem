//! 测试辅助：脚本化后端 + 由它构造的引擎，并提供调用记录。
//!
//! 刻意不做成 `soul-mem-llm` 的公共 feature：目前只有本 crate 需要它，
//! 等第二个 crate 也需要时再提升（YAGNI）。

use super::*;
use async_trait::async_trait;
use parking_lot::Mutex;
use soul_mem_llm::{BackendInfo, ChatBackend, Completion, EventStream};

pub enum Outcome {
    Text(&'static str),
    Fail(LlmErrorKind),
}

struct ScriptedBackend {
    outcomes: Mutex<VecDeque<Outcome>>,
    seen: Mutex<Vec<Task>>,
}

/// 引擎 + 其背后的脚本后端（测试直接持有两者，无需任何 downcast 接口）。
pub struct Scripted {
    engine: Arc<LlmEngine>,
    backend: Arc<ScriptedBackend>,
}

impl Scripted {
    /// 关闭引擎的整调用重试：本模块测的是"窗口在摘要失败时如何处置"，
    /// 重试属于 `soul-mem-llm` 的职责，混在一起会让断言含糊。
    pub fn new(outcomes: Vec<Outcome>) -> Self {
        Self::with_whole_call_retries(outcomes, 0)
    }

    /// 指定引擎的整调用重试预算（用于验证两层的衔接）。
    pub fn with_whole_call_retries(outcomes: Vec<Outcome>, retries: u32) -> Self {
        let backend = Arc::new(ScriptedBackend {
            outcomes: Mutex::new(outcomes.into()),
            seen: Mutex::new(Vec::new()),
        });
        let engine = Arc::new(LlmEngine::new(backend.clone()).with_whole_call_retries(retries));
        Self { engine, backend }
    }

    pub fn engine(&self) -> &Arc<LlmEngine> {
        &self.engine
    }

    /// 真实发生的 LLM 调用次数。
    pub fn calls(&self) -> usize {
        self.backend.seen.lock().len()
    }

    /// 最近一次调用的提示词（按角色逐行展开，便于断言）。
    pub fn last_prompt(&self) -> Option<String> {
        let seen = self.backend.seen.lock();
        seen.last().map(|task| {
            task.messages
                .iter()
                .map(|message| format!("[{}] {}", message.role.as_str(), message.text))
                .collect::<Vec<_>>()
                .join("\n")
        })
    }
}

#[async_trait]
impl ChatBackend for ScriptedBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo::new("scripted")
    }

    async fn complete(&self, task: Task) -> Result<Completion, LlmError> {
        self.seen.lock().push(task);
        match self.outcomes.lock().pop_front() {
            Some(Outcome::Text(text)) => Ok(Completion::new(text)),
            Some(Outcome::Fail(kind)) => Err(LlmError::new(kind, "脚本指定的失败")),
            None => Err(LlmError::new(
                LlmErrorKind::Internal,
                "脚本已耗尽：本用例不应发起 LLM 调用",
            )),
        }
    }

    async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
        Err(LlmError::unsupported("摘要任务不需要流式"))
    }
}
