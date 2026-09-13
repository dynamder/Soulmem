//! 滑动窗口与累加摘要。
//!
//! # LLM 调用去哪了
//!
//! 本模块只负责"什么时候该摘要、失败怎么办"，**不接触任何传输细节**：摘要通过
//! `soul_mem_llm::LlmEngine::complete` 发出（见 `SlidingWindow::summarize`，本文件私有），
//! 请求编码、超时、重试、错误分类、trace 都在 `soul-mem-llm` 里。
//! 完整链路见 `docs/architecture/llm-layer.md`。
//!
//! # 摘要失败的语义（本次重构修正）
//!
//! 旧实现是"先把消息出队，再调用 LLM 摘要"：一旦摘要失败，那条消息就永久消失，
//! 摘要也不再累加——历史被静默丢掉，而且没有任何信号。
//!
//! 现在改成**只有摘要成功才移除消息**：容量因此成为**软上限**——摘要失败时窗口可以
//! 短暂超出容量，下一次成功后再回落。代价是窗口大小不再严格等于容量；换来的是
//! "失败可以重试、数据不会消失"。
//!
//! # 其它
//!
//! - 摘要更新是"读旧摘要 → 调 LLM → 覆写摘要"的复合操作，且摘要是**累加**的。
//!   若不串行化，两个并发 push 各自读到同一份旧摘要、后写覆盖前写，被淘汰的历史同样永久丢失。
//!   这里用异步互斥锁把整个复合操作串行化（`parking_lot` 锁依旧一律不跨 await 持有）。
//! - 移除前会确认队首仍是那条消息，避免并发下删错对象。

use parking_lot::RwLock as ParkRwLock;
use soul_mem_llm::{Hints, LlmEngine, LlmError, LlmErrorKind, Role, Task};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 摘要任务的 system 提示。
const SUMMARY_SYSTEM_PROMPT: &str = "You are a summary and compact agent. \
    Based on the following conversation (which is happened before), provide a new summary.\n \
    Only Output the summary content, no other text.";

/// 滑动窗口（容器、容量、标记计数、摘要）。
#[derive(Debug)]
pub struct SlidingWindow {
    window: Arc<ParkRwLock<VecDeque<Information>>>,
    capacity: AtomicUsize,
    tag_count: AtomicUsize,
    summary: Arc<ParkRwLock<Summary>>,
    /// 摘要更新的串行化闸门（见模块注释）。只用于串行化该复合操作，不保护共享数据本身。
    summarize_gate: Arc<tokio::sync::Mutex<()>>,
}

impl Default for SlidingWindow {
    fn default() -> Self {
        Self::new(20)
    }
}

impl SlidingWindow {
    /// 新建。
    ///
    /// 不再在这里调用 `dotenv()`：构造函数不该有改变进程全局状态的副作用
    /// （且 `dotenvy` 不覆盖已有环境变量，配置会"看起来生效其实没有"）。
    pub fn new(capacity: usize) -> Self {
        Self {
            window: Arc::new(ParkRwLock::new(VecDeque::with_capacity(capacity + 1))),
            capacity: AtomicUsize::from(capacity),
            // 从 0 开始计数，与 clear() 及"每 capacity 次标记一次"的语义一致
            tag_count: AtomicUsize::from(0),
            summary: Arc::new(ParkRwLock::new(Summary::new())),
            summarize_gate: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// 信息滑入。超出容量时淘汰队首（被标记的队首必须先摘要成功）。
    pub async fn push(&self, value: &str, role: &str, engine: &LlmEngine) -> Result<(), LlmError> {
        let text = self.auto_tag(Information::new(value, role));
        self.window.write().push_back(text);
        self.enforce_capacity(engine).await
    }

    /// 信息滑出。被标记的队首在摘要成功后才移除。
    pub async fn pop(&self, engine: &LlmEngine) -> Result<(), LlmError> {
        let Some(front) = self.window.read().front().cloned() else {
            return Ok(());
        };
        if front.is_tagged() {
            self.summarize(engine, Some(&front)).await?;
        }
        self.remove_front(&front);
        Ok(())
    }

    /// 超出容量时淘汰队首，每次调用最多淘汰一条（与旧行为一致）。
    async fn enforce_capacity(&self, engine: &LlmEngine) -> Result<(), LlmError> {
        let capacity = self.capacity.load(Ordering::Acquire);
        let candidate = {
            let window = self.window.read();
            if window.len() <= capacity {
                return Ok(());
            }
            window.front().cloned()
        };
        let Some(candidate) = candidate else {
            return Ok(());
        };

        if candidate.is_tagged() {
            // 摘要是累加的，失败就必须保留消息：这里直接向上报错，
            // 由调用方决定是重试还是降级；窗口暂时处于超出容量的状态。
            self.summarize(engine, Some(&candidate)).await?;
        }
        self.remove_front(&candidate);
        Ok(())
    }

    /// 只在队首仍是这条消息时移除，返回是否真的移除了。
    ///
    /// 摘要期间窗口可能已被其他任务改动（异步锁不保护 window 本身），
    /// 无条件 `pop_front` 会删掉一条无关消息。
    fn remove_front(&self, expected: &Information) -> bool {
        let mut window = self.window.write();
        if window.front() == Some(expected) {
            window.pop_front();
            true
        } else {
            false
        }
    }

    pub fn window(&self) -> &Arc<ParkRwLock<VecDeque<Information>>> {
        &self.window
    }

    pub fn summary(&self) -> &Arc<ParkRwLock<Summary>> {
        &self.summary
    }

    pub fn get_windows(&self) -> Arc<[Information]> {
        let window = self.window.read();
        Arc::from(window.iter().cloned().collect::<Vec<_>>())
    }

    pub fn get_summary(&self) -> Arc<str> {
        Arc::from(self.summary.read().get())
    }

    /// 获取窗口大小。
    pub fn len(&self) -> usize {
        self.window.read().len()
    }

    /// 获取窗口容量。
    ///
    /// 注意这是**软上限**：摘要失败时实际长度可以超过它（见模块注释）。
    pub fn get_capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// 获取窗口容量（可变）。
    pub fn set_capacity(&self, val: usize) {
        self.capacity.store(val, Ordering::Release);
    }

    /// 获取窗口中指定索引的信息。
    pub fn get(&self, index: usize) -> Option<Information> {
        self.window.read().get(index).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.window.read().is_empty()
    }

    /// 清空窗口内容。
    pub fn clear(&self) {
        self.window.write().clear();
        self.tag_count.store(0, Ordering::Release);
    }

    /// 标记用。
    pub fn tag_information(&self, index: usize) {
        // 用 window 实际长度做边界检查，防止 pop 后索引越界 panic
        let mut window = self.window.write();
        if index < window.len() {
            window[index].tag_information();
        }
    }

    /// 取消标记用。
    pub fn untag_information(&self, index: usize) {
        let mut window = self.window.write();
        if index < window.len() {
            window[index].untag_information();
        }
    }

    /// 每滑入 capacity 次信息时进行一次标记。
    fn auto_tag(&self, mut value: Information) -> Information {
        let capacity = self.capacity.load(Ordering::Acquire);
        // 用 fetch_update 把"计数+判断+重置"合并为一次原子操作，避免并发下重复标记或漏标记
        let previous = self
            .tag_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                let next = count + 1;
                if next >= capacity {
                    Some(0) // 达到 capacity 时重置
                } else {
                    Some(next)
                }
            })
            .unwrap_or(0);
        if previous + 1 >= capacity {
            value.tag_information();
        }
        value
    }

    /// 把摘要记忆与当前滑动窗口信息合并成一次语义任务。
    fn prepare_task(&self, popped: Option<&Information>) -> Task {
        let mut snapshot = String::new();
        push_line(&mut snapshot, Role::Assistant, self.summary.read().get());
        // 被弹出的被标记消息也要纳入摘要，否则其内容会丢失
        if let Some(popped) = popped {
            push_line(&mut snapshot, popped.role(), popped.get_str());
        }
        for info in self.window.read().iter() {
            push_line(&mut snapshot, info.role(), info.get_str());
        }

        Task::system_user(SUMMARY_SYSTEM_PROMPT, snapshot)
            // 摘要要的是摘要本身，不是思考过程：对推理模型显式要求抑制 thinking
            .with_hints(Hints::default().with_suppress_reasoning(true))
    }

    /// 调用 LLM 更新摘要。
    ///
    /// 失败时**不修改任何状态**：消息不移除、旧摘要不被覆盖。
    async fn summarize(
        &self,
        engine: &LlmEngine,
        popped: Option<&Information>,
    ) -> Result<(), LlmError> {
        let _serialize = self.summarize_gate.lock().await;

        let task = self.prepare_task(popped);
        let completion = engine.complete(task).await?;

        // 空结果不能覆盖旧摘要：那等于把已经累积的历史清空
        if completion.text.trim().is_empty() {
            return Err(LlmError::new(
                LlmErrorKind::EmptyCompletion,
                "摘要结果为空，拒绝覆盖已有摘要",
            ));
        }
        self.summary.write().update(completion.text);
        Ok(())
    }
}

fn push_line(buffer: &mut String, role: Role, text: &str) {
    buffer.push_str(&format!("[{}]: {}\n", role.as_str(), text));
}

#[derive(Debug, Clone, PartialEq)]
pub enum Information {
    User(UserInformation),
    Assistant(AssistantInformation),
}

impl From<UserInformation> for Information {
    fn from(info: UserInformation) -> Self {
        Information::User(info)
    }
}

impl From<AssistantInformation> for Information {
    fn from(info: AssistantInformation) -> Self {
        Information::Assistant(info)
    }
}

impl Information {
    pub fn new(value: &str, role: &str) -> Self {
        // TODO: careful with this string compare
        match role {
            "user" => Information::User(UserInformation::new(value)),
            "assistant" => Information::Assistant(AssistantInformation::new(value)),
            _ => Information::User(UserInformation::new(value)),
        }
    }

    pub fn is_tagged(&self) -> bool {
        match self {
            Information::User(info) => info.tag,
            Information::Assistant(info) => info.tag,
        }
    }

    pub fn tag_information(&mut self) {
        match self {
            Information::User(info) => info.tag = true,
            Information::Assistant(info) => info.tag = true,
        }
    }

    pub fn untag_information(&mut self) {
        match self {
            Information::User(info) => info.tag = false,
            Information::Assistant(info) => info.tag = false,
        }
    }

    pub fn get_str(&self) -> &str {
        match self {
            Information::User(info) => &info.text,
            Information::Assistant(info) => &info.text,
        }
    }

    /// 语义角色。供拼接提示词用——不再暴露任何 async-openai 的消息类型。
    pub fn role(&self) -> Role {
        match self {
            Information::User(_) => Role::User,
            Information::Assistant(_) => Role::Assistant,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserInformation {
    pub text: Arc<str>,
    pub tag: bool,
}

impl UserInformation {
    pub fn new(text: &str) -> Self {
        Self {
            text: Arc::from(text),
            tag: false,
        }
    }

    pub fn get_str(&self) -> &str {
        &self.text
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantInformation {
    pub text: Arc<str>,
    pub tag: bool,
}

impl AssistantInformation {
    pub fn new(text: &str) -> Self {
        Self {
            text: Arc::from(text),
            tag: false,
        }
    }

    pub fn get_str(&self) -> &str {
        &self.text
    }
}

#[derive(Debug)]
pub struct Summary {
    summary: String,
}

impl Default for Summary {
    fn default() -> Self {
        Self::new()
    }
}

impl Summary {
    pub fn new() -> Self {
        Self {
            summary: String::new(),
        }
    }

    pub fn update(&mut self, content: impl Into<String>) {
        self.summary = content.into();
    }

    pub fn get(&self) -> &str {
        self.summary.as_str()
    }
}

#[cfg(test)]
mod scripted;
#[cfg(test)]
mod tests;
