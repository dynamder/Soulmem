//! 滑动窗口与累加摘要。
//!
//! | 操作 | 职责 | 会失败吗 |
//! |---|---|---|
//! | [`SlidingWindow::push`] | 入队，并告知是否该淘汰 | 否 |
//! | [`SlidingWindow::pop`] | 移除并返回队首 | 否 |
//! | [`SlidingWindow::summarize`] | 用被移除的那条更新累加摘要 | 会（依赖 LLM） |
//!
//! 淘汰由调用方分两步完成：`pop` 取走队首，再按 [`Information::is_tagged`] 决定要不要
//! `summarize`。窗口自身不调用 LLM，因此不存在"返回 Err 但消息其实已入队"或
//! "摘要失败导致该弹出的消息没弹出"这类歧义。
//!
//! `pop` **总是**移除并返回队首，摘要失败也不例外：被弹出的 [`Information`] 交到调用方
//! 手里，重试只需再调一次 [`SlidingWindow::summarize`]。摘要只在 LLM 返回非空结果后才覆写，
//! 所以失败也不会污染已有摘要。容量因此是**硬上限**（落在调用方完成淘汰之后）；从 `push`
//! 返回 [`PushOutcome::NeedsEviction`] 到调用方真正 `pop` 之间，窗口会短暂超出容量。
//!
//! 摘要更新是"读旧摘要 → 调 LLM → 覆写摘要"的复合操作，且摘要是**累加**的。若不串行化，
//! 两个并发淘汰各自读到同一份旧摘要、后写覆盖前写，被淘汰的历史同样永久丢失；这里用异步互斥锁
//! 把整个复合操作串行化（`parking_lot` 锁依旧一律不跨 await 持有）。

// 摘要经 `soul_mem_llm::LlmEngine::complete` 发出，编码/超时/重试/错误分类/trace 都在
// `soul-mem-llm`；完整链路见 `docs/architecture/llm-layer.md`。

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

/// `push` 的结果。
///
/// `push` 只负责入队；收到 [`PushOutcome::NeedsEviction`] 后，调用方用
/// [`SlidingWindow::pop`] 取出队首，再按 [`Information::is_tagged`] 决定是否
/// [`SlidingWindow::summarize`]。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "超过容量后必须淘汰队首，否则窗口会持续增长"]
pub enum PushOutcome {
    /// 未超过容量，无需动作。
    Enqueued,
    /// 已入队且超过容量；调用方应调用 [`SlidingWindow::pop`]。
    NeedsEviction,
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

    /// 信息滑入：只入队，不调用 LLM。
    pub fn push(&self, value: &str, role: &str) -> PushOutcome {
        let text = self.auto_tag(Information::new(value, role));
        let mut window = self.window.write();
        window.push_back(text);
        if window.len() > self.capacity.load(Ordering::Acquire) {
            PushOutcome::NeedsEviction
        } else {
            PushOutcome::Enqueued
        }
    }

    /// 信息滑出：移除并返回队首，不调用 LLM、也不会失败。
    ///
    /// 返回的 [`Information`] 是否带标记由调用方判断；带标记时用
    /// [`SlidingWindow::summarize`] 更新摘要。
    pub fn pop(&self) -> Option<Information> {
        self.window.write().pop_front()
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
    /// 从 `push` 返回 [`PushOutcome::NeedsEviction`] 到调用方 [`SlidingWindow::pop`]
    /// 之前，实际长度可以超过它（见模块注释）。
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

    /// 把摘要记忆、被弹出的消息与当前滑动窗口信息合并成一次语义任务。
    fn prepare_task(&self, evicted: &Information) -> Task {
        let mut snapshot = String::new();
        push_line(&mut snapshot, Role::Assistant, self.summary.read().get());
        // 已经出队的消息不在窗口里，必须单独带上，否则其内容会丢失
        push_line(&mut snapshot, evicted.role(), evicted.get_str());
        for info in self.window.read().iter() {
            push_line(&mut snapshot, info.role(), info.get_str());
        }

        Task::system_user(SUMMARY_SYSTEM_PROMPT, snapshot)
            // 摘要要的是摘要本身，不是思考过程：对推理模型显式要求抑制 thinking
            .with_hints(Hints::default().with_suppress_reasoning(true))
    }

    /// 用被弹出的消息更新累加摘要。
    ///
    /// 只读 `evicted` 与窗口**当前**内容；失败时旧摘要保持不变，调用方可以拿同一条
    /// 消息重试。
    pub async fn summarize(
        &self,
        engine: &LlmEngine,
        evicted: &Information,
    ) -> Result<(), LlmError> {
        let _serialize = self.summarize_gate.lock().await;

        let task = self.prepare_task(evicted);
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
        // 未知 role 一律按 user 处理：system/tool 会被静默误分类。
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
