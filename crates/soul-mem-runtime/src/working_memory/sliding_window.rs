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

/// 测试辅助：脚本化后端 + 由它构造的引擎，并提供调用记录。
///
/// 刻意不做成 `soul-mem-llm` 的公共 feature：目前只有本 crate 需要它，
/// 等第二个 crate 也需要时再提升（YAGNI）。
#[cfg(test)]
mod scripted {
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
}

#[cfg(test)]
mod tests {
    use super::scripted::{Outcome, Scripted};
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    // ---------- 窗口基本行为 ----------

    #[tokio::test]
    async fn push_and_get_keep_order_and_role() {
        let window = SlidingWindow::new(10);
        let scripted = Scripted::new(vec![]);

        window
            .push("user_info", "user", scripted.engine())
            .await
            .expect("push");
        window
            .push("assistant_info", "assistant", scripted.engine())
            .await
            .expect("push");

        assert_eq!(window.get(0).expect("第 0 条").get_str(), "user_info");
        assert_eq!(window.get(1).expect("第 1 条").get_str(), "assistant_info");
        assert_eq!(window.get_windows()[0].get_str(), "user_info");
        assert_eq!(window.get_windows()[1].get_str(), "assistant_info");
        assert_eq!(scripted.calls(), 0, "未超容量、未标记时不应调用 LLM");
    }

    #[tokio::test]
    async fn pop_removes_untagged_front_without_calling_llm() {
        let window = SlidingWindow::new(10);
        let scripted = Scripted::new(vec![]);

        window
            .push("user_info", "user", scripted.engine())
            .await
            .expect("push");
        window
            .push("assistant_info", "assistant", scripted.engine())
            .await
            .expect("push");
        window.untag_information(0);

        window.pop(scripted.engine()).await.expect("pop");

        assert_eq!(window.len(), 1);
        assert_eq!(window.get(0).expect("第 0 条").get_str(), "assistant_info");
        assert_eq!(scripted.calls(), 0);
    }

    #[tokio::test]
    async fn pop_on_empty_window_is_a_noop() {
        let window = SlidingWindow::new(10);
        let scripted = Scripted::new(vec![]);
        window
            .pop(scripted.engine())
            .await
            .expect("空窗口 pop 不应报错");
        assert!(window.is_empty());
    }

    // ---------- 摘要成功路径 ----------

    /// 容量为 1 时每次 push 都会被标记，因此第二次 push 必然触发摘要。
    #[tokio::test]
    async fn successful_summary_updates_summary_and_removes_the_message() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![Outcome::Text("第一段摘要")]);

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        assert_eq!(window.len(), 1, "未超容量");

        window
            .push("b", "user", scripted.engine())
            .await
            .expect("push b");

        assert_eq!(scripted.calls(), 1, "应摘要一次");
        assert_eq!(window.get_summary().as_ref(), "第一段摘要");
        assert_eq!(window.len(), 1, "成功后应回落到容量内");
        assert_eq!(window.get(0).expect("队首").get_str(), "b");
    }

    #[tokio::test]
    async fn summary_prompt_contains_window_in_order() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![Outcome::Text("摘要")]);

        window
            .push("第一句", "user", scripted.engine())
            .await
            .expect("push");
        window
            .push("第二句", "assistant", scripted.engine())
            .await
            .expect("push");

        let prompt = scripted.last_prompt().expect("应有调用");
        assert!(prompt.contains("summary"), "system 提示应在：{prompt}");
        let first_at = prompt.find("第一句").expect("旧消息应出现在提示词里");
        let second_at = prompt.find("第二句").expect("新消息应出现在提示词里");
        assert!(first_at < second_at, "消息顺序必须保持：{prompt}");
        assert!(prompt.contains("[user]"), "消息应带角色标签：{prompt}");
    }

    /// 第二次摘要必须带上上一次的摘要内容（摘要是累加的）。
    #[tokio::test]
    async fn summary_prompt_carries_the_previous_summary_forward() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![
            Outcome::Text("第一次摘要"),
            Outcome::Text("第二次摘要"),
        ]);

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        window
            .push("b", "user", scripted.engine())
            .await
            .expect("push b");
        assert_eq!(window.get_summary().as_ref(), "第一次摘要");

        window
            .push("c", "user", scripted.engine())
            .await
            .expect("push c");
        assert_eq!(window.get_summary().as_ref(), "第二次摘要");

        let prompt = scripted.last_prompt().expect("应有调用");
        assert!(
            prompt.contains("第一次摘要"),
            "旧摘要必须进入下一次摘要的提示词：{prompt}"
        );
    }

    // ---------- 摘要失败路径（本次重构的核心修正） ----------

    #[tokio::test]
    async fn summary_failure_keeps_the_evicted_message() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![Outcome::Fail(LlmErrorKind::Transport)]);

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        let error = window
            .push("b", "user", scripted.engine())
            .await
            .expect_err("摘要失败必须上报");

        assert_eq!(error.kind(), LlmErrorKind::Transport);
        assert!(error.is_unavailable(), "调用方应能据此降级");
        assert_eq!(window.len(), 2, "失败时不得丢消息：容量是软上限");
        assert_eq!(
            window.get(0).expect("队首").get_str(),
            "a",
            "被淘汰的消息必须还在"
        );
        assert_eq!(window.get_summary().as_ref(), "", "旧摘要不得被覆盖");
    }

    #[tokio::test]
    async fn summary_failure_keeps_the_popped_message() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![Outcome::Fail(LlmErrorKind::ServerError)]);

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        let error = window
            .pop(scripted.engine())
            .await
            .expect_err("摘要失败必须上报");

        assert_eq!(error.kind(), LlmErrorKind::ServerError);
        assert_eq!(window.len(), 1, "popped 的消息不得消失");
        assert_eq!(window.get(0).expect("队首").get_str(), "a");
    }

    /// 失败之后重试成功：消息才被移除，摘要才被更新。
    #[tokio::test]
    async fn a_failed_summary_can_be_retried_on_the_next_push() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![
            Outcome::Fail(LlmErrorKind::Timeout),
            Outcome::Text("补救摘要"),
        ]);

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        window
            .push("b", "user", scripted.engine())
            .await
            .expect_err("第一次失败");
        assert_eq!(window.len(), 2);

        window
            .push("c", "user", scripted.engine())
            .await
            .expect("第二次成功");

        assert_eq!(scripted.calls(), 2);
        assert_eq!(window.get_summary().as_ref(), "补救摘要");
        assert_eq!(
            window.get(0).expect("队首").get_str(),
            "b",
            "\"a\" 已被摘要淘汰"
        );
    }

    /// 引擎会先自行重试可重试的失败：窗口只有在重试预算耗尽后才看到错误。
    #[tokio::test]
    async fn a_transient_failure_is_retried_by_the_engine_before_push_reports_anything() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::with_whole_call_retries(
            vec![
                Outcome::Fail(LlmErrorKind::Transport),
                Outcome::Text("重试后摘要"),
            ],
            1,
        );

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        window
            .push("b", "user", scripted.engine())
            .await
            .expect("引擎重试后应当成功，窗口不该看到错误");

        assert_eq!(scripted.calls(), 2, "第一次失败 + 一次重试");
        assert_eq!(window.get_summary().as_ref(), "重试后摘要");
        assert_eq!(window.len(), 1, "只有摘要成功后才淘汰消息");
    }

    #[tokio::test]
    async fn empty_summary_is_rejected_and_keeps_everything() {
        let window = SlidingWindow::new(1);
        let scripted = Scripted::new(vec![Outcome::Text("   ")]);

        window
            .push("a", "user", scripted.engine())
            .await
            .expect("push a");
        let error = window
            .push("b", "user", scripted.engine())
            .await
            .expect_err("空摘要必须被拒绝");

        assert_eq!(error.kind(), LlmErrorKind::EmptyCompletion);
        assert_eq!(window.len(), 2, "拒绝空摘要时不得丢消息");
        assert_eq!(window.get_summary().as_ref(), "");
    }

    // ---------- 并发 ----------

    #[tokio::test]
    async fn concurrent_push_fills_the_window() {
        let window = Arc::new(SlidingWindow::new(100));
        let scripted = Scripted::new(vec![]);

        let window1 = window.clone();
        let engine1 = scripted.engine().clone();
        let handle = tokio::spawn(async move {
            for i in 0..50 {
                window1
                    .push(&format!("user_{i}"), "user", &engine1)
                    .await
                    .expect("push");
            }
        });

        for i in 50..100 {
            window
                .push(&format!("user_{i}"), "user", scripted.engine())
                .await
                .expect("push");
        }

        handle.await.expect("任务不应 panic");
        assert_eq!(window.len(), 100);
    }

    #[tokio::test]
    async fn concurrent_read_write_does_not_deadlock_or_lose_data() {
        let window = Arc::new(SlidingWindow::new(50));
        let scripted = Scripted::new(vec![]);

        let window_write = window.clone();
        let window_read = window.clone();
        let engine = scripted.engine().clone();

        let write_handle = tokio::spawn(async move {
            for i in 0..25 {
                window_write
                    .push(&format!("msg_{i}"), "user", &engine)
                    .await
                    .expect("push");
            }
        });

        let read_handle = tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            window_read.len()
        });

        write_handle.await.expect("写任务不应 panic");
        let len = read_handle.await.expect("读任务不应 panic");
        assert!(len > 0);
        assert_eq!(window.len(), 25);
    }

    #[tokio::test]
    async fn concurrent_pop_terminates_and_shrinks_the_window() {
        let window = Arc::new(SlidingWindow::new(10));
        let scripted = Scripted::new(vec![]);

        for i in 0..5 {
            window
                .push(&format!("initial_{i}"), "user", scripted.engine())
                .await
                .expect("push");
        }

        let window_clone = window.clone();
        let engine = scripted.engine().clone();
        let pop_handle = tokio::spawn(async move {
            for _ in 0..3 {
                window_clone.pop(&engine).await.expect("pop");
            }
        });

        pop_handle.await.expect("任务不应 panic");
        assert_eq!(window.len(), 2);
        assert_eq!(scripted.calls(), 0, "这些消息都未被标记");
    }

    #[tokio::test]
    async fn window_is_shareable_across_tasks() {
        let window = Arc::new(SlidingWindow::new(10));
        let scripted = Scripted::new(vec![]);

        let window1 = window.clone();
        let engine1 = scripted.engine().clone();
        let handle1 = tokio::spawn(async move {
            for i in 0..5 {
                window1.push(&format!("t1_{i}"), "user", &engine1).await?;
            }
            Ok::<(), LlmError>(())
        });

        let window2 = window.clone();
        let engine2 = scripted.engine().clone();
        let handle2 = tokio::spawn(async move {
            for i in 0..5 {
                window2.push(&format!("t2_{i}"), "user", &engine2).await?;
            }
            Ok::<(), LlmError>(())
        });

        handle1.await.expect("任务 1 不应 panic").expect("任务 1");
        handle2.await.expect("任务 2 不应 panic").expect("任务 2");
        assert_eq!(window.len(), 10);
    }

    #[tokio::test]
    async fn capacity_can_be_changed_concurrently() {
        let window = Arc::new(SlidingWindow::new(20));
        assert_eq!(window.get_capacity(), 20);

        window.set_capacity(30);
        assert_eq!(window.get_capacity(), 30);

        let window_clone = window.clone();
        let window_for_check = window.clone();
        let handle = tokio::spawn(async move {
            window_clone.set_capacity(15);
            window_clone.get_capacity()
        });

        let capacity = handle.await.expect("任务不应 panic");
        assert_eq!(capacity, 15);
        assert_eq!(window_for_check.get_capacity(), 15);
    }

    // ---------- 纯逻辑：Information / Summary / 标记 ----------

    #[test]
    fn information_new_maps_role_and_keeps_text() {
        let user = Information::new("hello", "user");
        assert!(matches!(user, Information::User(_)));
        assert_eq!(user.get_str(), "hello");
        assert_eq!(user.role(), Role::User);

        let assistant = Information::new("world", "assistant");
        assert!(matches!(assistant, Information::Assistant(_)));
        assert_eq!(assistant.get_str(), "world");
        assert_eq!(assistant.role(), Role::Assistant);

        // 未知角色按旧行为退化为 User
        let unknown = Information::new("fallback", "system");
        assert!(matches!(unknown, Information::User(_)));
        assert_eq!(unknown.role(), Role::User);
    }

    #[test]
    fn information_tag_roundtrip_for_both_variants() {
        for role in ["user", "assistant"] {
            let mut info = Information::new("text", role);
            assert!(!info.is_tagged(), "{role} 初始不应被标记");
            info.tag_information();
            assert!(info.is_tagged(), "{role} 应被标记");
            info.untag_information();
            assert!(!info.is_tagged(), "{role} 应取消标记");
        }
    }

    #[test]
    fn user_and_assistant_information_accessors() {
        let user = UserInformation::new("user text");
        assert_eq!(user.get_str(), "user text");
        assert!(!user.tag);

        let assistant = AssistantInformation::new("assistant text");
        assert_eq!(assistant.get_str(), "assistant text");
        assert!(!assistant.tag);
    }

    #[test]
    fn summary_update_and_get() {
        let mut summary = Summary::new();
        assert_eq!(summary.get(), "");
        summary.update("first summary");
        assert_eq!(summary.get(), "first summary");
        summary.update("second summary");
        assert_eq!(summary.get(), "second summary");
    }

    #[test]
    fn empty_window_reports_empty() {
        let window = SlidingWindow::new(10);
        assert!(window.is_empty());
        assert_eq!(window.len(), 0);
        assert!(window.get(0).is_none());
        assert_eq!(window.get_summary().as_ref(), "");
    }

    #[test]
    fn clear_empties_the_window_and_resets_tag_count() {
        let window = SlidingWindow::new(10);
        {
            let mut w = window.window.write();
            w.push_back(Information::new("a", "user"));
            w.push_back(Information::new("b", "user"));
        }
        assert_eq!(window.len(), 2);

        window.clear();
        assert!(window.is_empty());
        // 计数归零后，第一条不应被标记
        let first = window.auto_tag(Information::new("x", "user"));
        assert!(!first.is_tagged(), "clear 后计数必须重置");
    }

    #[test]
    fn tag_and_untag_are_bounds_checked() {
        let window = SlidingWindow::new(10);
        {
            let mut w = window.window.write();
            w.push_back(Information::new("a", "user"));
            w.push_back(Information::new("b", "user"));
        }

        window.tag_information(0);
        assert!(window.get(0).expect("第 0 条").is_tagged());
        assert!(!window.get(1).expect("第 1 条").is_tagged());

        // 越界（含 index == len）不应 panic，也不应影响任何元素
        window.tag_information(2);
        window.tag_information(99);
        window.untag_information(2);
        assert!(!window.get(1).expect("第 1 条").is_tagged());
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn untag_at_len_boundary_keeps_the_tag() {
        let window = SlidingWindow::new(10);
        {
            let mut w = window.window.write();
            let mut info = Information::new("a", "user");
            info.tag_information();
            w.push_back(info);
        }
        window.untag_information(1); // index == len
        assert!(window.get(0).expect("第 0 条").is_tagged());
    }

    #[test]
    fn auto_tag_marks_every_capacity_th_message() {
        let window = SlidingWindow::new(3);
        assert!(!window.auto_tag(Information::new("1", "user")).is_tagged());
        assert!(!window.auto_tag(Information::new("2", "user")).is_tagged());
        assert!(window.auto_tag(Information::new("3", "user")).is_tagged());
        // 计数重置后下一个周期第一条不再标记
        assert!(!window.auto_tag(Information::new("4", "user")).is_tagged());
    }

    #[test]
    fn auto_tag_with_capacity_one_tags_everything() {
        let window = SlidingWindow::new(1);
        assert!(window.auto_tag(Information::new("1", "user")).is_tagged());
        assert!(window.auto_tag(Information::new("2", "user")).is_tagged());
    }

    #[test]
    fn prompt_lines_are_labelled_by_role() {
        let mut buffer = String::new();
        push_line(&mut buffer, Role::User, "用户说的");
        push_line(&mut buffer, Role::Assistant, "助手说的");
        assert_eq!(buffer, "[user]: 用户说的\n[assistant]: 助手说的\n");
    }
}
