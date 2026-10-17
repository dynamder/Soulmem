use super::scripted::{Outcome, Scripted};
use super::*;
use std::time::Duration;
use tokio::time::sleep;

/// 调用方的成功路径：弹出队首，带标记时再摘要。返回被弹出的消息。
async fn evict_and_summarize(window: &SlidingWindow, engine: &LlmEngine) -> Information {
    let evicted = window.pop().expect("队首");
    if evicted.is_tagged() {
        window.summarize(engine, &evicted).await.expect("摘要");
    }
    evicted
}

#[tokio::test]
async fn push_and_get_keep_order_and_role() {
    let window = SlidingWindow::new(10);
    let scripted = Scripted::new(vec![]);

    assert_eq!(window.push("user_info", "user"), PushOutcome::Enqueued);
    assert_eq!(
        window.push("assistant_info", "assistant"),
        PushOutcome::Enqueued
    );

    assert_eq!(window.get(0).expect("第 0 条").get_str(), "user_info");
    assert_eq!(window.get(1).expect("第 1 条").get_str(), "assistant_info");
    assert_eq!(window.get_windows()[0].get_str(), "user_info");
    assert_eq!(window.get_windows()[1].get_str(), "assistant_info");
    assert_eq!(scripted.calls(), 0, "未超容量、未标记时不应调用 LLM");
}

#[tokio::test]
async fn push_never_calls_llm_even_when_over_capacity() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![]);

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);
    assert_eq!(window.len(), 2, "push 只入队，容量由 pop 负责");
    assert_eq!(scripted.calls(), 0, "淘汰之前不应调用 LLM");
}

#[test]
fn pop_returns_the_front_and_removes_it() {
    let window = SlidingWindow::new(10);
    assert_eq!(window.push("user_info", "user"), PushOutcome::Enqueued);
    assert_eq!(
        window.push("assistant_info", "assistant"),
        PushOutcome::Enqueued
    );

    let first = window.pop().expect("队首");
    assert_eq!(first.get_str(), "user_info");
    assert_eq!(window.len(), 1);
    assert_eq!(window.get(0).expect("第 0 条").get_str(), "assistant_info");

    assert_eq!(window.pop().expect("第二条").get_str(), "assistant_info");
    assert!(window.pop().is_none(), "空窗口返回 None");
    assert!(window.is_empty());
}

#[test]
fn pop_never_calls_the_llm_even_for_a_tagged_front() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![]);

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    let evicted = window.pop().expect("队首");

    assert!(evicted.is_tagged());
    assert!(window.is_empty());
    assert_eq!(scripted.calls(), 0, "pop 只出队，不接触 LLM");
}

#[tokio::test]
async fn summarize_updates_the_summary() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![Outcome::Text("第一段摘要")]);

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);
    evict_and_summarize(&window, scripted.engine()).await;

    assert_eq!(scripted.calls(), 1, "应摘要一次");
    assert_eq!(window.get_summary().as_ref(), "第一段摘要");
    assert_eq!(window.len(), 1);
    assert_eq!(window.get(0).expect("队首").get_str(), "b");
}

#[tokio::test]
async fn summary_prompt_contains_evicted_and_window_in_order() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![Outcome::Text("摘要")]);

    assert_eq!(window.push("第一句", "user"), PushOutcome::Enqueued);
    assert_eq!(
        window.push("第二句", "assistant"),
        PushOutcome::NeedsEviction
    );
    evict_and_summarize(&window, scripted.engine()).await;

    let prompt = scripted.last_prompt().expect("应有调用");
    assert!(prompt.contains("summary"), "system 提示应在：{prompt}");
    let first_at = prompt.find("第一句").expect("被弹出的消息也要进入提示词");
    let second_at = prompt.find("第二句").expect("窗口消息应出现在提示词里");
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

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);
    evict_and_summarize(&window, scripted.engine()).await;
    assert_eq!(window.get_summary().as_ref(), "第一次摘要");

    assert_eq!(window.push("c", "user"), PushOutcome::NeedsEviction);
    evict_and_summarize(&window, scripted.engine()).await;
    assert_eq!(window.get_summary().as_ref(), "第二次摘要");

    let prompt = scripted.last_prompt().expect("应有调用");
    assert!(
        prompt.contains("第一次摘要"),
        "旧摘要必须进入下一次摘要的提示词：{prompt}"
    );
}

#[tokio::test]
async fn empty_summary_is_rejected_and_keeps_the_old_one() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![Outcome::Text("   ")]);

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);
    let evicted = window.pop().expect("队首");
    let error = window
        .summarize(scripted.engine(), &evicted)
        .await
        .expect_err("空摘要必须被拒绝");

    assert_eq!(error.kind(), LlmErrorKind::EmptyCompletion);
    assert_eq!(window.get_summary().as_ref(), "", "旧摘要不得被覆盖");
}

#[tokio::test]
async fn summary_failure_keeps_the_evicted_message_in_the_callers_hand() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![Outcome::Fail(LlmErrorKind::Transport)]);

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);

    let evicted = window.pop().expect("队首");
    assert!(evicted.is_tagged());
    let error = window
        .summarize(scripted.engine(), &evicted)
        .await
        .expect_err("摘要失败必须上报");

    assert_eq!(error.kind(), LlmErrorKind::Transport);
    assert!(error.is_unavailable(), "调用方应能据此降级");
    assert_eq!(
        evicted.get_str(),
        "a",
        "被弹出的消息在调用方手里，不随摘要失败消失"
    );
    assert_eq!(window.len(), 1, "pop 已经出队，摘要失败不回滚");
    assert_eq!(window.get(0).expect("队首").get_str(), "b");
    assert_eq!(window.get_summary().as_ref(), "", "旧摘要不得被覆盖");
}

/// 失败之后用同一个被弹出的消息重试：摘要才更新，且不会再次出队。
#[tokio::test]
async fn a_failed_summary_can_be_retried_with_the_same_message() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::new(vec![
        Outcome::Fail(LlmErrorKind::Timeout),
        Outcome::Text("补救摘要"),
    ]);

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);
    let evicted = window.pop().expect("队首");

    window
        .summarize(scripted.engine(), &evicted)
        .await
        .expect_err("第一次失败");
    assert_eq!(window.get_summary().as_ref(), "");

    window
        .summarize(scripted.engine(), &evicted)
        .await
        .expect("第二次成功");

    assert_eq!(scripted.calls(), 2);
    assert_eq!(window.get_summary().as_ref(), "补救摘要");
    assert_eq!(window.len(), 1, "重试不会再次出队");
}

/// 引擎会先自行重试可重试的失败：窗口只有在重试预算耗尽后才看到错误。
#[tokio::test]
async fn a_transient_failure_is_retried_by_the_engine_before_the_window_sees_an_error() {
    let window = SlidingWindow::new(1);
    let scripted = Scripted::with_whole_call_retries(
        vec![
            Outcome::Fail(LlmErrorKind::Transport),
            Outcome::Text("重试后摘要"),
        ],
        1,
    );

    assert_eq!(window.push("a", "user"), PushOutcome::Enqueued);
    assert_eq!(window.push("b", "user"), PushOutcome::NeedsEviction);
    let evicted = evict_and_summarize(&window, scripted.engine()).await;

    assert_eq!(evicted.get_str(), "a");
    assert_eq!(scripted.calls(), 2, "第一次失败 + 一次重试");
    assert_eq!(window.get_summary().as_ref(), "重试后摘要");
    assert_eq!(window.len(), 1);
}

#[tokio::test]
async fn concurrent_push_fills_the_window() {
    let window = Arc::new(SlidingWindow::new(100));

    let window1 = window.clone();
    let handle = tokio::spawn(async move {
        for i in 0..50 {
            assert_eq!(
                window1.push(&format!("user_{i}"), "user"),
                PushOutcome::Enqueued
            );
        }
    });

    for i in 50..100 {
        assert_eq!(
            window.push(&format!("user_{i}"), "user"),
            PushOutcome::Enqueued
        );
    }

    handle.await.expect("任务不应 panic");
    assert_eq!(window.len(), 100);
}

#[tokio::test]
async fn concurrent_read_write_does_not_deadlock_or_lose_data() {
    let window = Arc::new(SlidingWindow::new(50));

    let window_write = window.clone();
    let window_read = window.clone();

    let write_handle = tokio::spawn(async move {
        for i in 0..25 {
            assert_eq!(
                window_write.push(&format!("msg_{i}"), "user"),
                PushOutcome::Enqueued
            );
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
        assert_eq!(
            window.push(&format!("initial_{i}"), "user"),
            PushOutcome::Enqueued
        );
    }

    let window_clone = window.clone();
    let pop_handle = tokio::spawn(async move {
        for _ in 0..3 {
            window_clone.pop();
        }
    });

    pop_handle.await.expect("任务不应 panic");
    assert_eq!(window.len(), 2);
    assert_eq!(scripted.calls(), 0, "这些消息都未被摘要");
}

#[tokio::test]
async fn window_is_shareable_across_tasks() {
    let window = Arc::new(SlidingWindow::new(10));

    let window1 = window.clone();
    let handle1 = tokio::spawn(async move {
        for i in 0..5 {
            assert_eq!(
                window1.push(&format!("t1_{i}"), "user"),
                PushOutcome::Enqueued
            );
        }
    });

    let window2 = window.clone();
    let handle2 = tokio::spawn(async move {
        for i in 0..5 {
            assert_eq!(
                window2.push(&format!("t2_{i}"), "user"),
                PushOutcome::Enqueued
            );
        }
    });

    handle1.await.expect("任务 1 不应 panic");
    handle2.await.expect("任务 2 不应 panic");
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
    assert!(window.pop().is_none());
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
