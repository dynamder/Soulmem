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
