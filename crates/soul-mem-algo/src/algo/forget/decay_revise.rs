use chrono::{DateTime, Utc};
use jieba_rs::Jieba;
use soul_mem_core::memory_note::{MemoryNote, MemoryType, situation_mem::SituationType};
use std::future::Future;

use super::decay_calculator::{DEFAULT_MAX_ACTIVATION_CAP, compute_missing_degree};
use super::mask;

// ========================================================================
// 默认参数
// ========================================================================

/// 基础半衰期（小时）
pub const DEFAULT_BASE_HALF_LIFE_HOURS: f32 = 24.0;
/// 活跃因子 —— 激活次数对半衰期的加成系数
pub const DEFAULT_ACTIVE_FACTOR: f32 = 0.1;
/// 缺失度低于此阈值时不执行任何遗忘操作
pub const MASK_THRESHOLD: f32 = 0.05;
/// 缺失度高于此阈值时触发 LLM 修订
pub const REVISE_THRESHOLD: f32 = 0.15;

// ========================================================================
// 遗忘操作结果
// ========================================================================

/// 对单次惰性遗忘结果的描述
#[derive(Debug)]
pub enum ForgetAction {
    /// 无需遗忘（节点类型不支持或缺失度低于 MASK_THRESHOLD）
    NoAction,
    /// 仅执行遮罩（缺失度中等，LLM 未被调用）
    MaskOnly {
        missing_degree: f32,
        masked_count: usize,
        masked_text: String,
    },
    /// 遮罩 → LLM 推测修订 → 内容已更新
    Revised {
        old_summary: String,
        new_summary: String,
        masked_text: String,
    },
}

// ========================================================================
// 惰性遗忘编排入口
// ========================================================================

/// 对节点执行惰性遗忘。
///
/// 仅在节点被激活时调用，根据时间跨度和激活次数计算缺失度：
/// - 缺失度 < MASK_THRESHOLD → `NoAction`（无需操作）
/// - MASK_THRESHOLD ≤ 缺失度 < REVISE_THRESHOLD → 仅遮罩概要，不调 LLM
/// - 缺失度 ≥ REVISE_THRESHOLD → 遮罩概要 + 调用 LLM 推测重建
///
/// 仅处理 `SpecificSituation.narrative` 和 `SemMemory.content`。
///
/// # 参数
/// - `node` — 可变的内存节点
/// - `current_time` — 当前时间
/// - `jieba` — Jieba 分词器实例
/// - `system_prompt` — 可选的自定义 LLM system prompt，`None` 使用默认值
/// - `llm_call` — LLM 调用闭包 `FnOnce(&str, &str) -> Future<Result<String>>`
pub async fn lazy_forget<F, Fut>(
    node: &mut MemoryNote,
    current_time: DateTime<Utc>,
    jieba: &Jieba,
    system_prompt: Option<&str>,
    llm_call: F,
) -> ForgetAction
where
    F: FnOnce(&str, &str) -> Fut,
    Fut: Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>>,
{
    if !matches!(
        node.mem_type(),
        MemoryType::Situation(SituationType::SpecificSituation(_)) | MemoryType::Semantic(_)
    ) {
        return ForgetAction::NoAction;
    }

    let md = compute_missing_degree(
        node.creation_time(),
        node.retrieval_count(),
        current_time,
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    );

    if md < MASK_THRESHOLD {
        return ForgetAction::NoAction;
    }

    let old_summary = match node.mem_type() {
        MemoryType::Situation(SituationType::SpecificSituation(s)) => s.get_narrative().clone(),
        MemoryType::Semantic(s) => s.content.clone(),
        _ => return ForgetAction::NoAction,
    };

    // 步骤一：分词遮罩（独立模块 mask）
    let mask_result = mask::mask_text(&old_summary, md, jieba);

    if md < REVISE_THRESHOLD {
        set_summary(node, &mask_result.masked_text);
        return ForgetAction::MaskOnly {
            missing_degree: md,
            masked_count: mask_result.masked_count,
            masked_text: mask_result.masked_text,
        };
    }

    // 步骤二：LLM 补全（独立模块 llm_completion）
    let masked_text = mask_result.masked_text;
    match super::llm_completion::reconstruct_summary(&masked_text, system_prompt, llm_call).await {
        Ok(new_summary) => {
            set_summary(node, &new_summary);
            ForgetAction::Revised {
                old_summary,
                new_summary,
                masked_text,
            }
        }
        Err(_) => {
            set_summary(node, &masked_text);
            ForgetAction::MaskOnly {
                missing_degree: md,
                masked_count: mask_result.masked_count,
                masked_text,
            }
        }
    }
}

// ========================================================================
// 内部辅助函数
// ========================================================================

fn set_summary(node: &mut MemoryNote, text: &str) {
    match node.mem_type_mut() {
        MemoryType::Situation(SituationType::SpecificSituation(s)) => {
            *s.get_mut_narrative() = text.to_string();
        }
        MemoryType::Semantic(s) => s.content = text.to_string(),
        _ => {}
    }
}

/// 获取节点的概要文本（narrative 或 content）
pub fn get_summary(node: &MemoryNote) -> Option<String> {
    match node.mem_type() {
        MemoryType::Situation(SituationType::SpecificSituation(s)) => {
            Some(s.get_narrative().clone())
        }
        MemoryType::Semantic(s) => Some(s.content.clone()),
        _ => None,
    }
}

// ========================================================================
// 字段对齐（重新导出 llm_completion 模块，对外保持同一调用入口）
// ========================================================================

pub use super::llm_completion::align_sem_fields;

// ========================================================================
// 单元测试
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{Context, Environment, SpecificSituation};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};

    fn make_old_semantic(content: &str, created: DateTime<Utc>) -> MemoryNote {
        let sem = SemMemory::new(
            content.to_string(),
            ConceptType::Entity,
            "测试描述".to_string(),
        );
        MemoryNoteBuilder::new(MemoryType::Semantic(sem))
            .create_time(created)
            .last_accessed_time(created)
            .build()
            .unwrap()
    }

    fn make_old_situation(narrative: &str, created: DateTime<Utc>) -> MemoryNote {
        let ctx = Context::new(
            None,
            vec![],
            vec![],
            vec![],
            Environment {
                atmosphere: "日常".to_string(),
                tone: "平静".to_string(),
            },
            vec![],
        );
        MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(
            SpecificSituation::new(narrative.to_string(), created, ctx),
        )))
        .create_time(created)
        .last_accessed_time(created)
        .build()
        .unwrap()
    }

    #[tokio::test]
    async fn test_no_action_for_procedure() {
        let mut node = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(Action::new(
            "test".to_string(),
            ActionType::Think,
        ))))
        .build()
        .unwrap();
        let result = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_, _| async {
            Ok("reconstructed".to_string())
        })
        .await;
        assert!(matches!(result, ForgetAction::NoAction));
    }

    #[tokio::test]
    async fn test_fresh_semantic_no_action() {
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "data".to_string(),
            ConceptType::Entity,
            "desc".to_string(),
        )))
        .build()
        .unwrap();
        let result = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_, _| async {
            Ok("reconstructed".to_string())
        })
        .await;
        assert!(matches!(result, ForgetAction::NoAction));
    }

    #[tokio::test]
    async fn test_semantic_high_missing_degree() {
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "鲁迅原名周树人浙江绍兴人".to_string(),
            ConceptType::Entity,
            "人物".to_string(),
        )))
        .create_time(past)
        .last_accessed_time(past)
        .build()
        .unwrap();
        let result = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_sys, user| {
            let u = user.to_string();
            async move {
                assert!(u.contains(mask::MASK_WORD.trim()));
                Ok("鲁迅是浙江绍兴人原名周树人".to_string())
            }
        })
        .await;
        match &result {
            ForgetAction::Revised {
                old_summary,
                new_summary,
                ..
            } => {
                assert_eq!(old_summary, "鲁迅原名周树人浙江绍兴人");
                assert_eq!(new_summary, "鲁迅是浙江绍兴人原名周树人");
            }
            ForgetAction::MaskOnly { missing_degree, .. } => {
                assert!(*missing_degree > REVISE_THRESHOLD)
            }
            ForgetAction::NoAction => panic!("old node should trigger forget"),
        }
    }

    #[tokio::test]
    async fn test_semantic_content_diff_over_time() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mut node = make_old_semantic(
            "今天下午我和张三在北京王府井的星巴克讨论了项目进展",
            created,
        );
        let before = get_summary(&node).unwrap();
        let result = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_, _| async {
            Err("mock".into())
        })
        .await;
        let after = get_summary(&node).unwrap();
        match &result {
            ForgetAction::NoAction => assert_eq!(before, after),
            ForgetAction::MaskOnly { .. } => {
                assert_ne!(before, after);
                assert!(after.contains(mask::MASK_WORD.trim()));
            }
            ForgetAction::Revised {
                old_summary,
                new_summary,
                ..
            } => assert_ne!(old_summary, new_summary),
        }
    }

    #[tokio::test]
    async fn test_situation_narrative_diff_over_time() {
        let created = Utc.with_ymd_and_hms(2024, 3, 15, 8, 0, 0).unwrap();
        let mut node = make_old_situation("早上八点我在公园慢跑看到一只金毛犬在湖边嬉水", created);
        let before = get_summary(&node).unwrap();
        let result = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_sys, user| {
            let u = user.to_string();
            async move {
                assert!(u.contains(mask::MASK_WORD.trim()));
                Ok("清晨在公园湖边慢跑时遇见一只金毛犬正在嬉水".to_string())
            }
        })
        .await;
        let after = get_summary(&node).unwrap();
        match &result {
            ForgetAction::NoAction => assert_eq!(before, after),
            ForgetAction::MaskOnly { .. } => assert_ne!(before, after),
            ForgetAction::Revised {
                old_summary,
                new_summary,
                ..
            } => assert_ne!(old_summary, new_summary),
        }
    }

    #[tokio::test]
    async fn test_progressive_forgetting_across_time() {
        let created = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let original = "张三上个月去杭州出差在西湖边吃了东坡肉和龙井虾仁";
        let mut node = make_old_semantic(original, created);
        for t in &[
            Utc.with_ymd_and_hms(2024, 6, 1, 3, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 6, 8, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap(),
        ] {
            let _ = lazy_forget(&mut node, *t, &Jieba::new(), None, |_, user| {
                let u = user.to_string();
                async move {
                    assert!(u.contains(mask::MASK_WORD.trim()));
                    Ok(u.replace(mask::MASK_WORD.trim(), "???"))
                }
            })
            .await;
        }
        assert_ne!(get_summary(&node).unwrap(), original);
    }

    #[tokio::test]
    async fn test_both_node_types_content_change() {
        let created = Utc.with_ymd_and_hms(2024, 5, 20, 12, 0, 0).unwrap();
        let jieba = Jieba::new();
        let now = Utc::now();
        let mut sem = make_old_semantic(
            "机器学习是人工智能的一个重要分支主要包括监督学习和无监督学习",
            created,
        );
        let mut sit = make_old_situation(
            "昨天下午我们团队在会议室开了三个小时的 Sprint 回顾会议",
            created,
        );
        let sem_before = get_summary(&sem).unwrap();
        let sit_before = get_summary(&sit).unwrap();
        let _ = lazy_forget(&mut sem, now, &jieba, None, |_, user| {
            let u = user.to_string();
            async move { Ok(u.replace(mask::MASK_WORD.trim(), "")) }
        })
        .await;
        let _ = lazy_forget(&mut sit, now, &jieba, None, |_, user| {
            let u = user.to_string();
            async move { Ok(u.replace(mask::MASK_WORD.trim(), "")) }
        })
        .await;
        assert_ne!(sem_before, get_summary(&sem).unwrap());
        assert_ne!(sit_before, get_summary(&sit).unwrap());
    }
}

// ========================================================================
// 真实 LLM 集成测试（需 API_KEY / API_BASE / MODEL 环境变量）
// 运行: cargo test -p soul-mem-algo -- "real_llm" --nocapture --ignored
// ========================================================================

#[cfg(test)]
mod real_llm_tests {
    use super::*;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_runtime::working_memory::llm::client::LlmClient;
    use soul_mem_runtime::working_memory::llm::config::LLMConfig;
    use std::sync::Arc;

    // 十六夜咲夜角色 prompts（测试专用，可自定义替换）
    const SAKUYA_RECONSTRUCT: Option<&str> = Some(
        "You are Sakuya Izayoi, the perfect and elegant maid of the Scarlet Devil Mansion. \
        You have the ability to manipulate time. Your character card defines who you are, but certain sections \
        have been deliberately removed — memories of specific individuals, particularly those connected to \
        Eientei and the moon, are no longer part of your recorded past. A segment of your memory has been \
        partially masked. As yourself, recall and reconstruct the complete memory naturally based on the \
        remaining fragments, relying only on what your current character card contains. Stay in character as \
        a composed maiden with a touch of elegance and pride. Output only the completed memory text in first \
        person, no explanation.",
    );
    const SAKUYA_ALIGN: Option<&str> = Some(
        "You are Sakuya Izayoi, the perfect and elegant maid of the Scarlet Devil Mansion. \
        Given a memory's content text from your own records, verify and if necessary correct the aliases, \
        description, and concept type fields so they match the content.\n\
        Respond ONLY in this exact format, one field per line:\n\
        Aliases: <comma-separated list>\n\
        Description: <short phrase>\n\
        ConceptType: Entity|Abstract\n\
        If the current values are already consistent with the content, keep them unchanged.\n\
        Do not add any explanation.",
    );

    fn try_create_llm_client() -> Option<LlmClient> {
        Some(LlmClient::new(LLMConfig::new(
            &std::env::var("API_KEY").ok()?,
            &std::env::var("API_BASE").ok()?,
            &std::env::var("MODEL").ok()?,
        )))
    }

    fn build_complete_sem_node(created: DateTime<Utc>) -> MemoryNote {
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Rust是一门由Mozilla主导研发的注重内存安全和零成本抽象的系统级编程语言也被称为Rust语言或Rust-lang它作为实体概念代表了现代系统编程的重要发展方向".to_string(),
            ConceptType::Entity, "系统级编程语言".to_string(),
        ))).create_time(created).last_accessed_time(created).build().unwrap();
        if let MemoryType::Semantic(s) = node.mem_type_mut() {
            s.aliases = vec!["Rust语言".to_string(), "Rust-lang".to_string()];
        }
        node
    }

    fn make_llm_closure(
        c: Arc<LlmClient>,
    ) -> impl FnOnce(
        &str,
        &str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<String, Box<dyn std::error::Error + Send + Sync>>,
                > + Send,
        >,
    > {
        move |sys: &str, user: &str| {
            let client = c.clone();
            let s = sys.to_string();
            let u = user.to_string();
            Box::pin(async move {
                use async_openai::types::chat::{
                    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
                };
                let mut resp = client
                    .call_llm(vec![
                        ChatCompletionRequestSystemMessage::from(s).into(),
                        ChatCompletionRequestUserMessage::from(u).into(),
                    ])
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
                Ok(resp.remove(0))
            })
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_real_llm_sem_forget_and_align() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        let now = Utc::now();
        let created = now - chrono::Duration::hours(20);
        let mut node = build_complete_sem_node(created);
        let before_content = get_summary(&node).unwrap();
        let (ba, bd, bc) = match node.mem_type() {
            MemoryType::Semantic(s) => (
                s.aliases.clone(),
                s.description.clone(),
                format!("{:?}", s.concept_type),
            ),
            _ => unreachable!(),
        };

        println!("\n========== 真实 LLM 遗忘 + 字段对齐演示 ==========");
        println!("原始 content: {}", before_content);
        println!(
            "原始 aliases: {:?} | description: {} | concept_type: {}",
            ba, bd, bc
        );
        println!(
            "缺失度: ~{:.0}%",
            compute_missing_degree(
                created,
                0,
                now,
                DEFAULT_BASE_HALF_LIFE_HOURS,
                DEFAULT_ACTIVE_FACTOR,
                DEFAULT_MAX_ACTIVATION_CAP
            ) * 100.0
        );
        println!();

        let result = lazy_forget(
            &mut node,
            now,
            &jieba,
            SAKUYA_RECONSTRUCT,
            make_llm_closure(client.clone()),
        )
        .await;
        match &result {
            ForgetAction::Revised {
                old_summary,
                new_summary,
                masked_text,
            } => {
                println!(
                    "【Revised】\n  原始: {}\n  遮罩: {}\n  LLM:  {}",
                    old_summary, masked_text, new_summary
                );
            }
            ForgetAction::MaskOnly { masked_text, .. } => {
                println!(
                    "【MaskOnly 降级】\n  原始: {}\n  遮罩: {}",
                    before_content, masked_text
                );
            }
            ForgetAction::NoAction => println!("【NoAction】"),
        }
        println!();

        let _ = align_sem_fields(&mut node, SAKUYA_ALIGN, make_llm_closure(client.clone())).await;
        let (fa, fd, fc) = match node.mem_type() {
            MemoryType::Semantic(s) => (
                s.aliases.clone(),
                s.description.clone(),
                format!("{:?}", s.concept_type),
            ),
            _ => unreachable!(),
        };
        println!(
            "【对齐后】content: {} | aliases: {:?} | description: {} | concept_type: {}",
            get_summary(&node).unwrap(),
            fa,
            fd,
            fc
        );

        let curr_md = compute_missing_degree(
            node.creation_time(),
            0,
            Utc::now(),
            DEFAULT_BASE_HALF_LIFE_HOURS,
            DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );
        if curr_md < 0.6 {
            assert!(fa.len() <= ba.len());
        }
        println!("=================================================\n");
        assert!(matches!(
            result,
            ForgetAction::Revised { .. } | ForgetAction::MaskOnly { .. }
        ));
    }

    #[tokio::test]
    #[ignore]
    async fn test_activation_slows_forgetting() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        let now = Utc::now();

        let gen_client = client.clone();
        let generated: String = {
            use async_openai::types::chat::{
                ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
            };
            let mut resp = gen_client.call_llm(vec![
                ChatCompletionRequestSystemMessage::from("你是红魔馆的女仆长十六夜咲夜。请以第一人称写一段你在幻想乡日常生活中的具体事件记忆，2~4句话，描述发生了什么、涉及谁、你的感受。只输出记忆文本，不要解释。".to_string()).into(),
                ChatCompletionRequestUserMessage::from("请讲述一件你在红魔馆经历过的难忘事件。".to_string()).into(),
            ]).await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() }).expect("LLM 生成失败");
            resp.remove(0)
        };
        let content: String = generated
            .chars()
            .filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace())
            .collect();
        let desc_prefix: String = generated
            .chars()
            .take(30)
            .filter(|c| !c.is_ascii_punctuation())
            .collect();
        let created = now - chrono::Duration::hours(48);

        let make_node = |rc: usize| -> MemoryNote {
            let mut n = build_complete_sem_node(created);
            if let MemoryType::Semantic(s) = n.mem_type_mut() {
                s.content = content.clone();
                s.aliases = vec!["紅魔館の思い出".to_string(), "咲夜の出来事".to_string()];
                s.description = format!("紅魔館メイド長の回想: {}", desc_prefix);
            }
            for _ in 0..rc {
                n.retrieval_increment();
            }
            n
        };

        let mut na = make_node(0);
        let mut nb = make_node(20);
        let mut nc = make_node(200);
        let mut nd = make_node(DEFAULT_MAX_ACTIVATION_CAP);

        let mda = compute_missing_degree(
            created,
            0,
            now,
            DEFAULT_BASE_HALF_LIFE_HOURS,
            DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );
        let mdb = compute_missing_degree(
            created,
            20,
            now,
            DEFAULT_BASE_HALF_LIFE_HOURS,
            DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );
        let mdc = compute_missing_degree(
            created,
            200,
            now,
            DEFAULT_BASE_HALF_LIFE_HOURS,
            DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );
        let mdd = compute_missing_degree(
            created,
            DEFAULT_MAX_ACTIVATION_CAP,
            now,
            DEFAULT_BASE_HALF_LIFE_HOURS,
            DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );

        println!("\n========== 激活次数对遗忘的影响 ==========");
        println!(
            "LLM 生成: {}\n缺失度: A={:.0}% B={:.0}% D={:.0}% C={:.0}%\n",
            generated,
            mda * 100.0,
            mdb * 100.0,
            mdd * 100.0,
            mdc * 100.0
        );

        for (node, label, md) in [
            (&mut na, "A(0次)", mda),
            (&mut nb, "B(20次)", mdb),
            (&mut nd, "D(cap)", mdd),
            (&mut nc, "C(200次,超cap)", mdc),
        ] {
            let before = get_summary(node).unwrap_or_default();
            let result = lazy_forget(
                node,
                now,
                &jieba,
                SAKUYA_RECONSTRUCT,
                make_llm_closure(client.clone()),
            )
            .await;
            println!(
                "【{} | 缺失度 {:.0}%】\n  原始: {}",
                label,
                md * 100.0,
                before
            );
            match &result {
                ForgetAction::Revised {
                    masked_text,
                    new_summary,
                    ..
                } => println!("  遮罩: {}\n  LLM:  {}", masked_text, new_summary),
                ForgetAction::MaskOnly { masked_text, .. } => {
                    println!("  遮罩: {} (LLM 失败)", masked_text)
                }
                ForgetAction::NoAction => println!("  (未触发)"),
            }
            println!();
        }

        assert!(mdb < mda);
        assert!((mdc - mdd).abs() < 0.001);
        assert!(mda > MASK_THRESHOLD);
        println!("=================================================\n");
    }
}
