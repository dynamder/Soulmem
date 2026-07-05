use chrono::{DateTime, Utc};
use jieba_rs::Jieba;
use soul_mem_core::memory_note::sem_mem::ConceptType;
use soul_mem_core::memory_note::{MemoryNote, MemoryType, situation_mem::SituationType};
use std::future::Future;

use super::decay_calculator::{compute_missing_degree, DEFAULT_MAX_ACTIVATION_CAP};
use super::mask;

/// 默认参数（半衰期 24 小时，活跃因子 0.1）
pub const DEFAULT_BASE_HALF_LIFE_HOURS: f32 = 24.0;
pub const DEFAULT_ACTIVE_FACTOR: f32 = 0.1;
/// 缺失度低于此阈值时不执行任何遗忘操作
pub const MASK_THRESHOLD: f32 = 0.05;
/// 缺失度高于此阈值时触发 LLM 修订
pub const REVISE_THRESHOLD: f32 = 0.15;
/// 遗忘度低于此值时 Vec 类字段（如 aliases）在对齐时不允许增加长度
pub const ALIGN_LENGTH_CAP_THRESHOLD: f32 = 0.6;

/// 遗忘操作的结果
#[derive(Debug)]
pub enum ForgetAction {
    /// 无需遗忘（节点类型不支持或缺失度太低）
    NoAction,
    /// 仅执行遮罩（缺失度中等，不调 LLM）
    MaskOnly {
        missing_degree: f32,
        masked_count: usize,
        masked_text: String,
    },
    /// 遮罩 + LLM 修订完成
    Revised {
        old_summary: String,
        new_summary: String,
        masked_text: String,
    },
}

/// 对节点执行惰性遗忘
///
/// # Arguments
/// * `node` - 待处理的内存节点（可变引用，可能修改 narrative/content）
/// * `current_time` - 当前时间
/// * `jieba` - jieba 分词器实例
/// * `llm_call` - LLM 调用闭包，签名: FnOnce(&str, &str) -> Future<Output = Result<String, Error>>
///
/// # Returns
/// `ForgetAction` 描述执行的操作
pub async fn lazy_forget<F, Fut>(
    node: &mut MemoryNote,
    current_time: DateTime<Utc>,
    jieba: &Jieba,
    llm_call: F,
) -> ForgetAction
where
    F: FnOnce(&str, &str) -> Fut,
    Fut: Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>>,
{
    // 只有 SpecificSituation 和 SemMemory 需要遗忘处理
    let can_forget = matches!(
        node.mem_type(),
        MemoryType::Situation(SituationType::SpecificSituation(_)) | MemoryType::Semantic(_)
    );
    if !can_forget {
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

    // 获取摘要文本（克隆以避免异步点借用问题）
    let old_summary = match node.mem_type() {
        MemoryType::Situation(SituationType::SpecificSituation(s)) => s.get_narrative().clone(),
        MemoryType::Semantic(s) => s.content.clone(),
        _ => return ForgetAction::NoAction,
    };

    // 执行遮罩
    let mask_result = mask::mask_text(&old_summary, md, jieba);

    if md < REVISE_THRESHOLD {
        // 缺失度中等，仅遮罩不调 LLM
        set_summary(node, &mask_result.masked_text);
        return ForgetAction::MaskOnly {
            missing_degree: md,
            masked_count: mask_result.masked_count,
            masked_text: mask_result.masked_text,
        };
    }

    // 缺失度高，遮罩并调 LLM 猜测
    let masked_text = mask_result.masked_text;
    let system_prompt = "You are a memory reconstruction assistant. A segment of memory text has been partially masked. Based on the context, infer and complete the [masked] parts. Output only the completed text, no explanation.";
    let user_prompt = format!("Masked text: {}", masked_text);

    match llm_call(system_prompt, &user_prompt).await {
        Ok(new_summary) => {
            set_summary(node, &new_summary);
            ForgetAction::Revised {
                old_summary,
                new_summary,
                masked_text,
            }
        }
        Err(_) => {
            // LLM 调用失败时至少保留遮罩后的文本
            set_summary(node, &masked_text);
            ForgetAction::MaskOnly {
                missing_degree: md,
                masked_count: mask_result.masked_count,
                masked_text,
            }
        }
    }
}

/// 更新节点的摘要文本（narrative 或 content）
fn set_summary(node: &mut MemoryNote, text: &str) {
    match node.mem_type_mut() {
        MemoryType::Situation(SituationType::SpecificSituation(s)) => {
            *s.get_mut_narrative() = text.to_string();
        }
        MemoryType::Semantic(s) => {
            s.content = text.to_string();
        }
        _ => {}
    }
}

/// 在 LLM 修订 SemMemory 的 content 之后，同步更新 aliases/description/concept_type，
/// 使其与新 content 保持一致。若文意一致则保留原值。
pub async fn align_sem_fields<F, Fut>(
    node: &mut MemoryNote,
    llm_call: F,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    F: FnOnce(&str, &str) -> Fut,
    Fut: Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>>,
{
    let (content, old_aliases, old_desc, old_ct) = match node.mem_type() {
        MemoryType::Semantic(s) => (
            s.content.clone(),
            s.aliases.clone(),
            s.description.clone(),
            format!("{:?}", s.concept_type),
        ),
        _ => return Ok(()),
    };

    let system = "You are a memory consistency checker. Given a memory's content text, verify and if necessary correct the aliases, description, and concept type fields so they match the content.\n\
        Respond ONLY in this exact format, one field per line:\n\
        Aliases: <comma-separated list>\n\
        Description: <short phrase>\n\
        ConceptType: Entity|Abstract\n\
        If the current values are already consistent with the content, keep them unchanged.\n\
        Do not add any explanation.";

    let user = format!(
        "Content: {}\nCurrent aliases: {:?}\nCurrent description: {}\nCurrent concept type: {}",
        content, old_aliases, old_desc, old_ct
    );

    let response = llm_call(system, &user).await?;
    let response = response.trim();

    let mut new_aliases: Option<Vec<String>> = None;
    let mut new_desc: Option<String> = None;
    let mut new_ct: Option<ConceptType> = None;

    for line in response.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("Aliases:") {
            let val = val.trim();
            if val.is_empty() || val.eq_ignore_ascii_case("none") {
                new_aliases = Some(vec![]);
            } else {
                new_aliases = Some(
                    val.split(',')
                        .map(|s| s.trim().trim_matches('"').to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                );
            }
        } else if let Some(val) = line.strip_prefix("Description:") {
            let val = val.trim();
            if !val.is_empty() && !val.eq_ignore_ascii_case("none") {
                new_desc = Some(val.to_string());
            }
        } else if let Some(val) = line.strip_prefix("ConceptType:") {
            let val = val.trim().to_lowercase();
            if val.contains("entity") {
                new_ct = Some(ConceptType::Entity);
            } else if val.contains("abstract") {
                new_ct = Some(ConceptType::Abstract);
            }
        }
    }

    // 计算当前缺失度，决定是否限制 Vec 长度增长
    let missing_degree = compute_missing_degree(
        node.creation_time(),
        node.retrieval_count(),
        Utc::now(),
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    );
    let cap_vec_length = missing_degree < ALIGN_LENGTH_CAP_THRESHOLD;

    if let MemoryType::Semantic(s) = node.mem_type_mut() {
        if let Some(aliases) = new_aliases {
            if cap_vec_length && aliases.len() > old_aliases.len() {
                // 遗忘度较低时不允许 aliases 增长
            } else if !aliases.is_empty() {
                s.aliases = aliases;
            }
        }
        if let Some(desc) = new_desc {
            if !desc.is_empty() {
                s.description = desc;
            }
        }
        if let Some(ct) = new_ct {
            s.concept_type = ct;
        }
    }

    Ok(())
}

/// 获取节点的摘要文本
pub fn get_summary(node: &MemoryNote) -> Option<String> {
    match node.mem_type() {
        MemoryType::Situation(SituationType::SpecificSituation(s)) => {
            Some(s.get_narrative().clone())
        }
        MemoryType::Semantic(s) => Some(s.content.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{Context, Environment, SpecificSituation};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};

    #[tokio::test]
    async fn test_no_action_for_procedure() {
        let proc = MemoryType::Procedure(ProcMemory::new(Action::new(
            "test".to_string(),
            ActionType::Think,
        )));
        let mut node = MemoryNoteBuilder::new(proc).build().unwrap();
        let jieba = Jieba::new();
        let result = lazy_forget(&mut node, Utc::now(), &jieba, |_, _| async {
            Ok("reconstructed".to_string())
        })
        .await;
        assert!(matches!(result, ForgetAction::NoAction));
    }

    #[tokio::test]
    async fn test_fresh_semantic_no_action() {
        let sem = SemMemory::new("data".to_string(), ConceptType::Entity, "desc".to_string());
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(sem))
            .build()
            .unwrap();
        let jieba = Jieba::new();
        let result = lazy_forget(&mut node, Utc::now(), &jieba, |_, _| async {
            Ok("reconstructed".to_string())
        })
        .await;
        assert!(matches!(result, ForgetAction::NoAction));
    }

    #[tokio::test]
    async fn test_semantic_high_missing_degree() {
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let sem = SemMemory::new(
            "鲁迅原名周树人浙江绍兴人".to_string(),
            ConceptType::Entity,
            "人物".to_string(),
        );
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(sem))
            .create_time(past)
            .last_accessed_time(past)
            .build()
            .unwrap();
        let jieba = Jieba::new();
        let now = Utc::now();
        let result = lazy_forget(&mut node, now, &jieba, |_sys, user| {
            let user_owned = user.to_string();
            async move {
                assert!(user_owned.contains(mask::MASK_WORD.trim()));
                Ok("鲁迅是浙江绍兴人原名周树人".to_string())
            }
        })
        .await;
        match &result {
            ForgetAction::Revised { old_summary, new_summary, .. } => {
                assert_eq!(old_summary, "鲁迅原名周树人浙江绍兴人");
                assert_eq!(new_summary, "鲁迅是浙江绍兴人原名周树人");
            }
            ForgetAction::MaskOnly { missing_degree, .. } => {
                assert!(*missing_degree > REVISE_THRESHOLD);
            }
            ForgetAction::NoAction => panic!("old node should trigger forget"),
        }
    }

    // ==================== 随时间变化前后内容差异展示 ====================

    /// 辅助：用当前内容创建一个时间倒退的 SemMemory 节点
    fn make_old_semantic(content: &str, created: DateTime<Utc>) -> MemoryNote {
        let sem = SemMemory::new(content.to_string(), ConceptType::Entity, "测试描述".to_string());
        MemoryNoteBuilder::new(MemoryType::Semantic(sem))
            .create_time(created)
            .last_accessed_time(created)
            .build()
            .unwrap()
    }

    /// 辅助：创建旧的 SpecificSituation 节点
    fn make_old_situation(narrative: &str, created: DateTime<Utc>) -> MemoryNote {
        let ctx = Context::new(
            None, vec![], vec![], vec![],
            Environment { atmosphere: "日常".to_string(), tone: "平静".to_string() },
            vec![],
        );
        let sit = SpecificSituation::new(narrative.to_string(), created, ctx);
        MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(sit)))
            .create_time(created)
            .last_accessed_time(created)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_semantic_content_diff_over_time() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let original_text = "今天下午我和张三在北京王府井的星巴克讨论了项目进展";
        let mut node = make_old_semantic(original_text, created);

        let jieba = Jieba::new();
        let now = Utc::now();

        // 遮罩前的原始内容
        let before = get_summary(&node).unwrap();

        let result = lazy_forget(&mut node, now, &jieba, |_, _| async {
            Err("mock LLM failure".into())
        })
        .await;

        // 遮罩/修订后的内容
        let after = get_summary(&node).unwrap();

        // 打印前后对比（在 test output 中可见）
        println!("===== SemMemory 遗忘前后对比 =====");
        println!("节点创建时间:  {:?}", created);
        println!("遗忘触发时间:  {:?}", now);
        println!("时间跨度:      ~{} 小时", (now - created).num_hours());
        println!("缺失度:        match-dependent");
        match &result {
            ForgetAction::NoAction => {
                println!("操作:          NoAction（未触发遗忘）");
                assert_eq!(before, after, "NoAction 不应修改内容");
            }
            ForgetAction::MaskOnly { missing_degree, masked_count, .. } => {
                println!("操作:          MaskOnly（仅遮罩）");
                println!("缺失度:        {:.4}", missing_degree);
                println!("遮罩词数:      {}/{}", masked_count, 0);
                println!("原始内容:      {}", before);
                println!("遮罩后内容:    {}", after);
                assert_ne!(before, after, "遮罩后内容应发生变化");
                assert!(after.contains(mask::MASK_WORD.trim()), "遮罩后应包含 [masked]");
            }
            ForgetAction::Revised { old_summary, new_summary, .. } => {
                println!("操作:          Revised（遮罩 + LLM 修订）");
                println!("修订前:        {}", old_summary);
                println!("修订后:        {}", new_summary);
                assert_ne!(old_summary, new_summary);
            }
        }
        println!("==================================");
    }

    #[tokio::test]
    async fn test_situation_narrative_diff_over_time() {
        let created = Utc.with_ymd_and_hms(2024, 3, 15, 8, 0, 0).unwrap();
        let original_narrative = "早上八点我在公园慢跑看到一只金毛犬在湖边嬉水";
        let mut node = make_old_situation(original_narrative, created);

        let jieba = Jieba::new();
        let now = Utc::now();

        let before = get_summary(&node).unwrap();

        let result = lazy_forget(&mut node, now, &jieba, |_sys, user| {
            let user_owned = user.to_string();
            async move {
                // 模拟 LLM 成功补全
                assert!(user_owned.contains(mask::MASK_WORD.trim()));
                Ok("清晨在公园湖边慢跑时遇见一只金毛犬正在嬉水".to_string())
            }
        })
        .await;

        let after = get_summary(&node).unwrap();

        println!("===== SpecificSituation 遗忘前后对比 =====");
        println!("节点创建时间:  {:?}", created);
        println!("遗忘触发时间:  {:?}", now);
        println!("时间跨度:      ~{} 小时", (now - created).num_hours());
        match &result {
            ForgetAction::NoAction => {
                println!("操作: NoAction");
                assert_eq!(before, after);
            }
            ForgetAction::MaskOnly { missing_degree, masked_count, .. } => {
                println!("操作:          MaskOnly");
                println!("缺失度:        {:.4}", missing_degree);
                println!("遮罩词数:      {}/{}", masked_count, 0);
                println!("原始叙述:      {}", before);
                println!("遮罩后叙述:    {}", after);
                assert_ne!(before, after);
            }
            ForgetAction::Revised { old_summary, new_summary, .. } => {
                println!("操作:          Revised（LLM 修订）");
                println!("修订前叙述:    {}", old_summary);
                println!("修订后叙述:    {}", new_summary);
                assert_ne!(old_summary, new_summary);
            }
        }
        println!("=========================================");
    }

    #[tokio::test]
    async fn test_progressive_forgetting_across_time() {
        // 模拟在不同时间点对同一个节点做遗忘，展示内容逐渐模糊/变化
        let created = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let original = "张三上个月去杭州出差在西湖边吃了东坡肉和龙井虾仁";
        let mut node = make_old_semantic(original, created);
        let jieba = Jieba::new();

        println!();
        println!("========== 遗忘进程演示（同一节点随时间推移） ==========");
        println!("原始内容:      {}", original);
        println!("创建时间:      {:?}", created);
        println!();

        // 定义 4 个时间点的检查点
        let checkpoints = [
            ("创建后  3 小时", Utc.with_ymd_and_hms(2024, 6, 1, 3, 0, 0).unwrap()),
            ("创建后  1 天  ", Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap()),
            ("创建后  7 天  ", Utc.with_ymd_and_hms(2024, 6, 8, 0, 0, 0).unwrap()),
            ("创建后 30 天  ", Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap()),
        ];

        for (label, checkpoint_time) in &checkpoints {
            let before = get_summary(&node).unwrap();

            // 每个时间点调用 lazy_forget（模拟 LLM 失败，仅遮罩）
            let _result = lazy_forget(&mut node, *checkpoint_time, &jieba, |_, user| {
                let user_owned = user.to_string();
                async move {
                    assert!(user_owned.contains(mask::MASK_WORD.trim()));
                    Ok(user_owned.replace(mask::MASK_WORD.trim(), "???"))
                }
            })
            .await;

            let after = get_summary(&node).unwrap();
            println!("【{}】", label);
            println!("  遗忘前: {}", before);
            println!("  遗忘后: {}", after);
            println!();
        }

        // 最终内容应与原始内容不同
        let final_content = get_summary(&node).unwrap();
        assert_ne!(final_content, original, "多次遗忘后内容应发生变化");
        println!("==========================================================");
    }

    #[tokio::test]
    async fn test_both_node_types_content_change() {
        // 同时对比 SemMemory 和 SpecificSituation 在遗忘后的差异
        let created = Utc.with_ymd_and_hms(2024, 5, 20, 12, 0, 0).unwrap();
        let jieba = Jieba::new();
        let now = Utc::now();

        // -- SemMemory --
        let sem_text = "机器学习是人工智能的一个重要分支主要包括监督学习和无监督学习";
        let mut sem_node = make_old_semantic(sem_text, created);
        let sem_before = get_summary(&sem_node).unwrap();

        let _ = lazy_forget(&mut sem_node, now, &jieba, |_, user| {
            let user_owned = user.to_string();
            async move { Ok(user_owned.replace(mask::MASK_WORD.trim(), "")) }
        })
        .await;

        let sem_after = get_summary(&sem_node).unwrap();

        // -- SpecificSituation --
        let sit_text = "昨天下午我们团队在会议室开了三个小时的 Sprint 回顾会议";
        let mut sit_node = make_old_situation(sit_text, created);
        let sit_before = get_summary(&sit_node).unwrap();

        let _ = lazy_forget(&mut sit_node, now, &jieba, |_, user| {
            let user_owned = user.to_string();
            async move { Ok(user_owned.replace(mask::MASK_WORD.trim(), "")) }
        })
        .await;

        let sit_after = get_summary(&sit_node).unwrap();

        println!("====== 双节点类型遗忘对比 ======");
        println!("SemMemory:");
        println!("  原始 content:   {}", sem_before);
        println!("  遗忘后 content: {}", sem_after);
        println!();
        println!("SpecificSituation:");
        println!("  原始 narrative: {}", sit_before);
        println!("  遗忘后 narrative:{}", sit_after);
        println!("=================================");

        // 两种节点类型的内容都应发生变化
        assert_ne!(sem_before, sem_after, "SemMemory 内容应因遗忘而变化");
        assert_ne!(sit_before, sit_after, "SpecificSituation 内容应因遗忘而变化");
    }
}

// ==================== 真实 LLM 集成测试 ====================
// 需要环境变量: API_KEY, API_BASE, MODEL
// 运行: cargo test -p soul-mem-algo -- "real_llm" --nocapture --ignored

#[cfg(test)]
mod real_llm_tests {
    use super::*;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_runtime::working_memory::llm::client::LlmClient;
    use soul_mem_runtime::working_memory::llm::config::LLMConfig;
    use std::sync::Arc;

    /// 从环境变量创建 LLM 客户端
    fn try_create_llm_client() -> Option<LlmClient> {
        let key = std::env::var("API_KEY").ok()?;
        let base = std::env::var("API_BASE").ok()?;
        let model = std::env::var("MODEL").ok()?;
        Some(LlmClient::new(LLMConfig::new(&key, &base, &model)))
    }

    /// 构建一个完整 SemMemory：content 用长句概括 aliases/concept_type/description
    fn build_complete_sem_node(created: DateTime<Utc>) -> MemoryNote {
        // content 是一句长话，其中嵌入了别名、类型、描述信息
        let content = "Rust是一门由Mozilla主导研发的注重内存安全和零成本抽象的系统级编程语言也被称为Rust语言或Rust-lang它作为实体概念代表了现代系统编程的重要发展方向"
            .to_string();
        let sem = SemMemory::new(
            content,
            ConceptType::Entity,
            "系统级编程语言".to_string(),
        );
        // 通过 builder 后追加 aliases（SemMemory::new 不接收 aliases）
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(sem))
            .create_time(created)
            .last_accessed_time(created)
            .build()
            .unwrap();
        // 补填 aliases
        if let MemoryType::Semantic(s) = node.mem_type_mut() {
            s.aliases = vec!["Rust语言".to_string(), "Rust-lang".to_string()];
        }
        node
    }

    #[tokio::test]
    #[ignore]
    async fn test_real_llm_sem_forget_and_align() {
        // 从 .env 加载环境变量
        dotenvy::dotenv().ok();
        let client = try_create_llm_client()
            .expect("请设置 API_KEY, API_BASE, MODEL 环境变量");
        let client = Arc::new(client);

        // ---- 步骤 1: 创建节点，20 小时前 → 缺失度约 44%，产生部分遮罩 ----
        let created = Utc::now() - chrono::Duration::hours(20);
        let mut node = build_complete_sem_node(created);
        let jieba = Jieba::new();
        let now = Utc::now();

        // 记录遗忘前完整状态
        let before_content = get_summary(&node).unwrap();
        let (before_aliases, before_desc, before_ct) = match node.mem_type() {
            MemoryType::Semantic(s) => (
                s.aliases.clone(),
                s.description.clone(),
                format!("{:?}", s.concept_type),
            ),
            _ => unreachable!(),
        };

        println!();
        println!("========== 真实 LLM 遗忘 + 字段对齐演示 ==========");
        println!("节点创建时间:      {:?}", created);
        println!("遗忘触发时间:      {:?}", now);
        println!("时间跨度（小时）:   {}", (now - created).num_hours());
        println!();
        println!("【遗忘前完整状态】");
        println!("  content:      {}", before_content);
        println!("  aliases:      {:?}", before_aliases);
        println!("  description:  {}", before_desc);
        println!("  concept_type: {}", before_ct);
        println!();

        // ---- 步骤 2: 执行惰性遗忘（遮罩 → LLM 推测内容） ----
        let client_for_forget = client.clone();
        let result = lazy_forget(&mut node, now, &jieba, move |sys, user| {
            let client = client_for_forget.clone();
            let sys = sys.to_string();
            let user = user.to_string();
            async move {
                use async_openai::types::chat::{
                    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
                    ChatCompletionRequestUserMessage,
                };
                let messages: Vec<ChatCompletionRequestMessage> = vec![
                    ChatCompletionRequestSystemMessage::from(sys).into(),
                    ChatCompletionRequestUserMessage::from(user).into(),
                ];
                let mut resp: Vec<String> = client.call_llm(messages).await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
                Ok(resp.remove(0))
            }
        })
        .await;

        // 遗忘后的 content
        let _after_content = get_summary(&node).unwrap();

        println!("【遗忘过程】");
        match &result {
            ForgetAction::NoAction => {
                println!("  操作:     NoAction（未触发）");
                println!("  缺失度不足: 需要检查时间设定");
            }
            ForgetAction::MaskOnly { missing_degree, masked_count, masked_text } => {
                println!("  操作:     MaskOnly（仅遮罩，LLM 失败降级）");
                println!("  缺失度:   {:.2}%", missing_degree * 100.0);
                println!("  遮罩词数: {}/{}", masked_count, 0);
                println!("  遮罩文本: {}", masked_text);
            }
            ForgetAction::Revised { old_summary, new_summary, masked_text } => {
                println!("  操作:     Revised（遮罩 → LLM 推测）");
                println!("  缺失度:   ~{:.0}%", compute_missing_degree(
                    node.creation_time(), 0, now,
                    DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR,
                    DEFAULT_MAX_ACTIVATION_CAP,
                ) * 100.0);
                println!("  遮罩文本: {}", masked_text);
                println!();
                println!("  原始 content: {}", old_summary);
                println!("  LLM 推测  →   {}", new_summary);
            }
        }

        // ---- 步骤 3: 遗忘后完整节点状态 ----
        let (after_aliases, after_desc, after_ct) = match node.mem_type() {
            MemoryType::Semantic(s) => (
                s.aliases.clone(),
                s.description.clone(),
                format!("{:?}", s.concept_type),
            ),
            _ => unreachable!(),
        };

        println!();
        println!("【遗忘后完整节点状态】");
        println!("  content:      {}", get_summary(&node).unwrap());
        println!("  aliases:      {:?}", after_aliases);
        println!("  description:  {}", after_desc);
        println!("  concept_type: {}", after_ct);
        println!();

        // ---- 步骤 4: 对齐 aliases/description/concept_type ----
        println!("----- 开始 align_sem_fields（根据新 content 修正其他字段） -----");

        let client_for_align = client.clone();
        match align_sem_fields(&mut node, move |sys, user| {
            let client = client_for_align.clone();
            let sys = sys.to_string();
            let user = user.to_string();
            async move {
                use async_openai::types::chat::{
                    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
                    ChatCompletionRequestUserMessage,
                };
                let messages: Vec<ChatCompletionRequestMessage> = vec![
                    ChatCompletionRequestSystemMessage::from(sys).into(),
                    ChatCompletionRequestUserMessage::from(user).into(),
                ];
                let mut resp: Vec<String> = client.call_llm(messages).await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
                Ok(resp.remove(0))
            }
        })
        .await {
            Ok(()) => {
                let (new_aliases, new_desc, new_ct) = match node.mem_type() {
                    MemoryType::Semantic(s) => (
                        s.aliases.clone(),
                        s.description.clone(),
                        format!("{:?}", s.concept_type),
                    ),
                    _ => unreachable!(),
                };
                println!();
                println!("【对齐后最终节点状态】");
                println!("  content:      {}", get_summary(&node).unwrap());
                println!("  aliases:      {:?}", new_aliases);
                println!("  description:  {}", new_desc);
                println!("  concept_type: {}", new_ct);
                println!();

                if new_aliases != before_aliases {
                    println!("  → aliases 已更新:");
                    println!("      旧: {:?}", before_aliases);
                    println!("      新: {:?}", new_aliases);
                } else {
                    println!("  → aliases 保持一致: {:?}", before_aliases);
                }
                if new_desc != before_desc {
                    println!("  → description 已更新:");
                    println!("      旧: {}", before_desc);
                    println!("      新: {}", new_desc);
                } else {
                    println!("  → description 保持一致: {}", before_desc);
                }
                if new_ct != before_ct {
                    println!("  → concept_type 已更新:");
                    println!("      旧: {}", before_ct);
                    println!("      新: {}", new_ct);
                } else {
                    println!("  → concept_type 保持一致: {}", before_ct);
                }
            }
            Err(e) => {
                println!("  align_sem_fields 调用失败: {}", e);
                println!("  aliases/description/concept_type 未被更新");
            }
        }

        // ---- 验证 aliases 长度约束 ----
        let (final_aliases, _final_desc, _final_ct) = match node.mem_type() {
            MemoryType::Semantic(s) => (
                s.aliases.clone(),
                s.description.clone(),
                format!("{:?}", s.concept_type),
            ),
            _ => unreachable!(),
        };
        let current_md = compute_missing_degree(
            node.creation_time(), 0, Utc::now(),
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR,
            DEFAULT_MAX_ACTIVATION_CAP,
        );
        if current_md < ALIGN_LENGTH_CAP_THRESHOLD {
            assert!(
                final_aliases.len() <= before_aliases.len(),
                "aliases 长度不应增加: 旧={}, 新={}, 缺失度={:.2}",
                before_aliases.len(),
                final_aliases.len(),
                current_md,
            );
            println!("  → ✓ aliases 长度约束验证通过（缺失度 {:.2} < 阈值 {}，长度未增长）",
                current_md, ALIGN_LENGTH_CAP_THRESHOLD);
        } else {
            println!("  → 跳过长度约束验证（缺失度 {:.2} >= 阈值 {}，允许增长）",
                current_md, ALIGN_LENGTH_CAP_THRESHOLD);
        }

        println!();
        println!("============================================================");
        println!();

        assert!(
            matches!(&result, ForgetAction::Revised { .. } | ForgetAction::MaskOnly { .. }),
            "遗忘应被触发，当前结果: {:?}",
            result
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_activation_slows_forgetting() {
        // 从 .env 加载环境变量
        dotenvy::dotenv().ok();
        let client = try_create_llm_client()
            .expect("请设置 API_KEY, API_BASE, MODEL 环境变量");
        let client = Arc::new(client);
        let jieba = Jieba::new();
        let now = Utc::now();

        // ---- 构建两个节点，相同的创建时间（48 小时前）但激活次数不同 ----
        let created = now - chrono::Duration::hours(48);
        let content = "Python是一门广泛应用于数据科学和人工智能的高级编程语言以其简洁的语法和丰富的生态著称也被称为Py或蟒蛇语言作为一种解释型语言适合快速原型开发";

        // 节点 A: 0 次激活
        let mut node_a = build_complete_sem_node(created);
        if let MemoryType::Semantic(s) = node_a.mem_type_mut() {
            s.content = content.to_string();
            s.aliases = vec!["Python".to_string(), "Py".to_string(), "蟒蛇语言".to_string()];
            s.description = "高级编程语言".to_string();
        }
        // retrieval_count 保持 0

        // 节点 B: 20 次激活
        let mut node_b = build_complete_sem_node(created);
        if let MemoryType::Semantic(s) = node_b.mem_type_mut() {
            s.content = content.to_string();
            s.aliases = vec!["Python".to_string(), "Py".to_string(), "蟒蛇语言".to_string()];
            s.description = "高级编程语言".to_string();
        }
        // 手动增加 retrieval_count
        for _ in 0..20 {
            node_b.retrieval_increment();
        }

        // 节点 C: 超出 cap 的激活次数（200 次），应与 cap（50）效果相同
        let mut node_c = build_complete_sem_node(created);
        if let MemoryType::Semantic(s) = node_c.mem_type_mut() {
            s.content = content.to_string();
            s.aliases = vec!["Python".to_string(), "Py".to_string(), "蟒蛇语言".to_string()];
            s.description = "高级编程语言".to_string();
        }
        for _ in 0..200 {
            node_c.retrieval_increment();
        }

        // 节点 D: 恰好 cap 次激活（50 次），应与节点 C 效果相同
        let mut node_d = build_complete_sem_node(created);
        if let MemoryType::Semantic(s) = node_d.mem_type_mut() {
            s.content = content.to_string();
            s.aliases = vec!["Python".to_string(), "Py".to_string(), "蟒蛇语言".to_string()];
            s.description = "高级编程语言".to_string();
        }
        for _ in 0..DEFAULT_MAX_ACTIVATION_CAP {
            node_d.retrieval_increment();
        }

        // ---- 打印基本信息 ----
        let md_a = compute_missing_degree(created, 0, now, DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP);
        let md_b = compute_missing_degree(created, 20, now, DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP);
        let md_c = compute_missing_degree(created, 200, now, DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP);
        let md_d = compute_missing_degree(created, DEFAULT_MAX_ACTIVATION_CAP, now, DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP);

        println!();
        println!("========== 激活次数对遗忘的影响 ==========");
        println!("创建时间:      {:?}", created);
        println!("遗忘触发时间:  {:?}", now);
        println!("时间跨度:      {} 小时", (now - created).num_hours());
        println!("原始 content:  {}", content);
        println!();
        println!("缺失度对比:");
        println!("  节点 A (激活 0 次):   {:.2}%", md_a * 100.0);
        println!("  节点 B (激活 20 次):  {:.2}%", md_b * 100.0);
        println!("  节点 D (激活 cap={} 次): {:.2}%", DEFAULT_MAX_ACTIVATION_CAP, md_d * 100.0);
        println!("  节点 C (激活 200 次): {:.2}%", md_c * 100.0);
        println!();

        // ---- 对节点 A, B, C 分别执行 lazy_forget ----
        // 辅助：为节点创建一个 LLM 调用闭包
        let make_llm_closure = |c: Arc<LlmClient>| {
            move |sys: &str, user: &str| {
                let client = c.clone();
                let sys = sys.to_string();
                let user = user.to_string();
                async move {
                    use async_openai::types::chat::{
                        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
                        ChatCompletionRequestUserMessage,
                    };
                    let messages: Vec<ChatCompletionRequestMessage> = vec![
                        ChatCompletionRequestSystemMessage::from(sys).into(),
                        ChatCompletionRequestUserMessage::from(user).into(),
                    ];
                    let mut resp: Vec<String> = client.call_llm(messages).await
                        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
                    Ok(resp.remove(0))
                }
            }
        };

        for (node, label, act_count) in [
            (&mut node_a, "节点 A（0 次激活）", 0usize),
            (&mut node_b, "节点 B（20 次激活）", 20),
            (&mut node_d, "节点 D（50 次激活，恰为 cap）", DEFAULT_MAX_ACTIVATION_CAP),
            (&mut node_c, "节点 C（200 次激活，超 cap）", 200),
        ] {
            let before = get_summary(node).unwrap();
            let result = lazy_forget(node, now, &jieba, make_llm_closure(client.clone())).await;
            let after = get_summary(node).unwrap();
            let md = compute_missing_degree(created, act_count, now,
                DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR,
                DEFAULT_MAX_ACTIVATION_CAP);

            println!("【{}  |  缺失度: {:.1}%】", label, md * 100.0);
            println!("  遗忘前: {}", before);
            match &result {
                ForgetAction::Revised { .. } => println!("  操作: Revised"),
                ForgetAction::MaskOnly { masked_text, .. } => {
                    println!("  操作: MaskOnly");
                    println!("  遮罩文本: {}", masked_text);
                }
                ForgetAction::NoAction => println!("  操作: NoAction"),
            }
            println!("  遗忘后: {}", after);
            println!();
        }

        // ---- 验证缺失度关系 ----
        assert!(md_b < md_a,
            "激活节点 B 的缺失度应低于未激活节点 A: B={:.4}, A={:.4}", md_b, md_a);
        assert!((md_c - md_d).abs() < 0.001,
            "超 cap 节点 C 的缺失度应与 cap 节点 D 接近: D={:.4}, C={:.4}", md_d, md_c);
        assert!(md_a > MASK_THRESHOLD,
            "节点 A 缺失度应超过遮罩阈值: {:.4}", md_a);

        println!("============================================================");
        println!();
    }
}
