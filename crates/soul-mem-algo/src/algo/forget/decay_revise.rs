use chrono::{DateTime, Utc};
use jieba_rs::Jieba;
use soul_mem_core::memory_links::MemoryLink;
use soul_mem_core::memory_note::{MemoryNote, MemoryType, situation_mem::SituationType};
use soul_mem_runtime::cluster::memory_cluster::{GraphMemoryLink, MemoryCluster};
use std::future::Future;

use super::decay_calculator::{
    update_missing_degree_incremental, DEFAULT_MAX_ACTIVATION_CAP,
};
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
/// 先在 `compute_and_update_missing_degree` 中刷新并存储缺失度（对所有节点生效），
/// 随后**仅**对 `SpecificSituation` 和 `SemMemory` 触发遮罩 / LLM：
/// - 缺失度 < MASK_THRESHOLD → `NoAction`（无需操作）
/// - MASK_THRESHOLD ≤ 缺失度 < REVISE_THRESHOLD → 仅遮罩概要，不调 LLM
/// - 缺失度 ≥ REVISE_THRESHOLD → 遮罩概要 + 调用 LLM 推测重建
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
    // 步骤〇：对所有节点刷新并存储当前缺失度
    let md = compute_and_update_missing_degree(node, current_time);

    // 仅 SpecificSituation 和 SemMemory 触发遮罩 / LLM，其余节点仅更新缺失度
    if !matches!(
        node.mem_type(),
        MemoryType::Situation(SituationType::SpecificSituation(_)) | MemoryType::Semantic(_)
    ) {
        return ForgetAction::NoAction;
    }

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

/// 计算并更新节点当前的遗忘缺失度，写入 `MemoryNote.missing_degree`。
///
/// 使用增量公式：基于上次存储的缺失度与时间差推算当前值，避免重复从头计算。
/// 适用于**所有**节点类型（SemMemory / SpecificSituation / Procedure 等），
/// 仅负责记录缺失度，不触发任何遮罩或 LLM 机制。
///
/// 返回更新后的缺失度（0.0 ~ 1.0）。
pub fn compute_and_update_missing_degree(
    node: &mut MemoryNote,
    current_time: DateTime<Utc>,
) -> f32 {
    let md = update_missing_degree_incremental(
        node.missing_degree(),
        node.last_forget_time(),
        current_time,
        node.retrieval_count(),
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    );
    node.set_missing_degree(md);
    node.set_last_forget_time(current_time);
    md
}

/// 只读计算节点当前遗忘缺失度，不写回节点。
///
/// 用于检索中对大量节点计算权重（读锁下安全，O(1)/节点）。
pub fn current_missing_degree(node: &MemoryNote, current_time: DateTime<Utc>) -> f32 {
    update_missing_degree_incremental(
        node.missing_degree(),
        node.last_forget_time(),
        current_time,
        node.retrieval_count(),
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    )
}

/// 只读计算边当前遗忘缺失度，不写回边。
///
/// 边无激活次数（retrieval_count = 0），衰减公式与节点一致。
pub fn current_edge_missing_degree(link: &GraphMemoryLink, current_time: DateTime<Utc>) -> f32 {
    update_missing_degree_incremental(
        link.missing_degree(),
        link.last_forget_time(),
        current_time,
        0,
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    )
}

/// 检索权重占位符：当前为 `(1 - missing_degree)`，后续可替换为更精细的权重模型。
pub fn weight_placeholder(missing_degree: f32) -> f64 {
    (1.0 - missing_degree.clamp(0.0, 1.0)) as f64
}

/// 衰减单条边（core 层 `MemoryLink`）：边自身独立增量衰减，与节点无关。
/// 返回衰减后的强度。
pub fn decay_edge(link: &mut MemoryLink, current_time: DateTime<Utc>) -> f64 {
    let md = update_missing_degree_incremental(
        link.missing_degree(),
        link.last_forget_time(),
        current_time,
        0,
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    );
    link.set_missing_degree(md);
    link.set_last_forget_time(current_time);
    link.intensity * (1.0 - md as f64)
}

/// 衰减图中单条边（`GraphMemoryLink`）：边自身独立增量衰减，与节点无关。
/// 返回衰减后的强度。
pub fn decay_graph_edge(link: &mut GraphMemoryLink, current_time: DateTime<Utc>) -> f64 {
    let md = update_missing_degree_incremental(
        link.missing_degree(),
        link.last_forget_time(),
        current_time,
        0,
        DEFAULT_BASE_HALF_LIFE_HOURS,
        DEFAULT_ACTIVE_FACTOR,
        DEFAULT_MAX_ACTIVATION_CAP,
    );
    link.set_missing_degree(md);
    link.set_last_forget_time(current_time);
    link.intensity() * (1.0 - md as f64)
}

/// 计算并更新**所有**节点与边的遗忘缺失度，不触发任何遮罩 / LLM 操作。
///
/// 适用于检索前的批量刷新：对图中每个节点与边就地写入最新缺失度与计算时间。
pub fn compute_all_missing_degrees(cluster: &mut MemoryCluster, current_time: DateTime<Utc>) {
    let graph = cluster.graph_mut();
    let node_indices: Vec<_> = graph.node_indices().collect();
    let edge_indices: Vec<_> = graph.edge_indices().collect();

    for node_idx in node_indices {
        if let Some(embedded) = graph.node_weight_mut(node_idx) {
            let node = &mut embedded.note;
            let md = update_missing_degree_incremental(
                node.missing_degree(),
                node.last_forget_time(),
                current_time,
                node.retrieval_count(),
                DEFAULT_BASE_HALF_LIFE_HOURS,
                DEFAULT_ACTIVE_FACTOR,
                DEFAULT_MAX_ACTIVATION_CAP,
            );
            node.set_missing_degree(md);
            node.set_last_forget_time(current_time);
        }
    }

    for edge_idx in edge_indices {
        if let Some(link) = graph.edge_weight_mut(edge_idx) {
            let md = update_missing_degree_incremental(
                link.missing_degree(),
                link.last_forget_time(),
                current_time,
                0,
                DEFAULT_BASE_HALF_LIFE_HOURS,
                DEFAULT_ACTIVE_FACTOR,
                DEFAULT_MAX_ACTIVATION_CAP,
            );
            link.set_missing_degree(md);
            link.set_last_forget_time(current_time);
        }
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
// 测试 —— 第一部分：节点强度衰减；第四部分：图结构实例
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::decay_calculator::node_intensity_after;
    use chrono::TimeZone;
    use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{Context, Environment, SpecificSituation};
    use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
    use std::time::Instant;

    // ------------------------------------------------------------------
    // 第一部分：节点强度衰减测试
    // 输入：时长、初始强度、已激活次数、激活次数影响系数、半衰期
    // 输出：节点强度；并给出案例注释
    // ------------------------------------------------------------------
    pub(crate) fn part1_intensity_report() {
        let t0 = Instant::now();
        // 输入参数
        let duration_hours = 48.0; // 时长 48 小时
        let initial_intensity = 1.0; // 初始强度
        let activation_count = 5; // 已激活次数
        let active_factor = 0.1; // 激活次数影响系数
        let half_life_hours = 24.0; // 半衰期 24 小时

        let intensity = node_intensity_after(
            duration_hours,
            initial_intensity,
            activation_count,
            active_factor,
            half_life_hours,
        );
        let elapsed = t0.elapsed();

        // 案例注释：
        // 半衰期 24h，激活 5 次（影响系数 0.1）→ 调整半衰期 = 24×(1+0.1×5) = 36h，
        // τ = 36/ln2 ≈ 51.94h，初始强度 1.0 经 48h 后强度 ≈ 0.397。
        println!("【第一部分】节点强度衰减");
        println!("  输入: 时长={}h, 初始强度={}, 激活次数={}, 影响系数={}, 半衰期={}h",
            duration_hours, initial_intensity, activation_count, active_factor, half_life_hours);
        println!("  输出: 节点强度 = {:.4}", intensity);
        println!("  案例: 半衰期24h×激活5次(系数0.1)→调整半衰期36h→经48h强度≈{:.4}", intensity);
        println!("  计算用时: {:?}", elapsed);
        println!();

        // 与理论值一致
        let adjusted_hl = half_life_hours * (1.0 + active_factor * activation_count as f32);
        let tau = adjusted_hl / std::f32::consts::LN_2;
        let expect = initial_intensity * (-duration_hours / tau).exp();
        assert!((intensity - expect).abs() < 1e-4);
    }

    #[test]
    fn test_part1_node_intensity_decay() {
        part1_intensity_report();
    }

    // ------------------------------------------------------------------
    // 第四部分：图结构实例 —— 节点和边种类不限，至少两种可遮罩节点，
    // 各节点强度不一；对所有节点和边一起计算强度（时间参数相同），
    // 输出列表，统计各节点强度差值。
    // ------------------------------------------------------------------
    #[test]
    fn test_part4_graph_all_decay() {
        use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType, sem_mem::SemMemLink};
        use soul_mem_query::embedding::{
            EmbeddingVec,
            note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant},
            sem::SemanticEmbedding,
        };
        use std::collections::HashMap;

        let t0 = Instant::now();
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap(); // 统一 Δt = 24h

        let make_embedded = |id: MemoryId, mem_type: MemoryType, initial_md: f32| -> EmbeddedMemoryNote {
            let note = MemoryNoteBuilder::new(mem_type)
                .id(id)
                .create_time(past)
                .last_accessed_time(past)
                .last_forget_time(past)
                .missing_degree(initial_md)
                .build()
                .unwrap();
            let embedding = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            EmbeddedMemoryNote { note, embedding }
        };

        // 三个节点，初始缺失度不一 → 初始强度(1-md)不一：1.0 / 0.7 / 0.4
        let id_sem = MemoryId::new();
        let id_sit = MemoryId::new();
        let id_proc = MemoryId::new();

        let mut cluster = MemoryCluster::new();
        cluster.add_single_node(make_embedded(
            id_sem,
            MemoryType::Semantic(SemMemory::new(
                "十六夜咲夜是红魔馆的女仆长拥有操纵时间的能力".to_string(), ConceptType::Entity, "红魔馆的女仆长".to_string(),
            )),
            0.0,
        ));
        cluster.add_single_node(make_embedded(
            id_sit,
            MemoryType::Situation(SituationType::SpecificSituation(
                SpecificSituation::new(
                    "午后蕾米莉亚大小姐在地下图书馆让帕秋莉小姐品尝我泡的红茶".to_string(), past,
                    Context::new(None, vec![], vec![], vec![],
                        Environment { atmosphere: "日常".to_string(), tone: "平静".to_string() }, vec![]),
                ),
            )),
            0.3,
        ));
        cluster.add_single_node(make_embedded(
            id_proc,
            MemoryType::Procedure(ProcMemory::new(Action::new(
                "红魔馆女仆长每日停止时间打扫洋馆再回收飞刀的工作流程".to_string(), ActionType::Think,
            ))),
            0.6,
        ));

        // 两条边，时间参数相同
        let mut link1 = MemoryLink::new(id_sem, id_sit, MemoryLinkType::Sem(SemMemLink::new("关联".to_string(), 1.0)));
        let mut link2 = MemoryLink::new(id_sit, id_proc, MemoryLinkType::Sem(SemMemLink::new("引发".to_string(), 1.0)));
        link1.set_last_forget_time(past);
        link2.set_last_forget_time(past);
        {
            let graph = cluster.graph_mut();
            for n in graph.node_weights_mut() {
                if n.note().id() == id_sem {
                    n.note.links_mut().push(link1.clone());
                } else if n.note().id() == id_sit {
                    n.note.links_mut().push(link2.clone());
                }
            }
        }
        cluster.refresh_node(&id_sem);
        cluster.refresh_node(&id_sit);
        let build_elapsed = t0.elapsed();

        // 记录各节点初始缺失度
        let init_md: HashMap<MemoryId, f32> = vec![(id_sem, 0.0), (id_sit, 0.3), (id_proc, 0.6)]
            .into_iter()
            .collect();

        // 对所有节点和边一起计算强度（不遮罩）
        let t1 = Instant::now();
        compute_all_missing_degrees(&mut cluster, now);
        let compute_elapsed = t1.elapsed();

        println!("【第四部分】图结构节点/边强度衰减");
        println!("  时间参数: 统一 Δt = {}h", (now - past).num_hours());
        println!("  节点强度 = 1 - 缺失度");
        println!();
        println!("  ┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
        println!("  │ 节点/边  │ 初始强度     │ 计算后强度   │ 强度差值     │ 类型          │");
        println!("  ├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

        let graph = cluster.graph();
        for node_idx in graph.node_indices() {
            let n = graph.node_weight(node_idx).unwrap();
            let id = n.note().id();
            let init = *init_md.get(&id).unwrap_or(&0.0);
            let final_md = n.note().missing_degree();
            let init_strength = 1.0 - init;
            let final_strength = 1.0 - final_md;
            let delta = init_strength - final_strength; // 强度衰减量
            let type_name = match n.note().mem_type() {
                MemoryType::Semantic(_) => "SemMemory",
                MemoryType::Situation(SituationType::SpecificSituation(_)) => "SpecificSituation",
                MemoryType::Procedure(_) => "Procedure",
                _ => "?",
            };
            let id_str = id.to_string();
            println!("  │ {:<8} │ {:<12.4} │ {:<12.4} │ {:<12.4} │ {:<12} │",
                &id_str[..8.min(id_str.len())], init_strength, final_strength, delta, type_name);
        }
        for e in graph.edge_weights() {
            let final_md_e = e.missing_degree();
            let init_strength = 1.0 - 0.0_f32;
            let final_strength = 1.0 - final_md_e;
            let delta = init_strength - final_strength;
            let id_str = e.id().to_string();
            println!("  │ {:<8} │ {:<12.4} │ {:<12.4} │ {:<12.4} │ {:<12} │",
                &id_str[..8.min(id_str.len())], init_strength, final_strength, delta, "Edge");
        }
        println!("  └──────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
        println!("  构建图用时: {:?}, 全图强度计算用时: {:?}", build_elapsed, compute_elapsed);
        println!();

        // 断言：所有节点缺失度已更新到 now 时刻，且大于初始值
        for (id, init) in &init_md {
            let n = graph.node_weight(
                graph.node_indices().find(|&i| graph.node_weight(i).map_or(false, |n| n.note().id() == *id)).unwrap(),
            ).unwrap();
            assert_eq!(n.note().last_forget_time(), now);
            assert!(n.note().missing_degree() > *init);
        }
        assert_eq!(graph.edge_count(), 2);
    }
}

// ========================================================================
// 测试 —— 第二部分：两种可遮罩节点完整流程；第三部分：三组遮罩；
//           第五部分：整体（按顺序执行第一、二、三部分）
// 需要环境变量 API_KEY / API_BASE / MODEL
// 运行: cargo test -p soul-mem-algo -- "part" --nocapture --ignored
// ========================================================================

#[cfg(test)]
mod real_llm_tests {
    use super::*;
    use super::tests::part1_intensity_report;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{Context, Environment, SpecificSituation};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_runtime::working_memory::llm::client::LlmClient;
    use soul_mem_runtime::working_memory::llm::config::LLMConfig;
    use std::sync::Arc;
    use std::time::Instant;

    // 十六夜咲夜角色 prompts（测试专用，可自定义替换）
    const SAKUYA_RECONSTRUCT: Option<&str> = Some(
        "You are Sakuya Izayoi, the perfect and elegant maid of the Scarlet Devil Mansion. \
        You have the ability to manipulate time. Your character card defines who you are, but certain sections \
        have been deliberately removed — memories of specific individuals, particularly those connected to \
        Eientei and the moon, are no longer part of your recorded past. A segment of your memory has been \
        partially masked, where some words are replaced with [masked]. \
        Output ONLY the original sentence with each [masked] slot filled in with a single plausible word or short phrase \
        that best fits the context, keeping every non-masked word EXACTLY as it is — do not reorder, rephrase, \
        rewrite, or generate a new sentence, and do not add or remove any other words. \
        Stay in character as a composed maiden with a touch of elegance and pride. \
        Output only the completed text, no explanation.",
    );

    fn try_create_llm_client() -> Option<LlmClient> {
        Some(LlmClient::new(LLMConfig::new(
            &std::env::var("API_KEY").ok()?,
            &std::env::var("API_BASE").ok()?,
            &std::env::var("MODEL").ok()?,
        )))
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

    /// 构建语义节点：创建于很久以前，last_forget_time 在 `forget_ago_hours` 前
    fn build_sem_node(content: &str, forget_ago_hours: i64) -> MemoryNote {
        let now = Utc::now();
        let created = now - chrono::Duration::hours(24 * 10);
        let forget_time = now - chrono::Duration::hours(forget_ago_hours);
        MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            content.to_string(),
            ConceptType::Entity,
            "测试描述".to_string(),
        )))
        .create_time(created)
        .last_accessed_time(created)
        .last_forget_time(forget_time)
        .build()
        .unwrap()
    }

    /// 构建具体情境节点
    fn build_situation_node(narrative: &str, forget_ago_hours: i64) -> MemoryNote {
        let now = Utc::now();
        let created = now - chrono::Duration::hours(24 * 10);
        let forget_time = now - chrono::Duration::hours(forget_ago_hours);
        let ctx = Context::new(
            None, vec![], vec![], vec![],
            Environment { atmosphere: "日常".to_string(), tone: "平静".to_string() },
            vec![],
        );
        MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(
            SpecificSituation::new(narrative.to_string(), created, ctx),
        )))
        .create_time(created)
        .last_accessed_time(created)
        .last_forget_time(forget_time)
        .build()
        .unwrap()
    }

    /// 打印遗忘流程的三段文本（原始 / 遮罩 / 推测），并给出耗时
    fn print_three_texts(label: &str, before: &str, result: &ForgetAction, md: f32) {
        println!("  【{}】缺失度 = {:.1}%", label, md * 100.0);
        println!("    原始文本: {}", before);
        match result {
            ForgetAction::Revised { masked_text, new_summary, .. } => {
                println!("    遮罩文本: {}", masked_text);
                println!("    推测文本: {}", new_summary);
            }
            ForgetAction::MaskOnly { masked_text, .. } => {
                println!("    遮罩文本: {} (LLM 失败降级)", masked_text);
            }
            ForgetAction::NoAction => println!("    (未触发遗忘)"),
        }
    }

    // ------------------------------------------------------------------
    // 第二部分 · SemMemory：从衰减到遮罩到 LLM 推测的完整流程
    // ------------------------------------------------------------------
    async fn run_part2_sem(client: Arc<LlmClient>, jieba: &Jieba) {
        let t0 = Instant::now();
        let content = "十六夜咲夜是红魔馆的女仆长拥有操纵时间的能力她可以停止时间在静止的世界中完成所有家务银质小刀是她惯用的武器大小姐为此深感满意";
        let mut node = build_sem_node(content, 20); // 20h 前衰减
        let before = get_summary(&node).unwrap();
        let t1 = Instant::now();

        let result = lazy_forget(&mut node, Utc::now(), jieba, SAKUYA_RECONSTRUCT, make_llm_closure(client)).await;
        let t2 = Instant::now();
        let md = node.missing_degree();

        println!("【第二部分 · SemMemory 完整遗忘流程】");
        println!("  构建节点用时: {:?}, 衰减+遮罩+LLM 用时: {:?}", t1 - t0, t2 - t1);
        print_three_texts("SemMemory", &before, &result, md);
        println!();
        assert!(matches!(result, ForgetAction::Revised { .. } | ForgetAction::MaskOnly { .. }));
    }

    // ------------------------------------------------------------------
    // 第二部分 · SpecificSituation：从衰减到遮罩到 LLM 推测的完整流程
    // ------------------------------------------------------------------
    async fn run_part2_situation(client: Arc<LlmClient>, jieba: &Jieba) {
        let t0 = Instant::now();
        let narrative = "傍晚我在红魔馆的庭院为大小姐斟茶蕾米莉亚坐在阳台的红伞下望着雾之湖畔天色渐暗四周渐渐安静下来";
        let mut node = build_situation_node(narrative, 20); // 20h 前衰减
        let before = get_summary(&node).unwrap();
        let t1 = Instant::now();

        let result = lazy_forget(&mut node, Utc::now(), jieba, SAKUYA_RECONSTRUCT, make_llm_closure(client)).await;
        let t2 = Instant::now();
        let md = node.missing_degree();

        println!("【第二部分 · SpecificSituation 完整遗忘流程】");
        println!("  构建节点用时: {:?}, 衰减+遮罩+LLM 用时: {:?}", t1 - t0, t2 - t1);
        print_three_texts("SpecificSituation", &before, &result, md);
        println!();
        assert!(matches!(result, ForgetAction::Revised { .. } | ForgetAction::MaskOnly { .. }));
    }

    // ------------------------------------------------------------------
    // 第三部分：对一个原始文本做三组遮罩（遗忘度由低到高）
    // ------------------------------------------------------------------
    async fn run_part3_mask_levels(client: Arc<LlmClient>, jieba: &Jieba) {
        let content = "红魔馆的女仆长十六夜咲夜擅长投掷银质小刀她害怕烫的食物是众所周知的猫舌大小姐为此常常感到无可奈何却又乐在其中";
        println!("【第三部分】遮罩测试（遗忘度由低到高）");
        println!("  原始文本: {}", content);
        println!();

        let levels: [(i64, &str); 3] = [
            (8, "低遗忘度 (Δt=8h, 缺失度≈21%)"),
            (24, "中遗忘度 (Δt=24h, 缺失度≈50%)"),
            (72, "高遗忘度 (Δt=72h, 缺失度≈87%)"),
        ];

        for (i, (ago, label)) in levels.iter().enumerate() {
            let t0 = Instant::now();
            let mut node = build_sem_node(content, *ago);
            let before = get_summary(&node).unwrap();
            let t1 = Instant::now();
            let result = lazy_forget(&mut node, Utc::now(), jieba, SAKUYA_RECONSTRUCT, make_llm_closure(client.clone())).await;
            let t2 = Instant::now();
            let md = node.missing_degree();

            println!("  {} · {}", i + 1, label);
            println!("    构建用时 {:?}, 遮罩+LLM 用时 {:?}", t1 - t0, t2 - t1);
            print_three_texts(&format!("第{}组", i + 1), &before, &result, md);
            println!();
        }
    }

    // ------------------------------------------------------------------
    // 第二部分测试：SemMemory
    // ------------------------------------------------------------------
    #[tokio::test]
    #[ignore]
    async fn test_part2_sem_forget_flow() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        run_part2_sem(client, &jieba).await;
    }

    // ------------------------------------------------------------------
    // 第二部分测试：SpecificSituation
    // ------------------------------------------------------------------
    #[tokio::test]
    #[ignore]
    async fn test_part2_situation_forget_flow() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        run_part2_situation(client, &jieba).await;
    }

    // ------------------------------------------------------------------
    // 第三部分测试：三组遮罩
    // ------------------------------------------------------------------
    #[tokio::test]
    #[ignore]
    async fn test_part3_mask_levels() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        run_part3_mask_levels(client, &jieba).await;
    }

    // ------------------------------------------------------------------
    // 第五部分：整体测试 —— 按顺序执行第一、二、三部分（不含第四）
    // ------------------------------------------------------------------
    #[tokio::test]
    #[ignore]
    async fn test_part5_overall() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        let overall_start = Instant::now();

        println!("========== 第五部分：整体测试（按顺序执行第一、二、三部分）==========");
        println!();
        println!("【开始第一部分】");
        part1_intensity_report();

        println!("【开始第二部分 · SemMemory】");
        run_part2_sem(client.clone(), &jieba).await;

        println!("【开始第二部分 · SpecificSituation】");
        run_part2_situation(client.clone(), &jieba).await;

        println!("【开始第三部分】");
        run_part3_mask_levels(client, &jieba).await;

        println!("========== 第五部分结束，总用时 {:?} ==========", overall_start.elapsed());
    }
}
