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
// 单元测试
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::decay_calculator::compute_missing_degree;
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
            .last_forget_time(created)
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
        .last_forget_time(created)
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
        .last_forget_time(past)
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

    /// 验证 lazy_forget 后将缺失度持久化到 MemoryNote.missing_degree 字段
    #[tokio::test]
    async fn test_missing_degree_stored_in_node() {
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "鲁迅原名周树人浙江绍兴人".to_string(),
            ConceptType::Entity,
            "人物".to_string(),
        )))
        .create_time(past)
        .last_accessed_time(past)
        .last_forget_time(past)
        .build()
        .unwrap();
        // 遗忘前缺失度为 0
        assert_eq!(node.missing_degree(), 0.0);

        let _ = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_, _| async {
            Ok("重建".to_string())
        })
        .await;

        // 遗忘后缺失度已写入 MemoryNote 字段，与增量计算结果一致
        let stored = node.missing_degree();
        let expected = compute_missing_degree(
            past, 0, Utc::now(),
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP,
        );
        assert!((stored - expected).abs() < 0.001, "stored={stored}, expected={expected}");
    }

    /// 验证仅计算缺失度（不触发遮罩/LLM）的接口：Procedure 节点也会更新缺失度
    #[tokio::test]
    async fn test_procedure_updates_missing_degree_only() {
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let mut node = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
            Action::new("test".to_string(), ActionType::Think),
        )))
        .create_time(past)
        .last_accessed_time(past)
        .last_forget_time(past)
        .build()
        .unwrap();

        // lazy_forget 对 Procedure 返回 NoAction，但仍会更新缺失度
        let result = lazy_forget(&mut node, Utc::now(), &Jieba::new(), None, |_, _| async {
            Ok("x".to_string())
        })
        .await;
        assert!(matches!(result, ForgetAction::NoAction));
        assert!(node.missing_degree() > MASK_THRESHOLD);
    }

    /// 验证连续两次仅计算缺失度位于同一条遗忘曲线上：
    /// 未触发巩固时，增量计算结果应与从创建时间直接计算的曲线值一致。
    #[tokio::test]
    async fn test_missing_degree_curve_continuity() {
        let create_time = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let t1 = Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap(); // 创建后 24h
        let t2 = Utc.with_ymd_and_hms(2024, 6, 3, 0, 0, 0).unwrap(); // 创建后 48h
        let mut node = make_old_semantic("鲁迅原名周树人浙江绍兴人", create_time);

        // 两次都仅计算缺失度（不触发遮罩 / LLM）
        let md1 = compute_and_update_missing_degree(&mut node, t1);
        let md2 = compute_and_update_missing_degree(&mut node, t2);

        // 与从创建时间直接计算的曲线值对比
        let expect1 = compute_missing_degree(
            create_time, 0, t1,
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP,
        );
        let expect2 = compute_missing_degree(
            create_time, 0, t2,
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP,
        );
        assert!((md1 - expect1).abs() < 0.01, "md1={md1} expect1={expect1}");
        assert!((md2 - expect2).abs() < 0.01, "md2={md2} expect2={expect2}");
        // 遗忘随时间单调增加
        assert!(md2 > md1);
    }

    /// 验证边缺失度独立衰减：不依赖节点缺失度，公式与节点一致
    #[test]
    fn test_edge_decay_stores_missing_degree() {
        use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType, sem_mem::SemMemLink};
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap(); // +24h
        let mut link = MemoryLink::new(
            Default::default(),
            Default::default(),
            MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 1.0)),
        );
        link.set_last_forget_time(past);
        assert_eq!(link.missing_degree(), 0.0);

        let new_intensity = decay_edge(&mut link, now);
        let expect_md = 1.0 - (-24.0_f32 / (24.0 / std::f32::consts::LN_2)).exp();
        assert!((link.missing_degree() - expect_md).abs() < 0.01, "md={}", link.missing_degree());
        assert_eq!(link.last_forget_time(), now);
        assert!((new_intensity - (1.0 - expect_md as f64)).abs() < 0.01);
    }

    /// 验证图中边（GraphMemoryLink）独立衰减
    #[test]
    fn test_graph_edge_decay() {
        use soul_mem_runtime::cluster::memory_cluster::GraphMemoryLink;
        use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType, sem_mem::SemMemLink};
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap();
        let core_link = MemoryLink::new(
            Default::default(),
            Default::default(),
            MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 1.0)),
        );
        let mut link: GraphMemoryLink = core_link.into();
        link.set_last_forget_time(past);

        let new_intensity = decay_graph_edge(&mut link, now);
        let expect_md = 1.0 - (-24.0_f32 / (24.0 / std::f32::consts::LN_2)).exp();
        assert!((link.missing_degree() - expect_md).abs() < 0.01);
        assert_eq!(link.last_forget_time(), now);
        assert!((new_intensity - (1.0 - expect_md as f64)).abs() < 0.01);
    }

    /// 验证只读缺失度计算不写回节点、且结果与曲线一致
    #[tokio::test]
    async fn test_current_missing_degree_readonly() {
        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap();
        let node = make_old_semantic("测试节点", past);

        let md = current_missing_degree(&node, now);
        // 节点存储字段未被修改
        assert_eq!(node.missing_degree(), 0.0);
        assert_eq!(node.last_forget_time(), past);
        // 与曲线值一致
        let expect = compute_missing_degree(
            past, 0, now,
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP,
        );
        assert!((md - expect).abs() < 0.01);
    }

    /// 验证权重占位符返回 (1 - md)
    #[test]
    fn test_weight_placeholder() {
        assert_eq!(weight_placeholder(0.0), 1.0);
        assert_eq!(weight_placeholder(0.5), 0.5);
        assert_eq!(weight_placeholder(1.0), 0.0);
        assert_eq!(weight_placeholder(2.0), 0.0); // 越界被 clamp
    }

    /// 验证全图节点/边遗忘度批量计算（不触发遮罩）
    #[test]
    fn test_compute_all_missing_degrees() {
        use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType, sem_mem::SemMemLink};
        use soul_mem_core::memory_note::MemoryId;
        use soul_mem_query::embedding::{
            EmbeddingVec,
            note::{EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant},
            sem::SemanticEmbedding,
        };

        let past = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2024, 6, 2, 0, 0, 0).unwrap();
        let mut cluster = MemoryCluster::new();
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        let make_embedded = |id: MemoryId, content: &str| -> EmbeddedMemoryNote {
            let mem_type = MemoryType::Semantic(SemMemory::new(
                content.to_string(),
                ConceptType::Entity,
                "desc".to_string(),
            ));
            let note = MemoryNoteBuilder::new(mem_type)
                .id(id)
                .create_time(past)
                .last_accessed_time(past)
                .last_forget_time(past)
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
        cluster.add_single_node(make_embedded(id1, "节点A"));
        cluster.add_single_node(make_embedded(id2, "节点B"));

        // 添加一条边并使其 last_forget_time 回到 past
        let mut link = MemoryLink::new(
            id1, id2,
            MemoryLinkType::Sem(SemMemLink::new("related".to_string(), 1.0)),
        );
        link.set_last_forget_time(past);
        {
            let graph = cluster.graph_mut();
            for n in graph.node_weights_mut() {
                if n.note().id() == id1 {
                    n.note.links_mut().push(link.clone());
                }
            }
        }
        cluster.refresh_node(&id1);

        // 全图批量计算（不遮罩）
        compute_all_missing_degrees(&mut cluster, now);

        let expect = 1.0 - (-24.0_f32 / (24.0 / std::f32::consts::LN_2)).exp();
        let graph = cluster.graph();
        for n in graph.node_weights() {
            assert!((n.note().missing_degree() - expect).abs() < 0.01, "node md={}", n.note().missing_degree());
            assert_eq!(n.note().last_forget_time(), now);
        }
        let mut edge_count = 0;
        for e in graph.edge_weights() {
            edge_count += 1;
            assert!((e.missing_degree() - expect).abs() < 0.01, "edge md={}", e.missing_degree());
            assert_eq!(e.last_forget_time(), now);
        }
        assert_eq!(edge_count, 1);
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
    use super::super::decay_calculator::compute_missing_degree;
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
        ))).create_time(created).last_accessed_time(created).last_forget_time(created).build().unwrap();
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

    /// 共享辅助：对指定初始内容执行 4 个激活档位的遗忘对比并打印三段文本
    async fn run_activation_compare(
        client: Arc<LlmClient>,
        jieba: &Jieba,
        now: DateTime<Utc>,
        content: String,
        title: &str,
    ) {
        let created = now - chrono::Duration::hours(48);
        let desc_prefix: String = content.chars().take(30).collect();

        let make_node = |rc: usize| -> MemoryNote {
            let mut n = build_complete_sem_node(created);
            if let MemoryType::Semantic(s) = n.mem_type_mut() {
                s.content = content.clone();
                s.aliases = vec!["红魔馆的记忆".to_string(), "咲夜的日常".to_string()];
                s.description = format!("{}: {}", title, desc_prefix);
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

        println!("\n========== 激活次数对遗忘的影响（{}）==========", title);
        println!(
            "初始内容: {}\n缺失度: A={:.0}% B={:.0}% D={:.0}% C={:.0}%\n",
            content,
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
                jieba,
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

    /// 激活次数对遗忘的影响 —— 中文初始内容（LLM 生成）
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
            let mut resp = gen_client
                .call_llm(vec![
                    ChatCompletionRequestSystemMessage::from("你是红魔馆的女仆长十六夜咲夜。请以第一人称写一段你在幻想乡日常生活中的具体事件记忆，2~4句话，描述发生了什么、涉及谁、你的感受。只输出记忆文本，不要解释。".to_string()).into(),
                    ChatCompletionRequestUserMessage::from("请讲述一件你在红魔馆经历过的难忘事件。".to_string()).into(),
                ])
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
                .expect("LLM 生成失败");
            resp.remove(0)
        };
        // 中文无需空格，去除标点与空白便于 jieba 分词
        let content: String = generated
            .chars()
            .filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace())
            .collect();

        run_activation_compare(client, &jieba, now, content, "中文").await;
    }

    /// 激活次数对遗忘的影响 —— 纯英文初始内容
    #[tokio::test]
    #[ignore]
    async fn test_activation_slows_forgetting_english() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        let now = Utc::now();

        // 保留空格以让 jieba 按单词切分
        let content = "On a quiet night, I spent hours polishing the silver cutlery of the Scarlet Devil Mansion, when Patchouli suddenly asked me to brew a pot of Darjeeling tea. The library glowed with candlelight and the scent of old books."
            .to_string();

        run_activation_compare(client, &jieba, now, content, "纯英文").await;
    }

    /// 激活次数对遗忘的影响 —— 中英混合初始内容（中文为主 + 重要英文名词）
    #[tokio::test]
    #[ignore]
    async fn test_activation_slows_forgetting_mixed() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();
        let now = Utc::now();

        let content = "那天深夜我在红魔馆的大厅擦洗 silver cutlery，Patchouli 小姐突然要我泡一壶 Darjeeling tea。图书馆里 candlelight 摇曳，空气中弥漫着 old books 的味道，我停下脚步享受了片刻的宁静。"
            .to_string();

        run_activation_compare(client, &jieba, now, content, "中英混合").await;
    }

    /// 两阶段遗忘：第一次仅计算缺失度，第二次计算缺失度并触发遮罩遗忘。
    /// 输出两次的缺失度与时间差，以及遮罩遗忘过程中的原文本 / 遮罩文本 / 推测文本。
    /// 同时验证两次缺失度位于同一条遗忘曲线上（无巩固时不偏离曲线）。
    #[tokio::test]
    #[ignore]
    async fn test_two_phase_forget_curve_and_texts() {
        dotenvy::dotenv().ok();
        let client = Arc::new(try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL"));
        let jieba = Jieba::new();

        // 节点创建于 72 小时前，两次计算时刻分别为创建后 24h、72h
        let create_time = Utc::now() - chrono::Duration::hours(72);
        let t1 = Utc::now() - chrono::Duration::hours(48);
        let t2 = Utc::now();

        let mut node = build_complete_sem_node(create_time);

        // 第一次：仅计算缺失度（不触发遮罩 / LLM）
        let md1 = compute_and_update_missing_degree(&mut node, t1);
        let delta1 = (t1 - create_time).num_hours();

        // 第二次：计算缺失度 + 遮罩遗忘（LLM 推测）
        let before = get_summary(&node).unwrap();
        let result = lazy_forget(
            &mut node,
            t2,
            &jieba,
            SAKUYA_RECONSTRUCT,
            make_llm_closure(client.clone()),
        )
        .await;
        let md2 = node.missing_degree();
        let delta2 = (t2 - create_time).num_hours();

        println!("\n========== 两阶段遗忘曲线一致性 ==========");
        println!("创建时间:          {:?}", create_time);
        println!("第一次(仅计算):     Δt = {}h   缺失度 = {:.2}%", delta1, md1 * 100.0);
        println!("第二次(计算+遮罩):  Δt = {}h   缺失度 = {:.2}%", delta2, md2 * 100.0);
        println!();
        println!("【遮罩遗忘过程】");
        println!("  原始文本: {}", before);
        match &result {
            ForgetAction::Revised {
                masked_text,
                new_summary,
                ..
            } => {
                println!("  遮罩文本: {}", masked_text);
                println!("  推测文本: {}", new_summary);
            }
            ForgetAction::MaskOnly { masked_text, .. } => {
                println!("  遮罩文本: {} (LLM 失败降级)", masked_text);
            }
            ForgetAction::NoAction => println!("  (未触发遗忘)"),
        }
        println!("=========================================\n");

        // 曲线一致性：两次缺失度均应与从创建时间直接计算的曲线值一致
        let expect1 = compute_missing_degree(
            create_time, 0, t1,
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP,
        );
        let expect2 = compute_missing_degree(
            create_time, 0, t2,
            DEFAULT_BASE_HALF_LIFE_HOURS, DEFAULT_ACTIVE_FACTOR, DEFAULT_MAX_ACTIVATION_CAP,
        );
        assert!((md1 - expect1).abs() < 0.01, "md1={md1} expect1={expect1}");
        assert!((md2 - expect2).abs() < 0.01, "md2={md2} expect2={expect2}");
        assert!(md2 > md1, "缺失度应随遗忘单调增加");
        assert!(matches!(
            result,
            ForgetAction::Revised { .. } | ForgetAction::MaskOnly { .. }
        ));
    }
}
