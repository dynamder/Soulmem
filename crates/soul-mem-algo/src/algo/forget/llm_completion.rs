use soul_mem_core::memory_note::sem_mem::ConceptType;
use soul_mem_core::memory_note::{MemoryNote, MemoryType};
use std::future::Future;

/// 遗忘度低于此值时 Vec 类字段（如 aliases）在对齐时不允许增加长度
pub const ALIGN_LENGTH_CAP_THRESHOLD: f32 = 0.6;

/// 默认记忆重建 system prompt（中立通用）。
///
/// 关键约束：
/// - 每个 `[masked]` 占位符对应原文的一个词（jieba token），占位符数量即缺失信息量；
/// - **全部**占位符必须被补全，输出中不得残留 `[masked]`；
/// - 文本全为遮罩（无任何可推断上下文）时，直接输出固定句。
pub const DEFAULT_RECONSTRUCT_SYSTEM_PROMPT: &str = "You are a memory reconstruction assistant. \
    A segment of memory text has been partially masked with [masked] placeholders. \
    Each [masked] placeholder corresponds to exactly one missing word/token of the original text, \
    so the number of placeholders tells you how much information is missing. \
    Based on the remaining context, infer and fill in ALL placeholders naturally; \
    the output must contain NO [masked] placeholders — every one must be completed. \
    If the text is entirely masked with no remaining context to infer from, \
    output exactly: \"I totally forget it and cannot recall anything.\" \
    Output only the completed text, no explanation.";

/// 全遮罩时 LLM 应输出的固定句（无上下文可推断 → 明确表示遗忘）。
pub const FULLY_MASKED_REPLY: &str = "I totally forget it and cannot recall anything.";

// ========================================================================
// 记忆重建（遮罩文本 → LLM → 完整文本）
// ========================================================================

/// 统计遮罩文本中的 `[masked]` 占位符数量。
pub fn count_mask_placeholders(masked_text: &str) -> usize {
    masked_text.matches("[masked]").count()
}

/// 判断遮罩文本是否**全部**被遮罩（去掉占位符后无剩余上下文）。
pub fn is_fully_masked(masked_text: &str) -> bool {
    let cleaned: String = masked_text.replace("[masked]", "");
    cleaned.trim().is_empty()
}

/// 构建记忆重建的 system + user prompt。
/// `system_prompt` 控制 LLM 的角色设定与行为，传入 `None` 使用默认值。
/// user 提示中附带占位符数量（= 缺失信息量），强化"必须全部补全"的约束。
pub fn build_reconstruct_prompt(
    masked_text: &str,
    system_prompt: Option<&str>,
) -> (String, String) {
    let system = system_prompt
        .unwrap_or(DEFAULT_RECONSTRUCT_SYSTEM_PROMPT)
        .to_string();
    let count = count_mask_placeholders(masked_text);
    let user = format!(
        "Masked text: {}\nThis text contains {} [masked] placeholder(s), each standing for one missing token. Fill them ALL in.",
        masked_text, count
    );
    (system, user)
}

/// 调用 LLM 重建遮罩的记忆文本。
/// `system_prompt` 控制 LLM 的角色设定与行为，传入 `None` 使用默认值。
///
/// 全遮罩（无剩余上下文）时不调用 LLM，直接返回 [`FULLY_MASKED_REPLY`]，
/// 保证结果确定且符合"无法回忆"语义。
pub async fn reconstruct_summary<F, Fut>(
    masked_text: &str,
    system_prompt: Option<&str>,
    llm_call: F,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnOnce(&str, &str) -> Fut,
    Fut: Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>>,
{
    if is_fully_masked(masked_text) {
        return Ok(FULLY_MASKED_REPLY.to_string());
    }
    let (system, user) = build_reconstruct_prompt(masked_text, system_prompt);
    llm_call(&system, &user).await
}

/// 默认字段对齐 system prompt（中立通用）
pub const DEFAULT_ALIGN_SYSTEM_PROMPT: &str = "You are a memory consistency checker. \
    Given a memory's content text, verify and if necessary correct the aliases, description, \
    and concept type fields so they match the content.\n\
    Respond ONLY in this exact format, one field per line:\n\
    Aliases: <comma-separated list>\n\
    Description: <short phrase>\n\
    ConceptType: Entity|Abstract\n\
    If the current values are already consistent with the content, keep them unchanged.\n\
    Do not add any explanation.";

// ========================================================================
// 字段对齐（SemMemory 的 aliases / description / concept_type 修正）
// ========================================================================

/// 构建字段对齐的 prompt。
/// `system_prompt` 控制 LLM 的角色设定与行为，传入 `None` 使用默认值。
pub fn build_align_prompt(
    content: &str,
    aliases: &[String],
    description: &str,
    concept_type: &str,
    system_prompt: Option<&str>,
) -> (String, String) {
    let system = system_prompt
        .unwrap_or(DEFAULT_ALIGN_SYSTEM_PROMPT)
        .to_string();
    let user = format!(
        "Content: {}\nCurrent aliases: {:?}\nCurrent description: {}\nCurrent concept type: {}",
        content, aliases, description, concept_type,
    );
    (system, user)
}

/// 解析 LLM 返回的结构化字段对齐结果。
/// 返回 (new_aliases, new_description, new_concept_type)，未被 LLM 提及的字段为 None。
pub fn parse_align_response(
    response: &str,
) -> (Option<Vec<String>>, Option<String>, Option<ConceptType>) {
    let mut new_aliases: Option<Vec<String>> = None;
    let mut new_desc: Option<String> = None;
    let mut new_ct: Option<ConceptType> = None;

    for line in response.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("Aliases:") {
            let val = val.trim();
            new_aliases = if val.is_empty() || val.eq_ignore_ascii_case("none") {
                Some(vec![])
            } else {
                Some(
                    val.split(',')
                        .map(|s| s.trim().trim_matches('"').to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                )
            };
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

    (new_aliases, new_desc, new_ct)
}

/// 调用 LLM 执行 SemMemory 字段对齐：根据新 content 修正 aliases / description / concept_type。
///
/// - `system_prompt` 控制 LLM 的角色设定与行为，传入 `None` 使用默认值
/// - 当缺失度 < `ALIGN_LENGTH_CAP_THRESHOLD` 时，aliases 的长度不允许增长
/// - 当缺失度 ≥ 阈值时，允许自由增长
pub async fn align_sem_fields<F, Fut>(
    node: &mut MemoryNote,
    system_prompt: Option<&str>,
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

    let (system, user) =
        build_align_prompt(&content, &old_aliases, &old_desc, &old_ct, system_prompt);
    let response = llm_call(&system, &user).await?;

    let (new_aliases, new_desc, new_ct) = parse_align_response(response.trim());

    // 使用节点当前存储的缺失度，决定是否限制 Vec 长度
    let missing_degree = node.missing_degree();
    let cap_vec_length = missing_degree < ALIGN_LENGTH_CAP_THRESHOLD;

    // 应用解析结果到节点
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
