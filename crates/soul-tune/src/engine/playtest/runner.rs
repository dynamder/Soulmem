use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;

use soul_mem_algo::algo::retrieve::{
    association::AssociationConfig,
    complex::{AssociateWithActionConfig, RetrAssociateWithAction},
    similarity::{RetrSimilarity, SimilarityConfig},
    RetrStrategy,
};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
};
use soul_mem_query::embedding::{Embeddable, EmbeddingModel};
use soul_mem_query::query::retrieve::{
    EnvironmentQueryUnit, EventQueryUnit, LocationQueryUnit, MemoryRetrieveQuery,
    MemoryRetrieveQueryVariant, ParticipantQueryUnit, PrioritizedMemoryRetrieveQuery,
    SemanticQueryUnit, SituationQueryUnit,
};
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::engine::llm::LlmBackend;
use crate::engine::loader::{cached_load_graph, get_bge_model};
use crate::engine::retrieve::data::NodeSummary;

use super::repair::{
    extract_balanced_array, extract_think_content, robust_json_extract, run_paw, strip_think_block,
    RawQuery, RawSemUnit, RawSitUnit, RawVariant,
};
use super::trace::{HitStage, QueryTrace, RetrievalTrace, TracedNode};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PlayConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub action_top_k: usize,
    pub ppr_top_k: usize,
    pub damping_factor: f64,
    pub residue_threshold: f64,
    pub runs_per_turn: usize,
    pub merged_top_k: usize,
}

const CHAT_INSTRUCTION: &str = "注意：这是短信聊天场景，回复必须自然口语化，像真人发消息。\
严禁使用括号描述动作、神态或心理活动，如（笑）、（叹气）、*摇头*等。\
只输出对话内容，不加任何表演注释。\
回复必须简短，一句话即可，不要重复用户的话，不要解释你的回复。";

const ENTITY_EXTRACT_SLUG: &str = "soul-tune-entity-extract-v1";
const ENTITY_EXTRACT_SPEC: &str = r#"You are an entity extraction tool for a character memory retrieval system.
Extract all key entities from the user's message that the character needs to recall in order to respond naturally.

Extract:
- Person names: people, characters, friends or foes mentioned or strongly implied
- Location names: places, venues, regions
- Concrete concepts: rules, terms, titles, skills
- Items and objects: tools, weapons, artifacts

Rules:
- Each entity must be a concrete noun phrase, e.g. "弹幕规则", "神社", "博丽灵梦", "阴阳玉"
- Do NOT extract abstract categories like "爱好", "技能", "朋友"
- Only include entities present in or strongly implied by the message
- Keep the original language of the message

Output format: a JSON array of strings, e.g. ["博丽灵梦", "神社", "弹幕规则"]
Output ONLY the JSON array. No markdown, no explanations."#;

/// 生成后校验的兜底分常量（默认与 SimilarityConfig / PlayConfig 兜底分一致）：
/// top-1 命中分低于该值的查询视为无对应记忆，直接丢弃。
pub const QUERY_VALIDATION_FLOOR: f32 = 0.35;

/// 记忆线索固定条数（k 固定为 5，提示词开销不随图规模增长）。
pub const HINT_TOP_K: usize = 5;

/// 记忆线索多样性替换的考察范围：top-k 内无 Situation 时，在 top-lookahead
/// 内寻找分数达标的 Situation 节点替换第 k 个 hint。
pub const HINT_LOOKAHEAD_K: usize = 10;

/// hint 检索耗时护栏（release）：实测超过该阈值时自动把 k 降为 3。
pub const HINT_MAX_ELAPSED: Duration = Duration::from_millis(100);

/// 耗时超限时的降级 k。
pub const HINT_FALLBACK_K: usize = 3;

/// 单条记忆线索内容摘要的最大字符数（超出截断，控制提示词长度）。
pub const HINT_CONTENT_MAX_CHARS: usize = 50;

/// 查询数量上限（4-8 的上界；下界由提示词引导，不强制）。
pub const QUERY_MAX_COUNT: usize = 8;

/// 空回退：主查询全部被丢弃或为空时，用提取实体构造 Semantic 兜底查询的条数。
pub const FALLBACK_ENTITY_TOP: usize = 3;

/// 空回退查询的 priority（低于正常重要度，避免淹没主检索）。
pub const FALLBACK_QUERY_PRIORITY: u32 = 2;

/// priority 小偏移上限：分数接近（≤0.05）时保护重要查询的命中不被淹没，
/// 分数差距较大时仍由分数主导。
const PRIORITY_BONUS_MAX: f64 = 0.05;

/// priority → 排序偏移：`PRIORITY_BONUS_MAX × (p / p_max)`。
fn priority_bonus(p: u32, p_max: u32) -> f64 {
    if p_max == 0 {
        return 0.0;
    }
    PRIORITY_BONUS_MAX * (p as f64 / p_max as f64)
}

/// 将某条查询的结果按"分数主导 + priority 小偏移"并入全局合并表。
/// 合并键 = `原始分 + bonus`；同一节点被多条查询命中时保留键最大的那条，
/// TracedNode.score 始终为原始融合分（展示用，0–1 量纲）。
fn fold_priority_nodes(
    merged: &mut HashMap<MemoryId, (f64, TracedNode)>,
    nodes: Vec<TracedNode>,
    bonus: f64,
) {
    for node in nodes {
        let key = node.score + bonus;
        match merged.entry(node.id) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let (best_key, best) = e.get_mut();
                if key > *best_key {
                    *best_key = key;
                    *best = node;
                }
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((key, node));
            }
        }
    }
}

/// 将合并表按排序键（原始分 + priority 偏移）降序转为节点列表，展示分为原始分。
fn finish_merged(merged: HashMap<MemoryId, (f64, TracedNode)>) -> Vec<TracedNode> {
    let mut nodes: Vec<(f64, TracedNode)> = merged.into_values().collect();
    nodes.sort_by(|a, b| {
        b.0
            .partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nodes.into_iter().map(|(_, node)| node).collect()
}

/// 一条已完成生成后校验的查询：嵌入结果随查询缓存，检索阶段直接复用，避免二次嵌入。
/// `dropped = true` 表示校验未通过（top-1 低于兜底分或嵌入失败），仅保留用于 trace 标记。
#[derive(Debug, Clone)]
struct PreparedQuery {
    query: PrioritizedMemoryRetrieveQuery,
    embedding: Option<MemoryRetrieveQueryEmbedding>,
    dropped: bool,
}

/// 生成一个仅用于 trace 的 dropped 标记查询轨迹（不参与检索）。
fn dropped_trace(query: &MemoryRetrieveQuery) -> QueryTrace {
    QueryTrace {
        query: query.clone(),
        sim_nodes: vec![],
        sim_elapsed: Duration::ZERO,
        ppr_nodes: vec![],
        ppr_elapsed: Duration::ZERO,
        action_nodes: vec![],
        action_elapsed: Duration::ZERO,
        total_elapsed: Duration::ZERO,
        dropped: true,
    }
}

/// 记忆线索候选：来自相似度检索的命中节点（已按分数降序）。
#[derive(Debug, Clone)]
struct HintHit {
    id: MemoryId,
    score: f32,
    is_situation: bool,
    summary: String,
}

/// 从相似度结果中选择记忆线索：
/// - 仅当 top-1 分数 ≥ 兜底分时才注入（寒暄等无关轮次不注入，防止用无关记忆诱导幻觉）；
/// - 默认取 top-k；若 top-k 中没有 Situation 节点，且 top-lookahead 内存在分数 ≥ 兜底分
///   的 Situation 节点，则用最佳 Situation 替换第 k 个 hint（多样性规则）。
fn select_hints(
    hits: Vec<HintHit>,
    k: usize,
    lookahead: usize,
    floor: f32,
) -> Vec<HintHit> {
    if hits.first().map(|h| h.score) < Some(floor) {
        return Vec::new();
    }
    let mut out: Vec<HintHit> = hits.iter().take(k).cloned().collect();
    if out.is_empty() {
        return out;
    }
    if !out.iter().any(|h| h.is_situation) {
        let best_situation = hits
            .iter()
            .take(lookahead)
            .filter(|h| h.is_situation && h.score >= floor)
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .cloned();
        if let Some(best) = best_situation {
            if out.len() >= k {
                out[k - 1] = best;
            } else {
                out.push(best);
            }
        }
    }
    out
}

/// 将解析后的查询数组截断到数量上限（4-8 中的上界 8）。
fn cap_raw_queries(mut raw: Vec<RawQuery>) -> Vec<RawQuery> {
    raw.truncate(QUERY_MAX_COUNT);
    raw
}

/// 空回退：用提取的实体构造 Semantic 兜底查询（priority=2，取前 top 个实体）。
/// 实体为空时返回空数组，检索阶段自然表现为"无检索"。
fn build_fallback_queries(entities: &[String], top: usize) -> Vec<PrioritizedMemoryRetrieveQuery> {
    entities
        .iter()
        .filter(|e| !e.trim().is_empty())
        .take(top)
        .map(|e| {
            MemoryRetrieveQuery::new(
                vec!["实体".to_string()],
                MemoryRetrieveQueryVariant::Semantic(vec![
                    SemanticQueryUnit::new().with_concept_identifier(e.clone()),
                ]),
            )
            .with_priority(FALLBACK_QUERY_PRIORITY)
        })
        .collect()
}

fn raw_sem_to_query(u: RawSemUnit) -> SemanticQueryUnit {
    let mut q = SemanticQueryUnit::new();
    if let Some(c) = u.concept_identifier.filter(|c| !c.trim().is_empty()) {
        q = q.with_concept_identifier(c);
    }
    if let Some(d) = u.description.filter(|d| !d.trim().is_empty()) {
        q = q.with_description(d);
    }
    q
}

fn raw_sit_to_query(u: RawSitUnit) -> SituationQueryUnit {
    let mut q = SituationQueryUnit::new();
    if let Some(n) = u.narrative.filter(|n| !n.trim().is_empty()) {
        q = q.with_narrative(n);
    }
    if let Some(locations) = u.location {
        let units: Vec<LocationQueryUnit> = locations
            .into_iter()
            .filter_map(|l| {
                l.name.filter(|n| !n.trim().is_empty()).map(|name| {
                    let mut lu = LocationQueryUnit::new(name);
                    if let Some(c) = l.coordinates.filter(|c| !c.trim().is_empty()) {
                        lu = lu.with_coordinates(c);
                    }
                    lu
                })
            })
            .collect();
        if !units.is_empty() {
            q = q.with_location(units);
        }
    }
    if let Some(participants) = u.participants {
        let units: Vec<ParticipantQueryUnit> = participants
            .into_iter()
            .map(|p| {
                let mut pu = ParticipantQueryUnit::new();
                if let Some(n) = p.name.filter(|n| !n.trim().is_empty()) {
                    pu = pu.with_name(n);
                }
                if let Some(r) = p.role.filter(|r| !r.trim().is_empty()) {
                    pu = pu.with_role(r);
                }
                pu
            })
            .collect();
        if !units.is_empty() {
            q = q.with_participants(units);
        }
    }
    if let Some(env) = u.environment {
        let mut eu = EnvironmentQueryUnit::new();
        let mut any = false;
        if let Some(a) = env.atmosphere.filter(|a| !a.trim().is_empty()) {
            eu = eu.with_atmosphere(a);
            any = true;
        }
        if let Some(t) = env.tone.filter(|t| !t.trim().is_empty()) {
            eu = eu.with_tone(t);
            any = true;
        }
        if any {
            q = q.with_environment(eu);
        }
    }
    if let Some(events) = u.event {
        let units: Vec<EventQueryUnit> = events
            .into_iter()
            .filter_map(|e| {
                e.action.filter(|a| !a.trim().is_empty()).map(|action| {
                    let mut eu = EventQueryUnit::new(action);
                    if let Some(i) = e.initiator.filter(|i| !i.trim().is_empty()) {
                        eu = eu.with_initiator(i);
                    }
                    if let Some(t) = e.target.filter(|t| !t.trim().is_empty()) {
                        eu = eu.with_target(t);
                    }
                    eu
                })
            })
            .collect();
        if !units.is_empty() {
            q = q.with_event(units);
        }
    }
    q
}

/// Strip 思维链 content from response and trim to a uniform max length for fair comparison.
const RESPONSE_MAX_CHARS: usize = 200;

fn strip_and_trim(s: &str) -> String {
    let mut text = s.to_string();
    while let Some(start) = text.find("<｜end▁of▁thinking｜>") {
        let end = text[start..]
            .find(" response")
            .map(|p| start + p + 7)
            .or_else(|| text[start..].find("<think/>").map(|p| start + p + 8))
            .unwrap_or(text.len());
        text.replace_range(start..end, "");
    }
    let trimmed = text.trim().to_string();
    if trimmed.chars().count() > RESPONSE_MAX_CHARS {
        trimmed.chars().take(RESPONSE_MAX_CHARS).collect()
    } else {
        trimmed
    }
}

impl Default for PlayConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: QUERY_VALIDATION_FLOOR,
            max_results: 4,
            action_top_k: 3,
            ppr_top_k: 8,
            damping_factor: 0.65,
            residue_threshold: 1e-5,
            runs_per_turn: 5,
            merged_top_k: 10,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct DialogueFile {
    pub name: Option<String>,
    pub graph_path: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub config: Option<PlayConfig>,
    pub conversations: Vec<ConversationEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConversationEntry {
    pub user_message: String,
}

#[derive(Debug, Clone)]
pub struct PlayRunSnapshot {
    pub embedding_response: Option<String>,
    pub fullpipeline_response: Option<String>,
    pub swap: bool,
    pub human_pick: Option<u8>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PlayTurnResult {
    pub index: usize,
    pub user_message: String,
    pub system_prompt: String,
    pub generated_queries_json: String,
    pub query_think_content: Option<String>,
    pub embedding_trace: Option<RetrievalTrace>,
    pub fullpipeline_trace: Option<RetrievalTrace>,
    pub runs: Vec<PlayRunSnapshot>,
}

#[derive(Debug, Clone)]
pub struct PlayTestResult {
    pub character_name: String,
    pub config: PlayConfig,
    pub turns: Vec<PlayTurnResult>,
    pub human_role: Option<String>,
}

pub struct PlayTestRunner {
    pub wm: Arc<WorkingMemory>,
    pub system_prompt: String,
    pub graph_names: Arc<HashMap<MemoryId, String>>,
    pub id_names: Arc<HashMap<MemoryId, NodeSummary>>,
    pub config: PlayConfig,
    pub human_role: Option<String>,
}

impl PlayTestRunner {
    pub fn load(graph_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_path = graph_dir.join("graph.json");
        let (wm, id_map) = cached_load_graph(&graph_path)?;

        let reverse_map: HashMap<MemoryId, String> =
            id_map.iter().map(|(k, v)| (*v, k.clone())).collect();

        let id_names = Arc::new(wm.memory_cluster().read_or_compute(|cluster| {
            cluster
                .graph()
                .node_weights()
                .map(|node| {
                    let note = node.note();
                    let id = note.id();
                    let tags = note.tags().to_vec();
                    let (type_label, primary, secondary) = match note.mem_type() {
                        MemoryType::Semantic(sem) => (
                            String::from("语义"),
                            sem.content.clone(),
                            sem.description.clone(),
                        ),
                        MemoryType::Situation(SituationType::SpecificSituation(s)) => (
                            String::from("情境"),
                            s.get_narrative().clone(),
                            s.get_time_span().to_string(),
                        ),
                        MemoryType::Situation(_) => {
                            (String::from("情境"), String::new(), String::new())
                        }
                        MemoryType::Procedure(_) => {
                            (String::from("流程"), String::new(), String::new())
                        }
                    };
                    (
                        id,
                        NodeSummary {
                            tags,
                            type_label,
                            primary,
                            secondary,
                        },
                    )
                })
                .collect::<HashMap<_, _>>()
        }));

        let system_prompt = Self::extract_system_prompt(&wm, &id_map, &reverse_map);
        let config = PlayConfig::default();

        Ok(Self {
            wm: Arc::new(wm),
            system_prompt,
            graph_names: Arc::new(reverse_map),
            id_names,
            config,
            human_role: None,
        })
    }

    pub fn with_config(mut self, config: PlayConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_human_role(mut self, role: Option<String>) -> Self {
        self.human_role = role;
        self
    }

    fn extract_system_prompt(
        wm: &WorkingMemory,
        id_map: &HashMap<String, MemoryId>,
        _reverse_map: &HashMap<MemoryId, String>,
    ) -> String {
        if let Some(self_id) = id_map.get("sem_self") {
            let cluster = wm.memory_cluster();
            let result = cluster.read_or_compute(|c| {
                c.graph().node_weights().find_map(|node| {
                    if node.note().id() == *self_id {
                        match node.note().mem_type() {
                            MemoryType::Semantic(sem) => {
                                Some(format!("{}。{}", sem.content, sem.description))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
            });
            if let Some(prompt) = result {
                return prompt;
            }
        }
        "你是一个角色扮演AI。请根据你的记忆和性格自然地回应。".into()
    }

    pub fn process_turn(
        &self,
        entry: &ConversationEntry,
        turn_index: usize,
        llm: &mut dyn LlmBackend,
    ) -> PlayTurnResult {
        let gen_query_result = self.generate_queries(entry, llm);
        let (queries, queries_json, query_think_content) = match gen_query_result {
            Ok((q, j, tc)) => (q, j, tc),
            Err(e) => {
                return PlayTurnResult {
                    index: turn_index,
                    user_message: entry.user_message.clone(),
                    system_prompt: self.system_prompt.clone(),
                    generated_queries_json: String::new(),
                    query_think_content: None,
                    embedding_trace: None,
                    fullpipeline_trace: None,
                    runs: vec![PlayRunSnapshot {
                        embedding_response: None,
                        fullpipeline_response: None,
                        swap: false,
                        human_pick: None,
                        error: Some(format!("Query generation failed: {}", e)),
                    }],
                };
            }
        };

        let embedding_trace = self.run_embedding_retrieval(&queries);
        let fullpipeline_trace = self.run_fullpipeline_retrieval(&queries);

        let emb_nodes = embedding_trace
            .as_ref()
            .map(|t| t.merged_nodes.clone())
            .unwrap_or_default();
        let full_nodes = fullpipeline_trace
            .as_ref()
            .map(|t| t.merged_nodes.clone())
            .unwrap_or_default();

        let emb_context = self.format_nodes(&emb_nodes);
        let full_context = self.format_nodes(&full_nodes);

        let mut chat_prompt = format!("{}\n\n{}", self.system_prompt, CHAT_INSTRUCTION);
        if let Some(ref role) = self.human_role {
            chat_prompt = format!("{}\n\n现在与你对话的是: {}", chat_prompt, role);
        }
        let emb_prompt = format!("{}\n\n相关记忆:\n{}", chat_prompt, emb_context);
        let full_prompt = format!("{}\n\n相关记忆:\n{}", chat_prompt, full_context);

        let mut runs: Vec<PlayRunSnapshot> = Vec::with_capacity(self.config.runs_per_turn);
        let user_text = match &self.human_role {
            Some(role) => format!(
                "（对方身份: {}）对方发来消息: \"{}\"",
                role, entry.user_message
            ),
            None => format!("\"{}\"", entry.user_message),
        };
        //runs_per_turn可能被配置为0，导致runs为空使评分页越界panic，这里保证至少执行一轮
        let run_count = self.config.runs_per_turn.max(1);
        for _run_idx in 0..run_count {
            let resp_emb = llm.chat(&emb_prompt, &user_text, 512);
            let resp_full = llm.chat(&full_prompt, &user_text, 512);

            let mut errors: Vec<String> = Vec::new();
            let embedding_response = match resp_emb {
                Ok(s) => Some(strip_and_trim(&s)),
                Err(e) => {
                    errors.push(format!("Embedding 响应失败: {}", e));
                    None
                }
            };
            let fullpipeline_response = match resp_full {
                Ok(s) => Some(strip_and_trim(&s)),
                Err(e) => {
                    errors.push(format!("FullPipeline 响应失败: {}", e));
                    None
                }
            };

            let swap = rand::random::<bool>();

            runs.push(PlayRunSnapshot {
                embedding_response,
                fullpipeline_response,
                swap,
                human_pick: None,
                error: if errors.is_empty() {
                    None
                } else {
                    Some(errors.join("; "))
                },
            });
        }

        PlayTurnResult {
            index: turn_index,
            user_message: entry.user_message.clone(),
            system_prompt: self.system_prompt.clone(),
            generated_queries_json: queries_json,
            query_think_content,
            embedding_trace,
            fullpipeline_trace,
            runs,
        }
    }

    /// 通过 PAW 提取消息中的关键实体，用于增强查询生成提示词，防止检索漏掉关键实体。
    /// PAW 不可用或提取为空时，暂时用主对话 LLM 顶上；均失败则返回空列表（不阻断流程）。
    fn extract_entities(&self, user_message: &str, llm: &mut dyn LlmBackend) -> Vec<String> {
        let prompt = format!("消息内容：\n{}\n\n请提取关键实体：", user_message);
        let parse = |raw: &str| {
            serde_json::from_str::<Vec<String>>(raw)
                .ok()
                .or_else(|| {
                    extract_balanced_array(raw)
                        .and_then(|j| serde_json::from_str::<Vec<String>>(&j).ok())
                })
                .unwrap_or_default()
        };
        // 实体列表很短，限制 token 数避免小模型填满上下文导致数分钟等待
        if let Some(raw) = run_paw(ENTITY_EXTRACT_SLUG, ENTITY_EXTRACT_SPEC, &prompt, Some(128)) {
            let entities = parse(&raw);
            if !entities.is_empty() {
                return entities;
            }
        }
        let raw = match llm.chat(ENTITY_EXTRACT_SPEC, &prompt, 128) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        parse(&raw)
    }

    /// 记忆锚点检索：用用户消息全文构造一条 Semantic 查询，经相似度检索取 top-5
    /// 命中节点作为 hint（真实记忆片段，注入提示词防止无中生有）。
    /// - 仅当 top-1 命中分 ≥ 兜底分时才注入（寒暄等无关轮次不注入）；
    /// - Situation 多样性替换：top-5 无 Situation 且 top-10 有达标 Situation 时替换；
    /// - 运行时护栏：实测检索耗时超过阈值时自动降 k 至 3。
    fn retrieve_hints(&self, user_message: &str) -> Vec<String> {
        let start = Instant::now();
        let model = match get_bge_model() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };

        let query = MemoryRetrieveQuery::new(
            Vec::new(),
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier(user_message.to_string()),
            ]),
        );
        let embedded = match query.embed(model) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let sim_config = SimilarityConfig {
            similarity_threshold: self.config.similarity_threshold,
            max_results: HINT_LOOKAHEAD_K,
        };
        let sim_req = sim_config.into_request(
            self.wm.clone(),
            EmbeddedMemoryRetrieveQuery {
                embedding: embedded,
                query: query.clone(),
            },
        );
        let hits: Vec<HintHit> = RetrSimilarity {}
            .retrieve(sim_req)
            .into_iter()
            .map(|(id, score)| HintHit {
                id,
                score,
                is_situation: self
                    .id_names
                    .get(&id)
                    .map(|s| s.type_label == "情境")
                    .unwrap_or(false),
                summary: self.hint_summary(id),
            })
            .collect();

        // 运行时护栏：实测 hint 检索耗时（含嵌入）超过阈值时降 k 至 3
        let k = if start.elapsed() > HINT_MAX_ELAPSED {
            HINT_FALLBACK_K
        } else {
            HINT_TOP_K
        };

        select_hints(hits, k, HINT_LOOKAHEAD_K, self.config.similarity_threshold)
            .into_iter()
            .map(|h| {
                let mut content: String = h.summary.chars().take(HINT_CONTENT_MAX_CHARS).collect();
                if content.trim().is_empty() {
                    content = self
                        .graph_names
                        .get(&h.id)
                        .cloned()
                        .unwrap_or_default();
                }
                format!("- {}", content)
            })
            .collect()
    }

    /// 节点内容摘要（hint 文本）：Semantic 取 content、Situation 取 narrative，
    /// 为空时回退到节点名。
    fn hint_summary(&self, id: MemoryId) -> String {
        self.id_names
            .get(&id)
            .map(|s| s.primary.clone())
            .filter(|p| !p.trim().is_empty())
            .unwrap_or_else(|| self.graph_names.get(&id).cloned().unwrap_or_default())
    }

    /// 构建查询生成提示词：包含字段说明、当前场景说明、记忆线索（真实记忆锚点），
    /// 并以角色自身的视角引导回忆，同时加入防幻觉条款。
    /// 设计依据：question.json 的理想查询中，Semantic 的 concept_identifier 是 graph 节点
    /// aliases 的特征性别名（如 "金发的魔法使" 命中 sem_marisa），Situation 的 narrative
    /// 是 graph 节点 narrative 的 1-2 句压缩转述。因此提示词引导 LLM：
    /// - Semantic 用"身边人怎么称呼"的别名式短语，而非照搬正式名称
    /// - Situation 只用 narrative 讲完整小故事，不填冗余子字段
    /// - 同一概念用多个不同描述覆盖不同角度，提升召回
    /// - 只基于记忆线索与对话内容回想，不编造记忆片段中不存在的事实要素
    fn build_query_prompt(&self, user_message: &str, entities: &[String], hints: &[String]) -> String {
        let scene = self
            .human_role
            .as_ref()
            .map(|r| format!("{}正在与你对话。", r))
            .unwrap_or_else(|| "有人正在与你对话。".to_string());

        let entities_text = if entities.is_empty() {
            String::new()
        } else {
            format!("消息中的关键实体: {}", entities.join("、"))
        };

        // 记忆线索小节：有真实记忆锚点才注入，无关轮次（无 hint）时整节省略
        let clues_section = if hints.is_empty() {
            String::new()
        } else {
            format!(
                "【记忆线索】\n\
                 以下片段来自你的记忆（只作回想线索，不要直接引用原文）：\n{}\n\n",
                hints.join("\n")
            )
        };

        format!(
            "当前场景：{}\n\
             对方说: \"{}\"{}\n\n\
             {}请以角色自身的视角，回想回应这句话所需的相关记忆，输出一个 JSON 数组，4-8 条，每条代表一个回忆方向。\n\n\
             【每条查询的字段】\n\
             - tag: 类型+子类，如 [\"人物\", \"挚友\"]、[\"事件\", \"异变\"]、[\"概念\", \"规则\"]、[\"物品\", \"秘宝\"]、[\"地点\", \"神社\"]、[\"日常\", \"习惯\"]\n\
             - variant: 二选一：\n\
               * Semantic: concept_identifier 用角色视角的特征性别名/转述——就像身边人平时怎么称呼这个人、这件东西、这条规则，不要照搬正式名称。description 可选补充说明。\n\
               * Situation: narrative 用一两句话讲一个完整的小故事（谁、发生了什么、结果如何），只填 narrative 一个字段，不要填 location/participants/environment/event。\n\
             - priority: 整数，越大表示这条回忆越重要。\n\n\
             【示例】\n\
             [\n\
               {{\"tag\": [\"人物\", \"挚友\"], \"variant\": {{\"Semantic\": [{{\"concept_identifier\": \"金发的魔法使\", \"description\": \"经常来神社蹭茶喝的魔法使\"}}]}}, \"priority\": 9}},\n\
               {{\"tag\": [\"物品\", \"秘宝\"], \"variant\": {{\"Semantic\": [{{\"concept_identifier\": \"又硬又重的勾玉\", \"description\": \"神社里最硬的那块\"}}]}}, \"priority\": 7}},\n\
               {{\"tag\": [\"概念\", \"规则\"], \"variant\": {{\"Semantic\": [{{\"concept_identifier\": \"弹幕规则\", \"description\": \"让妖怪和人类不危及性命的对决规矩\"}}]}}, \"priority\": 7}},\n\
               {{\"tag\": [\"事件\", \"异变\"], \"variant\": {{\"Situation\": [{{\"narrative\": \"吸血鬼因为讨厌太阳制造了红雾，灵梦冲进城堡教训了她一顿\"}}]}}, \"priority\": 9}},\n\
               {{\"tag\": [\"事件\", \"挚友\"], \"variant\": {{\"Situation\": [{{\"narrative\": \"魔理沙被怨灵附身消失了，灵梦当时很着急\"}}]}}, \"priority\": 8}},\n\
               {{\"tag\": [\"日常\", \"习惯\"], \"variant\": {{\"Situation\": [{{\"narrative\": \"灵梦每天在神社喝茶扫地，检查空空的赛钱箱\"}}]}}, \"priority\": 4}}\n\
             ]\n\n\
             【要点】\n\
             - 同一概念可以用多个不同描述的查询覆盖不同角度，提升召回\n\
             - Situation 只填 narrative，不要填其他子字段\n\
             - 只基于记忆线索与对话内容回想，不要编造线索中不存在的人物、事件、细节或关系\n\
             - Situation 的 narrative 必须是真实记忆的转述：可以换措辞，但事实要素必须来自记忆线索\n\
             - 如果当前对话没有任何对应记忆，只输出 1-3 条实体/概念查询，或输出空数组 []\n\
             只输出 JSON 数组，不要其他内容。",
            scene, user_message, entities_text, clues_section
        )
    }

    fn generate_queries(
        &self,
        entry: &ConversationEntry,
        llm: &mut dyn LlmBackend,
    ) -> Result<(Vec<PreparedQuery>, String, Option<String>), String> {
        // 第一步：PAW 提取关键实体，补充到提示词中防止漏掉关键实体
        let entities = self.extract_entities(&entry.user_message, llm);

        // 第二步：记忆锚点 hint 检索（真实记忆片段，防止无中生有）
        let hints = self.retrieve_hints(&entry.user_message);

        // 第三步：构建含记忆线索与防幻觉条款的查询提示词
        let query_prompt = self.build_query_prompt(&entry.user_message, &entities, &hints);

        let text = llm
            .chat(&self.system_prompt, &query_prompt, 2048)
            .map_err(|e| format!("LLM query gen failed: {}", e))?;

        let debug_path = std::env::temp_dir().join("soul_tune_llm_output.txt");
        let entity_text = if entities.is_empty() {
            String::from("(无)")
        } else {
            entities.join("、")
        };
        let debug_entry = format!(
            "=== 用户: {} ===\n提取实体: {}\n查询提示词:\n{}\n完整输出:\n{}\n<|end|>\n\n",
            entry.user_message, entity_text, query_prompt, text
        );
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&debug_path)
        {
            use std::io::Write;
            let _ = f.write_all(debug_entry.as_bytes());
        }

        let think_content = extract_think_content(&text);
        let clean = strip_think_block(&text);

        let json_str = robust_json_extract(&clean, llm).ok_or_else(|| {
            format!(
                "No JSON array found in LLM output (think stripped): {}\n---完整原始输出---\n{}",
                clean, text
            )
        })?;

        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&debug_path)
        {
            use std::io::Write;
            let _ = writeln!(f, "提取的 JSON:\n{}\n<|end_json|>\n\n", json_str);
        }

        let raw: Vec<RawQuery> = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(orig_err) => {
                // 单个畸形 query 不应拖垮整轮：尝试逐条解析，跳过无效项，保留有效项。
                let salvaged: Vec<RawQuery> =
                    match serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                        Ok(items) => items
                            .into_iter()
                            .filter_map(|v| serde_json::from_value::<RawQuery>(v).ok())
                            .collect(),
                        Err(_) => Vec::new(),
                    };
                if !salvaged.is_empty() {
                    salvaged
                } else if let Some(repaired) = super::repair::repair_json(&json_str, llm) {
                    if let Ok(v) = serde_json::from_str(&repaired) {
                        v
                    } else {
                        return Err(format!(
                            "Failed to parse LLM query JSON (PAW repair also failed): {}\n  提取的 JSON: {}\n  PAW修复后: {}\n  完整输出: {}",
                            orig_err, json_str, repaired, text
                        ));
                    }
                } else {
                    return Err(format!(
                        "Failed to parse LLM query JSON (PAW unavailable): {}\n  提取的 JSON: {}\n  完整输出: {}",
                        orig_err, json_str, text
                    ));
                }
            }
        };

        // 查询数量上限：4-8 中的上界 8（减少无效生成）
        let raw = cap_raw_queries(raw);

        let queries: Vec<PrioritizedMemoryRetrieveQuery> = raw
            .into_iter()
            .map(|r| {
                let variant = match r.variant {
                    RawVariant::Semantic { Semantic: units } => {
                        MemoryRetrieveQueryVariant::Semantic(
                            units.into_iter().map(raw_sem_to_query).collect(),
                        )
                    }
                    RawVariant::Situation { Situation: units } => {
                        MemoryRetrieveQueryVariant::Situation(
                            units.into_iter().map(raw_sit_to_query).collect(),
                        )
                    }
                    RawVariant::SemanticSingle(u) | RawVariant::BareSingle(u) => {
                        MemoryRetrieveQueryVariant::Semantic(vec![raw_sem_to_query(u)])
                    }
                    RawVariant::SituationSingle(u) => {
                        MemoryRetrieveQueryVariant::Situation(vec![raw_sit_to_query(u)])
                    }
                    RawVariant::BareArray(units) => MemoryRetrieveQueryVariant::Semantic(
                        units.into_iter().map(raw_sem_to_query).collect(),
                    ),
                };
                MemoryRetrieveQuery::new(r.tag, variant).with_priority(r.priority)
            })
            .collect();

        // 第四步：生成后校验 + 空回退——逐条嵌入并检查 top-1 兜底分，低于兜底分
        // 丢弃并在 trace/日志标记 dropped；嵌入结果缓存进查询对象，检索阶段直接
        // 复用不二次嵌入。主查询全部被丢弃（或 LLM 返回空数组）时用提取实体兜底。
        let prepared = self.prepare_queries(queries, &entities);

        let dropped_count = prepared.iter().filter(|p| p.dropped).count();
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&debug_path)
        {
            use std::io::Write;
            let _ = writeln!(
                f,
                "校验丢弃: {} 条 / 共 {} 条（含空回退）\n",
                dropped_count,
                prepared.len()
            );
        }

        Ok((prepared, json_str, think_content))
    }

    /// 生成后校验 + 空回退：
    /// - 对主查询逐条嵌入并检查 top-1 兜底分，达标保留，否则丢弃并标记 dropped；
    /// - 若主查询全部被丢弃（或为空数组），用提取实体构造 Semantic 兜底查询
    ///   （priority=2，top-3 实体）并同样校验；仍全空才真正无检索。
    /// 嵌入模型不可用时跳过校验（保持旧行为，检索阶段自行嵌入）。
    fn prepare_queries(
        &self,
        main_queries: Vec<PrioritizedMemoryRetrieveQuery>,
        entities: &[String],
    ) -> Vec<PreparedQuery> {
        let model: Option<&dyn EmbeddingModel> = get_bge_model().ok().map(|m| m as &dyn EmbeddingModel);
        let mut prepared: Vec<PreparedQuery> = Vec::new();

        for q in main_queries {
            match model {
                Some(m) => match self.validate_query(q.clone(), m) {
                    Some(p) => prepared.push(p),
                    None => prepared.push(PreparedQuery {
                        query: q,
                        embedding: None,
                        dropped: true,
                    }),
                },
                None => prepared.push(PreparedQuery {
                    query: q,
                    embedding: None,
                    dropped: false,
                }),
            }
        }

        // 空回退：主查询全部被丢弃或为空时，用提取实体兜底
        if !prepared.iter().any(|p| !p.dropped) {
            for q in build_fallback_queries(entities, FALLBACK_ENTITY_TOP) {
                match model {
                    Some(m) => match self.validate_query(q.clone(), m) {
                        Some(p) => prepared.push(p),
                        None => prepared.push(PreparedQuery {
                            query: q,
                            embedding: None,
                            dropped: true,
                        }),
                    },
                    None => prepared.push(PreparedQuery {
                        query: q,
                        embedding: None,
                        dropped: false,
                    }),
                }
            }
        }

        prepared
    }

    /// 生成后校验：嵌入查询并用相似度检索取 top-1 原始分，低于兜底分视为
    /// 无对应记忆（幻觉候选）直接丢弃；校验时已嵌入的结果随 PreparedQuery 返回，
    /// 检索阶段直接复用。
    fn validate_query(
        &self,
        query: PrioritizedMemoryRetrieveQuery,
        model: &dyn EmbeddingModel,
    ) -> Option<PreparedQuery> {
        let embedded = query.query().embed(model).ok()?;
        let sim_config = SimilarityConfig {
            // 校验需要拿到原始 top-1 分（含低于兜底分的情况），不用兜底分过滤
            similarity_threshold: 0.0,
            max_results: 1,
        };
        let sim_req = sim_config.into_request(
            self.wm.clone(),
            EmbeddedMemoryRetrieveQuery {
                embedding: embedded.clone(),
                query: query.query().clone(),
            },
        );
        let top1 = RetrSimilarity {}.retrieve(sim_req).into_iter().next();
        if matches!(top1, Some((_, score)) if score >= self.config.similarity_threshold) {
            Some(PreparedQuery {
                query,
                embedding: Some(embedded),
                dropped: false,
            })
        } else {
            None
        }
    }

    fn run_embedding_retrieval(
        &self,
        prepared: &[PreparedQuery],
    ) -> Option<RetrievalTrace> {
        if prepared.is_empty() {
            return None;
        }
        let total_start = Instant::now();
        let mut per_query = Vec::new();
        let mut merged_map: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();

        // priority 只作小偏移：分数接近时保护重要查询，分数差距大时由分数主导
        let p_max = prepared
            .iter()
            .filter(|p| !p.dropped)
            .map(|p| p.query.priority())
            .max()
            .unwrap_or(0);

        for pq in prepared {
            let q_start = Instant::now();
            let bonus = priority_bonus(pq.query.priority(), p_max);

            if pq.dropped {
                per_query.push(dropped_trace(pq.query.query()));
                continue;
            }
            let embedded = match pq.embedding.as_ref() {
                Some(e) => e.clone(),
                None => {
                    per_query.push(dropped_trace(pq.query.query()));
                    continue;
                }
            };
            let query = pq.query.query();

            let sim_config = SimilarityConfig {
                similarity_threshold: self.config.similarity_threshold,
                max_results: self.config.max_results,
            };
            let sim_req = sim_config.into_request(
                self.wm.clone(),
                EmbeddedMemoryRetrieveQuery {
                    embedding: embedded,
                    query: query.clone(),
                },
            );
            let sim_result = RetrSimilarity {}.retrieve(sim_req);

            let sim_elapsed = q_start.elapsed();

            let sim_nodes: Vec<TracedNode> = sim_result
                .into_iter()
                .map(|(id, score)| {
                    let name = self.graph_names.get(&id).cloned().unwrap_or_default();
                    let content = self
                        .id_names
                        .get(&id)
                        .map(|s| s.primary.clone())
                        .unwrap_or_default();
                    TracedNode {
                        id,
                        name,
                        content,
                        score: score as f64,
                        stage: HitStage::Similarity,
                    }
                })
                .collect();

            let query_trace = QueryTrace {
                query: query.clone(),
                sim_nodes: sim_nodes.clone(),
                sim_elapsed,
                ppr_nodes: vec![],
                ppr_elapsed: Duration::ZERO,
                action_nodes: vec![],
                action_elapsed: Duration::ZERO,
                total_elapsed: sim_elapsed,
                dropped: false,
            };
            per_query.push(query_trace);
            fold_priority_nodes(&mut merged_map, sim_nodes, bonus);
        }

        if per_query.is_empty() {
            return None;
        }

        let mut all_nodes = finish_merged(merged_map);
        all_nodes.truncate(self.config.merged_top_k);

        Some(RetrievalTrace {
            mode: RetrieveMode::Embedding,
            total_elapsed: total_start.elapsed(),
            merged_nodes: all_nodes,
            per_query,
        })
    }

    fn run_fullpipeline_retrieval(
        &self,
        prepared: &[PreparedQuery],
    ) -> Option<RetrievalTrace> {
        if prepared.is_empty() {
            return None;
        }
        let total_start = Instant::now();

        let mut per_query = Vec::new();
        let mut merged_map: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();

        // priority 只作小偏移：分数接近时保护重要查询，分数差距大时由分数主导
        let p_max = prepared
            .iter()
            .filter(|p| !p.dropped)
            .map(|p| p.query.priority())
            .max()
            .unwrap_or(0);

        for pq in prepared {
            let q_start = Instant::now();
            let bonus = priority_bonus(pq.query.priority(), p_max);

            if pq.dropped {
                per_query.push(dropped_trace(pq.query.query()));
                continue;
            }
            let embedded = match pq.embedding.as_ref() {
                Some(e) => e.clone(),
                None => {
                    per_query.push(dropped_trace(pq.query.query()));
                    continue;
                }
            };
            let query = pq.query.query();

            let sim_config = SimilarityConfig {
                similarity_threshold: self.config.similarity_threshold,
                max_results: self.config.max_results,
            };
            let sim_req = sim_config.into_request(
                self.wm.clone(),
                EmbeddedMemoryRetrieveQuery {
                    embedding: embedded,
                    query: query.clone(),
                },
            );
            let sim_result = RetrSimilarity {}.retrieve(sim_req);
            let sim_elapsed = Instant::now().duration_since(q_start);

            let sim_set: HashSet<MemoryId> = sim_result.iter().map(|(id, _)| *id).collect();

            let sim_nodes: Vec<TracedNode> = sim_result
                .into_iter()
                .map(|(id, score)| {
                    let name = self.graph_names.get(&id).cloned().unwrap_or_default();
                    let content = self
                        .id_names
                        .get(&id)
                        .map(|s| s.primary.clone())
                        .unwrap_or_default();
                    TracedNode {
                        id,
                        name,
                        content,
                        score: score as f64,
                        stage: HitStage::Similarity,
                    }
                })
                .collect();

            let ppr_start = Instant::now();
            let aa_config = AssociateWithActionConfig {
                association: AssociationConfig {
                    damping_factor: self.config.damping_factor,
                    residue_threshold: self.config.residue_threshold,
                    top_k: self.config.ppr_top_k,
                    ..Default::default()
                },
                action_top_k: self.config.action_top_k,
            };
            let aa_req = aa_config.into_request(
                self.wm.clone(),
                sim_nodes.iter().map(|n| (n.id, n.score as f32)).collect(),
            );

            let aa_result = RetrAssociateWithAction {}.retrieve(aa_req);
            let ppr_elapsed = Instant::now().duration_since(ppr_start);

            let mut ppr_nodes: Vec<TracedNode> = aa_result
                .memory
                .into_iter()
                .map(|(id, score)| {
                    let name = self.graph_names.get(&id).cloned().unwrap_or_default();
                    let stage = if sim_set.contains(&id) {
                        HitStage::Both
                    } else {
                        HitStage::Ppr
                    };
                    TracedNode {
                        id,
                        name,
                        content: self
                            .id_names
                            .get(&id)
                            .map(|s| s.primary.clone())
                            .unwrap_or_default(),
                        score,
                        stage,
                    }
                })
                .collect();
            ppr_nodes.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut action_nodes: Vec<TracedNode> = aa_result
                .action
                .into_iter()
                .map(|(id, score)| {
                    let name = self.graph_names.get(&id).cloned().unwrap_or_default();
                    let stage = if sim_set.contains(&id) {
                        HitStage::Both
                    } else {
                        HitStage::Action
                    };
                    TracedNode {
                        id,
                        name,
                        content: self
                            .id_names
                            .get(&id)
                            .map(|s| s.primary.clone())
                            .unwrap_or_default(),
                        score,
                        stage,
                    }
                })
                .collect();
            action_nodes.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let qt = QueryTrace {
                query: query.clone(),
                sim_nodes: sim_nodes.clone(),
                sim_elapsed,
                ppr_nodes: ppr_nodes.clone(),
                ppr_elapsed,
                action_nodes: action_nodes.clone(),
                action_elapsed: Duration::ZERO,
                total_elapsed: q_start.elapsed(),
                dropped: false,
            };
            per_query.push(qt);

            let mut merged: Vec<TracedNode> = Vec::new();
            for n in sim_nodes.into_iter().chain(ppr_nodes).chain(action_nodes) {
                if let Some(existing) = merged.iter_mut().find(|e: &&mut TracedNode| e.id == n.id) {
                    if n.score > existing.score {
                        existing.score = n.score;
                    }
                } else {
                    merged.push(n);
                }
            }
            merged.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            fold_priority_nodes(&mut merged_map, merged, bonus);
        }

        if per_query.is_empty() {
            return None;
        }

        let mut all_nodes = finish_merged(merged_map);
        all_nodes.truncate(self.config.merged_top_k);

        Some(RetrievalTrace {
            mode: RetrieveMode::FullPipeline,
            total_elapsed: total_start.elapsed(),
            merged_nodes: all_nodes,
            per_query,
        })
    }

    fn format_nodes(&self, nodes: &[TracedNode]) -> String {
        if nodes.is_empty() {
            return String::new();
        }
        nodes
            .iter()
            .map(|n| {
                let summary = self
                    .id_names
                    .get(&n.id)
                    .map(|s| format!("[{}] {}", s.type_label, s.primary))
                    .unwrap_or_else(|| n.name.clone());
                format!("- {}", summary)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::super::repair::{
        RawEnvironmentUnit, RawEventUnit, RawLocationUnit, RawParticipantUnit,
    };
    use super::*;

    #[test]
    fn raw_sem_unit_maps_concept_and_description() {
        let q = raw_sem_to_query(RawSemUnit {
            concept_identifier: Some("弹幕规则".into()),
            description: Some("补充".into()),
        });
        assert_eq!(q.concept_identifier(), Some("弹幕规则"));
        assert_eq!(q.description(), Some("补充"));
    }

    #[test]
    fn raw_sem_unit_skips_empty_fields() {
        let q = raw_sem_to_query(RawSemUnit {
            concept_identifier: Some("  ".into()),
            description: None,
        });
        assert_eq!(q.concept_identifier(), None);
        assert_eq!(q.description(), None);
    }

    #[test]
    fn raw_sit_unit_maps_all_dimensions() {
        let q = raw_sit_to_query(RawSitUnit {
            narrative: Some("在漫展".into()),
            location: Some(vec![RawLocationUnit {
                name: Some("漫展".into()),
                coordinates: Some("1,2".into()),
            }]),
            participants: Some(vec![RawParticipantUnit {
                name: Some("某人".into()),
                role: Some("朋友".into()),
            }]),
            environment: Some(RawEnvironmentUnit {
                atmosphere: Some("热闹".into()),
                tone: None,
            }),
            event: Some(vec![RawEventUnit {
                action: Some("逛展".into()),
                initiator: Some("某人".into()),
                target: None,
            }]),
        });
        assert_eq!(q.narrative().map(|s| s.as_str()), Some("在漫展"));
        assert_eq!(q.location().unwrap()[0].name(), "漫展");
        assert_eq!(q.participants().unwrap()[0].role(), Some("朋友"));
        assert_eq!(q.environment().unwrap().atmosphere(), Some("热闹"));
        assert_eq!(q.event().unwrap()[0].action(), "逛展");
        assert_eq!(q.event().unwrap()[0].initiator(), Some("某人"));
    }

    #[test]
    fn raw_sit_unit_drops_empty_sub_units() {
        let q = raw_sit_to_query(RawSitUnit {
            narrative: None,
            location: Some(vec![RawLocationUnit {
                name: None,
                coordinates: None,
            }]),
            participants: None,
            environment: None,
            event: Some(vec![RawEventUnit {
                action: None,
                initiator: None,
                target: None,
            }]),
        });
        assert_eq!(q.narrative(), None);
        assert_eq!(q.location(), None);
        assert_eq!(q.event(), None);
    }

    #[test]
    fn priority_bonus_scales_with_priority() {
        assert_eq!(priority_bonus(10, 10), PRIORITY_BONUS_MAX);
        assert_eq!(priority_bonus(5, 10), PRIORITY_BONUS_MAX / 2.0);
        assert_eq!(priority_bonus(0, 10), 0.0);
        assert_eq!(priority_bonus(7, 0), 0.0);
    }

    #[test]
    fn priority_merge_score_primary_with_small_bonus() {
        let id_high = MemoryId::new();
        let id_low_boost = MemoryId::new();
        let id_protected = MemoryId::new();
        let id_plain = MemoryId::new();
        let mk = |id: MemoryId, score: f64| TracedNode {
            id,
            name: String::new(),
            content: String::new(),
            score,
            stage: HitStage::Similarity,
        };

        let mut merged: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();
        // 分数差 0.1（>0.05）→ 分数主导：0.9 > 0.8+0.05
        fold_priority_nodes(&mut merged, vec![mk(id_high, 0.9)], 0.0);
        fold_priority_nodes(&mut merged, vec![mk(id_low_boost, 0.8)], PRIORITY_BONUS_MAX);
        // 分数差 0.01（≤0.05）→ priority 保护高重要度查询：0.81+0.05 > 0.82
        fold_priority_nodes(&mut merged, vec![mk(id_protected, 0.81)], PRIORITY_BONUS_MAX);
        fold_priority_nodes(&mut merged, vec![mk(id_plain, 0.82)], 0.0);

        let out = finish_merged(merged);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].id, id_high);
        assert_eq!(out[1].id, id_protected);
        assert_eq!(out[2].id, id_low_boost);
        assert_eq!(out[3].id, id_plain);
        // 展示分为原始融合分（0–1 量纲），不含 priority 偏移
        assert!((out[0].score - 0.9).abs() < 1e-9);
        assert!((out[1].score - 0.81).abs() < 1e-9);
    }

    #[test]
    fn priority_merge_same_node_keeps_strongest_evidence() {
        let id_a = MemoryId::new();
        let mk = |id: MemoryId, score: f64| TracedNode {
            id,
            name: String::new(),
            content: String::new(),
            score,
            stage: HitStage::Similarity,
        };
        let mut merged: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();
        fold_priority_nodes(&mut merged, vec![mk(id_a, 0.5)], 0.03);
        fold_priority_nodes(&mut merged, vec![mk(id_a, 0.4)], 0.02);
        let out = finish_merged(merged);
        assert_eq!(out.len(), 1);
        // 保留键最大的那条（0.5+0.03），展示分为原始分 0.5
        assert!((out[0].score - 0.5).abs() < 1e-9);
    }

    fn empty_runner() -> PlayTestRunner {
        PlayTestRunner {
            wm: Arc::new(WorkingMemory::new(10)),
            system_prompt: String::new(),
            graph_names: Arc::new(HashMap::new()),
            id_names: Arc::new(HashMap::new()),
            config: PlayConfig::default(),
            human_role: None,
        }
    }

    fn load_geluoxiu_runner() -> PlayTestRunner {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures/example_data/格蕾修_https_zh_moegirl_org_cn_E6_A0_BC_E8_95_BE_E4_BF_AE");
        PlayTestRunner::load(&dir).expect("格蕾修 graph should load")
    }

    fn hint(score: f32, is_situation: bool) -> HintHit {
        HintHit {
            id: MemoryId::new(),
            score,
            is_situation,
            summary: format!("m{:.2}", score),
        }
    }

    fn sem_query(concept: &str, priority: u32) -> PrioritizedMemoryRetrieveQuery {
        MemoryRetrieveQuery::new(
            vec!["测试".to_string()],
            MemoryRetrieveQueryVariant::Semantic(vec![
                SemanticQueryUnit::new().with_concept_identifier(concept.to_string()),
            ]),
        )
        .with_priority(priority)
    }

    #[test]
    fn test_select_hints_takes_top_k() {
        let hits: Vec<HintHit> = (0..8).map(|i| hint(0.90 - i as f32 * 0.05, false)).collect();
        let out = select_hints(hits, HINT_TOP_K, HINT_LOOKAHEAD_K, 0.35);
        assert_eq!(out.len(), HINT_TOP_K);
        assert!(out[0].score >= out[4].score);
    }

    #[test]
    fn test_select_hints_skips_when_below_floor() {
        let hits = vec![hint(0.30, false), hint(0.29, true)];
        let out = select_hints(hits, HINT_TOP_K, HINT_LOOKAHEAD_K, 0.35);
        assert!(out.is_empty(), "top-1 低于兜底分不应注入 hint");
    }

    #[test]
    fn test_select_hints_situation_diversity_replaces_fifth() {
        let mut hits: Vec<HintHit> = (0..10).map(|i| hint(0.95 - i as f32 * 0.02, false)).collect();
        hits[7] = HintHit {
            id: MemoryId::new(),
            score: 0.80,
            is_situation: true,
            summary: "情境记忆".into(),
        };
        let out = select_hints(hits, HINT_TOP_K, HINT_LOOKAHEAD_K, 0.35);
        assert_eq!(out.len(), HINT_TOP_K);
        assert!(out[4].is_situation, "第 5 个 hint 应被最佳 Situation 替换");
        assert_eq!(out[4].summary, "情境记忆");
    }

    #[test]
    fn test_select_hints_keeps_order_when_situation_in_top_k() {
        let mut hits: Vec<HintHit> = (0..6).map(|i| hint(0.90 - i as f32 * 0.05, false)).collect();
        hits[1] = HintHit {
            id: MemoryId::new(),
            score: 0.85,
            is_situation: true,
            summary: "情境1".into(),
        };
        let out = select_hints(hits, HINT_TOP_K, HINT_LOOKAHEAD_K, 0.35);
        assert_eq!(out.len(), HINT_TOP_K);
        assert!(out[1].is_situation);
        assert!(!out[4].is_situation, "top-k 已有 Situation 时不替换");
    }

    #[test]
    fn test_select_hints_no_replacement_when_situation_below_floor() {
        let mut hits: Vec<HintHit> = (0..8).map(|i| hint(0.80 - i as f32 * 0.02, false)).collect();
        hits[6] = HintHit {
            id: MemoryId::new(),
            score: 0.20,
            is_situation: true,
            summary: "低分情境".into(),
        };
        let out = select_hints(hits, HINT_TOP_K, HINT_LOOKAHEAD_K, 0.35);
        assert_eq!(out.len(), HINT_TOP_K);
        assert!(!out.iter().any(|h| h.is_situation), "低于兜底分的 Situation 不应被替换进来");
    }

    #[test]
    fn test_cap_raw_queries_limits_to_eight() {
        let raw: Vec<RawQuery> = (0..12)
            .map(|i| RawQuery {
                tag: vec![format!("t{}", i)],
                variant: RawVariant::Semantic {
                    Semantic: vec![RawSemUnit {
                        concept_identifier: Some(format!("c{}", i)),
                        description: None,
                    }],
                },
                priority: 1,
            })
            .collect();
        let capped = cap_raw_queries(raw);
        assert_eq!(capped.len(), QUERY_MAX_COUNT);
        assert_eq!(capped[7].tag[0], "t7");
    }

    #[test]
    fn test_build_fallback_queries_top3_priority2() {
        let entities: Vec<String> = (0..5).map(|i| format!("实体{}", i)).collect();
        let queries = build_fallback_queries(&entities, FALLBACK_ENTITY_TOP);
        assert_eq!(queries.len(), FALLBACK_ENTITY_TOP);
        for (i, q) in queries.iter().enumerate() {
            assert_eq!(q.priority(), FALLBACK_QUERY_PRIORITY);
            let expected = format!("实体{}", i);
            if let MemoryRetrieveQueryVariant::Semantic(units) = q.query().variant() {
                assert_eq!(
                    units[0].concept_identifier(),
                    Some(expected.as_str())
                );
            } else {
                panic!("兜底查询应为 Semantic variant");
            }
        }
    }

    #[test]
    fn test_build_query_prompt_with_hints_includes_clues_and_anti_hallucination() {
        let runner = empty_runner();
        let prompt = runner.build_query_prompt(
            "早上好",
            &["博丽灵梦".to_string()],
            &["- 格蕾修在画画".to_string()],
        );
        assert!(prompt.contains("【记忆线索】"));
        assert!(prompt.contains("- 格蕾修在画画"));
        assert!(prompt.contains("4-8 条"));
        assert!(prompt.contains("不要编造线索中不存在的人物、事件、细节或关系"));
        assert!(prompt.contains("Situation 的 narrative 必须是真实记忆的转述"));
        assert!(prompt.contains("如果当前对话没有任何对应记忆"));
        assert!(prompt.contains("消息中的关键实体: 博丽灵梦"));
    }

    #[test]
    fn test_build_query_prompt_without_hints_omits_clues_section() {
        let runner = empty_runner();
        let prompt = runner.build_query_prompt("早上好", &[], &[]);
        assert!(!prompt.contains("【记忆线索】"), "无 hint 时应省略记忆线索小节");
        // 防幻觉条款与 4-8 上限不随 hint 有无而变化
        assert!(prompt.contains("不要编造线索中不存在的人物、事件、细节或关系"));
        assert!(prompt.contains("4-8 条"));
    }

    #[test]
    fn test_retrieval_trace_marks_dropped_queries() {
        let runner = empty_runner();
        let prepared = vec![PreparedQuery {
            query: sem_query("不存在", 3),
            embedding: None,
            dropped: true,
        }];
        let emb = runner
            .run_embedding_retrieval(&prepared)
            .expect("dropped-only 也应产生 trace 以便观察丢弃");
        assert_eq!(emb.per_query.len(), 1);
        assert!(emb.per_query[0].dropped);
        assert!(emb.merged_nodes.is_empty());

        let full = runner
            .run_fullpipeline_retrieval(&prepared)
            .expect("dropped-only 也应产生 full pipeline trace");
        assert_eq!(full.per_query.len(), 1);
        assert!(full.per_query[0].dropped);
        assert!(full.merged_nodes.is_empty());
    }

    #[test]
    fn test_retrieve_hints_injects_when_relevant() {
        let runner = load_geluoxiu_runner();
        let hints = runner.retrieve_hints("格蕾修喜欢画画吗");
        assert!(!hints.is_empty(), "相关消息应注入记忆线索");
        for line in &hints {
            assert!(line.starts_with("- "));
            assert!(
                line.chars().count() <= HINT_CONTENT_MAX_CHARS + 2,
                "hint 内容应限制在 50 字以内: {line}"
            );
        }
    }

    #[test]
    fn test_validate_query_keeps_relevant_drops_irrelevant() {
        let runner = load_geluoxiu_runner();
        let model = get_bge_model().expect("BGE 模型应可用");

        let kept = runner
            .validate_query(sem_query("格蕾修", 5), model)
            .expect("命中角色自身的查询应通过校验");
        assert!(!kept.dropped);
        assert!(kept.embedding.is_some(), "校验嵌入结果应缓存供检索复用");

        assert!(
            runner
                .validate_query(sem_query("量子力学", 5), model)
                .is_none(),
            "无对应记忆的查询应被丢弃"
        );
    }

    #[test]
    fn test_prepare_queries_drops_below_floor_with_trace_marker() {
        let runner = load_geluoxiu_runner();
        let prepared = runner.prepare_queries(vec![sem_query("量子力学", 5)], &[]);
        assert_eq!(prepared.len(), 1);
        assert!(prepared[0].dropped);

        let trace = runner
            .run_embedding_retrieval(&prepared)
            .expect("全丢弃也应有 trace");
        assert!(trace.per_query[0].dropped);
        assert!(trace.merged_nodes.is_empty());
    }

    #[test]
    fn test_prepare_queries_fallback_uses_entities_when_empty() {
        let runner = load_geluoxiu_runner();
        // 等价于 LLM 返回空数组：主查询为空 → 用实体兜底
        let prepared = runner.prepare_queries(Vec::new(), &["格蕾修".to_string()]);
        assert_eq!(prepared.len(), 1);
        assert!(!prepared[0].dropped, "实体兜底查询应通过校验");
        assert_eq!(prepared[0].query.priority(), FALLBACK_QUERY_PRIORITY);
        if let MemoryRetrieveQueryVariant::Semantic(units) = prepared[0].query.query().variant() {
            assert_eq!(units[0].concept_identifier(), Some("格蕾修"));
        } else {
            panic!("兜底查询应为 Semantic variant");
        }
    }

    #[test]
    fn test_prepare_queries_fallback_when_all_main_dropped() {
        let runner = load_geluoxiu_runner();
        let prepared = runner.prepare_queries(
            vec![sem_query("量子力学", 5)],
            &["格蕾修".to_string()],
        );
        assert_eq!(prepared.len(), 2);
        assert!(prepared[0].dropped, "主查询低于兜底分应被丢弃");
        assert!(!prepared[1].dropped, "实体兜底查询应通过校验");
        assert_eq!(prepared[1].query.priority(), FALLBACK_QUERY_PRIORITY);
    }

    #[test]
    fn test_prepare_queries_skips_fallback_when_kept_exists() {
        let runner = load_geluoxiu_runner();
        let prepared = runner.prepare_queries(
            vec![sem_query("格蕾修", 5)],
            &["格蕾修".to_string()],
        );
        assert_eq!(prepared.len(), 1, "有保留查询时不应追加兜底");
        assert!(!prepared[0].dropped);
    }

}
