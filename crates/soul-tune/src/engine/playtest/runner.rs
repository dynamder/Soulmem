use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

use soul_mem_algo::algo::retrieve::{
    association::AssociationConfig,
    complex::{AssociateWithActionConfig, RetrAssociateWithAction},
    similarity::{RetrSimilarity, SimilarityConfig},
    RetrStrategy,
};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::proc_mem::ActionType;
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_core::memory_links::MemoryLinkType;
use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
};
use soul_mem_query::embedding::{Embeddable, EmbeddingModel};
use soul_mem_query::query::retrieve::{
    EnvironmentQueryUnit, MemoryRetrieveQuery, MemoryRetrieveQueryVariant,
    PrioritizedMemoryRetrieveQuery, SemanticQueryUnit, SituationQueryUnit,
};
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::engine::llm::LlmBackend;
use crate::engine::loader::{cached_load_graph, get_bge_model};
use crate::engine::retrieve::data::NodeSummary;

use super::repair::{extract_balanced_array, extract_balanced_object, run_paw, strip_think_block};
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

const CHAT_INSTRUCTION: &str = "这是即时通讯软件上的聊天场景：你通过聊天软件（IM）与对方互发消息；\
对方是谁、你们是什么关系，由对话设定决定，按设定自然回应。\
回复要像真人用聊天软件发消息一样自然、口语化，简短，通常一句话即可，不要重复用户的话。\
严禁使用括号或星号描述动作、神态、心理活动（如（笑）、（叹气）、*摇头*），只输出消息内容。\
不要解释你的回复，不要写长篇大论或旁白。";

/// v2：输入加入对方身份（管线输入），且支持"无法准确命名的实体用简练特征描述"。
/// slug 升级会触发 PAW 服务重新编译，避免命中 v1 的旧缓存函数。
const ENTITY_EXTRACT_SLUG: &str = "soul-tune-entity-extract-v2";
const ENTITY_EXTRACT_SPEC: &str = r#"You are an entity extraction tool for a character memory retrieval system.
Extract all key entities that the character needs to recall in order to respond naturally.

The character knows the identity of the person they are talking to (对方身份).
The partner's identity (name/称呼/关系/特征) is key information: always include the partner
as an entity when the message relates to them, even if their name is not in the message.

Extract:
- Person names: people, characters, friends or foes mentioned or strongly implied, including the partner (对方身份)
- Location names: places, venues, regions
- Concrete concepts: rules, terms, titles, skills
- Items and objects: tools, weapons, artifacts
- Unnameable entities: when something has no clear name but is described by its features,
  output a concise descriptive noun phrase of its salient features, e.g. "红色方形的旋转物"

Rules:
- Each entity must be a concrete noun phrase, e.g. "弹幕规则", "神社", "博丽灵梦", "阴阳玉", "红色方形的旋转物"
- Do NOT extract abstract categories like "爱好", "技能", "朋友"
- Only include entities present in or strongly implied by the message and the partner identity
- Keep the original language of the message
- For unnameable entities, describe salient features (color/shape/behavior/function) concisely as a noun phrase, not a full sentence

Output format: a JSON array of strings, e.g. ["博丽灵梦", "神社", "弹幕规则"]
Output ONLY the JSON array. No markdown, no explanations."#;

const ATMOSPHERE_EXTRACT_SLUG: &str = "soul-tune-atmosphere-extract-v1";
const ATMOSPHERE_EXTRACT_SPEC: &str = r#"You are an atmosphere extraction tool for a character memory retrieval system.
Given the identity of the person the character is talking to and the recent conversation history,
sense the CURRENT conversation atmosphere (氛围) and tone.

Output a JSON object:
{"atmosphere": "<一言概括当前会话氛围，如"深夜谈心"、"互相调侃"、"冷战">", "tone": "<对话语气，如"轻松"、"紧张"、"温暖">"}

Rules:
- Atmosphere and tone must come ONLY from the given dialogue context; do not invent emotions or details.
- Keep each value to a short noun/adjective phrase (2-8 chars).
- Output ONLY the JSON object. No markdown, no explanations."#;

/// 生成后校验的兜底分常量（默认与 SimilarityConfig / PlayConfig 兜底分一致）：
/// top-1 命中分低于该值的查询视为无对应记忆，直接丢弃。
pub const QUERY_VALIDATION_FLOOR: f32 = 0.35;

/// 查询生成可见的最近对话轮数（含助手回复；每轮两条消息）。
pub const HISTORY_TURNS: usize = 6;

/// 历史窗口最大消息数 = 轮数 × 2（对方 + 自己）。
pub const HISTORY_MAX_MESSAGES: usize = HISTORY_TURNS * 2;

/// 实体查询 priority 基准：首个实体最高，逐条递减（保护排序靠前的重要实体）。
pub const ENTITY_QUERY_PRIORITY_BASE: u32 = 10;

/// 氛围查询 priority：低于最高优先实体，高于一般实体查询。
pub const ATMOSPHERE_QUERY_PRIORITY: u32 = 9;

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

/// 一条对话历史消息（角色 + 内容），用于查询生成时的会话氛围提取。
#[derive(Debug, Clone, PartialEq)]
struct HistoryEntry {
    role: &'static str,
    text: String,
}

/// 将一条消息推入历史窗口，超出上限时丢弃最旧消息。
fn push_history(history: &mut VecDeque<HistoryEntry>, role: &'static str, text: String) {
    history.push_back(HistoryEntry { role, text });
    while history.len() > HISTORY_MAX_MESSAGES {
        history.pop_front();
    }
}

/// 实体查询：每个提取实体一条 Semantic 查询（tag=["实体"]），
/// priority 从基准递减（首实体最高），过滤空白实体。
fn build_entity_queries(entities: &[String]) -> Vec<PrioritizedMemoryRetrieveQuery> {
    entities
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.trim().is_empty())
        .map(|(i, e)| {
            MemoryRetrieveQuery::new(
                vec!["实体".to_string()],
                MemoryRetrieveQueryVariant::Semantic(vec![
                    SemanticQueryUnit::new().with_concept_identifier(e.trim().to_string()),
                ]),
            )
            .with_priority(ENTITY_QUERY_PRIORITY_BASE.saturating_sub(i as u32).max(1))
        })
        .collect()
}

/// PAW/LLM 气氛提取的输出：当前会话氛围（一句话概括）与语气。
#[derive(Debug, Clone, Deserialize)]
pub struct AtmosphereInfo {
    #[serde(default)]
    pub atmosphere: Option<String>,
    #[serde(default)]
    pub tone: Option<String>,
}

impl AtmosphereInfo {
    /// 氛围是否有效（atmosphere 字段非空）。
    pub fn is_empty(&self) -> bool {
        self.atmosphere
            .as_ref()
            .map(|a| a.trim().is_empty())
            .unwrap_or(true)
    }
}

/// 氛围查询：一条 Situation 查询，只携带 environment（atmosphere/tone），
/// 不引用对话原文；驱动 compute.rs 的氛围评分通道（sit_env_atmosphere）。
fn build_atmosphere_query(info: &AtmosphereInfo) -> Option<PrioritizedMemoryRetrieveQuery> {
    if info.is_empty() {
        return None;
    }
    let mut env = EnvironmentQueryUnit::new();
    let mut any_env = false;
    if let Some(a) = info.atmosphere.as_ref().filter(|a| !a.trim().is_empty()) {
        env = env.with_atmosphere(a.trim());
        any_env = true;
    }
    if let Some(t) = info.tone.as_ref().filter(|t| !t.trim().is_empty()) {
        env = env.with_tone(t.trim());
        any_env = true;
    }
    if !any_env {
        return None;
    }
    let sit = SituationQueryUnit::new().with_environment(env);
    Some(
        MemoryRetrieveQuery::new(
            vec!["氛围".to_string()],
            MemoryRetrieveQueryVariant::Situation(vec![sit]),
        )
        .with_priority(ATMOSPHERE_QUERY_PRIORITY),
    )
}

/// 解析 PAW/LLM 输出的氛围 JSON：容忍前后杂讯与 markdown 围栏，丢弃无效氛围。
fn parse_atmosphere(raw: &str) -> Option<AtmosphereInfo> {
    let clean = strip_think_block(raw);
    serde_json::from_str::<AtmosphereInfo>(&clean)
        .ok()
        .or_else(|| {
            extract_balanced_object(&clean)
                .and_then(|j| serde_json::from_str::<AtmosphereInfo>(&j).ok())
        })
        .filter(|a| !a.is_empty())
}

/// 查询的展示 JSON（GUI 树形渲染）：GUI 的 _QueryCard 期望顶层
/// tags（复数）/ variant / priority / dropped。
fn query_to_json(p: &PreparedQuery) -> serde_json::Value {
    serde_json::json!({
        "tags": p.query.query().tag(),
        "tag": p.query.query().tag(),
        "priority": p.query.priority(),
        "dropped": p.dropped,
        "variant": p.query.query().variant(),
    })
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
    /// 角色自身节点 id（sem_self）：身份只随系统提示词在会话开始时加载一次，
    /// 回复的"相关记忆"中过滤该节点，避免身份信息每轮重复注入引发身份说教。
    self_id: Option<MemoryId>,
    /// 最近对话历史（对方/自己消息交替），供查询生成提取会话氛围。
    history: Mutex<VecDeque<HistoryEntry>>,
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
                        MemoryType::Procedure(proc) => (
                            String::from("流程"),
                            proc.get_action().get_content().to_string(),
                            String::new(),
                        ),
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
            self_id: id_map.get("sem_self").copied(),
            history: Mutex::new(VecDeque::new()),
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
                // 查询生成失败也要记录本轮用户消息，保持历史连续性。
                self.record_turn(entry, None);
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
        let full_actions = fullpipeline_trace
            .as_ref()
            .map(|t| t.action_nodes.clone())
            .unwrap_or_default();
        let full_speech = fullpipeline_trace
            .as_ref()
            .map(|t| t.speech_nodes.clone())
            .unwrap_or_default();
        let full_think = fullpipeline_trace
            .as_ref()
            .map(|t| t.think_nodes.clone())
            .unwrap_or_default();

        let emb_context = self.format_nodes(&emb_nodes);
        let full_context = self.format_nodes(&full_nodes);
        let full_action_text = self.format_action_nodes(&full_actions);
        let full_speech_text = self.format_action_nodes(&full_speech);
        let full_think_text = self.format_action_nodes(&full_think);

        let mut chat_prompt = format!("{}\n\n{}", self.system_prompt, CHAT_INSTRUCTION);
        if let Some(ref role) = self.human_role {
            chat_prompt = format!("{}\n\n现在与你对话的是: {}", chat_prompt, role);
        }
        let emb_prompt = format!("{}\n\n相关记忆:\n{}", chat_prompt, emb_context);
        let mut full_prompt = format!("{}\n\n相关记忆:\n{}", chat_prompt, full_context);
        if !full_speech_text.is_empty() {
            full_prompt = format!("{}\n\n说话风格:\n{}", full_prompt, full_speech_text);
        }
        if !full_think_text.is_empty() {
            full_prompt = format!("{}\n\n思维习惯:\n{}", full_prompt, full_think_text);
        }
        if !full_action_text.is_empty() {
            full_prompt = format!("{}\n\n当前行为倾向:\n{}", full_prompt, full_action_text);
        }

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

        // 记录本轮对话：用户消息 + 第一条可用的完整管线回复（无则退回 embedding 回复）。
        let assistant_reply = runs
            .first()
            .and_then(|r| r.fullpipeline_response.clone())
            .or_else(|| runs.first().and_then(|r| r.embedding_response.clone()));
        self.record_turn(entry, assistant_reply);

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

    /// 将当前轮用户消息与助手回复写入历史窗口（溢出时丢弃最旧）。
    fn record_turn(&self, entry: &ConversationEntry, assistant_reply: Option<String>) {
        let mut history = self.history.lock().unwrap();
        push_history(&mut history, "对方", entry.user_message.clone());
        if let Some(reply) = assistant_reply {
            push_history(&mut history, "你", reply);
        }
    }

    /// 通过 PAW 提取消息中的关键实体（对方身份 + 用户消息为输入），
    /// 随后由 [`build_entity_queries`] 直接生成实体查询。
    /// 对方身份是管线输入：与对方相关的实体（称呼/关系/特征）即使未出现在消息中也应提取。
    /// 无法准确命名的实体应输出简练的特征描述名词短语。
    /// PAW 不可用或提取为空时，暂时用主对话 LLM 顶上；均失败则返回空列表（不阻断流程）。
    fn extract_entities(&self, user_message: &str, llm: &mut dyn LlmBackend) -> Vec<String> {
        let partner = self
            .human_role
            .as_ref()
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| "（对方身份未指定）".to_string());
        let prompt = format!(
            "与你对话的人（对方身份）: {}\n消息内容：\n{}\n\n请提取关键实体（包括与对方身份相关的实体；无法准确命名的实体用简练特征描述）：",
            partner, user_message
        );
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

    /// 通过 PAW 提取当前会话氛围（用户角色 + 对话历史 + 当前消息），用于生成氛围查询。
    /// PAW 不可用或解析为空时，暂时用主对话 LLM 顶上（管线不变，仅执行者由 PAW 换成 LLM）；
    /// 均失败返回 None（该轮不生成氛围查询，不阻断流程）。
    fn extract_atmosphere(
        &self,
        user_message: &str,
        llm: &mut dyn LlmBackend,
    ) -> Option<AtmosphereInfo> {
        let history: Vec<HistoryEntry> = self.history.lock().unwrap().iter().cloned().collect();
        let partner = self
            .human_role
            .as_ref()
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| "（对方身份未指定）".to_string());
        let history_text = if history.is_empty() {
            String::from("（暂无对话历史）")
        } else {
            history
                .iter()
                .map(|h| format!("{}: {}", h.role, h.text))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let prompt = format!(
            "与你对话的人（对方身份）: {}\n对方最近消息: \"{}\"\n\n【最近对话】\n{}\n\n请输出当前对话的氛围。",
            partner, user_message, history_text
        );
        if let Some(raw) =
            run_paw(ATMOSPHERE_EXTRACT_SLUG, ATMOSPHERE_EXTRACT_SPEC, &prompt, Some(128))
        {
            if let Some(info) = parse_atmosphere(&raw) {
                return Some(info);
            }
        }
        let raw = llm.chat(ATMOSPHERE_EXTRACT_SPEC, &prompt, 128).ok()?;
        parse_atmosphere(&raw)
    }

    fn generate_queries(
        &self,
        entry: &ConversationEntry,
        llm: &mut dyn LlmBackend,
    ) -> Result<(Vec<PreparedQuery>, String, Option<String>), String> {
        // 第一步：PAW 实体提取（用户角色 + 用户消息）→ 实体查询
        let entities = self.extract_entities(&entry.user_message, llm);
        let entity_queries = build_entity_queries(&entities);

        // 第二步：PAW 气氛提取（用户角色 + 对话历史 + 当前消息）→ 氛围查询
        let atmosphere = self.extract_atmosphere(&entry.user_message, llm);
        let atmosphere_query = atmosphere.as_ref().and_then(build_atmosphere_query);

        // 第三步：整合查询（实体查询 + 氛围查询）
        let mut main_queries = entity_queries;
        if let Some(aq) = atmosphere_query {
            main_queries.push(aq);
        }

        // 第四步：生成后校验（嵌入 + top-1 兜底分），嵌入结果缓存供检索复用
        let prepared = self.prepare_queries(main_queries);

        // 展示 JSON（GUI 树形渲染）：提取实体 / 实体查询 / 提取氛围 / 氛围查询 / 最终查询
        let atmosphere_json = atmosphere.as_ref().map(|a| {
            serde_json::json!({
                "atmosphere": a.atmosphere.clone().unwrap_or_default(),
                "tone": a.tone.clone().unwrap_or_default(),
            })
        });
        let queries_json = serde_json::json!({
            "提取实体": &entities,
            "实体查询": prepared
                .iter()
                .filter(|p| matches!(p.query.query().variant(), MemoryRetrieveQueryVariant::Semantic(_)))
                .map(query_to_json)
                .collect::<Vec<_>>(),
            "提取氛围": atmosphere_json,
            "氛围查询": prepared
                .iter()
                .filter(|p| matches!(p.query.query().variant(), MemoryRetrieveQueryVariant::Situation(_)))
                .map(query_to_json)
                .collect::<Vec<_>>(),
            "最终查询": prepared.iter().map(query_to_json).collect::<Vec<_>>(),
            // GUI _QuerySection 期望的键：查询列表
            "queries": prepared.iter().map(query_to_json).collect::<Vec<_>>(),
        })
        .to_string();

        // 调试落盘：实体 / 氛围 / 查询与校验结果
        let debug_path = std::env::temp_dir().join("soul_tune_llm_output.txt");
        let entity_text = if entities.is_empty() {
            String::from("(无)")
        } else {
            entities.join("、")
        };
        let atmosphere_text = match &atmosphere {
            Some(a) => format!("atmosphere={:?} tone={:?}", a.atmosphere, a.tone),
            None => String::from("(无)"),
        };
        let debug_entry = format!(
            "=== 用户: {} ===\n提取实体: {}\n提取氛围: {}\n查询JSON:\n{}\n\n",
            entry.user_message, entity_text, atmosphere_text, queries_json
        );
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&debug_path)
        {
            use std::io::Write;
            let _ = f.write_all(debug_entry.as_bytes());
        }

        Ok((prepared, queries_json, None))
    }

    /// 生成后校验：对主查询逐条嵌入并检查 top-1 兜底分，达标保留，否则丢弃并标记
    /// dropped；嵌入结果缓存进查询对象，检索阶段直接复用不二次嵌入。
    /// 嵌入模型不可用时跳过校验（保持旧行为，检索阶段自行嵌入）。
    fn prepare_queries(
        &self,
        main_queries: Vec<PrioritizedMemoryRetrieveQuery>,
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
            action_nodes: vec![],
            speech_nodes: vec![],
            think_nodes: vec![],
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
        // 动作节点按 ActionType 拆成三路独立通道（不参与记忆分数合并）：
        // Speak 语气 / Think 思维习惯各至多 1 条，其余行为类按 action_top_k
        let mut merged_speech: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();
        let mut merged_think: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();
        let mut merged_behavior: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();
        let action_type_map: HashMap<MemoryId, ActionType> = self
            .wm
            .memory_cluster()
            .read_or_compute(|c| {
                c.graph()
                    .node_weights()
                    .filter_map(|n| match n.note().mem_type() {
                        MemoryType::Procedure(p) => Some((
                            n.note().id(),
                            p.get_action().get_action_type().clone(),
                        )),
                        _ => None,
                    })
                    .collect()
            });

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
                ..Default::default()
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

            // 按 ActionType 分流：Speak→语气、Think（非 proc_none）→思维习惯、其余→行为
            let mut speech = Vec::new();
            let mut think = Vec::new();
            let mut behavior = Vec::new();
            for n in action_nodes {
                match action_type_map.get(&n.id) {
                    Some(ActionType::Speak) => speech.push(n),
                    Some(ActionType::Think)
                        if self
                            .graph_names
                            .get(&n.id)
                            .map(|s| s.as_str())
                            != Some("proc_none") =>
                    {
                        think.push(n)
                    }
                    _ => behavior.push(n),
                }
            }

            let mut merged: Vec<TracedNode> = Vec::new();
            for n in sim_nodes.into_iter().chain(ppr_nodes) {
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
            fold_priority_nodes(&mut merged_speech, speech, bonus);
            fold_priority_nodes(&mut merged_think, think, bonus);
            fold_priority_nodes(&mut merged_behavior, behavior, bonus);
        }

        if per_query.is_empty() {
            return None;
        }

        let mut all_nodes = finish_merged(merged_map);
        all_nodes.truncate(self.config.merged_top_k);
        // 行为类动作独立 top-k：即使分数低于记忆节点也保留进最终结果
        let mut action_nodes = finish_merged(merged_behavior);
        action_nodes.truncate(self.config.action_top_k);
        // 语气 / 思维习惯单席：只取分数最高 1 条，避免多个 Speak 内容互相矛盾
        let mut speech_nodes = finish_merged(merged_speech);
        speech_nodes.truncate(1);
        if speech_nodes.is_empty() {
            if let Some((id, score)) = self.trait_fallback_scores(&ActionType::Speak) {
                speech_nodes.push(self.traced_action_node(id, score));
            }
        }
        let mut think_nodes = finish_merged(merged_think);
        think_nodes.truncate(1);
        if think_nodes.is_empty() {
            if let Some((id, score)) = self.trait_fallback_scores(&ActionType::Think) {
                if self.graph_names.get(&id).map(|s| s.as_str()) != Some("proc_none") {
                    think_nodes.push(self.traced_action_node(id, score));
                }
            }
        }

        Some(RetrievalTrace {
            mode: RetrieveMode::FullPipeline,
            total_elapsed: total_start.elapsed(),
            merged_nodes: all_nodes,
            action_nodes,
            speech_nodes,
            think_nodes,
            per_query,
        })
    }

    /// 常驻语气/思维习惯兜底：Bayes 未检出对应类型时，从图中选择
    /// "广泛抽象触发总概率最高"的 Speak/Think proc。持久特质与对话恒相关，
    /// 单席注入不会带来矛盾内容。
    fn trait_fallback_scores(&self, want: &ActionType) -> Option<(MemoryId, f64)> {
        self.wm.memory_cluster().read_or_compute(|c| {
            let mut proc_type: HashMap<MemoryId, ActionType> = HashMap::new();
            for n in c.graph().node_weights() {
                if let MemoryType::Procedure(p) = n.note().mem_type() {
                    proc_type.insert(n.note().id(), p.get_action().get_action_type().clone());
                }
            }
            let mut scores: HashMap<MemoryId, f64> = HashMap::new();
            for n in c.graph().node_weights() {
                if !matches!(
                    n.note().mem_type(),
                    MemoryType::Situation(SituationType::AbstractSituation(_))
                ) {
                    continue;
                }
                for link in n.note().links() {
                    if let MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction {
                        prob,
                        ..
                    })) = link.link_type()
                    {
                        if proc_type.get(&link.to()) == Some(want) {
                            *scores.entry(link.to()).or_insert(0.0) += prob;
                        }
                    }
                }
            }
            scores
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
        })
    }

    fn traced_action_node(&self, id: MemoryId, score: f64) -> TracedNode {
        TracedNode {
            id,
            name: self.graph_names.get(&id).cloned().unwrap_or_default(),
            content: self
                .id_names
                .get(&id)
                .map(|s| s.primary.clone())
                .unwrap_or_default(),
            score,
            stage: HitStage::Action,
        }
    }

    fn format_nodes(&self, nodes: &[TracedNode]) -> String {
        if nodes.is_empty() {
            return String::new();
        }
        nodes
            .iter()
            // 身份节点（sem_self）已随系统提示词加载，回复的"相关记忆"中过滤，
            // 避免模型把自我身份当"记忆"引用而进入身份说教模式。
            .filter(|n| Some(n.id) != self.self_id)
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

    /// 动作节点（procedure）的独立上下文格式：动作有自己的分数通道，
    /// 不参与记忆节点的合并截断，始终以「当前行为倾向」进入提示词。
    fn format_action_nodes(&self, nodes: &[TracedNode]) -> String {
        if nodes.is_empty() {
            return String::new();
        }
        nodes
            .iter()
            .map(|n| {
                let content = self
                    .id_names
                    .get(&n.id)
                    .map(|s| s.primary.clone())
                    .filter(|c| !c.trim().is_empty())
                    .unwrap_or_else(|| n.name.clone());
                format!("- {}", content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_entity_queries_priorities_descend_and_filter_empty() {
        let queries = build_entity_queries(&[
            "桑多涅".to_string(),
            "  ".to_string(),
            "哥伦比娅".to_string(),
        ]);
        assert_eq!(queries.len(), 2, "空白实体应被过滤");
        assert_eq!(queries[0].priority(), ENTITY_QUERY_PRIORITY_BASE);
        assert_eq!(queries[1].priority(), ENTITY_QUERY_PRIORITY_BASE - 1);
        assert_eq!(queries[0].query().tag().first().map(|s| s.as_str()), Some("实体"));
        if let MemoryRetrieveQueryVariant::Semantic(units) = queries[0].query().variant() {
            assert_eq!(units[0].concept_identifier(), Some("桑多涅"));
        } else {
            panic!("实体查询应为 Semantic variant");
        }
    }

    #[test]
    fn build_atmosphere_query_maps_environment_only() {
        let info = AtmosphereInfo {
            atmosphere: Some("深夜谈心".to_string()),
            tone: Some("温暖".to_string()),
        };
        let q = build_atmosphere_query(&info).expect("有效氛围应产出查询");
        assert_eq!(q.priority(), ATMOSPHERE_QUERY_PRIORITY);
        assert_eq!(q.query().tag().first().map(|s| s.as_str()), Some("氛围"));
        if let MemoryRetrieveQueryVariant::Situation(units) = q.query().variant() {
            assert_eq!(units[0].narrative(), None, "氛围查询不应携带对话原文");
            let env = units[0].environment().expect("应带 environment");
            assert_eq!(env.atmosphere(), Some("深夜谈心"));
            assert_eq!(env.tone(), Some("温暖"));
        } else {
            panic!("氛围查询应为 Situation variant");
        }
    }

    #[test]
    fn build_atmosphere_query_empty_info_returns_none() {
        let info = AtmosphereInfo {
            atmosphere: Some("  ".to_string()),
            tone: None,
        };
        assert!(build_atmosphere_query(&info).is_none());
    }

    #[test]
    fn parse_atmosphere_tolerates_noise_and_fences() {
        let raw = "```json\n{\"atmosphere\": \"互相调侃\", \"tone\": \"轻松\"}\n```";
        let info = parse_atmosphere(raw).expect("应解析出氛围");
        assert_eq!(info.atmosphere.as_deref(), Some("互相调侃"));
        assert_eq!(info.tone.as_deref(), Some("轻松"));
    }

    #[test]
    fn parse_atmosphere_rejects_empty() {
        assert!(parse_atmosphere("{}").is_none());
        assert!(parse_atmosphere("{\"atmosphere\": \"\"}").is_none());
        assert!(parse_atmosphere("no json here").is_none());
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
            self_id: None,
            history: Mutex::new(VecDeque::new()),
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
    fn test_push_history_window_truncates() {
        let mut history: VecDeque<HistoryEntry> = VecDeque::new();
        for i in 0..(HISTORY_MAX_MESSAGES + 4) {
            push_history(&mut history, "对方", format!("m{}", i));
        }
        assert_eq!(history.len(), HISTORY_MAX_MESSAGES);
        assert_eq!(history.front().unwrap().text, "m4");
        assert_eq!(
            history.back().unwrap().text,
            format!("m{}", HISTORY_MAX_MESSAGES + 3)
        );
    }

    #[test]
    fn test_push_history_preserves_alternating_order() {
        let mut history: VecDeque<HistoryEntry> = VecDeque::new();
        push_history(&mut history, "对方", "你好".to_string());
        push_history(&mut history, "你", "嗨".to_string());
        push_history(&mut history, "对方", "在吗".to_string());
        let roles: Vec<&str> = history.iter().map(|h| h.role).collect();
        assert_eq!(roles, vec!["对方", "你", "对方"]);
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
        let prepared = runner.prepare_queries(vec![sem_query("量子力学", 5)]);
        assert_eq!(prepared.len(), 1);
        assert!(prepared[0].dropped);

        let trace = runner
            .run_embedding_retrieval(&prepared)
            .expect("全丢弃也应有 trace");
        assert!(trace.per_query[0].dropped);
        assert!(trace.merged_nodes.is_empty());
    }

    #[test]
    fn test_prepare_queries_keeps_relevant() {
        let runner = load_geluoxiu_runner();
        let prepared = runner.prepare_queries(vec![sem_query("格蕾修", 5)]);
        assert_eq!(prepared.len(), 1);
        assert!(!prepared[0].dropped, "命中角色自身的查询应通过校验");
    }

    #[test]
    fn test_prepare_queries_empty_input_yields_empty_output() {
        let runner = load_geluoxiu_runner();
        let prepared = runner.prepare_queries(Vec::new());
        assert!(prepared.is_empty(), "无主查询（实体/氛围均缺失）时不应凭空产生查询");
    }

    #[test]
    fn test_fullpipeline_actions_split_by_type() {
        let runner = load_geluoxiu_runner();
        let model = get_bge_model().expect("BGE 模型应可用");
        // 情境查询：命中 sit_watch_movies_on_ark → 触发 proc_learn_from_movies 等
        let query = MemoryRetrieveQuery::new(
            vec!["日常".to_string()],
            MemoryRetrieveQueryVariant::Situation(vec![SituationQueryUnit::new()
                .with_narrative("我在方舟上的时候喜欢看科幻电影".to_string())]),
        )
        .with_priority(5);
        let prepared = runner
            .validate_query(query, model)
            .expect("情境查询应通过校验");
        let trace = runner
            .run_fullpipeline_retrieval(&[prepared])
            .expect("full pipeline 应有 trace");

        // Speak / Think 单席：各至多 1 条
        assert!(trace.speech_nodes.len() <= 1, "说话风格通道至多 1 条");
        assert!(trace.think_nodes.len() <= 1, "思维习惯通道至多 1 条");
        // 常驻兜底：格蕾修图存在 Speak 与 Think proc，即使 Bayes 未命中也应各有 1 条
        assert_eq!(trace.speech_nodes.len(), 1, "Speak 常驻兜底应生效");
        assert_eq!(trace.think_nodes.len(), 1, "Think 常驻兜底应生效");
        // 行为类通道受 action_top_k 限制
        assert!(
            trace.action_nodes.len() <= runner.config.action_top_k,
            "行为类动作数量应受 action_top_k 限制"
        );
        // 至少一路检出动作（Think 通道应包含学习/睡觉习惯）
        assert!(
            trace.speech_nodes.len() + trace.think_nodes.len() + trace.action_nodes.len() >= 1,
            "应有动作被检出"
        );
        assert!(
            trace.think_nodes.iter().all(|n| n.name != "proc_none"),
            "proc_none 不应进入思维习惯通道"
        );
        assert!(
            trace
                .speech_nodes
                .iter()
                .chain(trace.think_nodes.iter())
                .chain(trace.action_nodes.iter())
                .all(|n| !n.content.is_empty()),
            "动作内容应来自 proc 节点"
        );
    }

}
