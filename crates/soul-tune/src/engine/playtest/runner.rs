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
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
use soul_mem_query::embedding::Embeddable;
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
    extract_balanced_array, extract_think_content, robust_json_extract, strip_think_block,
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

/// 将某条查询的结果按归一化权重加权并入全局合并表。
/// 同一节点被多条查询命中时累加 weighted 分（与 suite 的 merge_by_priority 一致），
/// 展示用的节点保留原始分最高的那条（stage/content 信息更全）。
fn fold_priority_nodes(
    merged: &mut HashMap<MemoryId, (f64, TracedNode)>,
    nodes: Vec<TracedNode>,
    weight: f64,
) {
    for node in nodes {
        let weighted = weight * node.score;
        match merged.entry(node.id) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let (w, best) = e.get_mut();
                *w += weighted;
                if node.score > best.score {
                    *best = node;
                }
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((weighted, node));
            }
        }
    }
}

/// 将合并表转换为按加权分降序的节点列表。
fn finish_merged(merged: HashMap<MemoryId, (f64, TracedNode)>) -> Vec<TracedNode> {
    let mut nodes: Vec<TracedNode> = merged
        .into_values()
        .map(|(weighted, mut node)| {
            node.score = weighted;
            node
        })
        .collect();
    nodes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nodes
}

/// 将同一批查询的 priority 归一化为概率分布（softmax）。
/// priority 是相对值，直接乘会放大绝对数值差异，先转成概率分布再乘。
fn softmax_normalize(priorities: &[u32]) -> Vec<f64> {
    if priorities.is_empty() {
        return Vec::new();
    }
    let vals: Vec<f64> = priorities.iter().map(|&p| p as f64).collect();
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = vals.iter().map(|v| (v - max).exp()).sum();
    vals.iter().map(|v| (v - max).exp() / exp_sum).collect()
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
            similarity_threshold: 0.7,
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

    /// 通过主对话 LLM 提取消息中的关键实体，用于增强查询生成提示词，防止检索漏掉关键实体。
    /// LLM 不可用或解析失败时返回空列表（不阻断流程）。
    fn extract_entities(&self, user_message: &str, llm: &mut dyn LlmBackend) -> Vec<String> {
        let prompt = format!("消息内容：\n{}\n\n请提取关键实体：", user_message);
        // 实体列表很短，限制 token 数
        let raw = match llm.chat(ENTITY_EXTRACT_SPEC, &prompt, 128) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        serde_json::from_str::<Vec<String>>(&raw)
            .ok()
            .or_else(|| {
                extract_balanced_array(&raw)
                    .and_then(|j| serde_json::from_str::<Vec<String>>(&j).ok())
            })
            .unwrap_or_default()
    }

    /// 构建查询生成提示词：包含字段说明、当前场景说明，并以角色自身的视角引导回忆。
    /// 设计依据：question.json 的理想查询中，Semantic 的 concept_identifier 是 graph 节点
    /// aliases 的特征性别名（如 "金发的魔法使" 命中 sem_marisa），Situation 的 narrative
    /// 是 graph 节点 narrative 的 1-2 句压缩转述。因此提示词引导 LLM：
    /// - Semantic 用"身边人怎么称呼"的别名式短语，而非照搬正式名称
    /// - Situation 只用 narrative 讲完整小故事，不填冗余子字段
    /// - 同一概念用多个不同描述覆盖不同角度，提升召回
    fn build_query_prompt(&self, user_message: &str, entities: &[String]) -> String {
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

        format!(
            "当前场景：{}\n\
             对方说: \"{}\"{}\n\n\
             请以角色自身的视角，回想回应这句话所需的相关记忆，输出一个 JSON 数组，5-8 条，每条代表一个回忆方向。\n\n\
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
             只输出 JSON 数组，不要其他内容。",
            scene, user_message, entities_text
        )
    }

    fn generate_queries(
        &self,
        entry: &ConversationEntry,
        llm: &mut dyn LlmBackend,
    ) -> Result<(Vec<PrioritizedMemoryRetrieveQuery>, String, Option<String>), String> {
        // 第一步：PAW 提取关键实体，补充到提示词中防止漏掉关键实体
        let entities = self.extract_entities(&entry.user_message, llm);

        // 第二步：构建含字段说明、场景与角色视角的查询提示词
        let query_prompt = self.build_query_prompt(&entry.user_message, &entities);

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

        Ok((queries, json_str, think_content))
    }

    fn run_embedding_retrieval(
        &self,
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> Option<RetrievalTrace> {
        let model = get_bge_model().ok()?;
        let total_start = Instant::now();
        let mut per_query = Vec::new();
        let mut merged_map: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();

        // priority 是相对值，先归一化为概率分布再加权合并
        let priorities: Vec<u32> = queries.iter().map(|pq| pq.priority()).collect();
        let weights = softmax_normalize(&priorities);

        for (pq, &weight) in queries.iter().zip(&weights) {
            let query = pq.query();
            let q_start = Instant::now();

            let embedded = match query.embed(model) {
                Ok(e) => e,
                Err(_) => continue,
            };

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
            };
            per_query.push(query_trace);
            fold_priority_nodes(&mut merged_map, sim_nodes, weight);
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
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> Option<RetrievalTrace> {
        let model = get_bge_model().ok()?;
        let total_start = Instant::now();

        let mut per_query = Vec::new();
        let mut merged_map: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();

        // priority 是相对值，先归一化为概率分布再加权合并
        let priorities: Vec<u32> = queries.iter().map(|pq| pq.priority()).collect();
        let weights = softmax_normalize(&priorities);

        for (pq, &weight) in queries.iter().zip(&weights) {
            let query = pq.query();
            let q_start = Instant::now();

            let embedded = match query.embed(model) {
                Ok(e) => e,
                Err(_) => continue,
            };

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
            fold_priority_nodes(&mut merged_map, merged, weight);
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
    fn priority_merge_sums_weighted_scores_and_keeps_top() {
        let mut merged: HashMap<MemoryId, (f64, TracedNode)> = HashMap::new();
        let mk = |id: u8, score: f64| TracedNode {
            id: MemoryId::new(),
            name: format!("n{id}"),
            content: String::new(),
            score,
            stage: HitStage::Similarity,
        };
        // 同一 id 命中两条不同权重的查询 → 加权分累加
        let id_a = MemoryId::new();
        let mut n1 = mk(1, 0.5);
        n1.id = id_a;
        fold_priority_nodes(&mut merged, vec![n1], 3.0);
        let mut n2 = mk(2, 0.4);
        n2.id = id_a;
        fold_priority_nodes(&mut merged, vec![n2], 2.0);

        let n3 = mk(3, 0.9);
        fold_priority_nodes(&mut merged, vec![n3], 1.0);

        let out = finish_merged(merged);
        assert_eq!(out.len(), 2);
        // id_a: 3.0*0.5 + 2.0*0.4 = 2.3；n3: 1.0*0.9 = 0.9
        assert_eq!(out[0].id, id_a);
        assert!((out[0].score - 2.3).abs() < 1e-9);
        assert!((out[1].score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn softmax_normalize_produces_probability_distribution() {
        let weights = softmax_normalize(&[10, 5, 1]);
        assert_eq!(weights.len(), 3);
        // 概率分布：和为 1，保持相对排序
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);

        // 等优先级 → 均匀分布
        let uniform = softmax_normalize(&[5, 5, 5]);
        assert!((uniform[0] - 1.0 / 3.0).abs() < 1e-9);

        // 空批次 → 空结果
        assert!(softmax_normalize(&[]).is_empty());
    }
}
