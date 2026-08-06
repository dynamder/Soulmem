use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;

use soul_mem_algo::algo::retrieve::{
    association::AssociationConfig,
    complex::{AssociateWithActionConfig, RetrAssociateWithAction},
    similarity::{RetrSimilarity, SimilarityConfig},
    RetrStrategy,
};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::query::retrieve::{
    MemoryRetrieveQuery, MemoryRetrieveQueryVariant, PrioritizedMemoryRetrieveQuery,
    SemanticQueryUnit,
};
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::eval::llama_server::LlamaServer;
use crate::eval::loader::cached_load_graph;
use crate::eval::retrieve_suite::NodeSummary;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PlayConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub action_top_k: usize,
    pub ppr_top_k: usize,
    pub damping_factor: f64,
    pub residue_threshold: f64,
}

const CHAT_INSTRUCTION: &str = "注意：这是短信聊天场景，回复必须自然口语化，像真人发消息。\
严禁使用括号描述动作、神态或心理活动，如（笑）、（叹气）、*摇头*等。\
只输出对话内容，不加任何表演注释。";

impl Default for PlayConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_results: 4,
            action_top_k: 3,
            ppr_top_k: 8,
            damping_factor: 0.65,
            residue_threshold: 1e-5,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct DialogueFile {
    pub name: Option<String>,
    pub graph_path: String,
    #[serde(default)]
    pub config: Option<PlayConfig>,
    pub conversations: Vec<ConversationEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConversationEntry {
    pub user_message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitStage {
    Similarity,
    Ppr,
    Action,
    Both,
}

#[derive(Debug, Clone)]
pub struct TracedNode {
    pub id: MemoryId,
    pub name: String,
    pub score: f64,
    pub stage: HitStage,
}

#[derive(Debug, Clone)]
pub struct QueryTrace {
    pub query: MemoryRetrieveQuery,
    pub sim_nodes: Vec<TracedNode>,
    pub sim_elapsed: Duration,
    pub ppr_nodes: Vec<TracedNode>,
    pub ppr_elapsed: Duration,
    pub action_nodes: Vec<TracedNode>,
    pub action_elapsed: Duration,
    pub total_elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct RetrievalTrace {
    pub mode: RetrieveMode,
    pub total_elapsed: Duration,
    pub merged_nodes: Vec<TracedNode>,
    pub per_query: Vec<QueryTrace>,
}

#[derive(Debug, Clone)]
pub struct PlayTurnResult {
    pub index: usize,
    pub user_message: String,
    pub system_prompt: String,
    pub generated_queries_json: String,
    pub think_content: Option<String>,
    pub embedding_trace: Option<RetrievalTrace>,
    pub fullpipeline_trace: Option<RetrievalTrace>,
    pub embedding_response: Option<String>,
    pub fullpipeline_response: Option<String>,
    pub swap: bool,
    pub human_pick: Option<u8>,
    pub error: Option<String>,
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
                        soul_mem_core::memory_note::MemoryType::Semantic(sem) => {
                            ("语义".into(), sem.content.clone(), sem.description.clone())
                        }
                        soul_mem_core::memory_note::MemoryType::Situation(
                            SituationType::SpecificSituation(s),
                        ) => (
                            "情境".into(),
                            s.get_narrative().clone(),
                            s.get_time_span().to_string(),
                        ),
                        soul_mem_core::memory_note::MemoryType::Situation(_) => {
                            ("情境".into(), String::new(), String::new())
                        }
                        soul_mem_core::memory_note::MemoryType::Procedure(_) => {
                            ("流程".into(), String::new(), String::new())
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
                            soul_mem_core::memory_note::MemoryType::Semantic(sem) => {
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
        llm: &LlamaServer,
    ) -> PlayTurnResult {
        let gen_query_result = self.generate_queries(entry, llm);
        let (queries, queries_json, think_content) = match gen_query_result {
            Ok((q, j, tc)) => (q, j, tc),
            Err(e) => {
                return PlayTurnResult {
                    index: turn_index,
                    user_message: entry.user_message.clone(),
                    system_prompt: self.system_prompt.clone(),
                    generated_queries_json: String::new(),
                    think_content: None,
                    embedding_trace: None,
                    fullpipeline_trace: None,
                    embedding_response: None,
                    fullpipeline_response: None,
                    swap: false,
                    human_pick: None,
                    error: Some(format!("Query generation failed: {}", e)),
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
            chat_prompt = format!("{}\n\n{}", chat_prompt, role);
        }
        let resp_emb =
            llm.generate_response(&chat_prompt, &emb_context, &entry.user_message);
        let resp_full =
            llm.generate_response(&chat_prompt, &full_context, &entry.user_message);

        let mut errors: Vec<String> = Vec::new();
        let embedding_response = match resp_emb {
            Ok(s) => Some(s),
            Err(e) => {
                errors.push(format!("Embedding 响应失败: {}", e));
                None
            }
        };
        let fullpipeline_response = match resp_full {
            Ok(s) => Some(s),
            Err(e) => {
                errors.push(format!("FullPipeline 响应失败: {}", e));
                None
            }
        };

        let swap = rand::random::<bool>();

        PlayTurnResult {
            index: turn_index,
            user_message: entry.user_message.clone(),
            system_prompt: self.system_prompt.clone(),
            generated_queries_json: queries_json,
            think_content,
            embedding_trace,
            fullpipeline_trace,
            embedding_response,
            fullpipeline_response,
            swap,
            human_pick: None,
            error: if errors.is_empty() {
                None
            } else {
                Some(errors.join("; "))
            },
        }
    }

    fn generate_queries(
        &self,
        entry: &ConversationEntry,
        llm: &LlamaServer,
    ) -> Result<(Vec<PrioritizedMemoryRetrieveQuery>, String, Option<String>), String> {
        let text = llm
            .generate_queries(&self.system_prompt, &entry.user_message)
            .map_err(|e| format!("LLM query gen failed: {}", e))?;

        // 调试：将 LLM 原始输出写入文件
        let debug_path = std::env::temp_dir().join("soul_tune_llm_output.txt");
        let debug_entry = format!(
            "=== 用户: {} ===\n完整输出:\n{}\n<|end|>\n\n",
            entry.user_message, text
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

        let json_str = extract_json_array(&clean).ok_or_else(|| {
            format!(
                "No JSON array found in LLM output (think stripped): {}\n---完整原始输出---\n{}",
                clean, text
            )
        })?;

        // 把提取到的 JSON 也写入调试文件
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&debug_path)
        {
            use std::io::Write;
            let _ = writeln!(f, "提取的 JSON:\n{}\n<|end_json|>\n\n", json_str);
        }

        let raw: Vec<RawQuery> = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(orig_err) => {
                if let Some(repaired) = paw_repair_json(json_str) {
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
                let units: Vec<SemanticQueryUnit> = match r.variant {
                    RawVariant::Semantic(units) | RawVariant::BareArray(units) => units
                        .into_iter()
                        .map(|u| {
                            SemanticQueryUnit::new()
                                .with_concept_identifier(u.concept_identifier.unwrap_or_default())
                        })
                        .collect(),
                    RawVariant::SemanticSingle(unit) | RawVariant::BareSingle(unit) => {
                        vec![SemanticQueryUnit::new()
                            .with_concept_identifier(unit.concept_identifier.unwrap_or_default())]
                    }
                };
                let variant = MemoryRetrieveQueryVariant::Semantic(units);
                MemoryRetrieveQuery::new(r.tag, variant).with_priority(r.priority)
            })
            .collect();

        Ok((queries, json_str.to_string(), think_content))
    }

    fn run_embedding_retrieval(
        &self,
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> Option<RetrievalTrace> {
        let model = crate::eval::loader::get_bge_model();
        let total_start = Instant::now();

        let mut per_query = Vec::new();
        let mut all_nodes: Vec<TracedNode> = Vec::new();

        for pq in queries {
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
            let sim_req = sim_config.into_request(self.wm.clone(), embedded);
            let sim_result = RetrSimilarity {}.retrieve(sim_req);

            let sim_elapsed = q_start.elapsed();

            let sim_nodes: Vec<TracedNode> = sim_result
                .into_iter()
                .map(|(id, score)| {
                    let name = self.graph_names.get(&id).cloned().unwrap_or_default();
                    TracedNode {
                        id,
                        name,
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
            all_nodes.extend(sim_nodes);
        }

        if per_query.is_empty() {
            return None;
        }

        all_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_nodes.dedup_by(|a, b| a.id == b.id);

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
        let model = crate::eval::loader::get_bge_model();
        let total_start = Instant::now();

        let mut per_query = Vec::new();
        let mut all_nodes: Vec<TracedNode> = Vec::new();

        for pq in queries {
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
            let sim_req = sim_config.into_request(self.wm.clone(), embedded);
            let sim_result = RetrSimilarity {}.retrieve(sim_req);
            let sim_elapsed = Instant::now().duration_since(q_start);

            let sim_set: HashSet<MemoryId> = sim_result.iter().map(|(id, _)| *id).collect();

            let sim_nodes: Vec<TracedNode> = sim_result
                .into_iter()
                .map(|(id, score)| {
                    let name = self.graph_names.get(&id).cloned().unwrap_or_default();
                    TracedNode {
                        id,
                        name,
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
            all_nodes.extend(merged);
        }

        if per_query.is_empty() {
            return None;
        }

        all_nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_nodes.dedup_by(|a, b| a.id == b.id);

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

#[derive(Debug, Deserialize)]
struct RawQuery {
    tag: Vec<String>,
    variant: RawVariant,
    priority: u32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawVariant {
    Semantic(Vec<RawSemUnit>),
    SemanticSingle(RawSemUnit),
    BareArray(Vec<RawSemUnit>),
    BareSingle(RawSemUnit),
}

#[derive(Debug, Deserialize)]
struct RawSemUnit {
    #[serde(default)]
    concept_identifier: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

/// Split a response into (think_content, body) by extracting ALL `<think>...</think>` blocks.
pub fn split_response(s: &str) -> (Option<String>, String) {
    let mut think_parts: Vec<String> = Vec::new();
    let mut body = s.to_string();
    loop {
        let start = body.find("<think>");
        let end = start.and_then(|s| body[s..].find("</think>").map(|e| s + e));
        match (start, end) {
            (Some(s), Some(e)) => {
                think_parts.push(body[s + 7..e].trim().to_string());
                body.replace_range(s..e + 8, "");
            }
            _ => break,
        }
    }
    let think = if think_parts.is_empty() {
        None
    } else {
        Some(think_parts.join("\n\n"))
    };
    (think, body.trim().to_string())
}

pub fn strip_think_block(s: &str) -> String {
    let mut result = s.to_string();
    loop {
        let start = result.find("<think>");
        let end = start.and_then(|s| result[s..].find("</think>").map(|e| s + e));
        match (start, end) {
            (Some(s), Some(e)) => {
                result.replace_range(s..e + 8, "");
            }
            _ => break,
        }
    }
    result.trim().to_string()
}

fn extract_think_content(s: &str) -> Option<String> {
    let start = s.find("<think>")?;
    let remaining = &s[start + 7..];
    let end = remaining.find("</think>")?;
    Some(remaining[..end].trim().to_string())
}

fn extract_json_array(s: &str) -> Option<&str> {
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    if end > start {
        Some(&s[start..=end])
    } else {
        None
    }
}

// ── PAW JSON Repair ───────────────────────────────────────────────────

const JSON_REPAIR_SLUG: &str = "soul-tune-json-repair-v1";
const JSON_REPAIR_SPEC: &str = r#"You are a JSON repair tool. Fix the malformed JSON array to produce valid JSON.
Fix these issues: trailing commas, unquoted keys, single quotes,
unclosed brackets/braces, extra text or markdown fences.

Correct output format:
[
  {"tag": ["personality"], "variant": [{"concept_identifier": "traits"}], "priority": 0},
  {"tag": ["event", "recent"], "variant": {"concept_identifier": "meeting", "description": "discussed timeline"}, "priority": 1}
]

Each object MUST have: <tag> (string array), <variant> (object or array of objects with concept_identifier and optional description), <priority> (integer).
Output ONLY the repaired JSON array. No markdown, no explanations."#;

struct PawState {
    rt: tokio::runtime::Runtime,
    inner: Mutex<Option<Box<dyn paw_rs::paw_core::PawFnTrait>>>,
    mapping_path: PathBuf,
    config: paw_rs::paw_core::PawConfig,
}

static PAW: OnceLock<PawState> = OnceLock::new();

fn init_paw_state() -> &'static PawState {
    PAW.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().expect("PAW tokio runtime");
        let config = paw_rs::paw_core::PawConfig::from_env();
        let mapping_path = config.cache_dir().join("paw_id_mapping.json");
        let inner = Mutex::new(init_paw_fn_blocking(&rt, &config, &mapping_path));
        PawState { rt, inner, mapping_path, config: config, }
    })
}

fn init_paw_fn_blocking(
    rt: &tokio::runtime::Runtime,
    config: &paw_rs::paw_core::PawConfig,
    mapping_path: &Path,
) -> Option<Box<dyn paw_rs::paw_core::PawFnTrait>> {
    rt.block_on(async {
        // Step 1: check local program_id mapping → load from cache (zero API)
        if let Ok(data) = std::fs::read_to_string(mapping_path) {
            if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
                if let Some(id) = map.get(JSON_REPAIR_SLUG) {
                    if let Ok(f) = paw_rs::PawFnBuilder::builder()
                        .config(config.clone())
                        .id(id)
                        .load().await
                    {
                        return Some(f);
                    }
                }
            }
        }

        // Step 2: compile new program (requires API)
        use paw_rs::paw_core::{CompileRequest, PawClient};
        let client = PawClient::new(config);
        let req = CompileRequest::builder()
            .spec(JSON_REPAIR_SPEC)
            .slug(JSON_REPAIR_SLUG)
            .ephemeral(false)
            .build().ok()?;
        let program = client.compile(req).await.ok()?;
        let _ = client.download_paw(&program.id).await.ok()?;

        // Step 3: save mapping for future cache hits
        let mut map: HashMap<String, String> = std::fs::read_to_string(mapping_path)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or_default();
        map.insert(JSON_REPAIR_SLUG.to_string(), program.id.clone());
        let _ = std::fs::write(mapping_path, serde_json::to_string(&map).unwrap_or_default());

        // Step 4: load the freshly saved program
        paw_rs::PawFnBuilder::builder()
            .config(config.clone())
            .id(&program.id)
            .load().await.ok()
    })
}

/// Repair a malformed JSON string using a PAW-compiled repair function.
/// Returns `None` if PAW is unavailable or repair fails.
fn paw_repair_json(bad_json: &str) -> Option<String> {
    let state = init_paw_state();
    let mut lock = state.inner.lock().ok()?;
    let f = lock.as_mut()?;
    let prompt = format!("Fix this JSON:\n{}\n\n---\nRepaired JSON:", bad_json);
    let raw = f.run(&prompt).ok()?;
    // safe-extract the JSON array from the output
    extract_json_array(&raw).map(|s| s.to_string())
}
