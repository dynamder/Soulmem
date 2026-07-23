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
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::query::retrieve::{
    MemoryRetrieveQuery, MemoryRetrieveQueryVariant, PrioritizedMemoryRetrieveQuery,
    SemanticQueryUnit,
};
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::engine::llm::LlmBackend;
use crate::engine::loader::{cached_load_graph, get_bge_model};
use crate::engine::retrieve::data::NodeSummary;

use super::repair::{
    extract_think_content, robust_json_extract, strip_think_block, RawQuery, RawVariant,
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
}

const CHAT_INSTRUCTION: &str = "注意：这是短信聊天场景，回复必须自然口语化，像真人发消息。\
严禁使用括号描述动作、神态或心理活动，如（笑）、（叹气）、*摇头*等。\
只输出对话内容，不加任何表演注释。\
回复必须简短，一句话即可，不要重复用户的话，不要解释你的回复。";

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
            similarity_threshold: 0.3,
            max_results: 8,
            action_top_k: 3,
            ppr_top_k: 8,
            damping_factor: 0.15,
            residue_threshold: 1e-5,
            runs_per_turn: 5,
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

        let mut runs: Vec<PlayRunSnapshot> = Vec::with_capacity(self.config.runs_per_turn);
        let user_text = match &self.human_role {
            Some(role) => format!(
                "（对方身份: {}）对方发来消息: \"{}\"",
                role, entry.user_message
            ),
            None => format!("\"{}\"", entry.user_message),
        };
        for _run_idx in 0..self.config.runs_per_turn {
            let resp_emb = llm.generate_response(&chat_prompt, &emb_context, &user_text);
            let resp_full = llm.generate_response(&chat_prompt, &full_context, &user_text);

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

    fn generate_queries(
        &self,
        entry: &ConversationEntry,
        llm: &mut dyn LlmBackend,
    ) -> Result<(Vec<PrioritizedMemoryRetrieveQuery>, String, Option<String>), String> {
        let text = llm
            .generate_queries(&self.system_prompt, &entry.user_message)
            .map_err(|e| format!("LLM query gen failed: {}", e))?;

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

        let json_str = robust_json_extract(&clean).ok_or_else(|| {
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
                if let Some(repaired) = super::repair::repair_json(&json_str) {
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

        Ok((queries, json_str, think_content))
    }

    fn run_embedding_retrieval(
        &self,
        queries: &[PrioritizedMemoryRetrieveQuery],
    ) -> Option<RetrievalTrace> {
        let model = get_bge_model();
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
        let model = get_bge_model();
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
