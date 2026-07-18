use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use rand::seq::SliceRandom;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
};

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
use soul_mem_runtime::working_memory::llm::client::LlmClient;
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::eval::loader::cached_load_graph;
use crate::eval::retrieve_suite::NodeSummary;

// ── Config ──────────────────────────────────────────────────────────────────

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

impl Default for PlayConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.3,
            max_results: 8,
            action_top_k: 3,
            ppr_top_k: 8,
            damping_factor: 0.15,
            residue_threshold: 1e-5,
        }
    }
}

// ── Dialogue File Format ────────────────────────────────────────────────────

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

// ── Retrieval Trace Types ───────────────────────────────────────────────────

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

// ── Turn & Session Result ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlayTurnResult {
    pub index: usize,
    pub user_message: String,
    pub system_prompt: String,
    pub generated_queries_json: String,
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
}

// ── PlayTest Runner ─────────────────────────────────────────────────────────

pub struct PlayTestRunner {
    pub wm: Arc<WorkingMemory>,
    pub system_prompt: String,
    pub graph_names: Arc<HashMap<MemoryId, String>>,
    pub id_names: Arc<HashMap<MemoryId, NodeSummary>>,
    pub config: PlayConfig,
}

impl PlayTestRunner {
    pub fn load(graph_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_path = graph_dir.join("graph.json");
        let (wm, id_map) = cached_load_graph(&graph_path)?;

        let reverse_map: HashMap<MemoryId, String> = id_map
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

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
                        soul_mem_core::memory_note::MemoryType::Situation(SituationType::SpecificSituation(s)) => {
                            ("情境".into(), s.get_narrative().clone(), s.get_time_span().to_string())
                        }
                        soul_mem_core::memory_note::MemoryType::Situation(_) => {
                            ("情境".into(), String::new(), String::new())
                        }
                        soul_mem_core::memory_note::MemoryType::Procedure(_) => {
                            ("流程".into(), String::new(), String::new())
                        }
                    };
                    (id, NodeSummary { tags, type_label, primary, secondary })
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
        })
    }

    pub fn with_config(mut self, config: PlayConfig) -> Self {
        self.config = config;
        self
    }

    fn extract_system_prompt(
        wm: &WorkingMemory,
        id_map: &HashMap<String, MemoryId>,
        reverse_map: &HashMap<MemoryId, String>,
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
        llm: &LlmClient,
        runtime: &tokio::runtime::Runtime,
    ) -> PlayTurnResult {
        let start = Instant::now();

        let gen_query_result = self.generate_queries(entry, llm, runtime);
        let (queries, queries_json) = match gen_query_result {
            Ok((q, j)) => (q, j),
            Err(e) => {
                return PlayTurnResult {
                    index: turn_index,
                    user_message: entry.user_message.clone(),
                    system_prompt: self.system_prompt.clone(),
                    generated_queries_json: String::new(),
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

        let (resp_emb, resp_full) = runtime.block_on(async {
            let f1 = llm.call_llm(self.build_chat_messages(&entry.user_message, &emb_nodes));
            let f2 = llm.call_llm(self.build_chat_messages(&entry.user_message, &full_nodes));
            tokio::join!(f1, f2)
        });

        let embedding_response = resp_emb
            .ok()
            .and_then(|v| v.into_iter().next());
        let fullpipeline_response = resp_full
            .ok()
            .and_then(|v| v.into_iter().next());

        let swap = rand::random::<bool>();

        PlayTurnResult {
            index: turn_index,
            user_message: entry.user_message.clone(),
            system_prompt: self.system_prompt.clone(),
            generated_queries_json: queries_json,
            embedding_trace,
            fullpipeline_trace,
            embedding_response,
            fullpipeline_response,
            swap,
            human_pick: None,
            error: None,
        }
    }

    fn generate_queries(
        &self,
        entry: &ConversationEntry,
        llm: &LlmClient,
        runtime: &tokio::runtime::Runtime,
    ) -> Result<(Vec<PrioritizedMemoryRetrieveQuery>, String), String> {
        let prompt = format!(
            "{}\n\n用户说: \"{}\"\n\n作为角色，请思考你需要从记忆中检索什么内容才能自然地回应这句话。\
             \n输出一个 JSON 数组，每项包含 tag(字符串数组)、variant(对象)、priority(整数)。\
             \nvariant 格式: {{\"Semantic\":[{{\"concept_identifier\":\"关键词\"}}]}}。\
             \n例如: [{{\"tag\":[\"住所\"],\"variant\":{{\"Semantic\":[{{\"concept_identifier\":\"住处\"}}]}},\"priority\":1}}]\
             \n只输出 JSON，不要其他内容。",
            self.system_prompt, entry.user_message
        );

        let msg: ChatCompletionRequestMessage =
            ChatCompletionRequestUserMessage::from(prompt).into();

        let response = runtime
            .block_on(llm.simple_call(msg))
            .map_err(|e| format!("LLM query gen failed: {}", e))?;

        let text = response
            .first()
            .cloned()
            .ok_or_else(|| "Empty LLM response".to_string())?;

        let raw: Vec<RawQuery> = serde_json::from_str(&text)
            .map_err(|e| format!("Failed to parse LLM query JSON: {} — raw: {}", e, text))?;

        let queries: Vec<PrioritizedMemoryRetrieveQuery> = raw
            .into_iter()
            .map(|r| {
                let variant = match r.variant {
                    RawVariant::Semantic(units) => {
                        MemoryRetrieveQueryVariant::Semantic(
                            units.into_iter().map(|u| {
                                SemanticQueryUnit::new()
                                    .with_concept_identifier(u.concept_identifier.unwrap_or_default())
                            }).collect()
                        )
                    }
                };
                MemoryRetrieveQuery::new(r.tag, variant)
                    .with_priority(r.priority)
            })
            .collect();

        Ok((queries, text))
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

        all_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
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

            // Stage 1: Similarity
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

            // Stage 2: Association + Action
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

            // Build ppr nodes
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
                    TracedNode { id, name, score, stage }
                })
                .collect();
            ppr_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

            // Action nodes
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
                    TracedNode { id, name, score, stage }
                })
                .collect();
            action_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

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

            // Merge: sim + ppr + action, dedup by id keeping highest score
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
            merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            all_nodes.extend(merged);
        }

        if per_query.is_empty() {
            return None;
        }

        all_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_nodes.dedup_by(|a, b| a.id == b.id);

        Some(RetrievalTrace {
            mode: RetrieveMode::FullPipeline,
            total_elapsed: total_start.elapsed(),
            merged_nodes: all_nodes,
            per_query,
        })
    }

    fn build_chat_messages(
        &self,
        user_message: &str,
        context_nodes: &[TracedNode],
    ) -> Vec<ChatCompletionRequestMessage> {
        let system_content = if context_nodes.is_empty() {
            self.system_prompt.clone()
        } else {
            let ctx: Vec<String> = context_nodes
                .iter()
                .map(|n| {
                    let summary = self
                        .id_names
                        .get(&n.id)
                        .map(|s| format!("[{}] {}", s.type_label, s.primary))
                        .unwrap_or_else(|| n.name.clone());
                    format!("- {}", summary)
                })
                .collect();
            format!(
                "{}\n\n相关记忆:\n{}",
                self.system_prompt,
                ctx.join("\n")
            )
        };

        let user_msg: ChatCompletionRequestMessage =
            ChatCompletionRequestUserMessage::from(user_message.to_string()).into();

        if !system_content.is_empty() {
            vec![
                ChatCompletionRequestSystemMessage::from(system_content).into(),
                user_msg,
            ]
        } else {
            vec![user_msg]
        }
    }

    pub fn to_order_label(&self, turn: &PlayTurnResult) -> (String, String) {
        if turn.swap {
            ("FullPipeline".into(), "Embedding".into())
        } else {
            ("Embedding".into(), "FullPipeline".into())
        }
    }
}

// ── Raw JSON parsing for LLM query output ───────────────────────────────────

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
}

#[derive(Debug, Deserialize)]
struct RawSemUnit {
    #[serde(default)]
    concept_identifier: Option<String>,
    #[serde(default)]
    description: Option<String>,
}
