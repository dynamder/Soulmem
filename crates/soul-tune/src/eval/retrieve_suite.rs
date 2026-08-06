use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use soul_mem_algo::algo::retrieve::association::{AssociationRequest, RetrAssociation};
use soul_mem_algo::algo::retrieve::similarity::{RetrSimilarity, SimilarityConfig};
use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_query::embedding::blend_weights::BlendWeights;
use soul_mem_query::embedding::query::note::MemoryRetrieveQueryEmbedding;
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::eval::dataset::{PerQueryExpectation, SubQuery, TestCaseConfig, TestCaseQuery};
use crate::eval::loader::{cached_load_graph, get_bge_model};
use crate::eval::metrics::ranking::{compute_action_metrics, compute_ranking_metrics};
use crate::eval::runner::{DetailRow, MetricGroup, SuiteReport, TestCaseOutcome, TestSuite};

// ─── Raw deserialization types ───────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SubQueryRaw {
    pub priority: u32,
    pub tag: Vec<String>,
    pub variant: MemoryRetrieveQueryVariant,
}

#[derive(Debug, Deserialize)]
pub struct PerQueryExpectationRaw {
    #[serde(rename = "q")]
    pub query_index: usize,
    pub ranking: Vec<String>,
    #[serde(default)]
    pub bonus_ranking: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct TestCaseQueryRaw {
    pub name: String,
    pub description: Option<String>,
    pub sub_queries: Vec<SubQueryRaw>,
    pub expected_per_query: Vec<PerQueryExpectationRaw>,
    pub expected_combined_ranking: Vec<String>,
    #[serde(default)]
    pub bonus_combined_ranking: Vec<String>,
    #[serde(default)]
    pub expected_actions: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct TestConfigRaw {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub test_k_values: Vec<usize>,
}

/// 单个权重对的 JSON 格式。缺失的字段使用 BlendWeights::default()
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BlendPairRaw {
    #[serde(default)]
    pub tag: Option<f32>,
    #[serde(default)]
    pub variant: Option<f32>,
    #[serde(default)]
    pub sem_concept_main: Option<f32>,
    #[serde(default)]
    pub sem_concept_aliases: Option<f32>,
    #[serde(default)]
    pub sem_concept: Option<f32>,
    #[serde(default)]
    pub sem_description: Option<f32>,
    #[serde(default)]
    pub sit_location_name: Option<f32>,
    #[serde(default)]
    pub sit_location_coord: Option<f32>,
    #[serde(default)]
    pub sit_participant_name: Option<f32>,
    #[serde(default)]
    pub sit_participant_role: Option<f32>,
    #[serde(default)]
    pub sit_env_atmosphere: Option<f32>,
    #[serde(default)]
    pub sit_env_tone: Option<f32>,
    #[serde(default)]
    pub sit_event_initiator: Option<f32>,
    #[serde(default)]
    pub sit_event_target: Option<f32>,
    #[serde(default)]
    pub sit_event_action: Option<f32>,
    #[serde(default)]
    pub sit_event_initiator_only_action: Option<f32>,
    #[serde(default)]
    pub sit_event_target_only_action: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlendSweepRaw {
    /// 快捷方式：指定 tag 权重列表，自动生成 (tag, 1-tag) pairs
    #[serde(default)]
    pub tag_sweep: Vec<f32>,
    /// 显式权重对列表
    #[serde(default)]
    pub pairs: Vec<BlendPairRaw>,
}

#[derive(Debug, Deserialize)]
pub struct RetrQueryFileRaw {
    pub name: String,
    pub description: String,
    pub graph_path: PathBuf,
    pub config: TestConfigRaw,
    pub test_cases: Vec<TestCaseQueryRaw>,
    #[serde(default)]
    pub blend_sweep: Option<BlendSweepRaw>,
}

// ─── Internal types ──────────────────────────────────────────────

struct TestCaseWithWeights {
    query: TestCaseQuery,
    tag_weight: f32,
    variant_weight: f32,
}

// ─── Suite-specific types ────────────────────────────────────────

#[derive(Clone)]
pub struct RankingMetrics {
    pub recall_at: Vec<(usize, f64)>,
    pub precision_at: Vec<(usize, f64)>,
    pub mrr: f64,
    pub ndcg_at: Vec<(usize, f64)>,
    pub hit_rate: f64,
}

#[derive(Clone)]
pub struct ActionMetrics {
    pub action_hit_rate: f64,
    pub action_recall_at: Vec<(usize, f64)>,
}

#[derive(Clone)]
pub struct PerQueryMetrics {
    pub query_index: usize,
    pub ranking_metrics: RankingMetrics,
}

#[derive(Clone)]
pub struct RetrieveCaseData {
    pub case_name: String,
    pub description: String,
    pub combined_retrieved_ids: Vec<MemoryId>,
    pub combined_ranking_metrics: RankingMetrics,
    pub per_query_metrics: Vec<PerQueryMetrics>,
    pub action_metrics: ActionMetrics,
    pub tag_weight: f32,
    pub variant_weight: f32,
    pub id_names: Option<Arc<HashMap<MemoryId, NodeSummary>>>,
    pub expected_combined_ranking: Vec<MemoryId>,
    pub bonus_combined_ranking: Vec<MemoryId>,
    pub graph_names: Option<Arc<HashMap<MemoryId, String>>>,
    pub sub_queries: Vec<SubQuery>,
}

#[derive(Clone)]
pub struct NodeSummary {
    pub tags: Vec<String>,
    pub type_label: String, // e.g. "语义·Entity", "情境", "流程"
    pub primary: String,    // content / narrative / action text
    pub secondary: String,  // description / time / action_type
}

// ─── Sweep pair expansion ────────────────────────────────────────

/// 将配置文件中的 sweep 展开为权重对列表。
/// - 如果 blend_sweep 为 None → 默认一对 (0.4, 0.6)
/// - 如果 blend_sweep.tag_sweep 非空 → 生成 (tag, 1-tag) pairs
/// - 如果 blend_sweep.pairs 非空 → 叠加 BlendPairRaw 覆盖默认值
fn expand_sweep_pairs(sweep: Option<BlendSweepRaw>) -> Vec<BlendWeights> {
    let raw = match sweep {
        Some(s) => s,
        None => return vec![BlendWeights::default()],
    };

    let default_bw = BlendWeights::default();

    let raw_pairs: Vec<BlendWeights> = if !raw.pairs.is_empty() {
        raw.pairs
            .into_iter()
            .map(|pair| apply_overrides(&default_bw, &pair))
            .collect()
    } else if !raw.tag_sweep.is_empty() {
        raw.tag_sweep
            .into_iter()
            .map(|tag| BlendWeights {
                tag,
                variant: 1.0 - tag,
                ..default_bw.clone()
            })
            .collect()
    } else {
        vec![BlendWeights::default()]
    };

    raw_pairs
}

fn apply_overrides(base: &BlendWeights, pair: &BlendPairRaw) -> BlendWeights {
    BlendWeights {
        tag: pair.tag.unwrap_or(base.tag),
        variant: pair.variant.unwrap_or(base.variant),
        sem_concept_main: pair.sem_concept_main.unwrap_or(base.sem_concept_main),
        sem_concept_aliases: pair.sem_concept_aliases.unwrap_or(base.sem_concept_aliases),
        sem_concept: pair.sem_concept.unwrap_or(base.sem_concept),
        sem_description: pair.sem_description.unwrap_or(base.sem_description),
        sit_location_name: pair.sit_location_name.unwrap_or(base.sit_location_name),
        sit_location_coord: pair.sit_location_coord.unwrap_or(base.sit_location_coord),
        sit_participant_name: pair
            .sit_participant_name
            .unwrap_or(base.sit_participant_name),
        sit_participant_role: pair
            .sit_participant_role
            .unwrap_or(base.sit_participant_role),
        sit_env_atmosphere: pair.sit_env_atmosphere.unwrap_or(base.sit_env_atmosphere),
        sit_env_tone: pair.sit_env_tone.unwrap_or(base.sit_env_tone),
        sit_event_initiator: pair.sit_event_initiator.unwrap_or(base.sit_event_initiator),
        sit_event_target: pair.sit_event_target.unwrap_or(base.sit_event_target),
        sit_event_action: pair.sit_event_action.unwrap_or(base.sit_event_action),
        sit_event_initiator_only_action: pair
            .sit_event_initiator_only_action
            .unwrap_or(base.sit_event_initiator_only_action),
        sit_event_target_only_action: pair
            .sit_event_target_only_action
            .unwrap_or(base.sit_event_target_only_action),
    }
}

// ─── RetrieveSuite ───────────────────────────────────────────────

pub struct RetrieveSuite {
    wm: Arc<WorkingMemory>,
    test_cases: Vec<TestCaseWithWeights>,
    meta: TestCaseConfig,
    query_embeddings: Vec<Vec<MemoryRetrieveQueryEmbedding>>,
    pipeline_mode: RetrieveMode,
    id_names: Arc<HashMap<MemoryId, NodeSummary>>,
    graph_names: Arc<HashMap<MemoryId, String>>,
}

impl RetrieveSuite {
    pub fn load(query_path: &Path, mode: RetrieveMode) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_with_params(query_path, mode, None)
    }

    pub fn load_with_params(
        query_path: &Path,
        mode: RetrieveMode,
        params: Option<&HashMap<String, String>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(query_path)?;
        let reader = std::io::BufReader::new(file);
        let raw: RetrQueryFileRaw = serde_json::from_reader(reader)?;

        // Load graph
        let graph_dir = query_path.parent().unwrap_or(Path::new("."));
        let graph_path = graph_dir.join(&raw.graph_path);
        let (wm, id_map) = cached_load_graph(&graph_path)?;

        // Build reverse name map for drill-down display
        let graph_names: Arc<HashMap<MemoryId, String>> = Arc::new(
            id_map
                .iter()
                .map(|(name, id)| (*id, name.clone()))
                .collect(),
        );

        // Config meta (apply user params overrides if provided)
        let mut meta = TestCaseConfig {
            similarity_threshold: raw.config.similarity_threshold,
            max_results: raw.config.max_results,
            test_k_values: raw.config.test_k_values,
        };
        if let Some(p) = params {
            if let Some(v) = p.get("threshold") {
                if let Ok(f) = v.parse() {
                    meta.similarity_threshold = f;
                }
            }
            if let Some(v) = p.get("top_k") {
                if let Ok(n) = v.parse() {
                    meta.max_results = n;
                }
            }
        }

        // Resolve sweep pairs (only for Embedding mode)
        let sweep_pairs = if mode == RetrieveMode::Embedding {
            expand_sweep_pairs(raw.blend_sweep)
        } else {
            expand_sweep_pairs(None)
        };

        // Build one TestCaseQuery per raw test case
        let base_cases: Vec<TestCaseQuery> = raw
            .test_cases
            .into_iter()
            .map(|tc| {
                let sub_queries: Vec<SubQuery> = tc
                    .sub_queries
                    .into_iter()
                    .map(|sq| SubQuery {
                        priority: sq.priority,
                        tags: sq.tag,
                        variant: sq.variant,
                    })
                    .collect();

                let expected_per_query: Vec<PerQueryExpectation> = tc
                    .expected_per_query
                    .into_iter()
                    .map(|epq| PerQueryExpectation {
                        query_index: epq.query_index,
                        ranking: resolve_ids(&epq.ranking, &id_map),
                        bonus_ranking: resolve_ids(&epq.bonus_ranking, &id_map),
                    })
                    .collect();

                TestCaseQuery {
                    name: tc.name,
                    description: tc.description.unwrap_or_default(),
                    sub_queries,
                    expected_per_query,
                    expected_combined_ranking: resolve_ids(&tc.expected_combined_ranking, &id_map),
                    bonus_combined_ranking: resolve_ids(&tc.bonus_combined_ranking, &id_map),
                    expected_actions: resolve_ids(&tc.expected_actions, &id_map),
                }
            })
            .collect();

        // Expand: base_case × sweep_pair
        let mut test_cases: Vec<TestCaseWithWeights> = Vec::new();
        let model = get_bge_model();
        let mut query_embeddings: Vec<Vec<MemoryRetrieveQueryEmbedding>> = Vec::new();

        for base in &base_cases {
            // Embed base queries once (without custom weights)
            let base_embs: Vec<MemoryRetrieveQueryEmbedding> = base
                .sub_queries
                .iter()
                .map(|sq| {
                    let mq = MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone());
                    mq.embed(model).expect("Query embed failed")
                })
                .collect();

            for bw in &sweep_pairs {
                let label = format!(" [w=tag:{:.1}/var:{:.1}]", bw.tag, bw.variant);
                let named = TestCaseQuery {
                    name: format!("{}{}", base.name, label),
                    ..base.clone()
                };

                // Apply weights to each sub-query embedding
                let embs: Vec<MemoryRetrieveQueryEmbedding> = base_embs
                    .iter()
                    .map(|emb| emb.clone().with_weights(bw.clone()))
                    .collect();

                test_cases.push(TestCaseWithWeights {
                    query: named,
                    tag_weight: bw.tag,
                    variant_weight: bw.variant,
                });
                query_embeddings.push(embs);
            }
        }

        // Build node summary map for drill-down display
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
                            format!("语义"),
                            sem.content.clone(),
                            sem.description.clone(),
                        ),
                        MemoryType::Situation(SituationType::SpecificSituation(s)) => (
                            "情境".into(),
                            s.get_narrative().clone(),
                            s.get_time_span().to_string(),
                        ),
                        MemoryType::Situation(_) => ("情境".into(), String::new(), String::new()),
                        MemoryType::Procedure(_) => ("流程".into(), String::new(), String::new()),
                    };
                    let summary = NodeSummary {
                        tags,
                        type_label,
                        primary,
                        secondary,
                    };
                    (id, summary)
                })
                .collect::<HashMap<_, _>>()
        }));

        Ok(Self {
            wm: Arc::new(wm),
            test_cases,
            meta,
            query_embeddings,
            pipeline_mode: mode,
            id_names,
            graph_names,
        })
    }
}

impl TestSuite for RetrieveSuite {
    fn case_count(&self) -> usize {
        self.test_cases.len()
    }

    fn run_case(&self, index: usize) -> TestCaseOutcome {
        let tcw = &self.test_cases[index];
        let test_case = &tcw.query;
        let query_embs = &self.query_embeddings[index];

        let mut per_query_metrics = Vec::new();
        // Store similarity results per sub-query for all modes
        let mut all_similarity: Vec<Vec<(MemoryId, f32)>> = Vec::new();

        for (sq_idx, sq) in test_case.sub_queries.iter().enumerate() {
            let emb = &query_embs[sq_idx];
            let config = SimilarityConfig {
                similarity_threshold: self.meta.similarity_threshold,
                max_results: self.meta.max_results,
            };
            let request = config.into_request(Arc::clone(&self.wm), emb.clone());
            let result = RetrSimilarity {}.retrieve(request);

            let expected = test_case
                .expected_per_query
                .iter()
                .find(|e| e.query_index == sq_idx)
                .map(|e| &e.ranking);

            let per_metrics = if let Some(expected_ranking) = expected {
                let ids: Vec<MemoryId> = result.iter().map(|(id, _)| *id).collect();
                let ranking_metrics =
                    compute_ranking_metrics(&ids, expected_ranking, &self.meta.test_k_values);
                PerQueryMetrics {
                    query_index: sq_idx,
                    ranking_metrics: RankingMetrics {
                        recall_at: ranking_metrics.recall_at,
                        precision_at: ranking_metrics.precision_at,
                        mrr: ranking_metrics.mrr,
                        ndcg_at: ranking_metrics.ndcg_at,
                        hit_rate: ranking_metrics.hit_rate,
                    },
                }
            } else {
                PerQueryMetrics {
                    query_index: sq_idx,
                    ranking_metrics: RankingMetrics {
                        recall_at: self.meta.test_k_values.iter().map(|&k| (k, 0.0)).collect(),
                        precision_at: self.meta.test_k_values.iter().map(|&k| (k, 0.0)).collect(),
                        mrr: 0.0,
                        ndcg_at: self.meta.test_k_values.iter().map(|&k| (k, 0.0)).collect(),
                        hit_rate: 0.0,
                    },
                }
            };
            per_query_metrics.push(per_metrics);
            all_similarity.push(result);
        }

        // ── Pipeline-mode-specific: convert similarity results to final ranking ──
        let (combined_ids, combined_ranking, passed) = match self.pipeline_mode {
            RetrieveMode::Embedding => {
                let all_retrieved: Vec<(MemoryId, f32, u32)> = all_similarity
                    .into_iter()
                    .enumerate()
                    .flat_map(|(sq_idx, results)| {
                        let priority = test_case.sub_queries[sq_idx].priority;
                        results
                            .into_iter()
                            .map(move |(id, score)| (id, score, priority))
                    })
                    .collect();
                let merged = merge_by_priority(all_retrieved, self.meta.max_results);
                let ids: Vec<MemoryId> = merged.iter().map(|(id, _)| *id).collect();
                let (full_metrics, must_hit) = compute_split_metrics(
                    &ids,
                    &test_case.expected_combined_ranking,
                    &test_case.bonus_combined_ranking,
                    &self.meta.test_k_values,
                );
                (ids, full_metrics, must_hit)
            }
            RetrieveMode::Association | RetrieveMode::FullPipeline => {
                const EMBED_PPR_BLEND: f32 = 0.5;

                let mut all_blended: Vec<(MemoryId, f32, u32)> = Vec::new();

                for (sq_idx, results) in all_similarity.into_iter().enumerate() {
                    let priority = test_case.sub_queries[sq_idx].priority;
                    if results.is_empty() {
                        continue;
                    }

                    let embed_map: HashMap<MemoryId, f32> =
                        results.iter().copied().collect();

                    let source: Vec<(MemoryId, f32)> = results;
                    let req = AssociationRequest::new(Arc::clone(&self.wm), source)
                        .with_top_k(self.meta.max_results);
                    let ppr_result = RetrAssociation {}.retrieve(req);

                    let ppr_map: HashMap<MemoryId, f64> =
                        ppr_result.iter().map(|(id, s)| (*id, *s)).collect();

                    let all_ids: std::collections::HashSet<MemoryId> = embed_map
                        .keys()
                        .chain(ppr_map.keys())
                        .copied()
                        .collect();

                    for id in all_ids {
                        let embed_s = embed_map.get(&id).copied().unwrap_or(0.0);
                        let ppr_s = ppr_map.get(&id).copied().unwrap_or(0.0) as f32;
                        let blended = EMBED_PPR_BLEND * embed_s
                            + (1.0 - EMBED_PPR_BLEND) * ppr_s;
                        all_blended.push((id, blended, priority));
                    }
                }

                let merged = merge_by_priority(all_blended, self.meta.max_results);
                let ids: Vec<MemoryId> = merged.iter().map(|(id, _)| *id).collect();
                let (full_metrics, must_hit) = compute_split_metrics(
                    &ids,
                    &test_case.expected_combined_ranking,
                    &test_case.bonus_combined_ranking,
                    &self.meta.test_k_values,
                );
                (ids, full_metrics, must_hit)
            }
        };

        let action_metrics = if test_case.expected_actions.is_empty() {
            ActionMetrics {
                action_hit_rate: 1.0,
                action_recall_at: self.meta.test_k_values.iter().map(|&k| (k, 1.0)).collect(),
            }
        } else {
            let action_res =
                compute_action_metrics(&[], &test_case.expected_actions, &self.meta.test_k_values);
            ActionMetrics {
                action_hit_rate: action_res.action_hit_rate,
                action_recall_at: action_res.action_recall_at,
            }
        };

        TestCaseOutcome {
            case_name: test_case.name.clone(),
            description: test_case.description.clone(),
            passed,
            data: Box::new(RetrieveCaseData {
                case_name: test_case.name.clone(),
                description: test_case.description.clone(),
                combined_retrieved_ids: combined_ids,
                combined_ranking_metrics: combined_ranking,
                per_query_metrics,
                action_metrics,
                tag_weight: tcw.tag_weight,
                variant_weight: tcw.variant_weight,
                id_names: Some(self.id_names.clone()),
                expected_combined_ranking: test_case.expected_combined_ranking.clone(),
                bonus_combined_ranking: test_case.bonus_combined_ranking.clone(),
                graph_names: Some(self.graph_names.clone()),
                sub_queries: test_case.sub_queries.clone(),
            }),
        }
    }

    fn build_report(
        &self,
        outcomes: Vec<TestCaseOutcome>,
        _elapsed: Duration,
        total: usize,
        passed: usize,
        _failed: usize,
    ) -> SuiteReport {
        let _pass_rate = if total > 0 {
            passed as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        // Extract data and group by (tag_weight, variant_weight)
        let mut by_weight: HashMap<(u32, u32), Vec<RetrieveCaseData>> = HashMap::new();
        for outcome in &outcomes {
            if let Some(data) = outcome.data.downcast_ref::<RetrieveCaseData>() {
                let key = (
                    (data.tag_weight * 100.0).round() as u32,
                    (data.variant_weight * 100.0).round() as u32,
                );
                by_weight.entry(key).or_default().push(RetrieveCaseData {
                    case_name: data.case_name.clone(),
                    description: data.description.clone(),
                    combined_retrieved_ids: data.combined_retrieved_ids.clone(),
                    combined_ranking_metrics: RankingMetrics {
                        recall_at: data.combined_ranking_metrics.recall_at.clone(),
                        precision_at: data.combined_ranking_metrics.precision_at.clone(),
                        mrr: data.combined_ranking_metrics.mrr,
                        ndcg_at: data.combined_ranking_metrics.ndcg_at.clone(),
                        hit_rate: data.combined_ranking_metrics.hit_rate,
                    },
                    per_query_metrics: Vec::new(),
                    action_metrics: ActionMetrics {
                        action_hit_rate: data.action_metrics.action_hit_rate,
                        action_recall_at: data.action_metrics.action_recall_at.clone(),
                    },
                    tag_weight: data.tag_weight,
                    variant_weight: data.variant_weight,
                    id_names: data.id_names.clone(),
                    expected_combined_ranking: data.expected_combined_ranking.clone(),
                    bonus_combined_ranking: data.bonus_combined_ranking.clone(),
                    graph_names: data.graph_names.clone(),
                    sub_queries: data.sub_queries.clone(),
                });
            }
        }

        // Compute per-weight summary
        let mut summary_groups = Vec::new();
        let mut detail_rows = Vec::new();

        let mut keys: Vec<_> = by_weight.keys().copied().collect();
        keys.sort();
        for (tag_n, var_n) in keys {
            let tag_w = tag_n as f64 / 100.0;
            let var_w = var_n as f64 / 100.0;

            let group_data = &by_weight[&(tag_n, var_n)];
            let n = group_data.len() as f64;

            let avg_mrr = group_data
                .iter()
                .map(|d| d.combined_ranking_metrics.mrr)
                .sum::<f64>()
                / n;
            let avg_hit = group_data
                .iter()
                .map(|d| d.combined_ranking_metrics.hit_rate)
                .sum::<f64>()
                / n;
            let avg_recall3 = group_data
                .iter()
                .filter_map(|d| {
                    d.combined_ranking_metrics
                        .recall_at
                        .iter()
                        .find(|(k, _)| *k == 3)
                        .map(|(_, v)| v)
                })
                .sum::<f64>()
                / n;

            summary_groups.push(MetricGroup {
                label: format!("权重 tag={:.1}, variant={:.1}", tag_w, var_w),
                items: vec![
                    ("平均 MRR".into(), format!("{:.4}", avg_mrr)),
                    ("平均 Hit".into(), format!("{:.2}", avg_hit)),
                    ("平均 Recall@3".into(), format!("{:.4}", avg_recall3)),
                    ("用例数".into(), format!("{}", group_data.len())),
                ],
            });

            for data in group_data {
                let hit = data.combined_ranking_metrics.hit_rate;
                let status = if hit > 0.0 { "✓" } else { "✗" };
                let mrr = data.combined_ranking_metrics.mrr;
                let name = if data.case_name.len() > 28 {
                    format!("{}..", &data.case_name[..26])
                } else {
                    format!("{:28}", data.case_name)
                };
                detail_rows.push(DetailRow {
                    text: format!("  {:28}  {:.4}  {:.2}    {}", name, mrr, hit, status),
                    has_error: hit <= 0.0 && !data.case_name.contains("无意义"),
                });
            }
        }

        // Sort groups by tag weight descending
        summary_groups.sort_by(|a, b| b.label.cmp(&a.label));

        SuiteReport {
            summary_groups,
            detail_header: "  用例                         MRR     Hit     状态".into(),
            detail_rows,
            outcomes,
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

fn resolve_ids(ids: &[String], id_map: &HashMap<String, MemoryId>) -> Vec<MemoryId> {
    ids.iter().filter_map(|s| id_map.get(s).copied()).collect()
}

fn merge_by_priority(results: Vec<(MemoryId, f32, u32)>, top_k: usize) -> Vec<(MemoryId, f32)> {
    let mut merged: HashMap<MemoryId, f32> = HashMap::new();
    for (id, score, priority) in results {
        *merged.entry(id).or_insert(0.0) += priority as f32 * score;
    }
    let mut sorted: Vec<(MemoryId, f32)> = merged.into_iter().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    sorted.truncate(top_k);
    sorted
}

/// 分拆评估：must_include 决定 pass/fail，must+bonus 合起来算完整指标
fn compute_split_metrics(
    ids: &[MemoryId],
    must: &[MemoryId],
    bonus: &[MemoryId],
    k: &[usize],
) -> (RankingMetrics, bool) {
    let must_metrics = compute_ranking_metrics(ids, must, k);
    let must_hit = must_metrics.hit_rate > 0.0;
    let full_gt: Vec<MemoryId> = must.iter().chain(bonus.iter()).copied().collect();
    let full_metrics = if full_gt.is_empty() {
        must_metrics
    } else {
        compute_ranking_metrics(ids, &full_gt, k)
    };
    (full_metrics, must_hit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_file_raw_deserialize() {
        let json = r#"
        {
          "name": "test",
          "description": "desc",
          "graph_path": "graph.json",
          "config": {
            "similarity_threshold": 0.7,
            "max_results": 4,
            "test_k_values": [1, 3, 5]
          },
          "test_cases": [
            {
              "name": "case1",
              "description": "desc1",
              "sub_queries": [
                { "priority": 1, "tag": ["rust"], "variant": { "Semantic": [] } }
              ],
              "expected_per_query": [{"q": 0, "ranking": ["mem_a"]}],
              "expected_combined_ranking": ["mem_a"],
              "expected_actions": []
            }
          ]
        }
        "#;
        let raw: RetrQueryFileRaw = serde_json::from_str(json).unwrap();
        assert_eq!(raw.test_cases.len(), 1);
        assert_eq!(raw.test_cases[0].sub_queries[0].tag, vec!["rust"]);
    }

    #[test]
    fn test_blend_sweep_tag_sweep() {
        let json = r#"{"tag_sweep": [0.3, 0.5, 0.7], "pairs": []}"#;
        let raw: BlendSweepRaw = serde_json::from_str(json).unwrap();
        let pairs = expand_sweep_pairs(Some(raw));
        assert_eq!(pairs.len(), 3);
        assert!((pairs[0].tag - 0.3).abs() < 1e-6);
        assert!((pairs[0].variant - 0.7).abs() < 1e-6);
        assert!((pairs[1].tag - 0.5).abs() < 1e-6);
        assert!((pairs[2].tag - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_blend_sweep_pairs() {
        let json = r#"{"pairs": [{"tag": 0.3, "variant": 0.7}], "tag_sweep": []}"#;
        let raw: BlendSweepRaw = serde_json::from_str(json).unwrap();
        let pairs = expand_sweep_pairs(Some(raw));
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].tag - 0.3).abs() < 1e-6);
        // Other fields should be defaults
        assert!((pairs[0].sem_concept - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_expand_sweep_none() {
        let pairs = expand_sweep_pairs(None);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].tag - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_retrieve_suite_load_character_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures/example_data/test_batch_output-serde-fix/zh_moegirl_org_cn_E9_BB_91_E8_B0_B7_E5_B1_B1_E5_A5_B3/question.json");
        let suite = RetrieveSuite::load(&path, RetrieveMode::Embedding)
            .expect("Failed to load character query fixture");
        let count = suite.case_count();
        assert!(count > 0, "Should have test cases");
        // Run the first test case
        if count > 0 {
            let outcome = suite.run_case(0);
            assert!(!outcome.case_name.is_empty());
        }
    }

    #[test]
    fn test_full_pipeline_character_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures/example_data/test_batch_output-serde-fix/zh_moegirl_org_cn_E9_BB_91_E8_B0_B7_E5_B1_B1_E5_A5_B3/question.json");
        let suite = RetrieveSuite::load(&path, RetrieveMode::Embedding)
            .expect("Failed to load character query fixture");
        let n = suite.case_count();
        assert!(n > 0, "Should have test cases");

        let start = std::time::Instant::now();
        let mut passed = 0;
        let mut outcomes = Vec::with_capacity(n);
        for i in 0..n {
            let outcome = suite.run_case(i);
            if outcome.passed {
                passed += 1;
            }
            outcomes.push(outcome);
        }
        let elapsed = start.elapsed();

        let report = suite.build_report(outcomes, elapsed, n, passed, n - passed);
        let pass_rate = if n > 0 {
            passed as f64 / n as f64 * 100.0
        } else {
            0.0
        };

        println!(
            "\n=== Full Pipeline Test ===\nTotal: {} | Passed: {} | Failed: {} | Rate: {:.1}% | Time: {:.2}s\n",
            n,
            passed,
            n - passed,
            pass_rate,
            elapsed.as_secs_f64(),
        );
        for group in &report.summary_groups {
            println!("  {}:", group.label);
            for (k, v) in &group.items {
                println!("    {}: {}", k, v);
            }
        }
        println!();

        assert!(!report.summary_groups.is_empty(), "Should have summary");
        assert!(!report.detail_rows.is_empty(), "Should have detail rows");
    }

    #[test]
    fn test_weight_sweep_smoke_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures/queries/retr_sim_smoke_zh_blend.json");
        let suite = RetrieveSuite::load(&path, RetrieveMode::Embedding)
            .expect("Failed to load blend sweep query fixture");
        let n = suite.case_count();
        // 3 base cases × 3 sweep pairs = 9 expanded cases
        assert_eq!(n, 9, "3 base cases × 3 sweep pairs should yield 9");

        let start = std::time::Instant::now();
        let mut passed = 0;
        let mut outcomes = Vec::with_capacity(n);
        for i in 0..n {
            let outcome = suite.run_case(i);
            if outcome.passed {
                passed += 1;
            }
            outcomes.push(outcome);
        }
        let elapsed = start.elapsed();

        // Verify report has groups for each sweep weight
        let report = suite.build_report(outcomes, elapsed, n, passed, n - passed);
        println!(
            "\n=== Weight Sweep Test ===\nTotal: {} | Passed: {} | Failed: {} | Rate: {:.1}% | Time: {:.2}s\n",
            n, passed, n - passed,
            if n > 0 { passed as f64 / n as f64 * 100.0 } else { 0.0 },
            elapsed.as_secs_f64(),
        );
        for group in &report.summary_groups {
            println!("  {}:", group.label);
            for (k, v) in &group.items {
                println!("    {}: {}", k, v);
            }
        }
        println!();

        // Should have 3 summary groups (one per sweep pair: tag=0.3, 0.5, 0.7)
        assert_eq!(
            report.summary_groups.len(),
            3,
            "Should have 3 weight groups for 3 sweep pairs"
        );
        assert_eq!(report.detail_rows.len(), 9, "Should have 9 detail rows");
    }

    #[test]
    fn test_merge_by_priority() {
        let id_a = MemoryId::new();
        let id_b = MemoryId::new();
        let results = vec![(id_a, 0.8, 1), (id_b, 0.5, 2), (id_a, 0.3, 2)];
        let merged = merge_by_priority(results, 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].0, id_a);
    }
}
