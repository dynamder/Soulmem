use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use soul_mem_algo::algo::retrieve::association::{AssociationRequest, RetrAssociation};
use soul_mem_algo::algo::retrieve::complex::{AssociateWithActionConfig, DefaultPipelineConfig, RetrDefaultPipeline};
use soul_mem_algo::algo::retrieve::short_only::ShortOnlyConfig;
use soul_mem_algo::algo::retrieve::similarity::{RetrSimilarity, SimilarityConfig};
use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_query::embedding::blend_weights::BlendWeights;
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
};
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveMode;
use crate::engine::dataset::TestCaseConfig;
use crate::engine::loader::{cached_load_graph, get_bge_model};
use crate::engine::metrics::ranking::{compute_action_metrics, compute_ranking_metrics};
use crate::engine::retrieve::data::{
    ActionMetrics, NodeSummary, PerQueryMetrics, RankingMetrics, RetrieveCaseData,
};
use crate::engine::retrieve::dataset::{PerQueryExpectation, SubQuery, TestCaseQuery};
use crate::engine::suite::{key_value_metric, DetailRow, SuiteReport, TestCaseOutcome, TestSuite};

#[derive(Debug, Clone, Copy)]
pub struct RetrieveConfig {
    pub mode: RetrieveMode,
}

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

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BlendPairRaw {
    #[serde(default)]
    pub tag: Option<f32>,
    #[serde(default)]
    pub variant: Option<f32>,
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
    #[serde(default)]
    pub tag_sweep: Vec<f32>,
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

struct TestCaseWithWeights {
    query: TestCaseQuery,
    tag_weight: f32,
    variant_weight: f32,
}

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

        let graph_dir = query_path.parent().unwrap_or(Path::new("."));
        let graph_path = graph_dir.join(&raw.graph_path);
        let (wm, id_map) = cached_load_graph(&graph_path)?;

        let graph_names: Arc<HashMap<MemoryId, String>> = Arc::new(
            id_map
                .iter()
                .map(|(name, id)| (*id, name.clone()))
                .collect(),
        );

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

        let sweep_pairs = if mode == RetrieveMode::Embedding {
            expand_sweep_pairs(raw.blend_sweep)
        } else {
            expand_sweep_pairs(None)
        };

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

        let mut test_cases: Vec<TestCaseWithWeights> = Vec::new();
        let model = get_bge_model()?;
        let mut query_embeddings: Vec<Vec<MemoryRetrieveQueryEmbedding>> = Vec::new();

        for base in &base_cases {
            let base_embs: Vec<MemoryRetrieveQueryEmbedding> = base
                .sub_queries
                .iter()
                .map(|sq| {
                    let mq = MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone());
                    mq.embed(model)
                        .map_err(|e| format!("Query embed failed: {e}"))
                })
                .collect::<Result<_, _>>()?;

            for bw in &sweep_pairs {
                let label = format!(" [w=tag:{:.1}/var:{:.1}]", bw.tag, bw.variant);
                let named = TestCaseQuery {
                    name: format!("{}{}", base.name, label),
                    ..base.clone()
                };

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

        // 抽象节点集合与期望抽象列表（供"抽象检出率 / 抽象直接命中率"指标使用）
        let abstract_ids: std::collections::HashSet<MemoryId> = self
            .wm
            .memory_cluster()
            .read_or_compute(|c| {
                c.graph()
                    .node_weights()
                    .filter_map(|n| match n.note().mem_type() {
                        MemoryType::Situation(SituationType::AbstractSituation(_)) => {
                            Some(n.note().id())
                        }
                        _ => None,
                    })
                    .collect()
            });
        let expected_abstract: Vec<MemoryId> = test_case
            .expected_combined_ranking
            .iter()
            .chain(test_case.bonus_combined_ranking.iter())
            .copied()
            .filter(|id| abstract_ids.contains(id))
            .collect();
        let has_expected_abstract = !expected_abstract.is_empty();

        let mut per_query_metrics = Vec::new();
        let mut all_similarity: Vec<Vec<(MemoryId, f32)>> = Vec::new();
        // FullPipeline 模式：每个子查询执行真正的 DefaultPipeline，收集其合并记忆与动作输出
        let mut all_full_memory: Vec<(MemoryId, f64, u32)> = Vec::new();
        let mut all_full_action: Vec<(MemoryId, f64, u32)> = Vec::new();

        for (sq_idx, sq) in test_case.sub_queries.iter().enumerate() {
            let emb = &query_embs[sq_idx];
            let priority = sq.priority;

            let expected = test_case
                .expected_per_query
                .iter()
                .find(|e| e.query_index == sq_idx)
                .map(|e| &e.ranking);

            let ids: Vec<MemoryId> = if self.pipeline_mode == RetrieveMode::FullPipeline {
                // full 即 DefaultPipeline：ShortOnly(窗口/摘要) + Similarity + AssociateWithAction
                let pipeline_config = DefaultPipelineConfig {
                    short_mem_with_history: ShortOnlyConfig {
                        clipping_length: None,
                        include_summary: true,
                    },
                    similarity: SimilarityConfig {
                        similarity_threshold: self.meta.similarity_threshold,
                        max_results: self.meta.max_results,
                    },
                    assoc_with_action: AssociateWithActionConfig {
                        association: Default::default(),
                        action_top_k: 3,
                        ..Default::default()
                    },
                };
                let pipeline_request = pipeline_config.into_request(
                    Arc::clone(&self.wm),
                    EmbeddedMemoryRetrieveQuery {
                        embedding: emb.clone(),
                        query: MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone()),
                    },
                    priority,
                );
                let pipeline_res = RetrDefaultPipeline {}.retrieve(pipeline_request);
                all_full_action.extend(
                    pipeline_res
                        .action
                        .into_iter()
                        .map(|(id, score)| (id, score, priority)),
                );
                let ids: Vec<MemoryId> = pipeline_res
                    .association
                    .iter()
                    .map(|(id, _)| *id)
                    .collect();
                all_full_memory.extend(
                    pipeline_res
                        .association
                        .into_iter()
                        .map(|(id, score)| (id, score, priority)),
                );
                // 抽象直接命中率观测：仅对期望含抽象节点的用例额外跑一次相似度，
                // 避免全量双跑相似度拖慢 suite（字符串分是 O(N) 开销）。
                if has_expected_abstract {
                    let sim_config = SimilarityConfig {
                        similarity_threshold: self.meta.similarity_threshold,
                        max_results: self.meta.max_results,
                    };
                    let sim_req = sim_config.into_request(
                        Arc::clone(&self.wm),
                        EmbeddedMemoryRetrieveQuery {
                            embedding: emb.clone(),
                            query: MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone()),
                        },
                    );
                    all_similarity.push(RetrSimilarity {}.retrieve(sim_req));
                }
                ids
            } else {
                let config = SimilarityConfig {
                    similarity_threshold: self.meta.similarity_threshold,
                    max_results: self.meta.max_results,
                };
                let request = config.into_request(
                    Arc::clone(&self.wm),
                    EmbeddedMemoryRetrieveQuery {
                        embedding: emb.clone(),
                        query: MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone()),
                    },
                );
                let result = RetrSimilarity {}.retrieve(request);
                let ids: Vec<MemoryId> = result.iter().map(|(id, _)| *id).collect();
                all_similarity.push(result);
                ids
            };

            let per_metrics = if let Some(expected_ranking) = expected {
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
        }

        // 相似度直接命中集合（供抽象直接命中率观测；FullPipeline 模式下仅
        // 期望含抽象节点的用例有数据，其余模式每子查询都有）。
        let sim_ids: std::collections::HashSet<MemoryId> = all_similarity
            .iter()
            .flat_map(|results| results.iter().map(|(id, _)| *id))
            .collect();

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
            RetrieveMode::Association => {
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
            RetrieveMode::FullPipeline => {
                let all_retrieved: Vec<(MemoryId, f32, u32)> = all_full_memory
                    .into_iter()
                    .map(|(id, score, priority)| (id, score as f32, priority))
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
        };

        // ── 抽象检出指标 ──
        let combined_set: std::collections::HashSet<MemoryId> =
            combined_ids.iter().copied().collect();
        let abstract_detected = if has_expected_abstract {
            Some(expected_abstract.iter().any(|id| combined_set.contains(id)))
        } else {
            None
        };
        let abstract_direct_hit = if has_expected_abstract {
            Some(expected_abstract.iter().any(|id| sim_ids.contains(id)))
        } else {
            None
        };

        let action_metrics = if test_case.expected_actions.is_empty() {
            ActionMetrics {
                action_hit_rate: 1.0,
                action_recall_at: self.meta.test_k_values.iter().map(|&k| (k, 1.0)).collect(),
                has_expected_actions: false,
            }
        } else if self.pipeline_mode == RetrieveMode::FullPipeline {
            // 动作评测使用 DefaultPipeline 实际输出的 action 节点
            let all_actions: Vec<(MemoryId, f32, u32)> = all_full_action
                .into_iter()
                .map(|(id, score, priority)| (id, score as f32, priority))
                .collect();
            let merged_actions = merge_by_priority(all_actions, self.meta.max_results);
            let actual_action_ids: Vec<MemoryId> =
                merged_actions.iter().map(|(id, _)| *id).collect();
            let action_res = compute_action_metrics(
                &actual_action_ids,
                &test_case.expected_actions,
                &self.meta.test_k_values,
            );
            ActionMetrics {
                action_hit_rate: action_res.action_hit_rate,
                action_recall_at: action_res.action_recall_at,
                has_expected_actions: true,
            }
        } else {
            // 非 FullPipeline 模式暂不输出动作检索，保持占位
            let action_res =
                compute_action_metrics(&[], &test_case.expected_actions, &self.meta.test_k_values);
            ActionMetrics {
                action_hit_rate: action_res.action_hit_rate,
                action_recall_at: action_res.action_recall_at,
                has_expected_actions: true,
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
                has_expected_abstract,
                abstract_detected,
                abstract_direct_hit,
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
                        has_expected_actions: data.action_metrics.has_expected_actions,
                    },
                    has_expected_abstract: data.has_expected_abstract,
                    abstract_detected: data.abstract_detected,
                    abstract_direct_hit: data.abstract_direct_hit,
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

        let mut metrics: Vec<Box<dyn crate::engine::suite::ReportMetric>> = Vec::new();
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

            let group_label = format!("权重 tag={:.1}, variant={:.1}", tag_w, var_w);
            metrics.push(Box::new(key_value_metric(
                "平均 MRR",
                group_label.clone(),
                format!("{:.4}", avg_mrr),
            )));
            metrics.push(Box::new(key_value_metric(
                "平均 Hit",
                group_label.clone(),
                format!("{:.2}", avg_hit),
            )));
            metrics.push(Box::new(key_value_metric(
                "平均 Recall@3",
                group_label.clone(),
                format!("{:.4}", avg_recall3),
            )));
            let abstract_total = group_data
                .iter()
                .filter(|d| d.has_expected_abstract)
                .count();
            if abstract_total > 0 {
                let abstract_detected_rate = group_data
                    .iter()
                    .filter(|d| d.abstract_detected == Some(true))
                    .count() as f64
                    / abstract_total as f64;
                let abstract_direct_hit_rate = group_data
                    .iter()
                    .filter(|d| d.abstract_direct_hit == Some(true))
                    .count() as f64
                    / abstract_total as f64;
                metrics.push(Box::new(key_value_metric(
                    "抽象检出率",
                    group_label.clone(),
                    format!("{:.2}", abstract_detected_rate),
                )));
                metrics.push(Box::new(key_value_metric(
                    "抽象直接命中率",
                    group_label.clone(),
                    format!("{:.2}", abstract_direct_hit_rate),
                )));
            }
            metrics.push(Box::new(key_value_metric(
                "用例数",
                group_label,
                format!("{}", group_data.len()),
            )));

            for data in group_data {
                let hit = data.combined_ranking_metrics.hit_rate;
                let status = if hit > 0.0 { "✓" } else { "✗" };
                let mrr = data.combined_ranking_metrics.mrr;
                let name = if data.case_name.chars().count() > 28 {
                    let trimmed: String = data.case_name.chars().take(26).collect();
                    format!("{}..", trimmed)
                } else {
                    format!("{:28}", data.case_name)
                };
                detail_rows.push(DetailRow {
                    text: format!("  {:28}  {:.4}  {:.2}    {}", name, mrr, hit, status),
                    has_error: hit <= 0.0 && !data.case_name.contains("无意义"),
                });
            }
        }

        SuiteReport {
            metrics,
            detail_header: String::from("  用例                         MRR     Hit     状态"),
            detail_rows,
            outcomes,
        }
    }
}

fn expand_sweep_pairs(sweep: Option<BlendSweepRaw>) -> Vec<BlendWeights> {
    let raw = match sweep {
        Some(s) => s,
        None => return vec![BlendWeights::default()],
    };

    let default_bw = BlendWeights::default();

    if !raw.pairs.is_empty() {
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
    }
}

fn apply_overrides(base: &BlendWeights, pair: &BlendPairRaw) -> BlendWeights {
    BlendWeights {
        tag: pair.tag.unwrap_or(base.tag),
        variant: pair.variant.unwrap_or(base.variant),
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
        string_blend_alpha: base.string_blend_alpha,
    }
}

fn resolve_ids(ids: &[String], id_map: &HashMap<String, MemoryId>) -> Vec<MemoryId> {
    ids.iter().filter_map(|s| id_map.get(s).copied()).collect()
}

/// priority 小偏移上限：分数接近（≤0.05）时保护重要查询的命中不被淹没，
/// 分数差距较大时仍由分数主导（与 playtest runner 的合并语义一致）。
const PRIORITY_BONUS_MAX: f64 = 0.05;

fn priority_bonus(p: u32, p_max: u32) -> f64 {
    if p_max == 0 {
        return 0.0;
    }
    PRIORITY_BONUS_MAX * (p as f64 / p_max as f64)
}

/// 跨查询合并："分数主导 + priority 小偏移"。
/// 合并键 = `原始分 + priority_bonus`，同一节点跨查询命中时保留键最大的那条，
/// 返回值为原始融合分（0–1 量纲，用于排序与指标）。
fn merge_by_priority(results: Vec<(MemoryId, f32, u32)>, top_k: usize) -> Vec<(MemoryId, f32)> {
    let p_max = results.iter().map(|(_, _, p)| *p).max().unwrap_or(0);
    let mut merged: HashMap<MemoryId, (f32, f32)> = HashMap::new(); // id -> (key, raw)
    for (id, score, priority) in results {
        let key = score + priority_bonus(priority, p_max) as f32;
        match merged.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let (best_key, best_raw) = e.get_mut();
                if key > *best_key {
                    *best_key = key;
                    *best_raw = score;
                }
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((key, score));
            }
        }
    }
    let mut sorted: Vec<(MemoryId, f32, f32)> =
        merged.into_iter().map(|(id, (key, raw))| (id, key, raw)).collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    sorted.into_iter().take(top_k).map(|(id, _, raw)| (id, raw)).collect()
}

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
        assert!((pairs[0].sem_concept - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_expand_sweep_none() {
        let pairs = expand_sweep_pairs(None);
        assert_eq!(pairs.len(), 1);
        // 默认权重：tag 0.3 / variant 0.7（BlendWeights::default）
        assert!((pairs[0].tag - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_merge_by_priority() {
        let id_a = MemoryId::new();
        let id_b = MemoryId::new();
        let results = vec![(id_a, 0.8, 1), (id_b, 0.5, 2), (id_a, 0.3, 2)];
        let merged = merge_by_priority(results, 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].0, id_a);
        // 返回原始融合分（0–1 量纲）：id_a = 0.8，而非累加/加权值
        assert!((merged[0].1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_merge_by_priority_tie_uses_priority_offset() {
        let id_low = MemoryId::new();
        let id_high = MemoryId::new();
        // 分数差 0.02（≤0.05）：高 priority 命中靠偏移胜出
        let results = vec![(id_low, 0.82, 2), (id_high, 0.80, 10)];
        let merged = merge_by_priority(results, 10);
        assert_eq!(merged[0].0, id_high);
        // 分数差 0.1（>0.05）：分数主导，低 priority 高分节点胜出
        let results2 = vec![(id_high, 0.70, 10), (id_low, 0.80, 2)];
        let merged2 = merge_by_priority(results2, 10);
        assert_eq!(merged2[0].0, id_low);
    }
}
