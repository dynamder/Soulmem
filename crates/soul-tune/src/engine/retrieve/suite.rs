use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_algo::algo::retrieve::association::{AssociationRequest, RetrAssociation};
use soul_mem_algo::algo::retrieve::complex::{
    AssociateWithActionConfig, DefaultPipelineConfig, RetrDefaultPipeline,
};
use soul_mem_algo::algo::retrieve::short_only::ShortOnlyConfig;
use soul_mem_algo::algo::retrieve::similarity::{RetrSimilarity, SimilarityConfig};
use soul_mem_algo::algo::retrieve::{DbPrefetchConfig, prefetch_db};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryType};
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::embedding::blend_weights::BlendWeights;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
};
use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
use soul_mem_runtime::storage::MemoryRepository;
use soul_mem_runtime::storage::surreal::SurrealRepository;
use soul_mem_runtime::working_memory::WorkingMemory;
use tokio::runtime::Runtime;

use crate::base::{RetrieveFlavor, RetrieveMode};
use crate::engine::dataset::TestCaseConfig;
use crate::engine::loader::{cached_load_graph, get_bge_model};
use crate::engine::metrics::ranking::{compute_action_metrics, compute_ranking_metrics};
use crate::engine::retrieve::data::{
    ActionMetrics, DbRecallDetail, NodeSummary, PerQueryMetrics, RankingMetrics, RetrieveCaseData,
};
use crate::engine::retrieve::dataset::{PerQueryExpectation, SubQuery, TestCaseQuery};
use crate::engine::suite::{DetailRow, SuiteReport, TestCaseOutcome, TestSuite, key_value_metric};

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
    /// DB 模式：每槽位 HNSW KNN 候选召回预算（可选，缺省用启发式默认值）。
    #[serde(default)]
    pub db_candidate_k: Option<usize>,
    /// DB 模式：邻居扩展跳数（可选，缺省 1；0 = 关闭扩展）。
    #[serde(default)]
    pub db_neighbor_depth: Option<usize>,
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

/// 一个测试用例 + 其权重（权重扫描会把同一用例展开成多份）。
pub(crate) struct TestCaseWithWeights {
    pub(crate) query: TestCaseQuery,
    pub(crate) tag_weight: f32,
    pub(crate) variant_weight: f32,
}

/// 数据集加载产物：与"记忆来源"无关的一切（全量图、查询用例、查询嵌入、元数据）。
///
/// 拆出这一层是为了让**加载**与**运行**解耦：`RetrieveSuite` 只负责"把某个记忆来源
/// 接到某条管线上"，而需要同一份加载与嵌入结果的分析路径
/// （如 [`crate::engine::retrieve::depth_audit`] 的几何审计）可以直接复用，
/// 不必复制一遍解析/嵌入逻辑——那份副本一定会随实现演进漂移。
///
/// 图工作记忆以全量图为基准：DB 模式后续只在**用例级子图**上跑管线，
/// 但全图仍保留在这里作为几何分析的参照系。
pub(crate) struct RetrDataset {
    /// 全量图工作记忆（直接模式的记忆来源；也是几何分析的全图基准）。
    pub wm: Arc<WorkingMemory>,
    /// 节点 id → 节点名（图里声明的可读名字）。
    pub graph_names: Arc<HashMap<MemoryId, String>>,
    /// 节点 id → 可读摘要（UI/日志用）。
    pub id_names: Arc<HashMap<MemoryId, NodeSummary>>,
    /// 全图抽象情境节点集合（供抽象检出/直接命中指标观测）。
    pub abstract_ids: std::collections::HashSet<MemoryId>,
    pub meta: TestCaseConfig,
    pub test_cases: Vec<TestCaseWithWeights>,
    /// 与 `test_cases` 一一对应：每个用例各子查询的嵌入。
    pub query_embeddings: Vec<Vec<MemoryRetrieveQueryEmbedding>>,
}

impl RetrDataset {
    /// 解析 question.json + 载入（带缓存）图 + 展开权重扫描 + 嵌入全部子查询。
    ///
    /// `flavor` 只影响权重扫描是否展开（扫描仅对相似度管线有意义），
    /// 与记忆来源无关。`params` 可覆盖 `threshold` / `top_k` /
    /// `db_candidate_k` / `db_neighbor_depth`。
    pub(crate) fn load(
        query_path: &Path,
        flavor: RetrieveFlavor,
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
            db_candidate_k: raw.config.db_candidate_k,
            db_neighbor_depth: raw.config.db_neighbor_depth,
        };
        if let Some(p) = params {
            if let Some(v) = p.get("threshold")
                && let Ok(f) = v.parse()
            {
                meta.similarity_threshold = f;
            }
            if let Some(v) = p.get("top_k")
                && let Ok(n) = v.parse()
            {
                meta.max_results = n;
            }
            if let Some(v) = p.get("db_candidate_k")
                && let Ok(n) = v.parse()
            {
                meta.db_candidate_k = Some(n);
            }
            if let Some(v) = p.get("db_neighbor_depth")
                && let Ok(n) = v.parse()
            {
                meta.db_neighbor_depth = Some(n);
            }
        }

        // 权重扫描（sweep）只对相似度管线有意义：直接与 DB 源行为一致
        let sweep_pairs = if flavor == RetrieveFlavor::Embedding {
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

        // 元数据（与记忆来源无关，加载时一次性预计算）：
        // - id_names：可读摘要（UI/日志用）
        // - abstract_ids：抽象情境节点集合（供抽象检出/直接命中指标观测）
        let id_names = Arc::new(wm.memory_cluster().read_or_compute(|cluster| {
            cluster
                .graph()
                .node_weights()
                .map(|node| {
                    let note = node.note();
                    (note.id(), note_summary(note))
                })
                .collect::<HashMap<_, _>>()
        }));
        let abstract_ids: std::collections::HashSet<MemoryId> =
            wm.memory_cluster().read_or_compute(|c| {
                c.graph()
                    .node_weights()
                    .filter(|&n| is_abstract_situation(n.note()))
                    .map(|n| n.note().id())
                    .collect()
            });

        Ok(RetrDataset {
            wm: Arc::new(wm),
            graph_names,
            id_names,
            abstract_ids,
            meta,
            test_cases,
            query_embeddings,
        })
    }

    /// 用例数量。
    pub(crate) fn case_count(&self) -> usize {
        self.test_cases.len()
    }
}

/// DB 模式后端：记忆仓库 + 驱动 async 操作的 runtime + 预取参数。
struct DbBackend {
    repo: SurrealRepository,
    rt: Runtime,
    /// DB 预取参数（每槽位候选预算 + 邻居扩展跳数）。
    config: DbPrefetchConfig,
}

pub struct RetrieveSuite {
    /// 直接模式：全量图工作记忆；DB 模式为 None。
    wm_direct: Option<Arc<WorkingMemory>>,
    /// DB 模式后端；直接模式为 None。
    db: Option<DbBackend>,
    test_cases: Vec<TestCaseWithWeights>,
    meta: TestCaseConfig,
    query_embeddings: Vec<Vec<MemoryRetrieveQueryEmbedding>>,
    /// 完整运行模式：管线（RetrieveFlavor）× 记忆来源（直接 / 数据库）。
    mode: RetrieveMode,
    /// 全图抽象情境节点集合（加载时预计算，两种来源共用，供抽象指标观测）。
    abstract_ids: std::collections::HashSet<MemoryId>,
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
        // 加载与"记忆来源"无关的部分（图 / 用例 / 嵌入 / 元数据）
        let RetrDataset {
            wm,
            graph_names,
            id_names,
            abstract_ids,
            meta,
            test_cases,
            query_embeddings,
        } = RetrDataset::load(query_path, mode.flavor(), params)?;

        // DB 模式：example_data 全量写入 mem 数据库（默认进程内 kv-mem；
        // params 传 db_path 时改用磁盘 SurrealKv 验证持久化读回）。
        let db = if mode.uses_db() {
            let notes: Vec<EmbeddedMemoryNote> = wm
                .memory_cluster()
                .read_or_compute(|c| c.graph().node_weights().cloned().collect());
            let rt = Runtime::new().map_err(|e| format!("创建 tokio runtime 失败: {e}"))?;
            let config = meta.db_prefetch_config();
            let db_path = params.and_then(|p| p.get("db_path")).map(PathBuf::from);
            let repo = rt
                .block_on(connect_and_seed_repo(notes, db_path.as_deref()))
                .map_err(|e| format!("mem 数据库初始化/写入失败: {e}"))?;
            Some(DbBackend { repo, rt, config })
        } else {
            None
        };

        Ok(Self {
            wm_direct: if mode.uses_db() { None } else { Some(wm) },
            db,
            test_cases,
            meta,
            query_embeddings,
            mode,
            abstract_ids,
            id_names,
            graph_names,
        })
    }

    /// DB 预取失败等无法执行检索时的失败用例（零指标 + 错误说明，避免 panic）。
    fn db_failed_outcome(&self, index: usize, message: String) -> TestCaseOutcome {
        let tcw = &self.test_cases[index];
        let test_case = &tcw.query;
        let zero_rows: Vec<(usize, f64)> =
            self.meta.test_k_values.iter().map(|&k| (k, 0.0)).collect();
        let zero_metrics = RankingMetrics {
            recall_at: zero_rows.clone(),
            precision_at: zero_rows.clone(),
            mrr: 0.0,
            ndcg_at: zero_rows.clone(),
            hit_rate: 0.0,
        };
        let data = RetrieveCaseData {
            case_name: test_case.name.clone(),
            description: format!("{}（{}）", test_case.description, message),
            combined_retrieved_ids: Vec::new(),
            combined_ranking_metrics: zero_metrics,
            db_recall: None,
            per_query_metrics: Vec::new(),
            action_metrics: ActionMetrics {
                action_hit_rate: 0.0,
                action_recall_at: zero_rows,
                has_expected_actions: false,
            },
            has_expected_abstract: false,
            abstract_detected: None,
            abstract_direct_hit: None,
            tag_weight: tcw.tag_weight,
            variant_weight: tcw.variant_weight,
            id_names: Some(self.id_names.clone()),
            expected_combined_ranking: test_case.expected_combined_ranking.clone(),
            bonus_combined_ranking: test_case.bonus_combined_ranking.clone(),
            graph_names: Some(self.graph_names.clone()),
            sub_queries: test_case.sub_queries.clone(),
        };
        TestCaseOutcome {
            case_name: data.case_name.clone(),
            description: data.description.clone(),
            passed: false,
            data: Box::new(data),
        }
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

        // 工作记忆解析：
        // - 直接模式：套件加载时全量载入的全图工作记忆；
        // - DB 模式：一次性预取该用例全部子查询（DB 端 HNSW 候选召回 +
        //   `db_neighbor_depth` 跳邻居扩展）到用例级临时工作记忆——
        //   管线只跑在 DB 召回子图上。
        // 召回观测直接取自 prefetch_db 的返回值（不再重跑一次相同查询：
        // 重跑的副本会随实现演进与真实预取静默漂移，也让时延观测失真）。
        let (wm, db_recall): (Arc<WorkingMemory>, Option<DbRecallDetail>) = match &self.db {
            Some(db) => {
                let queries: Vec<EmbeddedMemoryRetrieveQuery> = test_case
                    .sub_queries
                    .iter()
                    .enumerate()
                    .map(|(sq_idx, sq)| EmbeddedMemoryRetrieveQuery {
                        embedding: query_embs[sq_idx].clone(),
                        query: MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone()),
                    })
                    .collect();

                let case_wm = Arc::new(WorkingMemory::new(10));
                let outcome = match db
                    .rt
                    .block_on(prefetch_db(&db.repo, queries, db.config, &case_wm))
                {
                    Ok(outcome) => outcome,
                    Err(e) => {
                        return self.db_failed_outcome(index, format!("DB 预取失败: {e}"));
                    }
                };
                // 实际写入的子图以工作记忆为准（而非由返回值推断）：
                // 这样观测能暴露"召回到了却没写进工作记忆"这类写入侧偏差。
                let subgraph_ids: Vec<MemoryId> = case_wm
                    .memory_cluster()
                    .read_or_compute(|c| c.graph().node_weights().map(|n| n.note().id()).collect());
                let detail = DbRecallDetail::from_prefetch(
                    &outcome,
                    &subgraph_ids,
                    &test_case.expected_combined_ranking,
                    &test_case.bonus_combined_ranking,
                    test_case.sub_queries.len(),
                    db.config,
                );
                (case_wm, Some(detail))
            }
            None => (
                Arc::clone(self.wm_direct.as_ref().expect("direct mode working memory")),
                None,
            ),
        };
        let flavor = self.mode.flavor();

        // 期望抽象列表（供"抽象检出率 / 抽象直接命中率"指标使用）
        let expected_abstract: Vec<MemoryId> = test_case
            .expected_combined_ranking
            .iter()
            .chain(test_case.bonus_combined_ranking.iter())
            .copied()
            .filter(|id| self.abstract_ids.contains(id))
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

            let ids: Vec<MemoryId> = if flavor == RetrieveFlavor::FullPipeline {
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
                    Arc::clone(&wm),
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
                let ids: Vec<MemoryId> =
                    pipeline_res.association.iter().map(|(id, _)| *id).collect();
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
                        Arc::clone(&wm),
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
                    Arc::clone(&wm),
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

        let (combined_ids, combined_ranking, passed) = match flavor {
            RetrieveFlavor::Embedding => {
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
            RetrieveFlavor::Association => {
                const EMBED_PPR_BLEND: f32 = 0.5;

                let mut all_blended: Vec<(MemoryId, f32, u32)> = Vec::new();

                for (sq_idx, results) in all_similarity.into_iter().enumerate() {
                    let priority = test_case.sub_queries[sq_idx].priority;
                    if results.is_empty() {
                        continue;
                    }

                    let embed_map: HashMap<MemoryId, f32> = results.iter().copied().collect();

                    let source: Vec<(MemoryId, f32)> = results;
                    let req = AssociationRequest::new(Arc::clone(&wm), source)
                        .with_top_k(self.meta.max_results);
                    let ppr_result = RetrAssociation {}.retrieve(req);

                    let ppr_map: HashMap<MemoryId, f64> =
                        ppr_result.iter().map(|(id, s)| (*id, *s)).collect();

                    let all_ids: std::collections::HashSet<MemoryId> =
                        embed_map.keys().chain(ppr_map.keys()).copied().collect();

                    for id in all_ids {
                        let embed_s = embed_map.get(&id).copied().unwrap_or(0.0);
                        let ppr_s = ppr_map.get(&id).copied().unwrap_or(0.0) as f32;
                        let blended = EMBED_PPR_BLEND * embed_s + (1.0 - EMBED_PPR_BLEND) * ppr_s;
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
            RetrieveFlavor::FullPipeline => {
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
        } else if flavor == RetrieveFlavor::FullPipeline {
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
                db_recall,
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
        _total: usize,
        _passed: usize,
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
                    db_recall: data.db_recall.clone(),
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

        let mut metrics: Vec<crate::engine::suite::MetricEntry> = Vec::new();
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
            metrics.push(key_value_metric(
                "平均 MRR",
                group_label.clone(),
                format!("{:.4}", avg_mrr),
            ));
            metrics.push(key_value_metric(
                "平均 Hit",
                group_label.clone(),
                format!("{:.2}", avg_hit),
            ));
            metrics.push(key_value_metric(
                "平均 Recall@3",
                group_label.clone(),
                format!("{:.4}", avg_recall3),
            ));
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
                metrics.push(key_value_metric(
                    "抽象检出率",
                    group_label.clone(),
                    format!("{:.2}", abstract_detected_rate),
                ));
                metrics.push(key_value_metric(
                    "抽象直接命中率",
                    group_label.clone(),
                    format!("{:.2}", abstract_direct_hit_rate),
                ));
            }
            metrics.push(key_value_metric(
                "用例数",
                group_label,
                format!("{}", group_data.len()),
            ));

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
                // DB 模式：行尾附带回溯观测摘要（期望全入子图 / 漏召 n）
                let db_suffix = match data.db_recall.as_ref() {
                    Some(r) if r.expected_count > 0 && r.expected_missed.is_empty() => format!(
                        " | DB 期望 {}/{} 全入子图",
                        r.expected_in_subgraph, r.expected_count
                    ),
                    Some(r) if r.expected_count > 0 => {
                        format!(
                            " | DB 漏召 {}/{}",
                            r.expected_missed.len(),
                            r.expected_count
                        )
                    }
                    _ => String::new(),
                };
                detail_rows.push(DetailRow {
                    text: format!(
                        "  {:28}  {:.4}  {:.2}    {}{}",
                        name, mrr, hit, status, db_suffix
                    ),
                    has_error: hit <= 0.0 && !data.case_name.contains("无意义"),
                });
            }
        }

        // DB 模式汇总：prefetch_db 召回观测（候选/邻居/子图均值、期望覆盖、漏召用例数）。
        // 直接模式所有用例 db_recall 为 None，此块自动跳过。
        let db_observed: Vec<&DbRecallDetail> = outcomes
            .iter()
            .filter_map(|o| {
                o.data
                    .downcast_ref::<RetrieveCaseData>()
                    .and_then(|d| d.db_recall.as_ref())
            })
            .collect();
        if !db_observed.is_empty() {
            let n = db_observed.len() as f64;
            let avg_candidate = db_observed
                .iter()
                .map(|r| r.candidate_count as f64)
                .sum::<f64>()
                / n;
            let avg_neighbor = db_observed
                .iter()
                .map(|r| r.neighbor_count as f64)
                .sum::<f64>()
                / n;
            let avg_subgraph = db_observed
                .iter()
                .map(|r| r.subgraph_count as f64)
                .sum::<f64>()
                / n;
            metrics.push(key_value_metric(
                "候选/邻居/子图（均值）",
                "DB 召回观察",
                format!("{avg_candidate:.1} / {avg_neighbor:.1} / {avg_subgraph:.1}"),
            ));
            let expected_total: usize = db_observed.iter().map(|r| r.expected_count).sum();
            if expected_total > 0 {
                let covered_total: usize = db_observed.iter().map(|r| r.expected_in_subgraph).sum();
                metrics.push(key_value_metric(
                    "期望覆盖（进入子图）",
                    "DB 召回观察",
                    format!(
                        "{:.1}%",
                        covered_total as f64 / expected_total as f64 * 100.0
                    ),
                ));
            }
            let missed_cases = db_observed
                .iter()
                .filter(|r| !r.expected_missed.is_empty())
                .count();
            metrics.push(key_value_metric(
                "期望漏召用例",
                "DB 召回观察",
                format!("{missed_cases}/{}", db_observed.len()),
            ));
        }

        SuiteReport {
            metrics,
            detail_header: String::from("  用例                         MRR     Hit     状态"),
            detail_rows,
            outcomes,
        }
    }
}

/// 连接 mem 数据库（kv-mem 默认 / db_path 走磁盘 SurrealKv）+ 幂等建 schema
/// + 事务内全量写入 example_data 的 EmbeddedMemoryNote（含全部出边）。
async fn connect_and_seed_repo(
    notes: Vec<EmbeddedMemoryNote>,
    db_path: Option<&Path>,
) -> Result<SurrealRepository, String> {
    let repo = match db_path {
        Some(path) => SurrealRepository::connect(path, "soulmem")
            .await
            .map_err(|e| format!("Surreal 磁盘库连接失败: {e}"))?,
        None => SurrealRepository::connect_mem()
            .await
            .map_err(|e| format!("Surreal kv-mem 连接失败: {e}"))?,
    };
    repo.init_schema()
        .await
        .map_err(|e| format!("schema 初始化失败: {e}"))?;
    repo.upsert_notes(notes)
        .await
        .map_err(|e| format!("图数据写入 mem 数据库失败: {e}"))?;
    Ok(repo)
}

/// 节点可读摘要（直接模式与 DB 模式共用同一映射，保证 UI/日志显示一致）。
fn note_summary(note: &MemoryNote) -> NodeSummary {
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
        MemoryType::Situation(_) => (String::from("情境"), String::new(), String::new()),
        MemoryType::Procedure(_) => (String::from("流程"), String::new(), String::new()),
    };
    NodeSummary {
        tags,
        type_label,
        primary,
        secondary,
    }
}

/// 是否为抽象情境节点（抽象检出/直接命中指标观测用）。
fn is_abstract_situation(note: &MemoryNote) -> bool {
    matches!(
        note.mem_type(),
        MemoryType::Situation(SituationType::AbstractSituation(_))
    )
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
///
/// 供深度审计复用：审计在各深度的对照子图上跑同一条合并逻辑，
/// 才能与套件指标逐点可比。
pub(crate) fn merge_by_priority(
    results: Vec<(MemoryId, f32, u32)>,
    top_k: usize,
) -> Vec<(MemoryId, f32)> {
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
    let mut sorted: Vec<(MemoryId, f32, f32)> = merged
        .into_iter()
        .map(|(id, (key, raw))| (id, key, raw))
        .collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    sorted
        .into_iter()
        .take(top_k)
        .map(|(id, _, raw)| (id, raw))
        .collect()
}

/// must/bonus 拆分指标（与套件用例判定同源；供深度审计复用以保证口径一致）。
pub(crate) fn compute_split_metrics(
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
    use crate::engine::dataset::default_db_candidate_k;

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

    #[test]
    fn test_query_file_raw_deserialize_db_candidate_k() {
        // config 缺省 db_candidate_k → None（向后兼容）；显式给出 → 解析成功
        let raw_plain: TestConfigRaw = serde_json::from_str(
            r#"{"similarity_threshold":0.7,"max_results":4,"test_k_values":[1,3]}"#,
        )
        .unwrap();
        assert_eq!(raw_plain.db_candidate_k, None);
        let raw_with: TestConfigRaw = serde_json::from_str(
            r#"{"similarity_threshold":0.7,"max_results":4,"test_k_values":[1,3],"db_candidate_k":64}"#,
        )
        .unwrap();
        assert_eq!(raw_with.db_candidate_k, Some(64));
    }

    #[test]
    fn test_default_db_candidate_k_heuristic() {
        assert_eq!(default_db_candidate_k(4), 20); // max(2*4, 20)
        assert_eq!(default_db_candidate_k(20), 40);
        assert_eq!(default_db_candidate_k(0), 20);
    }

    #[test]
    fn test_note_summary_semantic_and_abstract_flag() {
        use soul_mem_core::memory_note::MemoryNoteBuilder;
        use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
        use soul_mem_core::memory_note::situation_mem::{AbstractSituation, Location};

        let sem_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
            content: "内容".into(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "描述".into(),
        }))
        .tags(vec!["t".into()])
        .build()
        .unwrap();
        let s = note_summary(&sem_note);
        assert_eq!(s.type_label, "语义");
        assert_eq!(s.primary, "内容");
        assert_eq!(s.tags, vec!["t"]);
        assert!(!is_abstract_situation(&sem_note));

        let abs_sit = MemoryNoteBuilder::new(MemoryType::Situation(
            AbstractSituation::Location(Location {
                name: "地点".into(),
                coordinates: "".into(),
            })
            .into(),
        ))
        .build()
        .unwrap();
        assert!(is_abstract_situation(&abs_sit));
        let s2 = note_summary(&abs_sit);
        assert_eq!(s2.type_label, "情境");
    }
}

/// DB 路径端到端集成测试：小图 fixture（真实 BGE 嵌入 + kv-mem 内存库）。
/// 需要模型缓存（与 loader 既有集成测试一致）；若离线且无缓存会失败。
#[cfg(test)]
mod db_suite_tests {
    use super::*;
    use std::fs;

    /// 写入小图（3 个语义节点，无链接）与 question.json（1 用例 / 1 子查询）。
    fn write_mini_dataset(dir: &Path, db_candidate_k: Option<usize>) {
        let graph = r#"[
          {
            "id": "node_fuhua",
            "tags": ["班长", "武术"],
            "mem_type": {
              "Semantic": {
                "content": "符华",
                "aliases": [],
                "concept_type": "Entity",
                "description": "逐火之蛾的班长，精通武术"
              }
            },
            "mem_links": []
          },
          {
            "id": "node_jizi",
            "tags": ["教师", "姬子"],
            "mem_type": {
              "Semantic": {
                "content": "姬子",
                "aliases": [],
                "concept_type": "Entity",
                "description": "圣芙蕾雅学园的教师"
              }
            },
            "mem_links": []
          },
          {
            "id": "node_teri",
            "tags": ["学园长", "修女"],
            "mem_type": {
              "Semantic": {
                "content": "德丽莎",
                "aliases": [],
                "concept_type": "Entity",
                "description": "德丽莎·阿波卡利斯"
              }
            },
            "mem_links": []
          }
        ]"#;
        fs::write(dir.join("graph.json"), graph).unwrap();

        let budget = db_candidate_k
            .map(|k| format!(",\n    \"db_candidate_k\": {k}"))
            .unwrap_or_default();
        let question = format!(
            r#"{{
              "name": "mini_db",
              "description": "db integration mini",
              "graph_path": "graph.json",
              "config": {{
                "similarity_threshold": 0.0,
                "max_results": 10,
                "test_k_values": [1, 3, 5]{budget}
              }},
              "test_cases": [
                {{
                  "name": "who_is_fuhua",
                  "description": "query about fuhua",
                  "sub_queries": [
                    {{
                      "priority": 1,
                      "tag": ["符华", "武术"],
                      "variant": {{
                        "Semantic": [
                          {{ "concept_identifier": "符华", "description": "逐火之蛾的班长" }}
                        ]
                      }}
                    }}
                  ],
                  "expected_per_query": [{{ "q": 0, "ranking": ["node_fuhua"] }}],
                  "expected_combined_ranking": ["node_fuhua"],
                  "bonus_combined_ranking": [],
                  "expected_actions": []
                }}
              ]
            }}"#
        );
        fs::write(dir.join("question.json"), question).unwrap();
    }

    /// 两个套件（直接 / DB）跑同一 question.json，逐用例断言完全一致
    /// （候选预算覆盖全图时 DB 路径不应丢信息）。
    fn assert_db_parity_with_full_budget(flavor: RetrieveFlavor) {
        let dir = tempfile::tempdir().unwrap();
        write_mini_dataset(dir.path(), Some(512));

        let direct_mode = match flavor {
            RetrieveFlavor::Embedding => RetrieveMode::Embedding,
            RetrieveFlavor::Association => RetrieveMode::Association,
            RetrieveFlavor::FullPipeline => RetrieveMode::FullPipeline,
        };
        let db_mode = direct_mode.db_mode().unwrap();
        let question = dir.path().join("question.json");

        let direct_suite = RetrieveSuite::load(&question, direct_mode).expect("direct load");
        let db_suite = RetrieveSuite::load(&question, db_mode).expect("db load");
        assert_eq!(direct_suite.case_count(), db_suite.case_count());

        for i in 0..direct_suite.case_count() {
            let direct = direct_suite.run_case(i);
            let db = db_suite.run_case(i);
            assert!(
                direct.passed,
                "direct case {} should pass: {}",
                i, direct.description
            );
            assert!(db.passed, "db case {} should pass: {}", i, db.description);
            let d = direct
                .data
                .downcast_ref::<RetrieveCaseData>()
                .expect("direct data");
            let b = db.data.downcast_ref::<RetrieveCaseData>().expect("db data");
            assert_eq!(
                d.combined_retrieved_ids, b.combined_retrieved_ids,
                "flavor {flavor}: db(全预算) 检索序列应与直接一致 (case {i})"
            );
            assert_eq!(
                d.combined_ranking_metrics.mrr, b.combined_ranking_metrics.mrr,
                "flavor {flavor}: MRR 应一致 (case {i})"
            );
            assert_eq!(
                d.combined_ranking_metrics.hit_rate, b.combined_ranking_metrics.hit_rate,
                "flavor {flavor}: Hit 应一致 (case {i})"
            );
        }
    }

    #[test]
    fn test_db_embedding_parity_with_full_budget() {
        assert_db_parity_with_full_budget(RetrieveFlavor::Embedding);
    }

    #[test]
    fn test_db_full_parity_with_full_budget() {
        assert_db_parity_with_full_budget(RetrieveFlavor::FullPipeline);
    }

    /// 默认启发式预算（max(2*10,20)=20 ≥ 3 节点 → 全图召回）：db/full 也应通过。
    #[test]
    fn test_db_full_runs_with_default_budget() {
        let dir = tempfile::tempdir().unwrap();
        write_mini_dataset(dir.path(), None);
        let question = dir.path().join("question.json");
        let suite =
            RetrieveSuite::load(&question, RetrieveMode::FullPipelineDb).expect("db/full load");
        assert!(suite.case_count() > 0);
        let outcome = suite.run_case(0);
        assert!(
            outcome.passed,
            "db/full 默认预算应通过: {}",
            outcome.description
        );
    }
}
