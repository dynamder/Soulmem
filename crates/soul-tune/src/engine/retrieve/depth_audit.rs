//! 检索子图深度的几何审计：回答「DB 预取的一跳邻居扩展够不够用」。
//!
//! # 为什么要单独做这件事
//!
//! `prefetch_db` 的子图是 `候选 ∪ N₁(候选) ∪ … ∪ N_d(候选)`（见
//! [`soul_mem_algo::algo::retrieve::prefetch_db`]）。工作记忆只物化**被写入的节点**，
//! 指向集合外的边被挂起、不会补建节点——因此**图上距候选集超过 `d` 跳的节点，
//! 对该子图上的任何算法（含 PPR）都不可达**。于是"深度够不够"是一个几何问题：
//!
//! > 直接模式真正用到的节点里，有多大比例位于候选集的 ≥2 跳位置？
//!
//! # 对照组：把深度做成单变量
//!
//! 直接模式的 PPR 种子是全图 `compute_fused` top-k；DB 模式的候选是**每槽位向量**
//! KNN 的 union。两者本来就不是同一个集合（字符串通道提分、HNSW 近似误差、槽位口径差
//! 都会造成差异），拿"真实 DB 候选"去归因深度只会把三种原因混在一起。因此本模块使用
//! **oracle 种子对照**：
//!
//! - 种子集 `S` = 直接模式在全图上的相似度 top-k（与管线第一步同源，不引入 DB 变量）；
//! - 对照子图 = `{ v : dist(S, v) ≤ d }`，`d` 取多个值；
//! - 全图（`DepthPoint::FullGraph`）作为基线。
//!
//! 这给出**深度本身的能力上界**。真实 DB 路径与它的差距属于召回侧问题，不在本模块。
//!
//! # 不碰数据库
//!
//! 整条审计只读图 + 查询 + 内存中的管线，不写/读 SurrealDB，因此确定性强、可复现。
//! 管线参数与合并逻辑直接复用套件（[`merge_by_priority`]、[`compute_split_metrics`]），
//! 保证 `DepthPoint::FullGraph` 一行与 `retrieve/full` 的指标逐点可比
//! （由 `tests::full_graph_matches_direct_suite` 锁定）。

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;

use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_algo::algo::retrieve::complex::{
    AssociateWithActionConfig, DefaultPipelineConfig, RetrDefaultPipeline,
};
use soul_mem_algo::algo::retrieve::short_only::ShortOnlyConfig;
use soul_mem_algo::algo::retrieve::similarity::{RetrSimilarity, SimilarityConfig};
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
};
use soul_mem_query::query::retrieve::MemoryRetrieveQuery;
use soul_mem_runtime::working_memory::WorkingMemory;

use crate::base::RetrieveFlavor;
use crate::engine::dataset::TestCaseConfig;
use crate::engine::metrics::ranking::compute_action_metrics;
use crate::engine::retrieve::data::{ActionMetrics, RankingMetrics};
use crate::engine::retrieve::dataset::TestCaseQuery;
use crate::engine::retrieve::suite::{RetrDataset, compute_split_metrics, merge_by_priority};

/// 审计对照点：子图是"种子 + d 跳内节点"还是"全图基线"。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DepthPoint {
    /// 种子集 `S` 加上 `d` 跳以内的节点（`0` = 只有种子本身）。
    Hops(usize),
    /// 全图（直接模式基线；用于校验与 `retrieve/full` 等价）。
    FullGraph,
}

impl DepthPoint {
    /// 人类可读标签（表格用）。
    pub fn label(&self) -> String {
        match self {
            DepthPoint::Hops(0) => "种子(0跳)".to_string(),
            DepthPoint::Hops(d) => format!("{d} 跳"),
            DepthPoint::FullGraph => "全图".to_string(),
        }
    }
}

/// 审计参数。
#[derive(Debug, Clone)]
pub struct DepthAuditConfig {
    /// 要评测的跳数上限（去重后升序即可，重复无害）。
    pub depths: Vec<usize>,
}

impl Default for DepthAuditConfig {
    /// 默认扫描 0/1/2/3 跳——1 是当前生产值，0/2/3 是它的上下界对照。
    fn default() -> Self {
        Self {
            depths: vec![0, 1, 2, 3],
        }
    }
}

/// 单个对照点的用例级结果。
#[derive(Debug, Clone, Serialize)]
pub struct DepthCaseRow {
    pub point: DepthPoint,
    /// 该对照子图的节点数。
    pub subgraph_nodes: usize,
    /// 子图占全图的比例（`[0, 1]`）。
    pub coverage: f64,
    /// must 期望是否至少命中一个（与套件 `passed` 同义）。
    pub passed: bool,
    /// must + bonus 去重后实际被召回的期望节点数。
    pub retrieved_expected: usize,
    /// 召回集合中**不在种子集内**的节点数（`|R \ S|`）。
    ///
    /// 全图对照点上的这个数就是"联想/PPR 对最终 top-k 成员集合的净贡献"：
    /// 为 0 意味着最终结果完全由相似度种子决定，子图规模与深度都无法改变成员集合。
    pub retrieved_outside_seeds: usize,
    /// 与套件 FullPipeline 分支同源的合并指标。
    pub metrics: RankingMetrics,
    /// 动作输出指标（`expected_actions` 为空时 `has_expected_actions = false`，占位不参与统计）。
    pub action_metrics: ActionMetrics,
}

/// 单个用例的审计结果：几何归因 + 各对照点的管线指标。
#[derive(Debug, Clone, Serialize)]
pub struct DepthAuditCase {
    pub case_name: String,
    /// 期望节点数（must + bonus 去重）。
    pub expected_count: usize,
    /// oracle 种子集规模 `|S|`（跨子查询 union）。
    pub seed_count: usize,
    /// 逐期望节点距种子集的无向最短跳数（`None` = 从种子集不可达，即跨组件）。
    ///
    /// 保持期望原序（must 在前，bonus 在后）。
    pub expected_hops: Vec<(MemoryId, Option<usize>)>,
    /// 直接模式（全图）召回到的期望节点数——深度损失的分母。
    pub direct_retrieved_expected: usize,
    /// 直接模式召回到、但位于种子集 ≥2 跳位置的期望节点数。
    ///
    /// 这是 **depth=1 的结构性损失上界**：这些节点在全图路径下确实被采用了，
    /// 而在一跳子图里连可达性都没有。
    pub direct_retrieved_beyond_one_hop: usize,
    pub rows: Vec<DepthCaseRow>,
}

/// 期望节点按"最小跳数"的直方图分桶。
#[derive(Debug, Clone, Default, Serialize)]
pub struct HopHistogram {
    /// 距离 = 0（落在种子集内）。
    pub at_seed: usize,
    /// 距离 = 1。
    pub one_hop: usize,
    /// 距离 = 2。
    pub two_hop: usize,
    /// 距离 ≥ 3。
    pub three_hop_or_more: usize,
    /// 从种子集不可达（跨组件；任何深度都救不回来）。
    pub unreachable: usize,
}

impl HopHistogram {
    /// 总期望节点数。
    pub fn total(&self) -> usize {
        self.at_seed + self.one_hop + self.two_hop + self.three_hop_or_more + self.unreachable
    }

    /// 累加另一份直方图（batch 汇总用）。
    pub fn merge(&mut self, other: &HopHistogram) {
        self.at_seed += other.at_seed;
        self.one_hop += other.one_hop;
        self.two_hop += other.two_hop;
        self.three_hop_or_more += other.three_hop_or_more;
        self.unreachable += other.unreachable;
    }
}

/// 单个对照点的跨用例聚合。
///
/// `sum_*` 只服务于合并（batch 汇总需要精确求和后再取均值，不能对均值再取均值），
/// 不进序列化产物。
#[derive(Debug, Clone, Serialize)]
pub struct DepthAggregateRow {
    pub point: DepthPoint,
    pub avg_subgraph_nodes: f64,
    pub avg_coverage: f64,
    /// must 命中的用例数（与套件 `passed` 口径一致）。
    pub passed_cases: usize,
    pub case_count: usize,
    pub total_expected: usize,
    /// 各用例召回期望节点数之和。
    pub retrieved_expected: usize,
    /// 各用例召回集合中不在种子集内的节点数之和（见 [`DepthCaseRow::retrieved_outside_seeds`]）。
    pub retrieved_outside_seeds: usize,
    pub avg_hit_rate: f64,
    pub avg_mrr: f64,
    /// `test_k_values` 里 `k=3` 的 recall 均值（不存在该 k 时为 0）。
    pub avg_recall3: f64,
    /// 带 `expected_actions` 的用例数（动作指标的统计分母）。
    pub action_case_count: usize,
    /// 动作命中的用例数。
    pub action_passed_cases: usize,
    /// 动作 Hit 均值（仅统计带期望动作的用例）。
    pub avg_action_hit_rate: f64,
    #[serde(skip)]
    pub sum_subgraph_nodes: f64,
    #[serde(skip)]
    pub sum_coverage: f64,
    #[serde(skip)]
    pub sum_hit_rate: f64,
    #[serde(skip)]
    pub sum_mrr: f64,
    #[serde(skip)]
    pub sum_recall3: f64,
    #[serde(skip)]
    pub sum_action_hit_rate: f64,
    /// 相对全图基线丢失的期望命中数（逐用例求和）。
    pub lost_vs_full: usize,
    /// 相对 `DepthPoint::Hops(1)` 的期望命中数增益（无 1 跳对照点时为 `None`）。
    pub gain_vs_one_hop: Option<i64>,
}

impl DepthAggregateRow {
    /// 由 `sum_*` 重算均值，并重算两个派生差值。
    ///
    /// 合并（batch）后必须重跑一次：均值不能相加，派生差值依赖全量分母。
    fn recompute_derived(&mut self, full_retrieved: usize, one_hop_retrieved: Option<usize>) {
        let divisor = if self.case_count == 0 {
            1.0
        } else {
            self.case_count as f64
        };
        self.avg_subgraph_nodes = self.sum_subgraph_nodes / divisor;
        self.avg_coverage = self.sum_coverage / divisor;
        self.avg_hit_rate = self.sum_hit_rate / divisor;
        self.avg_mrr = self.sum_mrr / divisor;
        self.avg_recall3 = self.sum_recall3 / divisor;
        // 动作均值只对带期望动作的用例取平均（分母与 hit/mrr 不同）
        self.avg_action_hit_rate = if self.action_case_count == 0 {
            0.0
        } else {
            self.sum_action_hit_rate / self.action_case_count as f64
        };
        self.lost_vs_full = full_retrieved.saturating_sub(self.retrieved_expected);
        self.gain_vs_one_hop =
            one_hop_retrieved.map(|one_hop| self.retrieved_expected as i64 - one_hop as i64);
    }
}

/// 跨用例聚合结果。
#[derive(Debug, Clone, Serialize)]
pub struct DepthAuditAggregate {
    pub case_count: usize,
    pub graph_nodes: usize,
    /// 期望节点跳数分布（全部用例求和）。
    pub hop_histogram: HopHistogram,
    /// 直接模式召回到的期望节点总数。
    pub direct_retrieved_expected: usize,
    /// 其中位于 ≥2 跳的数量。
    pub direct_retrieved_beyond_one_hop: usize,
    pub rows: Vec<DepthAggregateRow>,
}

impl DepthAuditAggregate {
    /// 取某个对照点的聚合行（不存在时返回 `None`）。
    pub fn row(&self, point: DepthPoint) -> Option<&DepthAggregateRow> {
        self.rows.iter().find(|row| row.point == point)
    }
}

/// 一次深度审计的完整报告。
#[derive(Debug, Clone, Serialize)]
pub struct DepthAuditReport {
    pub dataset: String,
    pub flavor: String,
    pub aggregate: DepthAuditAggregate,
    pub cases: Vec<DepthAuditCase>,
}

/// 全图的无向视图：审计的所有跳数计算都在这张图上做。
struct GraphView {
    /// 稠密下标 → 节点 id。
    ids: Vec<MemoryId>,
    /// id → 稠密下标。
    dense_of_id: HashMap<MemoryId, usize>,
    /// 无向邻接（已去重）。
    adjacency: Vec<Vec<usize>>,
}

impl GraphView {
    /// 从工作记忆构图（**无向**：边的两个端点互为邻居）。
    ///
    /// 邻接直接取自各节点的出边（`MemoryNote::links()`），而不是工作记忆内部的
    /// petgraph 结构：`fetch_neighbors` 在 DB 侧遍历的是 `memory_link` 表，
    /// 而那张表正是由这些出边写入的——这样审计的跳数与真实预取的 BFS 同语义。
    fn from_wm(wm: &WorkingMemory) -> Self {
        wm.memory_cluster().read_or_compute(|cluster| {
            let mut ids: Vec<MemoryId> = Vec::new();
            let mut dense_of_id: HashMap<MemoryId, usize> = HashMap::new();
            for node in cluster.graph().node_weights() {
                let id = node.note().id();
                dense_of_id.insert(id, ids.len());
                ids.push(id);
            }

            let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); ids.len()];
            for node in cluster.graph().node_weights() {
                for link in node.note().links() {
                    // 指向图外节点的边（正常不该有）直接跳过，不 panic
                    if let (Some(&a), Some(&b)) =
                        (dense_of_id.get(&link.from()), dense_of_id.get(&link.to()))
                        && a != b
                    {
                        adjacency[a].insert(b);
                        adjacency[b].insert(a);
                    }
                }
            }

            GraphView {
                ids,
                dense_of_id,
                adjacency: adjacency
                    .into_iter()
                    .map(|neighbors| neighbors.into_iter().collect())
                    .collect(),
            }
        })
    }

    /// 全图节点数。
    fn node_count(&self) -> usize {
        self.ids.len()
    }

    /// 无向多源 BFS：返回**可达**节点距 `sources` 的最短跳数（不可达的不出现在结果里）。
    ///
    /// 一次计算即可满足所有深度：`dist ≤ d` 就是 d 跳内的闭包。
    fn hop_distances(&self, sources: &[MemoryId]) -> HashMap<MemoryId, usize> {
        let mut dist: Vec<Option<usize>> = vec![None; self.ids.len()];
        let mut queue: VecDeque<usize> = VecDeque::new();

        for source in sources {
            if let Some(&dense) = self.dense_of_id.get(source)
                && dist[dense].is_none()
            {
                dist[dense] = Some(0);
                queue.push_back(dense);
            }
        }

        while let Some(current) = queue.pop_front() {
            let Some(current_dist) = dist[current] else {
                continue;
            };
            for &next in &self.adjacency[current] {
                if dist[next].is_none() {
                    dist[next] = Some(current_dist + 1);
                    queue.push_back(next);
                }
            }
        }

        self.ids
            .iter()
            .enumerate()
            .filter_map(|(dense, id)| dist[dense].map(|d| (*id, d)))
            .collect()
    }
}

/// `DefaultPipeline` 一次运行的输出（记忆 + 动作）。
struct PipelineOutput {
    /// 合并并截断后的记忆检索序列。
    merged: Vec<MemoryId>,
    /// 合并并截断后的动作节点序列。
    actions: Vec<MemoryId>,
}

/// 在给定工作记忆上跑 DefaultPipeline（参数与合并逻辑与套件 FullPipeline 分支一致）。
fn run_full_pipeline(
    wm: &Arc<WorkingMemory>,
    case: &TestCaseQuery,
    embeddings: &[MemoryRetrieveQueryEmbedding],
    meta: &TestCaseConfig,
) -> PipelineOutput {
    let mut all_memory: Vec<(MemoryId, f32, u32)> = Vec::new();
    let mut all_actions: Vec<(MemoryId, f32, u32)> = Vec::new();
    for (sq_idx, sq) in case.sub_queries.iter().enumerate() {
        let Some(emb) = embeddings.get(sq_idx) else {
            continue;
        };
        let pipeline_config = DefaultPipelineConfig {
            short_mem_with_history: ShortOnlyConfig {
                clipping_length: None,
                include_summary: true,
            },
            similarity: SimilarityConfig {
                similarity_threshold: meta.similarity_threshold,
                max_results: meta.max_results,
            },
            assoc_with_action: AssociateWithActionConfig {
                association: Default::default(),
                action_top_k: 3,
                ..Default::default()
            },
        };
        let request = pipeline_config.into_request(
            Arc::clone(wm),
            EmbeddedMemoryRetrieveQuery {
                embedding: emb.clone(),
                query: MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone()),
            },
            sq.priority,
        );
        let result = RetrDefaultPipeline {}.retrieve(request);
        all_memory.extend(
            result
                .association
                .into_iter()
                .map(|(id, score)| (id, score as f32, sq.priority)),
        );
        all_actions.extend(
            result
                .action
                .into_iter()
                .map(|(id, score)| (id, score as f32, sq.priority)),
        );
    }

    let ids_of = |pairs: Vec<(MemoryId, f32)>| -> Vec<MemoryId> {
        pairs.into_iter().map(|(id, _)| id).collect()
    };
    PipelineOutput {
        merged: ids_of(merge_by_priority(all_memory, meta.max_results)),
        actions: ids_of(merge_by_priority(all_actions, meta.max_results)),
    }
}

/// 动作指标（与套件 FullPipeline 分支同口径；无期望动作时返回占位并标记 `has_expected_actions = false`）。
fn action_metrics_of(
    case: &TestCaseQuery,
    actions: &[MemoryId],
    meta: &TestCaseConfig,
) -> ActionMetrics {
    if case.expected_actions.is_empty() {
        return ActionMetrics {
            action_hit_rate: 1.0,
            action_recall_at: meta.test_k_values.iter().map(|&k| (k, 1.0)).collect(),
            has_expected_actions: false,
        };
    }
    let result = compute_action_metrics(actions, &case.expected_actions, &meta.test_k_values);
    ActionMetrics {
        action_hit_rate: result.action_hit_rate,
        action_recall_at: result.action_recall_at,
        has_expected_actions: true,
    }
}

/// oracle 种子集：管线第一步（相似度）在**全图**上的输出，跨子查询 union（保持顺序去重）。
///
/// 用它而不是 DB 候选，是为了让深度成为单变量（见模块文档）。
fn oracle_seeds(
    wm: &Arc<WorkingMemory>,
    case: &TestCaseQuery,
    embeddings: &[MemoryRetrieveQueryEmbedding],
    meta: &TestCaseConfig,
) -> Vec<MemoryId> {
    let mut seeds: Vec<MemoryId> = Vec::new();
    let mut seen: HashSet<MemoryId> = HashSet::new();
    for (sq_idx, sq) in case.sub_queries.iter().enumerate() {
        let Some(emb) = embeddings.get(sq_idx) else {
            continue;
        };
        let config = SimilarityConfig {
            similarity_threshold: meta.similarity_threshold,
            max_results: meta.max_results,
        };
        let request = config.into_request(
            Arc::clone(wm),
            EmbeddedMemoryRetrieveQuery {
                embedding: emb.clone(),
                query: MemoryRetrieveQuery::new(sq.tags.clone(), sq.variant.clone()),
            },
        );
        let sim = RetrSimilarity {}.retrieve(request);
        for (id, _) in sim {
            if seen.insert(id) {
                seeds.push(id);
            }
        }
    }
    seeds
}

/// 期望集：must + bonus，保持顺序去重（与 `DbRecallDetail::from_prefetch` 同口径）。
fn expected_set(case: &TestCaseQuery) -> Vec<MemoryId> {
    let mut seen: HashSet<MemoryId> = HashSet::new();
    case.expected_combined_ranking
        .iter()
        .chain(case.bonus_combined_ranking.iter())
        .copied()
        .filter(|id| seen.insert(*id))
        .collect()
}

/// 由全图构造只含 `include` 的工作记忆子图。
///
/// 复制 `EmbeddedMemoryNote`（含链接与嵌入），因此子图内边与全图一致；
/// 指向子图外的边在工作记忆里成为挂起边，不补建节点——与 DB 路径行为一致。
fn build_subgraph_wm(full: &WorkingMemory, include: &HashSet<MemoryId>) -> Arc<WorkingMemory> {
    let notes: Vec<EmbeddedMemoryNote> = full.memory_cluster().read_or_compute(|cluster| {
        cluster
            .graph()
            .node_weights()
            .filter(|note| include.contains(&note.note().id()))
            .cloned()
            .collect()
    });
    let subgraph = WorkingMemory::new(10);
    subgraph.memory_cluster().write(|cluster| {
        for note in notes {
            cluster.add_single_node(note);
        }
    });
    Arc::new(subgraph)
}

/// 取 `k` 对应的 recall（缺该 k 时为 0）。
fn recall_at(metrics: &RankingMetrics, k: usize) -> f64 {
    metrics
        .recall_at
        .iter()
        .find(|(value, _)| *value == k)
        .map(|(_, v)| *v)
        .unwrap_or(0.0)
}

/// 跑一次深度审计。
///
/// 只支持 [`RetrieveFlavor::FullPipeline`]（生产路径 `prefetch_db` → `DefaultPipeline`）。
/// 其余 flavor 需要各自复刻套件的混合规则（如 association 的 embed/ppr 加权），
/// 复刻出来的副本会与套件静默漂移，因此这里显式拒绝而不是近似。
pub fn run_depth_audit(
    dataset_path: &Path,
    flavor: RetrieveFlavor,
    config: &DepthAuditConfig,
) -> Result<DepthAuditReport, String> {
    if flavor != RetrieveFlavor::FullPipeline {
        return Err(format!(
            "深度审计只支持 flavor=full（生产路径 prefetch_db → DefaultPipeline），收到 {flavor}"
        ));
    }

    let dataset = RetrDataset::load(dataset_path, flavor, None)
        .map_err(|e| format!("加载数据集失败: {e}"))?;

    let graph = GraphView::from_wm(&dataset.wm);
    let graph_nodes = graph.node_count();

    // 对照点：去重后按"跳数升序"，全图永远排在最后
    let mut depths: Vec<usize> = config.depths.clone();
    depths.sort_unstable();
    depths.dedup();
    let mut points: Vec<DepthPoint> = depths.into_iter().map(DepthPoint::Hops).collect();
    points.push(DepthPoint::FullGraph);

    let mut cases: Vec<DepthAuditCase> = Vec::with_capacity(dataset.case_count());
    for (case_index, tcw) in dataset.test_cases.iter().enumerate() {
        let case = &tcw.query;
        let embeddings = &dataset.query_embeddings[case_index];

        let seeds = oracle_seeds(&dataset.wm, case, embeddings, &dataset.meta);
        let distances = graph.hop_distances(&seeds);
        let expected = expected_set(case);

        let expected_hops: Vec<(MemoryId, Option<usize>)> = expected
            .iter()
            .map(|id| (*id, distances.get(id).copied()))
            .collect();

        // 各对照点的子图 + 管线指标
        let mut rows: Vec<DepthCaseRow> = Vec::with_capacity(points.len());
        let mut full_graph_hits: Option<HashSet<MemoryId>> = None;
        for point in &points {
            let subgraph = match point {
                DepthPoint::FullGraph => Arc::clone(&dataset.wm),
                DepthPoint::Hops(depth) => {
                    let include: HashSet<MemoryId> = distances
                        .iter()
                        .filter(|(_, hop)| **hop <= *depth)
                        .map(|(id, _)| *id)
                        .collect();
                    build_subgraph_wm(&dataset.wm, &include)
                }
            };

            let subgraph_nodes = subgraph
                .memory_cluster()
                .read_or_compute(|cluster| cluster.graph().node_count());
            let output = run_full_pipeline(&subgraph, case, embeddings, &dataset.meta);
            let retrieved_set: HashSet<MemoryId> = output.merged.iter().copied().collect();
            let retrieved_expected = expected
                .iter()
                .filter(|id| retrieved_set.contains(id))
                .count();
            let seed_set: HashSet<MemoryId> = seeds.iter().copied().collect();
            let retrieved_outside_seeds = output
                .merged
                .iter()
                .filter(|id| !seed_set.contains(id))
                .count();
            let (metrics, passed) = compute_split_metrics(
                &output.merged,
                &case.expected_combined_ranking,
                &case.bonus_combined_ranking,
                &dataset.meta.test_k_values,
            );
            let action_metrics = action_metrics_of(case, &output.actions, &dataset.meta);

            if *point == DepthPoint::FullGraph {
                full_graph_hits = Some(retrieved_set);
            }

            rows.push(DepthCaseRow {
                point: *point,
                subgraph_nodes,
                coverage: if graph_nodes == 0 {
                    0.0
                } else {
                    subgraph_nodes as f64 / graph_nodes as f64
                },
                passed,
                retrieved_expected,
                retrieved_outside_seeds,
                metrics,
                action_metrics,
            });
        }

        // 直接模式（全图）召回到的期望节点 —— 深度损失的分母与分子都取自同一集合，
        // 避免"召回了但指标没算进去"的口径错位
        let (direct_retrieved_expected, direct_retrieved_beyond_one_hop) = match &full_graph_hits {
            Some(hits) => {
                let retrieved: usize = expected.iter().filter(|id| hits.contains(id)).count();
                // 位于 ≥2 跳（含跨组件不可达）却仍被全图路径采用的期望节点：
                // depth=1 的结构性损失上界
                let beyond = expected_hops
                    .iter()
                    .filter(|(id, hop)| hits.contains(id) && hop.map(|h| h > 1).unwrap_or(true))
                    .count();
                (retrieved, beyond)
            }
            None => (0, 0),
        };

        cases.push(DepthAuditCase {
            case_name: case.name.clone(),
            expected_count: expected.len(),
            seed_count: seeds.len(),
            expected_hops,
            direct_retrieved_expected,
            direct_retrieved_beyond_one_hop,
            rows,
        });
    }

    let aggregate = aggregate_report(&cases, graph_nodes, &points);
    Ok(DepthAuditReport {
        dataset: dataset_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        flavor: flavor.to_string(),
        aggregate,
        cases,
    })
}

/// 跨用例聚合。
fn aggregate_report(
    cases: &[DepthAuditCase],
    graph_nodes: usize,
    points: &[DepthPoint],
) -> DepthAuditAggregate {
    let mut histogram = HopHistogram::default();
    let mut total_expected = 0usize;
    let mut direct_retrieved_expected = 0usize;
    let mut direct_retrieved_beyond_one_hop = 0usize;
    for case in cases {
        total_expected += case.expected_count;
        direct_retrieved_expected += case.direct_retrieved_expected;
        direct_retrieved_beyond_one_hop += case.direct_retrieved_beyond_one_hop;
        for (_, hop) in &case.expected_hops {
            match hop {
                Some(0) => histogram.at_seed += 1,
                Some(1) => histogram.one_hop += 1,
                Some(2) => histogram.two_hop += 1,
                Some(_) => histogram.three_hop_or_more += 1,
                None => histogram.unreachable += 1,
            }
        }
    }

    let rows: Vec<DepthAggregateRow> = points
        .iter()
        .map(|point| {
            let matching: Vec<&DepthCaseRow> = cases
                .iter()
                .filter_map(|case| case.rows.iter().find(|row| row.point == *point))
                .collect();
            let n = matching.len();
            let action_rows: Vec<&DepthCaseRow> = matching
                .iter()
                .copied()
                .filter(|row| row.action_metrics.has_expected_actions)
                .collect();

            DepthAggregateRow {
                point: *point,
                avg_subgraph_nodes: 0.0,
                avg_coverage: 0.0,
                passed_cases: matching.iter().filter(|row| row.passed).count(),
                case_count: n,
                total_expected,
                retrieved_expected: matching.iter().map(|row| row.retrieved_expected).sum(),
                retrieved_outside_seeds: matching
                    .iter()
                    .map(|row| row.retrieved_outside_seeds)
                    .sum(),
                avg_hit_rate: 0.0,
                avg_mrr: 0.0,
                avg_recall3: 0.0,
                action_case_count: action_rows.len(),
                action_passed_cases: action_rows
                    .iter()
                    .filter(|row| row.action_metrics.action_hit_rate > 0.0)
                    .count(),
                avg_action_hit_rate: 0.0,
                sum_subgraph_nodes: matching.iter().map(|row| row.subgraph_nodes as f64).sum(),
                sum_coverage: matching.iter().map(|row| row.coverage).sum(),
                sum_hit_rate: matching.iter().map(|row| row.metrics.hit_rate).sum(),
                sum_mrr: matching.iter().map(|row| row.metrics.mrr).sum(),
                sum_recall3: matching.iter().map(|row| recall_at(&row.metrics, 3)).sum(),
                sum_action_hit_rate: action_rows
                    .iter()
                    .map(|row| row.action_metrics.action_hit_rate)
                    .sum(),
                lost_vs_full: 0,
                gain_vs_one_hop: None,
            }
        })
        .collect();

    let mut aggregate = DepthAuditAggregate {
        case_count: cases.len(),
        graph_nodes,
        hop_histogram: histogram,
        direct_retrieved_expected,
        direct_retrieved_beyond_one_hop,
        rows,
    };
    finalize_aggregate(&mut aggregate);
    aggregate
}

/// 重算所有派生字段（均值与相对基线差值）。
fn finalize_aggregate(aggregate: &mut DepthAuditAggregate) {
    let full_retrieved = aggregate
        .rows
        .iter()
        .find(|row| row.point == DepthPoint::FullGraph)
        .map(|row| row.retrieved_expected);
    let one_hop_retrieved = aggregate
        .rows
        .iter()
        .find(|row| row.point == DepthPoint::Hops(1))
        .map(|row| row.retrieved_expected);
    for row in &mut aggregate.rows {
        // 无全图对照点时以自身为基线（差值恒 0，而不是伪造一个基线）
        row.recompute_derived(
            full_retrieved.unwrap_or(row.retrieved_expected),
            one_hop_retrieved,
        );
    }
}

/// 合并多份审计报告的聚合部分（batch 汇总用）。
///
/// 各报告必须来自同一份 `DepthAuditConfig`（对照点集合一致）；对照点只出现在
/// 其中一部分报告时按"逐对照点独立求和"处理，均值分母用各自的用例数。
pub fn merge_aggregates(reports: &[DepthAuditReport]) -> Option<DepthAuditAggregate> {
    let mut iter = reports.iter();
    let first = iter.next()?;
    let mut merged = first.aggregate.clone();
    for report in iter {
        merged.case_count += report.aggregate.case_count;
        merged.graph_nodes += report.aggregate.graph_nodes;
        merged.hop_histogram.merge(&report.aggregate.hop_histogram);
        merged.direct_retrieved_expected += report.aggregate.direct_retrieved_expected;
        merged.direct_retrieved_beyond_one_hop += report.aggregate.direct_retrieved_beyond_one_hop;

        for row in &report.aggregate.rows {
            match merged.rows.iter_mut().find(|m| m.point == row.point) {
                Some(target) => {
                    target.passed_cases += row.passed_cases;
                    target.case_count += row.case_count;
                    target.total_expected += row.total_expected;
                    target.retrieved_expected += row.retrieved_expected;
                    target.sum_subgraph_nodes += row.sum_subgraph_nodes;
                    target.sum_coverage += row.sum_coverage;
                    target.sum_hit_rate += row.sum_hit_rate;
                    target.sum_mrr += row.sum_mrr;
                    target.sum_recall3 += row.sum_recall3;
                    target.retrieved_outside_seeds += row.retrieved_outside_seeds;
                    target.action_case_count += row.action_case_count;
                    target.action_passed_cases += row.action_passed_cases;
                    target.sum_action_hit_rate += row.sum_action_hit_rate;
                }
                None => merged.rows.push(row.clone()),
            }
        }
    }
    finalize_aggregate(&mut merged);
    Some(merged)
}

/// 渲染人类可读报告（CLI 用）。
pub fn format_depth_audit(report: &DepthAuditReport) -> String {
    use std::fmt::Write as _;

    let agg = &report.aggregate;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== 深度审计: {} (flavor={}, 全图 {} 节点, {} 用例) ===",
        report.dataset, report.flavor, agg.graph_nodes, agg.case_count
    );

    let hist = &agg.hop_histogram;
    let total = hist.total();
    let pct = |n: usize| {
        if total == 0 {
            0.0
        } else {
            n as f64 / total as f64 * 100.0
        }
    };
    let _ = writeln!(
        out,
        "期望节点距种子集(全图相似度 top-k)的跳数分布（{} 个期望节点）:",
        total
    );
    let _ = writeln!(
        out,
        "  0 跳(种子内) {:>5} ({:>5.1}%)",
        hist.at_seed,
        pct(hist.at_seed)
    );
    let _ = writeln!(
        out,
        "  1 跳         {:>5} ({:>5.1}%)",
        hist.one_hop,
        pct(hist.one_hop)
    );
    let _ = writeln!(
        out,
        "  2 跳         {:>5} ({:>5.1}%)",
        hist.two_hop,
        pct(hist.two_hop)
    );
    let _ = writeln!(
        out,
        "  ≥3 跳        {:>5} ({:>5.1}%)",
        hist.three_hop_or_more,
        pct(hist.three_hop_or_more)
    );
    let _ = writeln!(
        out,
        "  跨组件不可达 {:>5} ({:>5.1}%)   ← 任何深度都救不回来（需提高候选召回）",
        hist.unreachable,
        pct(hist.unreachable)
    );
    let _ = writeln!(
        out,
        "直接模式召回的期望节点: {} | 其中位于 ≥2 跳: {}   ← depth=1 的结构性损失上界",
        agg.direct_retrieved_expected, agg.direct_retrieved_beyond_one_hop
    );
    if let Some(full) = agg.row(DepthPoint::FullGraph) {
        let _ = writeln!(
            out,
            "全图召回集合中不在种子集内的节点: {}   ← 联想(PPR)对最终 top-k 成员集合的净贡献",
            full.retrieved_outside_seeds
        );
    }

    let _ = writeln!(
        out,
        "\n{:<10} {:>7} {:>7} {:>9} {:>9} {:>8} {:>7} {:>7} {:>9} {:>7} {:>8} {:>13}",
        "深度",
        "子图",
        "覆盖率",
        "通过用例",
        "期望命中",
        "种子外",
        "Hit",
        "MRR",
        "Recall@3",
        "丢全图",
        "比1跳",
        "动作Hit"
    );
    for row in &agg.rows {
        let gain = row
            .gain_vs_one_hop
            .map(|g| format!("{g:+}"))
            .unwrap_or_else(|| "-".to_string());
        let action = if row.action_case_count == 0 {
            "N/A".to_string()
        } else {
            format!(
                "{:.3} {}/{}",
                row.avg_action_hit_rate, row.action_passed_cases, row.action_case_count
            )
        };
        let _ = writeln!(
            out,
            "{:<10} {:>7.1} {:>6.1}% {:>9} {:>9} {:>8} {:>7.3} {:>7.3} {:>9.3} {:>7} {:>8} {:>13}",
            row.point.label(),
            row.avg_subgraph_nodes,
            row.avg_coverage * 100.0,
            format!("{}/{}", row.passed_cases, row.case_count),
            format!("{}/{}", row.retrieved_expected, row.total_expected),
            row.retrieved_outside_seeds,
            row.avg_hit_rate,
            row.avg_mrr,
            row.avg_recall3,
            row.lost_vs_full,
            gain,
            action
        );
    }
    out
}

/// 每个用例的失败/漏召明细（CLI `--verbose` 用）。
pub fn format_depth_audit_cases(report: &DepthAuditReport) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for case in &report.cases {
        let _ = writeln!(
            out,
            "用例 {} | 期望 {} | 种子 {} | 直接召回期望 {} | ≥2 跳的 {}",
            case.case_name,
            case.expected_count,
            case.seed_count,
            case.direct_retrieved_expected,
            case.direct_retrieved_beyond_one_hop
        );
        let hops: Vec<String> = case
            .expected_hops
            .iter()
            .map(|(_, hop)| {
                hop.map(|h| h.to_string())
                    .unwrap_or_else(|| "不可达".to_string())
            })
            .collect();
        let _ = writeln!(out, "  期望节点跳数: {}", hops.join(", "));
        for row in &case.rows {
            let _ = writeln!(
                out,
                "  {:<10} 子图 {:>4} ({:>5.1}%) 通过 {:<5} 期望命中 {:>3} Hit {:.3} MRR {:.3}",
                row.point.label(),
                row.subgraph_nodes,
                row.coverage * 100.0,
                row.passed,
                row.retrieved_expected,
                row.metrics.hit_rate,
                row.metrics.mrr
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{MemoryEmbedding, MemoryEmbeddingVariant};
    use soul_mem_query::embedding::sem::SemanticEmbedding;

    const DIM: usize = 128;

    /// 链接链 `n0 → n1 → … → n{len-1}` 的工作记忆（纯几何，不含真实嵌入）。
    fn chain_wm(len: usize) -> (WorkingMemory, Vec<MemoryId>) {
        let wm = WorkingMemory::new(10);
        let ids: Vec<MemoryId> = (0..len).map(|_| MemoryId::new()).collect();
        let cluster = wm.memory_cluster();
        cluster.write(|c| {
            for i in 0..len {
                let links: Vec<MemoryLink> = if i + 1 < len {
                    vec![MemoryLink::new(
                        ids[i],
                        ids[i + 1],
                        MemoryLinkType::Sem(SemMemLink::new("rel".into(), 0.8)),
                    )]
                } else {
                    vec![]
                };
                let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                    content: format!("node-{i}"),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                }))
                .id(ids[i])
                .mem_links(links)
                .build()
                .unwrap();
                let embedding = MemoryEmbedding::new(
                    EmbeddingVec::zero(DIM),
                    MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                        EmbeddingVec::zero(DIM),
                        EmbeddingVec::zero(DIM),
                        EmbeddingVec::zero(DIM),
                    )),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding });
            }
        });
        (wm, ids)
    }

    #[test]
    fn test_hop_distances_undirected_chain() {
        let (wm, ids) = chain_wm(4);
        let graph = GraphView::from_wm(&wm);
        assert_eq!(graph.node_count(), 4);

        let distances = graph.hop_distances(&[ids[0]]);
        assert_eq!(distances.get(&ids[0]), Some(&0));
        assert_eq!(distances.get(&ids[1]), Some(&1));
        assert_eq!(distances.get(&ids[2]), Some(&2));
        assert_eq!(distances.get(&ids[3]), Some(&3));

        // 无向：从末端出发也能回到头（与 fetch_neighbors 的 in/out 同语义）
        let reverse = graph.hop_distances(&[ids[3]]);
        assert_eq!(reverse.get(&ids[0]), Some(&3));

        // 多源：取更近的那个源
        let multi = graph.hop_distances(&[ids[0], ids[3]]);
        assert_eq!(multi.get(&ids[2]), Some(&1));
    }

    #[test]
    fn test_hop_distances_reports_unreachable_as_absent() {
        let (wm, ids) = chain_wm(3);
        // 再加一个孤立节点：从链上任何节点都不可达
        let lonely = MemoryId::new();
        wm.memory_cluster().write(|c| {
            let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "lonely".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(lonely)
            .build()
            .unwrap();
            let embedding = MemoryEmbedding::new(
                EmbeddingVec::zero(DIM),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(DIM),
                    EmbeddingVec::zero(DIM),
                    EmbeddingVec::zero(DIM),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote { note, embedding });
        });

        let graph = GraphView::from_wm(&wm);
        let distances = graph.hop_distances(&[ids[0]]);
        assert!(
            !distances.contains_key(&lonely),
            "跨组件节点不应出现在距离表里"
        );
    }

    #[test]
    fn test_build_subgraph_keeps_induced_edges_only() {
        let (wm, ids) = chain_wm(3);
        let include: HashSet<MemoryId> = [ids[0], ids[1]].into_iter().collect();
        let subgraph = build_subgraph_wm(&wm, &include);

        let (nodes, edges) = subgraph
            .memory_cluster()
            .read_or_compute(|c| (c.graph().node_count(), c.graph().edge_count()));
        assert_eq!(nodes, 2, "只应写入被选中的节点");
        assert_eq!(edges, 1, "指向子图外的边应挂起，不产生边也不补建节点");
    }

    // ── 聚合与合并 ──

    fn metrics_of(hit: f64, mrr: f64, recall3: f64) -> RankingMetrics {
        RankingMetrics {
            recall_at: vec![(1, recall3), (3, recall3)],
            precision_at: vec![(1, recall3), (3, recall3)],
            mrr,
            ndcg_at: vec![],
            hit_rate: hit,
        }
    }

    fn row(point: DepthPoint, passed: bool, retrieved: usize, hit: f64, mrr: f64) -> DepthCaseRow {
        DepthCaseRow {
            point,
            subgraph_nodes: 2,
            coverage: 0.2,
            passed,
            retrieved_expected: retrieved,
            retrieved_outside_seeds: 0,
            metrics: metrics_of(hit, mrr, hit),
            action_metrics: ActionMetrics {
                action_hit_rate: 0.0,
                action_recall_at: vec![],
                has_expected_actions: false,
            },
        }
    }

    /// 带期望动作的行：用于校验动作均值只对这批用例取平均。
    fn row_with_action(point: DepthPoint, action_hit: f64) -> DepthCaseRow {
        DepthCaseRow {
            action_metrics: ActionMetrics {
                action_hit_rate: action_hit,
                action_recall_at: vec![(3, action_hit)],
                has_expected_actions: true,
            },
            ..row(point, true, 1, 1.0, 1.0)
        }
    }

    fn case_of(name: &str, hops: Vec<Option<usize>>, rows: Vec<DepthCaseRow>) -> DepthAuditCase {
        let full = rows
            .iter()
            .find(|r| r.point == DepthPoint::FullGraph)
            .map(|r| r.retrieved_expected)
            .unwrap_or(0);
        let beyond = hops
            .iter()
            .filter(|h| h.map(|d| d > 1).unwrap_or(true))
            .count();
        DepthAuditCase {
            case_name: name.to_string(),
            expected_count: hops.len(),
            seed_count: 2,
            expected_hops: hops.into_iter().map(|h| (MemoryId::new(), h)).collect(),
            direct_retrieved_expected: full,
            direct_retrieved_beyond_one_hop: beyond,
            rows,
        }
    }

    fn points_of() -> Vec<DepthPoint> {
        vec![DepthPoint::Hops(1), DepthPoint::FullGraph]
    }

    #[test]
    fn test_aggregate_histogram_and_derived_fields() {
        let cases = vec![
            case_of(
                "a",
                vec![Some(0), Some(1), Some(2), None],
                vec![
                    row(DepthPoint::Hops(1), true, 2, 0.5, 0.4),
                    row(DepthPoint::FullGraph, true, 4, 1.0, 0.9),
                ],
            ),
            case_of(
                "b",
                vec![Some(0), Some(3)],
                vec![
                    row(DepthPoint::Hops(1), false, 1, 0.0, 0.0),
                    row(DepthPoint::FullGraph, true, 2, 1.0, 1.0),
                ],
            ),
        ];

        let aggregate = aggregate_report(&cases, 10, &points_of());
        assert_eq!(aggregate.case_count, 2);
        assert_eq!(aggregate.graph_nodes, 10);
        let hist = &aggregate.hop_histogram;
        assert_eq!(hist.at_seed, 2);
        assert_eq!(hist.one_hop, 1);
        assert_eq!(hist.two_hop, 1);
        assert_eq!(hist.three_hop_or_more, 1);
        assert_eq!(hist.unreachable, 1);
        assert_eq!(hist.total(), 6);
        // 全图召回 6 个期望命中，其中 ≥2 跳/不可达的有 3 个（2跳、3跳、不可达）
        assert_eq!(aggregate.direct_retrieved_expected, 6);
        assert_eq!(aggregate.direct_retrieved_beyond_one_hop, 3);

        let one_hop = aggregate.row(DepthPoint::Hops(1)).unwrap();
        let full = aggregate.row(DepthPoint::FullGraph).unwrap();
        assert_eq!(one_hop.passed_cases, 1);
        assert_eq!(one_hop.retrieved_expected, 3);
        assert!((one_hop.avg_hit_rate - 0.25).abs() < 1e-9);
        assert!((one_hop.avg_mrr - 0.2).abs() < 1e-9);
        // lost_vs_full = 全图 6 − 1跳 3
        assert_eq!(one_hop.lost_vs_full, 3);
        assert_eq!(one_hop.gain_vs_one_hop, Some(0));
        assert_eq!(full.passed_cases, 2);
        assert_eq!(full.lost_vs_full, 0);
        assert_eq!(full.gain_vs_one_hop, Some(3));
    }

    #[test]
    fn test_merge_aggregates_sums_then_recomputes_means() {
        let points = points_of();
        let b_hop1 = DepthCaseRow {
            action_metrics: ActionMetrics {
                action_hit_rate: 0.0,
                action_recall_at: vec![(3, 0.0)],
                has_expected_actions: true,
            },
            ..row(DepthPoint::Hops(1), false, 0, 0.0, 0.0)
        };
        let report_a = DepthAuditReport {
            dataset: "a".into(),
            flavor: "full".into(),
            aggregate: aggregate_report(
                &[case_of(
                    "a1",
                    vec![Some(0)],
                    vec![
                        row_with_action(DepthPoint::Hops(1), 1.0),
                        row(DepthPoint::FullGraph, true, 1, 1.0, 1.0),
                    ],
                )],
                10,
                &points,
            ),
            cases: Vec::new(),
        };
        let report_b = DepthAuditReport {
            dataset: "b".into(),
            flavor: "full".into(),
            aggregate: aggregate_report(
                &[case_of(
                    "b1",
                    vec![Some(0), Some(2)],
                    vec![b_hop1, row(DepthPoint::FullGraph, true, 2, 1.0, 0.5)],
                )],
                30,
                &points,
            ),
            cases: Vec::new(),
        };

        let merged = merge_aggregates(&[report_a, report_b]).expect("合并应成功");
        assert_eq!(merged.case_count, 2);
        assert_eq!(merged.graph_nodes, 40, "图规模是跨数据集求和");
        assert_eq!(merged.hop_histogram.total(), 3);

        let one_hop = merged.row(DepthPoint::Hops(1)).unwrap();
        let full = merged.row(DepthPoint::FullGraph).unwrap();
        // 均值必须按用例数加权重算，而不是对两个均值再取平均
        assert!((one_hop.avg_hit_rate - 0.5).abs() < 1e-9);
        assert!((full.avg_hit_rate - 1.0).abs() < 1e-9);
        assert!((full.avg_mrr - 0.75).abs() < 1e-9);
        assert_eq!(one_hop.retrieved_expected, 1);
        // 派生差值用合并后的分母重算：全图 3 − 1跳 1 = 2
        assert_eq!(one_hop.lost_vs_full, 2);
        assert_eq!(full.gain_vs_one_hop, Some(2));
        // 动作均值只对带期望动作的用例取平均（2 个里 1 个命中 → 0.5）
        assert_eq!(one_hop.action_case_count, 2);
        assert_eq!(one_hop.action_passed_cases, 1);
        assert!((one_hop.avg_action_hit_rate - 0.5).abs() < 1e-9);
        assert_eq!(
            full.action_case_count, 0,
            "全图行没标动作期望 → 不参与动作统计"
        );
        assert_eq!(full.avg_action_hit_rate, 0.0);
    }

    #[test]
    fn test_merge_aggregates_empty_is_none() {
        assert!(merge_aggregates(&[]).is_none());
    }

    #[test]
    fn test_depth_point_labels() {
        assert_eq!(DepthPoint::Hops(0).label(), "种子(0跳)");
        assert_eq!(DepthPoint::Hops(2).label(), "2 跳");
        assert_eq!(DepthPoint::FullGraph.label(), "全图");
    }

    /// 自校验：`DepthPoint::FullGraph` 必须与 `retrieve/full` 套件逐用例一致。
    ///
    /// 这是审计实现的锚点——若这里的合并/指标口径与套件漂移，本测试会红。
    /// 需要嵌入模型（与套件其它检索测试同源）；离线冷启动下模型不可用时
    /// 明确跳过并说明原因，而不是伪装通过。
    #[test]
    fn test_full_graph_matches_direct_suite() {
        use crate::base::RetrieveMode;
        use crate::engine::loader::get_bge_model;
        use crate::engine::suite::TestSuite;

        if let Err(e) = get_bge_model() {
            eprintln!("跳过 test_full_graph_matches_direct_suite：嵌入模型不可用（{e}）");
            return;
        }

        let dataset_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|root| root.join("fixtures/queries/retr_sim_smoke_zh.json"))
            .expect("仓库根目录");
        if !dataset_path.exists() {
            eprintln!(
                "跳过 test_full_graph_matches_direct_suite：数据集不存在（{}）",
                dataset_path.display()
            );
            return;
        }

        let config = DepthAuditConfig { depths: vec![0] };
        let report = run_depth_audit(&dataset_path, RetrieveFlavor::FullPipeline, &config)
            .expect("审计应成功");
        let suite = crate::engine::retrieve::suite::RetrieveSuite::load(
            &dataset_path,
            RetrieveMode::FullPipeline,
        )
        .expect("直接套件应加载成功");

        assert_eq!(report.cases.len(), suite.case_count());
        for (index, case) in report.cases.iter().enumerate() {
            let outcome = suite.run_case(index);
            let data = outcome
                .data
                .downcast_ref::<crate::engine::retrieve::data::RetrieveCaseData>()
                .expect("套件产出应为 RetrieveCaseData");
            let full_row = case
                .rows
                .iter()
                .find(|r| r.point == DepthPoint::FullGraph)
                .expect("每个用例都应有全图对照行");

            assert_eq!(
                case.case_name, data.case_name,
                "用例顺序应与套件一致（第 {index} 个）"
            );
            assert_eq!(
                full_row.metrics.hit_rate, data.combined_ranking_metrics.hit_rate,
                "用例 {} 的 Hit 应与 retrieve/full 一致",
                case.case_name
            );
            assert_eq!(
                full_row.metrics.mrr, data.combined_ranking_metrics.mrr,
                "用例 {} 的 MRR 应与 retrieve/full 一致",
                case.case_name
            );
            assert_eq!(
                full_row.metrics.recall_at, data.combined_ranking_metrics.recall_at,
                "用例 {} 的 Recall 应与 retrieve/full 一致",
                case.case_name
            );
            assert_eq!(
                full_row.passed, outcome.passed,
                "用例 {} 的通过判定应与 retrieve/full 一致",
                case.case_name
            );
        }
    }
}
