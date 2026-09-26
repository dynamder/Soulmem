pub mod association;
pub mod bayes_action;
pub mod complex;
pub mod short_only;
pub mod similarity;

use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
use soul_mem_runtime::storage::{MemoryRepository, StorageResult};
use soul_mem_runtime::working_memory::WorkingMemory;

pub trait RetrStrategy: 'static {
    type Request: RetrRequest;
    type Return<'a>
    where
        Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_>;
}

pub trait RetrRequest {}

#[derive(serde::Deserialize)]
#[serde(tag = "type")]
pub enum RetrRequestConfig {
    Association(association::AssociationConfig),
    BayesAction(bayes_action::BayesActionConfig),
    AssociateWithAction(complex::assoc_with_action::AssociateWithActionConfig),
    ShortOnly(short_only::ShortOnlyConfig),
    Similarity(similarity::SimilarityConfig),
}

/// 数据库预取参数。
///
/// 两个字段都是裸数值且含义容易混淆，因此收进命名结构体而非位置参数：
/// `candidate_k` 是**每个槽位**的预算（查询会 fan-out 成多个槽位列，实际候选
/// 规模 ≈ 槽位数 × candidate_k 再去重），`neighbor_depth` 是**跳数**而非节点数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DbPrefetchConfig {
    /// `similarity_fetch` 的每个槽位 HNSW KNN 召回预算；
    /// DB 只做候选召回，精确重排与 top-k 截断由调用方在内存侧完成。
    pub candidate_k: usize,
    /// 邻居扩展跳数：`0` = 不扩展（只写相似命中），`1` = 一跳邻居。
    pub neighbor_depth: usize,
}

impl DbPrefetchConfig {
    /// 常用配置：给定每槽位预算与邻居跳数。
    pub fn new(candidate_k: usize, neighbor_depth: usize) -> Self {
        Self {
            candidate_k,
            neighbor_depth,
        }
    }
}

impl Default for DbPrefetchConfig {
    /// 默认：每槽位 20 个候选 + 一跳邻居扩展（与测试框架的启发式缺省一致）。
    fn default() -> Self {
        Self {
            candidate_k: 20,
            neighbor_depth: 1,
        }
    }
}

/// 一次数据库预取的观测结果：两次 DB 查询各自的原始产物（顺序与去重语义由仓储决定）。
///
/// 由 [`prefetch_db`] 一并返回，调用方**不需要**为了观测而重跑同样的查询——
/// 观测与实现同源，避免两者随实现演进静默漂移。
/// 两侧都只含 `MemoryId`，实际写入工作记忆的完整节点以工作记忆为准。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefetchOutcome {
    /// `similarity_fetch` 召回命中（跨槽位 union 去重，无相似度分数——重排在内存侧）。
    pub candidates: Vec<MemoryId>,
    /// 邻居扩展结果（`visited - 源节点`，与 `candidates` 天然无重叠）。
    pub neighbors: Vec<MemoryId>,
}

impl PrefetchOutcome {
    /// 候选 + 邻居的节点总数上限（两侧无重叠，故为直接相加）。
    pub fn node_count(&self) -> usize {
        self.candidates.len() + self.neighbors.len()
    }
}

/// 数据库预取：以一组查询嵌入在数据库执行相似度召回（`similarity_fetch`），
/// 再以召回命中为源做 `config.neighbor_depth` 跳的邻居扩展（`fetch_neighbors`），
/// 最后把两次数据库查询取回的 `EmbeddedMemoryNote` 全部写入工作记忆。
///
/// - `queries`：可携带多个查询（如多优先级子查询），全部展平为槽位向量后
///   交由 DB 召回，结果去重 union；
/// - `config.candidate_k`：`similarity_fetch` 每个槽位的召回预算（DB 只做候选召回，
///   精确重排与 top-k 截断由调用方完成，需按槽位数放大余量）；
/// - `config.neighbor_depth`：邻居扩展跳数。深度为 `d` 时写入的节点集合为
///   `候选 ∪ N₁(候选) ∪ … ∪ N_d(候选)`；由于工作记忆只物化被写入的节点
///   （指向集合外的边被挂起、不补建节点），**图上距候选集超过 `d` 跳的节点
///   对该子图上的任何算法都不可达**；
/// - 邻居扩展返回 `visited - 源节点`，与相似命中天然无重叠；
/// - 工作记忆侧按 `MemoryId` 合并（`add_single_node`），重复预取不会产生重复节点；
/// - 返回 [`PrefetchOutcome`] 供调用方观测召回规模与归因漏召，无需重跑查询。
pub async fn prefetch_db(
    repo: &dyn MemoryRepository,
    queries: Vec<EmbeddedMemoryRetrieveQuery>,
    config: DbPrefetchConfig,
    working_mem: &WorkingMemory,
) -> StorageResult<PrefetchOutcome> {
    // 1. 相似度召回：所有查询嵌入展平后做 DB 端 HNSW KNN 候选召回
    //    所有权消费性链路：owned 查询解构移动，无隐式克隆
    let query_embeddings: Vec<_> = queries.into_iter().map(|q| q.embedding).collect();
    let similar = repo
        .similarity_fetch(query_embeddings, config.candidate_k)
        .await?;

    // 2. 邻居扩展：以相似命中为源，恢复链接上下文（深度由配置决定；
    //    深度 0 时源集即全部命中，扩展结果为空）
    let source_ids: Vec<_> = similar.iter().map(|note| note.note().id()).collect();
    let neighbors = repo
        .fetch_neighbors(&source_ids, config.neighbor_depth)
        .await?;

    let outcome = PrefetchOutcome {
        candidates: source_ids,
        neighbors: neighbors.iter().map(|note| note.note().id()).collect(),
    };

    // 3. 两次查询的结果批量写入工作记忆：
    //    通过 memory_cluster 句柄一次性获取写锁批量合并，
    //    避免逐条 add_node 反复加解锁（loader 的图加载同款模式）
    let cluster = working_mem.memory_cluster();
    cluster.write(|c| {
        for embedded_note in similar.into_iter().chain(neighbors) {
            c.add_single_node(embedded_note);
        }
    });
    Ok(outcome)
}
