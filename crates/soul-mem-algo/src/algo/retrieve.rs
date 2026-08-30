pub mod association;
pub mod bayes_action;
pub mod complex;
pub mod short_only;
pub mod similarity;

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

/// 数据库预取：以一组查询嵌入在数据库执行相似度召回（`similarity_fetch`），
/// 再以召回命中为源做**深度 1** 的邻居扩展（`fetch_neighbors`），
/// 最后把两次数据库查询取回的 `EmbeddedMemoryNote` 全部写入工作记忆。
///
/// - `queries`：可携带多个查询（如多优先级子查询），全部展平为槽位向量后
///   交由 DB 召回，结果去重 union；
/// - `candidate_k`：`similarity_fetch` 每个槽位的召回预算（DB 只做候选召回，
///   精确重排与 top-k 截断由调用方完成，需按槽位数放大余量）；
/// - 邻居扩展返回 `visited - 源节点`，与相似命中天然无重叠；
/// - 工作记忆侧按 `MemoryId` 合并（`add_single_node`），重复预取不会产生重复节点。
pub async fn prefetch_db(
    repo: &dyn MemoryRepository,
    queries: Vec<EmbeddedMemoryRetrieveQuery>,
    candidate_k: usize,
    working_mem: &WorkingMemory,
) -> StorageResult<()> {
    // 1. 相似度召回：所有查询嵌入展平后做 DB 端 HNSW KNN 候选召回
    //    所有权消费性链路：owned 查询解构移动，无隐式克隆
    let query_embeddings: Vec<_> = queries.into_iter().map(|q| q.embedding).collect();
    let similar = repo
        .similarity_fetch(query_embeddings, candidate_k)
        .await?;

    // 2. 一跳邻居扩展：以相似命中为源，恢复链接上下文（深度 1）
    let source_ids: Vec<_> = similar.iter().map(|note| note.note().id()).collect();
    let neighbors = repo.fetch_neighbors(&source_ids, 1).await?;

    // 3. 两次查询的结果批量写入工作记忆：
    //    通过 memory_cluster 句柄一次性获取写锁批量合并，
    //    避免逐条 add_node 反复加解锁（loader 的图加载同款模式）
    let cluster = working_mem.memory_cluster();
    cluster.write(|c| {
        for embedded_note in similar.into_iter().chain(neighbors) {
            c.add_single_node(embedded_note);
        }
    });
    Ok(())
}
