//采用dfs，通过边权中的记忆向量来快速扩展子图信息，详见ReMindRAG

use soul_mem_core::memory_note::MemoryId;
use soul_mem_runtime::working_memory::WorkingMemory;
use std::sync::Arc;

use super::RetrRequest;
use super::RetrStrategy;

pub struct RetrCachedPath {
    pub max_depth: usize,      // dfs的最大深度
    pub expand_threshold: f64, //计算向量与查询向量的相似度大于此值，将被扩展
}

#[allow(dead_code)]
pub struct CachedPathRequest {
    working_mem: Arc<WorkingMemory>, //计算向量与查询向量的相似度大于此值，将被扩展
}
impl RetrRequest for CachedPathRequest {}

impl RetrStrategy for RetrCachedPath {
    type Request = CachedPathRequest;
    type Return<'a> = Vec<MemoryId>;
    fn retrieve(&self, _request: Self::Request) -> Self::Return<'_> {
        todo!()
    }
}
