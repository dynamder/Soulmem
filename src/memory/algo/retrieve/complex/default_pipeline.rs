use std::sync::Arc;

use serde::Deserialize;

use crate::memory::{
    algo::retrieve::{
        RetrRequest, RetrStrategy,
        complex::{AssociateWithActionConfig, RetrAssociateWithAction},
        short_only::{RetrShortOnly, ShortOnlyConfig},
        similarity::{RetrSimilarity, SimilarityConfig},
    },
    embedding::{Embeddable, EmbeddingModel, query::note::EmbeddedMemoryRetrieveQuery},
    memory_note::MemoryId,
    query::retrieve::PrioritizedMemoryRetrieveQuery,
    working_memory::{WorkingMemory, sliding_window::Information},
};

#[derive(Debug, Clone, Deserialize)]
pub struct DefaultPipelineConfig {
    short_mem_with_history: ShortOnlyConfig,
    similarity: SimilarityConfig,
    assoc_with_action: AssociateWithActionConfig,
}

impl DefaultPipelineConfig {
    pub fn into_request(
        self,
        working_memory: Arc<WorkingMemory>,
        query: EmbeddedMemoryRetrieveQuery,
        priority: u32,
    ) -> DefaultPipelineRequest {
        DefaultPipelineRequest {
            working_mem: working_memory,
            query,
            priority,
            pipeline_config: self,
        }
    }
}

pub struct RetrDefaultPipeline;

pub struct DefaultPipelineRequest {
    working_mem: Arc<WorkingMemory>,
    pipeline_config: DefaultPipelineConfig,
    query: EmbeddedMemoryRetrieveQuery,
    priority: u32,
}

pub struct DefaultPipelineResult {
    pub association: Vec<(MemoryId, f64)>,
    pub action: Vec<(MemoryId, f64)>,
    pub short_history: Arc<[Information]>,
    pub short_mem: Arc<str>,
    pub priority: u32,
}

impl RetrRequest for DefaultPipelineRequest {}

impl RetrStrategy for RetrDefaultPipeline {
    type Request = DefaultPipelineRequest;
    type Return<'a> = DefaultPipelineResult;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        //短期记忆：滑动窗口和摘要
        let short_mem_request = request
            .pipeline_config
            .short_mem_with_history
            .into_request(Arc::clone(&request.working_mem));
        let short_mem_res = RetrShortOnly {}.retrieve(short_mem_request);

        //向量相似性搜索，提取ppr源节点
        let EmbeddedMemoryRetrieveQuery {
            query,
            embedding: query_embedding,
        } = request.query;

        let query_embedding = Arc::new(query_embedding);

        let similarity_request = request.pipeline_config.similarity.into_request(
            Arc::clone(&request.working_mem),
            Arc::clone(&query_embedding),
        );
        let similarity_res = RetrSimilarity {}.retrieve(similarity_request);

        //PPR联想及动作概率推理
        let association_with_action_request = request
            .pipeline_config
            .assoc_with_action
            .into_request(Arc::clone(&request.working_mem), similarity_res);
        let association_with_action_res =
            RetrAssociateWithAction {}.retrieve(association_with_action_request);

        DefaultPipelineResult {
            association: association_with_action_res.memory,
            action: association_with_action_res.action,
            short_history: short_mem_res.0,
            short_mem: short_mem_res.1,
            priority: request.priority,
        }
    }
}
