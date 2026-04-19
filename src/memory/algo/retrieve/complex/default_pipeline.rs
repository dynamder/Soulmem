use serde::Deserialize;

use crate::memory::{algo::retrieve::RetrRequest, query::retrieve::PrioritizedMemoryRetrieveQuery};

#[derive(Debug, Clone, Deserialize)]
pub struct DefaultPipelineConfig {}

impl DefaultPipelineConfig {
    pub fn into_request(self, query: PrioritizedMemoryRetrieveQuery) -> DefaultPipelineRequest {
        DefaultPipelineRequest { query }
    }
}

pub struct RetrDefaultPipeline;

pub struct DefaultPipelineRequest {
    query: PrioritizedMemoryRetrieveQuery,
}

impl RetrRequest for DefaultPipelineRequest {}
