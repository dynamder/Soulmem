use serde::Deserialize;

use crate::memory::memory_note::MemoryId;
use crate::memory::working_memory::WorkingMemory;
use std::sync::Arc;

use super::RetrRequest;
use super::RetrStrategy;

#[derive(Debug, Clone, Deserialize)]
pub struct CachedPathConfig {
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    #[serde(default = "default_expand_threshold")]
    pub expand_threshold: f64,
}

fn default_max_depth() -> usize {
    3
}
fn default_expand_threshold() -> f64 {
    0.7
}

impl CachedPathConfig {
    pub fn into_request(self, working_mem: Arc<WorkingMemory>) -> CachedPathRequest {
        CachedPathRequest {
            working_mem,
            max_depth: self.max_depth,
            expand_threshold: self.expand_threshold,
        }
    }
}

pub struct RetrCachedPath;

pub struct CachedPathRequest {
    working_mem: Arc<WorkingMemory>,
    pub max_depth: usize,
    pub expand_threshold: f64,
}

impl RetrRequest for CachedPathRequest {}

impl RetrStrategy for RetrCachedPath {
    type Request = CachedPathRequest;
    type Return<'a> = Vec<MemoryId>;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        todo!("This will not be included in the MVP.")
    }
}
