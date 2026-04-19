use serde::Deserialize;

use crate::memory::{
    algo::retrieve::RetrRequest,
    memory_note::MemoryId,
    working_memory::{WorkingMemory, sliding_window::Information},
};
use std::sync::Arc;

use super::RetrStrategy;

#[derive(Debug, Clone, Deserialize)]
pub struct ShortOnlyConfig {
    #[serde(default)]
    pub clipping_length: Option<usize>,
    #[serde(default = "default_include_summary")]
    pub include_summary: bool,
}

fn default_include_summary() -> bool {
    false
}

impl ShortOnlyConfig {
    pub fn into_request(self, working_mem: Arc<WorkingMemory>) -> ShortOnlyRequest {
        ShortOnlyRequest {
            working_mem,
            clipping_length: self.clipping_length,
            include_summary: self.include_summary,
        }
    }
}

pub struct RetrShortOnly;

pub struct ShortOnlyRequest {
    working_mem: Arc<WorkingMemory>,
    pub clipping_length: Option<usize>,
    pub include_summary: bool,
}

impl RetrRequest for ShortOnlyRequest {}

impl RetrStrategy for RetrShortOnly {
    type Request = ShortOnlyRequest;
    type Return<'a>
        = (Arc<[Information]>, Arc<str>)
    where
        Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let window = {
            let mut window = request.working_mem.sliding_window().get_windows();
            if let Some(clipping_len) = request.clipping_length {
                window = window
                    .iter()
                    .rev()
                    .take(clipping_len)
                    .rev()
                    .cloned()
                    .collect::<Arc<_>>()
            }
            window
        };
        let summary = request.working_mem.sliding_window().get_summary();
        (window, summary)
    }
}
