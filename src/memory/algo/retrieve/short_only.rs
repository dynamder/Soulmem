use crate::memory::{
    algo::retrieve::RetrRequest, memory_note::MemoryId, working_memory::WorkingMemory,
};
use std::sync::Arc;

//仅提取短期记忆策略，即仅提取滑动窗口
use super::RetrStrategy;
pub struct RetrShortOnly {
    pub clipping_length: Option<usize>,
    pub include_summary: bool,
}
pub struct ShortOnlyRequest {
    working_mem: Arc<WorkingMemory>, //因为检索算法很可能需要并发执行，使用Arc而非引用确保可以Send
}

impl RetrRequest for ShortOnlyRequest {}

impl RetrStrategy for RetrShortOnly {
    type Request = ShortOnlyRequest;
    fn retrieve(&self, request: Self::Request) -> Vec<MemoryId> {
        todo!()
    }
}
