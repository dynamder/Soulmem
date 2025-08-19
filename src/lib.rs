//! This lib crate is inspired from A-mem, an agentic memory system

use crate::memory::long_term::MemoryLongTerm;
use crate::memory::MemoryNote;
use crate::memory::working::WorkingMemory;
use crate::time_wheel::{HierarchicalTimeWheel, TimeWheelRunner};
use anyhow::Result;

pub mod memory;
mod soul_embedding;
mod llm_driver;
mod db;
mod utils;
mod time_wheel;

pub struct SoulMemory {
    working: WorkingMemory,
    long_term: MemoryLongTerm,
    task_scheduler: TimeWheelRunner<HierarchicalTimeWheel>
}
impl SoulMemory {
    pub fn new(working: WorkingMemory, long_term: MemoryLongTerm, task_scheduler: TimeWheelRunner<HierarchicalTimeWheel>) -> Self {
        Self {
            working,
            long_term,
            task_scheduler
        }
    }
    ///根据给定信息，尝试回忆
    pub async fn recall_mem<RetIter>(&mut self, input: &str) -> RetIter 
    where
        RetIter: Iterator<Item = MemoryNote> + Send + 'static,
    {
        todo!()
    }
    ///根据给定的MemoryNote，将记忆直接写入长期记忆
    pub async fn flash_mem<MemNoteIter>(&mut self, mem_to_flash: MemNoteIter ) -> Result<()>
    where
        MemNoteIter: IntoIterator<Item = MemoryNote> + Send + 'static,
    {
        todo!()
    }

}
