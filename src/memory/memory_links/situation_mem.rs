use serde::{Deserialize, Serialize};

use crate::memory::memory_note::MemoryId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SituationMemLink {
    AbstractToSpecific(AbstractToSpecific),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractToSpecific {
    pub abstract_memory_id: MemoryId,
    pub specific_memory_id: Vec<MemoryId>,
}
impl AbstractToSpecific {
    pub fn new(abstract_memory_id: MemoryId, specific_memory_id: Vec<MemoryId>) -> Self {
        AbstractToSpecific {
            abstract_memory_id,
            specific_memory_id,
        }
    }
    pub fn add_specific_memory(&mut self, memory_id: MemoryId) {
        self.specific_memory_id.push(memory_id);
    }
    pub fn remove_specific_memory(&mut self, memory_id: MemoryId) {
        self.specific_memory_id.retain(|id| *id != memory_id);
    }
    pub fn clear_specific_memories(&mut self) {
        self.specific_memory_id.clear();
    }
    pub fn get_specific_memories(&self) -> &Vec<MemoryId> {
        &self.specific_memory_id
    }
    pub fn get_abstract_memory_id(&self) -> MemoryId {
        self.abstract_memory_id
    }
}
