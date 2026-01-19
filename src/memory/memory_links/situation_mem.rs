use serde::{Deserialize, Serialize};

use crate::memory::memory_note::MemoryId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SituationMemLink {
    AbstractToSpecific(AbstractToSpecific),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractToSpecific {
    pub abstract_memory_id: MemoryId,
    pub specific_memory_id: MemoryId,
}
impl AbstractToSpecific {
    pub fn new(abstract_memory_id: MemoryId, specific_memory_id: MemoryId) -> Self {
        AbstractToSpecific {
            abstract_memory_id,
            specific_memory_id,
        }
    }
    pub fn change_specific_memory(&mut self, memory_id: MemoryId) {
        self.specific_memory_id = memory_id;
    }
    pub fn change_abstract_memory(&mut self, memory_id: MemoryId) {
        self.abstract_memory_id = memory_id;
    }
    pub fn get_specific_memories(&self) -> &MemoryId {
        &self.specific_memory_id
    }
    pub fn get_abstract_memory_id(&self) -> MemoryId {
        self.abstract_memory_id
    }
}
