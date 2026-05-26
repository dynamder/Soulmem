use serde::{Deserialize, Serialize};

use crate::memory::memory_note::MemoryId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SituationMemLink {
    AbstractToSpecific(AbstractToSpecific),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractToSpecific {}
impl AbstractToSpecific {
    pub fn new() -> Self {
        AbstractToSpecific {}
    }
}
