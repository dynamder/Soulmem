use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SituationMemLink {
    AbstractToSpecific(AbstractToSpecific),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractToSpecific {}
impl Default for AbstractToSpecific {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractToSpecific {
    pub fn new() -> Self {
        AbstractToSpecific {}
    }
}
