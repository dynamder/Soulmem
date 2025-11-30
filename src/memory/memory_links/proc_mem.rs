use serde::{Deserialize, Serialize};

use crate::memory::memory_note::MemoryId;

///Procedural Memory Link
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcMemLink {
    TrigToAction(TrigToAction),
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrigToAction {
    pub prob: f32, //转移概率
}
impl TrigToAction {
    pub fn new(prob: f32) -> Self {
        TrigToAction { prob }
    }

    pub fn get_prob(&self) -> f32 {
        self.prob
    }
    pub fn set_prob(&mut self, prob: f32) {
        self.prob = prob;
    }
}
