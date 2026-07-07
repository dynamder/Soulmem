use serde::{Deserialize, Serialize};

///Procedural Memory Link
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcMemLink {
    TrigToAction(TrigToAction),
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrigToAction {
    pub prob: f64, //转移概率
}
impl TrigToAction {
    pub fn new(prob: f64) -> Self {
        TrigToAction { prob }
    }

    pub fn get_prob(&self) -> f64 {
        self.prob
    }
    pub fn set_prob(&mut self, prob: f64) {
        self.prob = prob;
    }
}
