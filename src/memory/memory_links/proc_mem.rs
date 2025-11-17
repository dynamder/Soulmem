use crate::memory::memory_note::MemoryId;

///Procedural Memory Link
#[derive(Debug, Clone, PartialEq)]
pub enum ProcMemLink {
    TrigToAction(TrigToAction),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrigToAction {
    pub trig: MemoryId,
    pub action: MemoryId,
    pub prob: f32, //转移概率
}
impl TrigToAction {
    pub fn new(trig: MemoryId, action: MemoryId, prob: f32) -> Self {
        TrigToAction { trig, action, prob }
    }
    pub fn get_trig(&self) -> &MemoryId {
        &self.trig
    }
    pub fn get_action(&self) -> &MemoryId {
        &self.action
    }
    pub fn get_prob(&self) -> f32 {
        self.prob
    }
    pub fn set_prob(&mut self, prob: f32) {
        self.prob = prob;
    }
    pub fn into_tuple(self) -> (MemoryId, MemoryId, f32) {
        (self.trig, self.action, self.prob)
    }
}
impl From<(MemoryId, MemoryId, f32)> for TrigToAction {
    fn from(t: (MemoryId, MemoryId, f32)) -> Self {
        TrigToAction::new(t.0, t.1, t.2)
    }
}
impl From<TrigToAction> for (MemoryId, MemoryId, f32) {
    fn from(t: TrigToAction) -> Self {
        t.into_tuple()
    }
}