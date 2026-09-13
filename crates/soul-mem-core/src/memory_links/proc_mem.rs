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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig_to_action_new_and_get() {
        let action = TrigToAction::new(0.5);
        assert_eq!(action.get_prob(), 0.5);
    }

    #[test]
    fn test_trig_to_action_set() {
        let mut action = TrigToAction::new(0.1);
        action.set_prob(0.9);
        assert_eq!(action.get_prob(), 0.9);
    }

    #[test]
    fn test_proc_mem_link_variant() {
        let action = TrigToAction::new(0.3);
        let link = ProcMemLink::TrigToAction(action);
        match link {
            ProcMemLink::TrigToAction(a) => assert_eq!(a.get_prob(), 0.3),
        }
    }
}
