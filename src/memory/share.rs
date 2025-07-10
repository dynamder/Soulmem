use std::collections::HashMap;
use mockall::automock;
use petgraph::graph::NodeIndex;
use crate::memory::NodeRefId;

#[allow(dead_code)]
#[automock]
pub trait NoteRecord {
    fn activation_count(&self) -> u32;
    fn record_activation(&mut self, act_ids: &[NodeRefId]);
    fn activation_history(&self) -> &HashMap<NodeRefId, u32>;
}