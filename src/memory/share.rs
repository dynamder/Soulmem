use std::collections::HashMap;
use mockall::automock;
use petgraph::graph::NodeIndex;

#[allow(dead_code)]
#[automock]
pub trait NoteRecord {
    fn activation_count(&self) -> u32;
    fn record_activation(&mut self, indexes: &[NodeIndex]);
    fn activation_history(&self) -> &HashMap<NodeIndex, u32>;
}