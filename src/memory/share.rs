use std::collections::HashMap;
use petgraph::graph::NodeIndex;

#[allow(dead_code)]
pub trait NoteRecord {
    fn activation_count(&self) -> u32;
    fn record_activation(&mut self, indexes: &[NodeIndex]);
    fn activation_history(&self) -> &HashMap<NodeIndex, u32>;
}