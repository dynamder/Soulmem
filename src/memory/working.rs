use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use crate::memory::{GraphMemoryLink, MemoryLink, MemoryNote};


pub struct RecordNewNode {
    pub index: NodeIndex,
    pub ref_count: u32,
}
pub struct SoulTask {
    pub summary: String,
    pub related_notes: Vec<NodeIndex>,
}

#[allow(dead_code)]
pub struct WorkingMemory {
    graph: StableGraph<MemoryNote, GraphMemoryLink>,
    temporary: Vec<RecordNewNode>,
    
}