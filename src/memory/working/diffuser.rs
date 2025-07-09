use nalgebra::{DMatrix, DVector};
use petgraph::graph::NodeIndex;
use petgraph::prelude::{StableDiGraph, StableGraph};
use petgraph::visit::{IntoEdgeReferences, IntoEdges, NodeIndexable, EdgeRef, IntoNodeReferences};
use crate::memory::{GraphMemoryLink, MemoryNote};

pub struct DiffuserConfig {
    damping: f32,
    max_iteration: usize,
    clamp_threshold: f32
}
impl Default for DiffuserConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iteration: 100,
            clamp_threshold: 1e-6
        }
    }
}
pub struct DiffuserConfigBuilder {
    damping: Option<f32>,
    max_iteration: Option<usize>,
    clamp_threshold: Option<f32>
}
impl DiffuserConfigBuilder {
    pub fn new() -> Self {
        Self {
            damping: None,
            max_iteration: None,
            clamp_threshold: None
        }
    }
    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = Some(damping);
        self
    }
    pub fn max_iteration(mut self, max_iteration: usize) -> Self {
        self.max_iteration = Some(max_iteration);
        self
    }
    pub fn clamp_threshold(mut self, clamp_threshold: f32) -> Self {
        self.clamp_threshold = Some(clamp_threshold);
        self
    }
    pub fn build(self) -> DiffuserConfig {
        DiffuserConfig {
            damping: self.damping.unwrap_or(0.85),
            max_iteration: self.max_iteration.unwrap_or(100),
            clamp_threshold: self.clamp_threshold.unwrap_or(1e-6)
        }
    }
}

//TODO:test it
#[derive(Clone, Debug)]
pub struct MemoryDiffuser {
    damping: f32,
    max_iteration: usize,
    clamp_threshold: f32
}
impl MemoryDiffuser {
    pub fn new(damping: f32, max_iteration: usize, clamp_threshold: f32) -> Self {
        Self {
            damping,
            max_iteration,
            clamp_threshold
        }
    }
    pub fn from_config(config: DiffuserConfig) -> Self {
        Self {
            damping: config.damping,
            max_iteration: config.max_iteration,
            clamp_threshold: config.clamp_threshold
        }
    }
    pub fn diffuse(&self, start_nodes: &[NodeIndex], mem_graph: &StableGraph<MemoryNote, GraphMemoryLink>) -> Vec<(NodeIndex,f32)> {
        let mut act0: DVector<f32> = DVector::zeros(mem_graph.node_bound());
        for &node in start_nodes {
            if let Some(item) = act0.get_mut(node.index()) {
                *item = 1.0 / start_nodes.len() as f32;
            }
        }
        let transition_matrix = self.build_transition_matrix(start_nodes, mem_graph);
        let mut act = act0.clone();
        for _ in 0..self.max_iteration {
            act = (1.0 - self.damping) * &act0 + self.damping * &transition_matrix * act;
            if (&act - &act0).abs().sum() < self.clamp_threshold {
                break;
            }
        }
        act.into_iter()
            .enumerate()
            .filter_map(|(index, &value)| {
                if value > self.clamp_threshold {
                    Some((NodeIndex::new(index), value))
                } else {
                    None
                }
            })
            .collect()
    }
    fn build_transition_matrix(&self, start_nodes: &[NodeIndex],mem_graph: &StableGraph<MemoryNote, GraphMemoryLink>) -> DMatrix<f32> {
        let node_bound = mem_graph.node_bound();
        let mut matrix: DMatrix<f32> = DMatrix::zeros(node_bound,node_bound);
        for source in mem_graph.node_references() {
            let total_intensity = mem_graph.edges(source.0).fold(0.0, |acc, edge| {
                acc + edge.weight().intensity
            });
            if total_intensity < f32::EPSILON {
                continue;
            }
            for edge in mem_graph.edges(source.0) {
                if let Some(item) = matrix.get_mut((edge.target().index(), source.0.index())) {
                    *item = edge.weight().intensity / total_intensity;
                }
            }
        }
        matrix
    }
    pub fn sparse_diffuse(&self, start_nodes: &[NodeIndex], mem_graph: &StableGraph<MemoryNote, GraphMemoryLink>) -> Vec<(NodeIndex,f32)> {
        todo!()
    }
    pub fn monte_carlo_walker() {
        todo!()
    }
}