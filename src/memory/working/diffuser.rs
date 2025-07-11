use std::collections::HashMap;
use nalgebra::{DVector};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use petgraph::graph::NodeIndex;
use petgraph::prelude::{StableGraph};
use petgraph::visit::{EdgeRef};
use crate::memory::{GraphMemoryLink, MemoryNote};

pub struct DiffuserConfig {
    damping: f32,
    max_iteration: usize,
    clamp_threshold: f32
}
impl Default for DiffuserConfig {
    fn default() -> Self {
        Self {
            damping: 0.5,
            max_iteration: 30,
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
    pub fn base_diffuse(&self, start_nodes: &[NodeIndex], mem_cluster: &StableGraph<MemoryNote, GraphMemoryLink>) -> Vec<(NodeIndex,f32)> {
        if start_nodes.is_empty() || mem_cluster.node_count() == 0 {
            return Vec::new();
        }
        let start_nodes = start_nodes.iter()
            .filter(|&&node| mem_cluster.contains_node(node))
            .copied()
            .collect::<Vec<_>>();

        if start_nodes.is_empty() {
            return Vec::new();
        }

        let index_map: HashMap<NodeIndex, usize> = HashMap::from_iter(
            mem_cluster.node_indices()
                .enumerate()
                .map(|(i, node)| (node, i))
        );
        let reverse_index_map: Vec<NodeIndex> = mem_cluster.node_indices().collect();

        let mut act0: DVector<f32> = DVector::zeros(mem_cluster.node_count());
        
        let start_prob = 1.0 / start_nodes.len() as f32;
        for &node in &start_nodes {
            if let Some(item) = act0.get_mut(index_map[&node]) {
                *item = start_prob;
            }
        }
        let transition_matrix = self.build_transition_matrix(&start_nodes,&index_map, mem_cluster);
        let mut act_current = act0.clone();
        for _ in 0..self.max_iteration {
            let act_next = self.damping * &act0 + (1.0 - self.damping) * &transition_matrix * &act_current;
            if (&act_next - &act_current).norm() < self.clamp_threshold {
                act_current = act_next;
                break;
            }
            act_current = act_next;
        }

        act_current.into_iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if value > self.clamp_threshold {
                    Some((reverse_index_map[i], value))
                } else {
                    None
                }
            })
            .collect()
    }
    fn build_transition_matrix(&self, start_nodes: &[NodeIndex], index_map: &HashMap<NodeIndex, usize> ,mem_cluster: &StableGraph<MemoryNote, GraphMemoryLink>) -> CsrMatrix<f32> {
        let node_count = mem_cluster.node_count();
        let mut matrix: CooMatrix<f32> = CooMatrix::zeros(node_count,node_count);

        for source in mem_cluster.node_indices() {
            let j = index_map[&source];

            let total_intensity = mem_cluster.edges(source).fold(0.0, |acc, edge| {
                acc + edge.weight().intensity
            });

            if total_intensity <= f32::EPSILON{
                //对于无出度或连接强度异常低的节点，转移到初始节点集满足Markov矩阵要求
                let prob = 1.0 / start_nodes.len() as f32;
                for &start in start_nodes {
                    let i = index_map[&start]; // 目标节点索引（行）
                    matrix.push(i, j, prob);  
                }
                continue;
            }

            for edge in mem_cluster.edges(source) {
                matrix.push(index_map[&edge.target()], j, edge.weight().intensity / total_intensity);
            }
        }
        CsrMatrix::from(&matrix)
    }

   
    pub fn monte_carlo_walker() {
        todo!()
    }
}

mod test {
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
    use crate::memory::{MemoryCluster, MemoryLink, MemoryNoteBuilder};
    use super::*;
    fn prepare_graph() -> MemoryCluster {
        let mut graph = MemoryCluster::new(
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(false)
            ).unwrap()
        );
        let mem = vec![
            MemoryNoteBuilder::new("test1")
                .id("test1")
                .links(
                    vec![
                        MemoryLink::new("test2", None::<String>,"test",10.0),
                        MemoryLink::new("test3", None::<String>,"test",10.0)
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test2")
                .id("test2")
                .links(
                    vec![
                        MemoryLink::new("test1", None::<String>,"test",8.0),
                        MemoryLink::new("test4", None::<String>,"test",15.0)
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test3")
                .id("test3")
                .links(
                    vec![
                        MemoryLink::new("test2", None::<String>,"test",10.0),
                    ]
                )
                .build(),
            MemoryNoteBuilder::new("test4")
                .id("test4")
                .links(
                    vec![
                        MemoryLink::new("test2", None::<String>,"test",10.0),
                    ]
                )
                .build(),
        ];
        graph.merge(mem);
        
        graph
    }
    #[test]
    fn test_diffuser() { 
        let graph = prepare_graph();
        let diffuser = MemoryDiffuser::from_config(DiffuserConfig::default());
        let start_nodes = vec![
            NodeIndex::new(0),
            NodeIndex::new(2)
        ];
        println!("{:?}", graph.graph.node_weight(NodeIndex::new(0)));
        println!("{:?}", graph.graph.node_weight(NodeIndex::new(1)));
        println!("{:?}", graph.graph.node_weight(NodeIndex::new(2)));
        println!("{:?}", graph.graph.node_weight(NodeIndex::new(3)));
        let result = diffuser.base_diffuse(&start_nodes, graph.graph());
        println!("{:?}",result);
    }
}