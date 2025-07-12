use std::collections::HashMap;
use anyhow::{Error,Result};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use petgraph::graph::NodeIndex;
use petgraph::prelude::{StableGraph};
use petgraph::visit::{EdgeRef};
use surrealdb::sql::Start;
use crate::memory::{GraphMemoryLink, MemoryCluster, MemoryNote, MemoryQuery};
use crate::utils::IteratorPipe;

fn cosine_similarity(a: &DVector<f32>, b: &DVector<f32>) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Error::msg("Vectors must have the same length"));
    }
    Ok(a.dot(b) / (a.norm() * b.norm()))
}

pub struct DiffuserConfig {
    damping: f32,
    max_iteration: usize,
    clamp_threshold: f32
}
impl Default for DiffuserConfig {
    fn default() -> Self {
        Self {
            damping: 0.5,
            max_iteration: 50,
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
#[derive(Debug,Copy,Clone)]
pub enum DiffuseType {
    Concept,
    Emotion,
    Context
}

#[derive(Debug,Copy,Clone)]
pub struct DiffuseBoostWeight {
    pub emotion: f32,
    pub context: f32
}
impl DiffuseBoostWeight {
    pub fn new(emotion: f32, context: f32) -> Self {
        Self {
            emotion,
            context
        }
    }
    pub fn from_preset(preset: DiffuseType) -> Self {
        match preset { //TODO:test these preset value's actual effect
            DiffuseType::Concept => Self::new(0.9, 0.9),
            DiffuseType::Emotion => Self::new(1.2, 0.6),
            DiffuseType::Context => Self::new(0.7, 1.1)
        }
    }
}
#[derive(Debug,Clone)]
pub struct DiffuseInitStruct<'a> {
    pub blend_weight: DiffuseBoostWeight,
    start_nodes: &'a [NodeIndex],
    emotion_vec: Option<DVector<f32>>,
    context_vec: Option<DVector<f32>>
}
impl<'a> DiffuseInitStruct<'a> {
    pub fn new(blend_weight: DiffuseBoostWeight, start_nodes: &'a [NodeIndex]) -> Self {
        Self {
            blend_weight,
            start_nodes,
            emotion_vec: None,
            context_vec: None
        }
    }
    pub fn start_nodes(&self) -> &'a [NodeIndex] {
        self.start_nodes
    }
    pub fn with_emotion(mut self, emotion_vec: DVector<f32>) -> Self {
        self.emotion_vec = Some(emotion_vec);
        self
    }
    pub fn with_context(mut self, context_vec: DVector<f32>) -> Self {
        self.context_vec = Some(context_vec);
        self
    }
    pub fn emotion_vec(&self) -> Option<&DVector<f32>> {
        self.emotion_vec.as_ref()
    }
    pub fn context_vec(&self) -> Option<&DVector<f32>> {
        self.context_vec.as_ref()
    }
}
#[derive(Debug,Clone)]
pub struct DiffuseResult {
    pub raw: Vec<(NodeIndex,f32)>,
    pub boosted: Vec<(NodeIndex,f32)>,
}
impl DiffuseResult {
    pub fn new(raw: Vec<(NodeIndex,f32)>, boosted: Vec<(NodeIndex,f32)>) -> Self {
        Self {
            raw,
            boosted,
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

        //println!("start_nodes: {:?}", start_nodes);

        if start_nodes.is_empty() {
            return Vec::new();
        }
        let reverse_index_map = mem_cluster.node_indices().collect::<Vec<_>>();
        let index_map: HashMap<NodeIndex, usize> = HashMap::from_iter(
            reverse_index_map.iter()
                .enumerate()
                .map(|(i, &node)| (node, i))
        );


        let mut act0: DVector<f32> = DVector::zeros(mem_cluster.node_count());
        
        let start_prob = 1.0 / start_nodes.len() as f32;
        for &node in &start_nodes {
            if let Some(item) = act0.get_mut(index_map[&node]) {
                *item = start_prob;
            }
        }
        let transition_matrix = self.build_transition_matrix(&start_nodes,&index_map, mem_cluster);
        //println!("transition_matrix: {}", DMatrix::from(&transition_matrix));
        let mut act_current = act0.clone();
        for i in 0..self.max_iteration {
            //println!("iteration: {}, current:{}", i, act_current);
            let act_next = self.damping * &act0 + (1.0 - self.damping) * &transition_matrix * &act_current;
            if (&act_next - &act_current).norm() < self.clamp_threshold {
                //println!("iteration: {}, norm: {}", i, (&act_next - &act_current).norm());
                //println!("iteration ended with: act_current: {}, act_next:{}", act_current,act_next);
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
    pub fn hybrid_diffuse(&self, memory_cluster: &MemoryCluster, init: DiffuseInitStruct) -> DiffuseResult {
        let base_result = self.base_diffuse(init.start_nodes(),memory_cluster.graph());
        let base_iter = base_result.clone().into_iter();

        let boosted_iter = base_iter
            .pipe_if(
                init.emotion_vec().is_some(),
                |iter| self.emotion_boost(
                    iter,
                    memory_cluster,
                    init.emotion_vec().unwrap(),
                    init.blend_weight.emotion
                )
            )
            .pipe_if(
                init.context_vec().is_some(),
                |iter| self.context_boost(
                    iter,
                    memory_cluster,
                    init.context_vec().unwrap(),
                    init.blend_weight.context
                )
            );
        DiffuseResult::new(base_result,boosted_iter.collect())
    }
    fn emotion_boost<I>(&self, raw: I, memory_cluster: &MemoryCluster, emotion_vec: &DVector<f32>, boost_factor: f32) -> impl Iterator<Item = (NodeIndex, f32)>
    where
        I: IntoIterator<Item = (NodeIndex, f32)>
    {
        raw.into_iter().map(move |(node_idx, base_score)| {
            if let Some(node) = memory_cluster.graph().node_weight(node_idx) {
                if let Some(embeddings) = memory_cluster.get_embedding(node.id()) {
                    if let Ok(cos_sim) = cosine_similarity(emotion_vec,&DVector::from_column_slice(&embeddings.emotion_embedding)) {
                        return (node_idx,base_score * ((1.0 + cos_sim) * boost_factor).max(0.8))
                    }
                }
            }
            (node_idx,base_score)
        })

    }
    fn context_boost<I>(&self, raw: I, memory_cluster: &MemoryCluster, context_vec: &DVector<f32>, boost_factor: f32) -> impl Iterator<Item=(NodeIndex,f32)>
    where
        I: IntoIterator<Item = (NodeIndex, f32)>
    {
        raw.into_iter().map(move |(node_idx, base_score)| {
            if let Some(node) = memory_cluster.graph().node_weight(node_idx) {
                if let Some(embeddings) = memory_cluster.get_embedding(node.id()) {
                    if let Ok(cos_sim) = cosine_similarity(context_vec,&DVector::from_column_slice(&embeddings.context_embedding)) {
                        return (node_idx,base_score * (0.8 + 0.4 * cos_sim) * boost_factor)
                    }
                }
            }
            (node_idx,base_score)
        })
    }
   
    pub fn monte_carlo_walker() {
        todo!()
    }
}

mod test {
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
    use crate::memory::{MemoryCluster, MemoryLink, MemoryNoteBuilder, NodeRefId};
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
    #[test]
    fn test_diffuser_with_invalid_start_node() {
        let graph = prepare_graph();
        let diffuser = MemoryDiffuser::from_config(DiffuserConfig::default());
        let start_nodes = vec![
            NodeIndex::new(0),
            NodeIndex::new(4),
        ];
        let result = diffuser.base_diffuse(&start_nodes, graph.graph());
        println!("{:?}",result);
    }
    #[test]
    fn test_diffuser_with_inconsistent_index() {
        let mut graph = prepare_graph();
        let mem2 = vec![
            MemoryNoteBuilder::new("test5")
                .id("test5")
                .links(
                    vec![
                        MemoryLink::new("test1", None::<String>,"test",8.0),
                        MemoryLink::new("test4", None::<String>,"test",15.0)
                    ]
                )
                .build(),
        ];

        graph.merge(mem2);
        graph.remove_single_node(NodeRefId::new("test3"));
        println!("{:?}",graph.graph.node_indices().map(|i| i.index()).collect::<Vec<usize>>());
        let diffuser = MemoryDiffuser::from_config(DiffuserConfig::default());
        let start_nodes = vec![
            NodeIndex::new(0),
            NodeIndex::new(4)
        ];
        let result = diffuser.base_diffuse(&start_nodes, graph.graph());
        println!("{:?}",result);
    }
}