use crate::{
    memory::{
        cluster::memory_cluster::GraphMemoryLink,
        embedding::note::EmbeddedMemoryNote,
        memory_note::{MemoryId, MemoryType},
    },
    utils::graph_algo::ord_float::OrdFloat,
};
use petgraph::{algo::UnitMeasure, visit::EdgeRef};
use qdrant_client::qdrant::ScoredPoint;
use serde::Deserialize;
use std::sync::Arc;

use petgraph::{
    prelude::StableDiGraph,
    stable_graph::EdgeReference,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};

use crate::{
    memory::{
        algo::retrieve::RetrRequest, memory_links::MemoryLinkType, memory_note::MemoryNote,
        query::retrieve::PrioritizedMemoryRetrieveQuery, working_memory::WorkingMemory,
    },
    utils::graph_algo::ppr::weighted_ppr_fp,
};

use super::RetrStrategy;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AssociationConfig {
    #[serde(default)]
    pub intensity_factor: Option<f64>,
    #[serde(default)]
    pub confidence_factor: Option<f64>,
    #[serde(default = "default_damping_factor")]
    pub damping_factor: f64,
    #[serde(default = "default_residue_threshold")]
    pub residue_threshold: f64,
    #[serde(default)]
    pub preference: TypePreference,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_damping_factor() -> f64 {
    0.15
}
fn default_residue_threshold() -> f64 {
    1e-5
}
fn default_top_k() -> usize {
    8
}

impl AssociationConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        source: Vec<(MemoryId, f64)>,
    ) -> AssociationRequest {
        AssociationRequest {
            working_mem,
            source,
            intensity_factor: self.intensity_factor,
            confidence_factor: self.confidence_factor,
            damping_factor: self.damping_factor,
            residue_threshold: self.residue_threshold,
            preference: self.preference,
            top_k: self.top_k,
        }
    }
}

pub struct AssociationRequest {
    pub working_mem: Arc<WorkingMemory>,
    pub source: Vec<(MemoryId, f64)>,
    pub intensity_factor: Option<f64>,
    pub confidence_factor: Option<f64>,
    pub damping_factor: f64,
    pub residue_threshold: f64,
    pub preference: TypePreference,
    pub top_k: usize,
}

impl AssociationRequest {
    pub fn new(working_mem: Arc<WorkingMemory>, source: Vec<(MemoryId, f64)>) -> Self {
        Self {
            working_mem,
            source,
            intensity_factor: None,
            confidence_factor: None,
            damping_factor: 0.15,
            residue_threshold: 1e-5,
            preference: TypePreference::default(),
            top_k: 8,
        }
    }
    pub fn with_preference(mut self, preference: TypePreference) -> Self {
        self.preference = preference;
        self
    }
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
    pub fn with_intensity_factor(mut self, intensity_factor: Option<f64>) -> Self {
        self.intensity_factor = intensity_factor;
        self
    }
    pub fn with_confidence_factor(mut self, confidence_factor: Option<f64>) -> Self {
        self.confidence_factor = confidence_factor;
        self
    }
    pub fn with_damping_factor(mut self, damping_factor: f64) -> Self {
        self.damping_factor = damping_factor;
        self
    }
    pub fn with_residue_threshold(mut self, residue_threshold: f64) -> Self {
        self.residue_threshold = residue_threshold;
        self
    }
}

pub struct RetrAssociation;

impl RetrRequest for AssociationRequest {}

impl RetrStrategy for RetrAssociation {
    type Request = AssociationRequest;
    type Return<'a> = Vec<(MemoryId, f64)>;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let dyn_weight_func = DynWeightFuncBuilder::new(request.preference)
            .option_intensity_factor(request.intensity_factor)
            .option_confidence_factor(request.confidence_factor)
            .build();

        let mem_cluster = request.working_mem.memory_cluster();

        let personalized_vec = mem_cluster.read_or_compute(|cluster| {
            request
                .source
                .into_iter()
                .filter_map(|(id, weight)| {
                    cluster
                        .get_mem_index(id)
                        .map(|index| (index, OrdFloat::from_f64(weight)))
                })
                .collect()
        });

        let res = mem_cluster.read_or_compute(|cluster| {
            weighted_ppr_fp(
                cluster.graph(),
                OrdFloat::from_f64(request.damping_factor),
                personalized_vec,
                OrdFloat::from_f64(request.residue_threshold),
                dyn_weight_func,
                None,
            )
        });

        let mut res = mem_cluster.read_or_compute(|cluster| {
            res.into_iter()
                .filter_map(|(index, score)| {
                    cluster
                        .graph()
                        .node_weight(index)
                        .map(|memory_note| (memory_note.note().id(), score))
                })
                .collect::<Vec<(MemoryId, OrdFloat<f64>)>>()
        });

        res.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

        res.into_iter()
            .take(request.top_k)
            .map(|(id, score)| (id, score.into_inner()))
            .collect()
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub enum TypePreference {
    Semantic,
    #[default]
    Situation,
}

type MemClusterEdgeRef<'a> = EdgeReference<'a, GraphMemoryLink>;
type MemClusterGraph = StableDiGraph<EmbeddedMemoryNote, GraphMemoryLink>;

pub struct DynWeightFuncBuilder {
    intensity_factor: Option<f64>,
    confidence_factor: Option<f64>,
    type_preference: [f64; 2], // [semantic, situation]
}
impl DynWeightFuncBuilder {
    pub fn new(preference: TypePreference) -> Self {
        let type_preference = match preference {
            TypePreference::Semantic => [0.7, 0.3],
            TypePreference::Situation => [0.3, 0.7],
        };
        Self {
            intensity_factor: None,
            confidence_factor: None,
            type_preference,
        }
    }
    pub fn intensity_factor(mut self, factor: f64) -> Self {
        self.intensity_factor = Some(factor);
        self
    }
    pub fn confidence_factor(mut self, factor: f64) -> Self {
        self.confidence_factor = Some(factor);
        self
    }

    pub fn option_intensity_factor(mut self, factor: Option<f64>) -> Self {
        self.intensity_factor = factor;
        self
    }
    pub fn option_confidence_factor(mut self, factor: Option<f64>) -> Self {
        self.confidence_factor = factor;
        self
    }

    //TODO: 测试其是否正确工作
    pub fn build(
        self,
    ) -> impl Fn(
        &MemClusterGraph,
        &MemClusterEdgeRef,
        Option<&PrioritizedMemoryRetrieveQuery>,
    ) -> OrdFloat<f64> {
        let intensity_factor = self.intensity_factor.unwrap_or(1.0);
        let confidence_factor = self.confidence_factor.unwrap_or(0.8);
        move |graph: &MemClusterGraph,
              edge: &MemClusterEdgeRef,
              _query: Option<&PrioritizedMemoryRetrieveQuery>| {
            if let Some(target_weight) = graph.node_weight(edge.target()) {
                match target_weight.note().mem_type() {
                    // Proc类型不能被ppr联想，将由触发的情境进行贝叶斯推理
                    MemoryType::Procedure(_) => return OrdFloat::from_f64(0.0),
                    _ => {}
                }
            }
            let intensity = edge.weight().intensity();
            let (confidence_boost, type_boost) = match edge.weight().link_type() {
                MemoryLinkType::Proc(_) => (0.0, 0.0), // Proc类型记忆不提升置信度, 设想一定程度抑制直接的Proc提取。
                MemoryLinkType::Situation(_) => (0.8, self.type_preference[1]), //TODO: 调整数值，暂时设定为0.8，不设定为0或1，因为这会导致Situation类型记忆占据优势或劣势
                MemoryLinkType::Sem(mem) => (mem.confidence, self.type_preference[0]),
            };
            let normalize_factor = intensity_factor + confidence_factor + type_boost;
            OrdFloat::from_f64(
                (intensity * intensity_factor
                    + confidence_boost as f64 * confidence_factor
                    + type_boost)
                    / normalize_factor,
            )
        }
    }
}
