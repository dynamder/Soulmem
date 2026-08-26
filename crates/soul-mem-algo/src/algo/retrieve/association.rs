use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_runtime::cluster::memory_cluster::GraphMemoryLink;

use petgraph::{algo::UnitMeasure, visit::EdgeRef};
use serde::Deserialize;
use std::sync::Arc;

use petgraph::{prelude::StableDiGraph, stable_graph::EdgeReference};

use crate::{
    algo::retrieve::RetrRequest,
    common::{ord_float::OrdFloat, ppr::weighted_ppr_fp},
};
use soul_mem_core::memory_links::MemoryLinkType;
use soul_mem_query::query::retrieve::PrioritizedMemoryRetrieveQuery;
use soul_mem_runtime::working_memory::WorkingMemory;

use super::RetrStrategy;

#[derive(Debug, Clone, Deserialize)]
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
impl Default for AssociationConfig {
    fn default() -> Self {
        AssociationConfig {
            intensity_factor: None,
            confidence_factor: None,
            damping_factor: 0.15,
            residue_threshold: 1e-5,
            preference: TypePreference::Situation,
            top_k: 8,
        }
    }
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
        source: Vec<(MemoryId, f32)>,
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
    pub source: Vec<(MemoryId, f32)>,
    pub intensity_factor: Option<f64>,
    pub confidence_factor: Option<f64>,
    pub damping_factor: f64,
    pub residue_threshold: f64,
    pub preference: TypePreference,
    pub top_k: usize,
}

impl AssociationRequest {
    pub fn new(working_mem: Arc<WorkingMemory>, source: Vec<(MemoryId, f32)>) -> Self {
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
        if request.source.is_empty() {
            return Vec::new();
        }

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
                        .map(|index| (index, OrdFloat::from_f32(weight)))
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
                MemoryLinkType::Sit(_) => (0.8, self.type_preference[1]), //TODO: 调整数值，暂时设定为0.8，不设定为0或1，因为这会导致Situation类型记忆占据优势或劣势
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

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::MemoryLink;
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_note::{
        MemoryNoteBuilder, MemoryType,
        sem_mem::{ConceptType, SemMemory},
    };
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_runtime::working_memory::WorkingMemory;

    fn create_mock_working_memory_with_links() -> (WorkingMemory, Vec<MemoryId>) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let ids: Vec<_> = (0..3).map(|_| MemoryId::new()).collect();

        let sem_link = SemMemLink::new("related".to_string(), 0.8);
        let link_type = MemoryLinkType::Sem(sem_link);
        let link1 = MemoryLink::new(ids[0], ids[1], link_type.clone());
        let link2 = MemoryLink::new(ids[1], ids[2], link_type);

        cluster.write(|c| {
            let note0 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "Memory 0".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(ids[0])
            .mem_links(vec![link1])
            .build()
            .unwrap();
            let embedding0 = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: note0,
                embedding: embedding0,
            });

            let note1 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "Memory 1".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(ids[1])
            .mem_links(vec![link2])
            .build()
            .unwrap();
            let embedding1 = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: note1,
                embedding: embedding1,
            });

            let note2 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "Memory 2".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(ids[2])
            .build()
            .unwrap();
            let embedding2 = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: note2,
                embedding: embedding2,
            });
        });

        (wm, ids)
    }

    #[test]
    fn test_retr_association_basic() {
        let (wm, ids) = create_mock_working_memory_with_links();
        let config = AssociationConfig::default();
        let request = config.into_request(Arc::new(wm), vec![(ids[0], 1.0)]);
        let result = RetrAssociation {}.retrieve(request);

        assert!(!result.is_empty(), "Result is: {:?}", result);
    }

    #[test]
    fn test_retr_association_with_empty_source() {
        let (wm, _ids) = create_mock_working_memory_with_links();
        let config = AssociationConfig::default();
        let request = config.into_request(Arc::new(wm), vec![]);
        let result = RetrAssociation {}.retrieve(request);

        assert!(result.is_empty());
    }

    #[test]
    fn test_retr_association_top_k() {
        let (wm, ids) = create_mock_working_memory_with_links();
        let config = AssociationConfig {
            top_k: 1,
            ..Default::default()
        };
        let request = config.into_request(Arc::new(wm), vec![(ids[0], 1.0)]);
        let result = RetrAssociation {}.retrieve(request);

        assert!(result.len() <= 1);
    }

    #[test]
    fn test_dyn_weight_func_builder() {
        let builder = DynWeightFuncBuilder::new(TypePreference::default());
        let weight_func = builder.intensity_factor(0.5).confidence_factor(0.3).build();
        let _weight = weight_func;
    }
}
