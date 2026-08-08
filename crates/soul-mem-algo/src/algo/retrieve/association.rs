use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_runtime::cluster::memory_cluster::GraphMemoryLink;

use petgraph::{algo::UnitMeasure, stable_graph::NodeIndex, visit::EdgeRef};
use serde::Deserialize;
use std::collections::HashMap;
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
            damping_factor: 0.65,
            residue_threshold: 1e-5,
            preference: TypePreference::Situation,
            top_k: 8,
        }
    }
}

fn default_damping_factor() -> f64 {
    0.65
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
            damping_factor: 0.65,
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

        let personalized_vec: HashMap<NodeIndex, OrdFloat<f64>> =
            mem_cluster.read_or_compute(|cluster| {
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

        //source非空但所有id都无法在cluster中解析时（陈旧或调用方提供的非法id），
        //personalized_vec为空，直接返回空结果，避免weighted_ppr_fp内部的assert panic。
        if personalized_vec.is_empty() {
            return Vec::new();
        }

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
            //normalize_factor为0时（如intensity_factor=0且confidence_factor=0且type_boost=0），
            //0/0会产生NaN污染PPR边权，这里退化为0权重
            if normalize_factor == 0.0 {
                return OrdFloat::from_f64(0.0);
            }
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
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_links::MemoryLink;
    use soul_mem_core::memory_note::{
        sem_mem::{ConceptType, SemMemory},
        MemoryNoteBuilder, MemoryType,
    };
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::EmbeddingVec;
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
        println!("wm: {:?}, ids: {:?}", wm, ids);
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
    fn test_dyn_weight_func_semantic_edge() {
        use soul_mem_core::memory_links::sem_mem::SemMemLink;

        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let id_a = MemoryId::new();
        let id_b = MemoryId::new();
        let link = MemoryLink::new(
            id_a,
            id_b,
            MemoryLinkType::Sem(SemMemLink::new("knows".into(), 0.7)),
        );

        cluster.write(|c| {
            let na = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "a".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(id_a)
            .mem_links(vec![link])
            .build()
            .unwrap();
            let ea = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: na,
                embedding: ea,
            });

            let nb = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "b".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(id_b)
            .build()
            .unwrap();
            let eb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: nb,
                embedding: eb,
            });
        });

        let weight = cluster.read_or_compute(|mem_cluster| {
            let graph = mem_cluster.graph();
            let edge = graph
                .edges(mem_cluster.get_mem_index(id_a).unwrap())
                .find(|e| e.target() == mem_cluster.get_mem_index(id_b).unwrap())
                .unwrap();

            let weight_fn = DynWeightFuncBuilder::new(TypePreference::Semantic)
                .intensity_factor(1.0)
                .confidence_factor(1.0)
                .build();

            let w: f64 = weight_fn(graph, &edge, None).into_inner();
            w
        });

        insta::assert_debug_snapshot!("dyn_weight_semantic_edge", weight);
    }

    fn create_graph_with_type_preference() -> (WorkingMemory, MemoryId) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let src_id = MemoryId::new();
        let id_b = MemoryId::new();
        let id_c = MemoryId::new();

        use soul_mem_core::memory_links::situation_mem::{AbstractToSpecific, SituationMemLink};
        let sem_link = MemoryLinkType::Sem(SemMemLink::new("related".into(), 0.8));
        let sit_link = MemoryLinkType::Situation(SituationMemLink::AbstractToSpecific(
            AbstractToSpecific::new(),
        ));
        let link1 = MemoryLink::new(src_id, id_b, sem_link);
        let link2 = MemoryLink::new(src_id, id_c, sit_link);

        cluster.write(|c| {
            for (i, (nid, links)) in [(src_id, vec![link1, link2]), (id_b, vec![]), (id_c, vec![])]
                .into_iter()
                .enumerate()
            {
                let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                    content: format!("Mem {i}"),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                }))
                .id(nid)
                .mem_links(links)
                .build()
                .unwrap();
                let embedding = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                    )),
                );
                c.add_single_node(EmbeddedMemoryNote { note, embedding });
            }
        });
        (wm, src_id)
    }

    #[test]
    fn test_type_preference_affects_ranking() {
        let (wm_sem, src_a) = create_graph_with_type_preference();
        let (wm_sit, src_b) = create_graph_with_type_preference();

        let sem_config = AssociationConfig {
            preference: TypePreference::Semantic,
            top_k: 5,
            ..Default::default()
        };
        let sem_req = sem_config.into_request(Arc::new(wm_sem), vec![(src_a, 1.0)]);
        let sem_result = RetrAssociation {}.retrieve(sem_req);

        let sit_config = AssociationConfig {
            preference: TypePreference::Situation,
            top_k: 5,
            ..Default::default()
        };
        let sit_req = sit_config.into_request(Arc::new(wm_sit), vec![(src_b, 1.0)]);
        let sit_result = RetrAssociation {}.retrieve(sit_req);

        assert!(!sem_result.is_empty());
        assert!(!sit_result.is_empty());

        let sem_scores: Vec<f64> = sem_result.iter().map(|(_, s)| *s).collect();
        let sit_scores: Vec<f64> = sit_result.iter().map(|(_, s)| *s).collect();
        insta::assert_debug_snapshot!("type_preference_semantic", sem_scores);
        insta::assert_debug_snapshot!("type_preference_situation", sit_scores);
    }

    #[test]
    fn test_multi_source_ppr() {
        let (wm, ids) = create_mock_working_memory_with_links();
        let sources = vec![(ids[0], 0.6), (ids[1], 0.4)];
        let config = AssociationConfig::default();
        let request = config.into_request(Arc::new(wm), sources);
        let result = RetrAssociation {}.retrieve(request);

        assert!(!result.is_empty());
        let scores: Vec<f64> = result.iter().map(|(_, s)| *s).collect();
        insta::assert_debug_snapshot!("multi_source_ppr", scores);
    }

    #[test]
    fn test_procedure_node_exclusion_in_ppr() {
        use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
        use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};

        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let sem_id = MemoryId::new();
        let proc_id = MemoryId::new();

        let proc_link = MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.5)));
        let link = MemoryLink::new(sem_id, proc_id, proc_link);

        cluster.write(|c| {
            let sem_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "sem".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(sem_id)
            .mem_links(vec![link])
            .build()
            .unwrap();
            let sem_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: sem_note,
                embedding: sem_emb,
            });

            let proc_note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                Action::new("act".into(), ActionType::new_speak()),
            )))
            .id(proc_id)
            .build()
            .unwrap();
            let proc_emb =
                MemoryEmbedding::new(EmbeddingVec::zero(128), MemoryEmbeddingVariant::Procedure());
            c.add_single_node(EmbeddedMemoryNote {
                note: proc_note,
                embedding: proc_emb,
            });
        });

        let config = AssociationConfig::default();
        let request = config.into_request(Arc::new(wm), vec![(sem_id, 1.0)]);
        let result = RetrAssociation {}.retrieve(request);

        let result_ids: Vec<MemoryId> = result
            .iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(id, _)| *id)
            .collect();
        assert!(
            !result_ids.contains(&proc_id),
            "Procedure nodes should have zero PPR score, but got nonzero in: {:?}",
            result
        );
    }

    #[test]
    fn test_ppr_forward_push_vs_naive_consistency() {
        use crate::common::ppr::{naive_ppr, weighted_ppr_fp};
        use petgraph::stable_graph::StableDiGraph;

        let mut g: StableDiGraph<(), ()> = StableDiGraph::new();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        let n3 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n0, n2, ());
        g.add_edge(n1, n2, ());
        g.add_edge(n2, n3, ());

        let mut source: std::collections::HashMap<_, OrdFloat<f64>> =
            std::collections::HashMap::new();
        source.insert(n0, OrdFloat::from_f64(1.0));

        let naive = naive_ppr(&g, OrdFloat::from_f64(0.15), source.clone(), 20);
        let no_query: Option<&()> = None;
        let fp = weighted_ppr_fp(
            &g,
            OrdFloat::from_f64(0.15),
            source,
            OrdFloat::from_f64(1e-6),
            |_, _, _| OrdFloat::from_f64(1.0),
            no_query,
        );

        let mut naive_sorted: Vec<_> = naive.into_iter().collect();
        naive_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut fp_sorted: Vec<_> = fp.into_iter().collect();
        fp_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let _ = (naive_sorted.len(), fp_sorted.len());

        assert_eq!(naive_sorted.len(), fp_sorted.len());
        for ((_, ns), (_, fs)) in naive_sorted.iter().zip(fp_sorted.iter()) {
            let diff = f64::abs(ns.into_inner() - fs.into_inner());
            assert!(
                diff < 0.05,
                "PPR difference too large: {diff} (naive={}, fp={})",
                ns.into_inner(),
                fs.into_inner()
            );
        }
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(default_damping_factor(), 0.65);
        assert_eq!(default_residue_threshold(), 1e-5);
        assert_eq!(default_top_k(), 8);
    }

    #[test]
    fn test_association_config_defaults() {
        let config = AssociationConfig::default();
        assert_eq!(config.damping_factor, 0.65);
        assert_eq!(config.residue_threshold, 1e-5);
        assert_eq!(config.top_k, 8);
        assert!(matches!(config.preference, TypePreference::Situation));
        assert!(config.intensity_factor.is_none());
        assert!(config.confidence_factor.is_none());
    }

    #[test]
    fn test_dyn_weight_func_zero_factors() {
        // 因子为 0 时权重仍应有限（不产生 NaN）
        let (wm, src_id) = create_graph_with_type_preference();
        wm.memory_cluster().read_or_compute(|mem_cluster| {
            let graph = mem_cluster.graph();
            let edge = graph
                .edges(mem_cluster.get_mem_index(src_id).unwrap())
                .next()
                .unwrap();

            let weight_fn = DynWeightFuncBuilder::new(TypePreference::Semantic)
                .intensity_factor(0.0)
                .confidence_factor(0.0)
                .build();
            let w: f64 = weight_fn(graph, &edge, None).into_inner();
            assert!(w.is_finite(), "weight must be finite, got {w}");
            assert!(w >= 0.0);
        });
    }

    #[test]
    fn test_dyn_weight_func_defaults() {
        // 默认 intensity_factor=1.0, confidence_factor=0.8
        let (wm, src_id) = create_graph_with_type_preference();
        wm.memory_cluster().read_or_compute(|mem_cluster| {
            let graph = mem_cluster.graph();
            let edge = graph
                .edges(mem_cluster.get_mem_index(src_id).unwrap())
                .next()
                .unwrap();

            let weight_fn = DynWeightFuncBuilder::new(TypePreference::Situation).build();
            let w: f64 = weight_fn(graph, &edge, None).into_inner();
            assert!(w.is_finite() && w > 0.0, "weight should be positive, got {w}");
        });
    }
}
