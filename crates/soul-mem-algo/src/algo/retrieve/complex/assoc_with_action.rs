use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;

use crate::algo::retrieve::{
    RetrRequest, RetrStrategy,
    association::{AssociationConfig, AssociationRequest, RetrAssociation},
    bayes_action::{ActionTypeSlots, BayesActionRequest, RetrBayesAction},
};
use soul_mem_core::memory_note::situation_mem::SituationType;
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_runtime::working_memory::WorkingMemory;

#[derive(Debug, Clone, Deserialize)]
pub struct AssociateWithActionConfig {
    #[serde(default)]
    pub association: AssociationConfig,
    /// 抽象情境源在 Bayes 动作提取中的权重倍率（抽象优先；具体源权重为 1.0）。
    #[serde(default = "default_abstract_source_priority")]
    pub abstract_source_priority: f64,
}

fn default_abstract_source_priority() -> f64 {
    2.0
}

impl Default for AssociateWithActionConfig {
    fn default() -> Self {
        Self {
            association: AssociationConfig::default(),
            abstract_source_priority: default_abstract_source_priority(),
        }
    }
}

impl AssociateWithActionConfig {
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        source: Vec<(MemoryId, f32)>,
    ) -> AssociateWithActionRequest {
        AssociateWithActionRequest {
            association: self
                .association
                .into_request(Arc::clone(&working_mem), source),
            abstract_source_priority: self.abstract_source_priority,
        }
    }
}

pub struct RetrAssociateWithAction;

pub struct AssociateWithActionRequest {
    association: AssociationRequest,
    abstract_source_priority: f64,
}

impl AssociateWithActionRequest {
    pub fn new(association: AssociationRequest) -> Self {
        Self {
            association,
            abstract_source_priority: default_abstract_source_priority(),
        }
    }
    pub fn with_abstract_source_priority(mut self, priority: f64) -> Self {
        self.abstract_source_priority = priority;
        self
    }
}

impl RetrRequest for AssociateWithActionRequest {}

pub struct AssociateWithActionResult {
    pub memory: Vec<(MemoryId, f64)>,
    /// 动作推理结果：按 `ActionType` 每类至多一个（见 [`ActionTypeSlots`]）。
    pub slots: ActionTypeSlots,
}

impl RetrStrategy for RetrAssociateWithAction {
    type Request = AssociateWithActionRequest;
    type Return<'a> = AssociateWithActionResult;

    #[hotpath::measure]
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let working_mem = Arc::clone(&request.association.working_mem);
        // 相似度种子（直接命中）与 PPR 关联结果的并集共同作为 Bayes 源候选，
        // 避免"具体情境直接命中但 PPR 未扩散"时丢失具体兜底。
        let sim_sources = request.association.source.clone();
        let association_res = RetrAssociation {}.retrieve(request.association);
        if association_res.is_empty() {
            return AssociateWithActionResult {
                memory: Vec::new(),
                slots: ActionTypeSlots::default(),
            };
        }

        // 只保留 Situation 节点（AbstractSituation + SpecificSituation）作为 Bayes 源：
        // Semantic 节点不触发行为（beta_ver：语义→proc 只作自我认知补充）。
        let merged = merge_situation_sources(&working_mem, &sim_sources, &association_res);
        // 抽象优先：抽象源权重 × abstract_source_priority；具体源权重 1.0。
        // 抽象源为空时退化为仅具体源（模式尚未巩固泛化时的兜底）。
        let mut bayes_sources: Vec<(MemoryId, f64)> = Vec::new();
        for (id, score, is_abstract) in merged {
            let weight = if is_abstract {
                score * request.abstract_source_priority
            } else {
                score
            };
            bayes_sources.push((id, weight));
        }
        if bayes_sources.is_empty() {
            return AssociateWithActionResult {
                memory: association_res,
                slots: ActionTypeSlots::default(),
            };
        }

        let normalized_bayes_sources = softmax(&bayes_sources);

        let action_request = BayesActionRequest::new(working_mem, normalized_bayes_sources);
        let slots = RetrBayesAction {}.retrieve(action_request);

        AssociateWithActionResult {
            memory: association_res,
            slots,
        }
    }
}

/// 合并相似度种子与 PPR 关联结果（同 id 取 max），只返回 Situation 节点，
/// 附带是否为抽象情境的标记（`true` = AbstractSituation，`false` = SpecificSituation）。
#[hotpath::measure]
fn merge_situation_sources(
    working_mem: &Arc<WorkingMemory>,
    sim_sources: &[(MemoryId, f32)],
    assoc: &[(MemoryId, f64)],
) -> Vec<(MemoryId, f64, bool)> {
    let type_map: HashMap<MemoryId, bool> = working_mem.memory_cluster().read_or_compute(|c| {
        c.graph()
            .node_weights()
            .filter_map(|n| match n.note().mem_type() {
                MemoryType::Situation(SituationType::AbstractSituation(_)) => {
                    Some((n.note().id(), true))
                }
                MemoryType::Situation(SituationType::SpecificSituation(_)) => {
                    Some((n.note().id(), false))
                }
                _ => None,
            })
            .collect()
    });

    let mut score_map: HashMap<MemoryId, (f64, bool)> = HashMap::new();
    let combined = sim_sources
        .iter()
        .map(|&(id, s)| (id, s as f64))
        .chain(assoc.iter().map(|&(id, s)| (id, s)));
    for (id, score) in combined {
        // PPR 会返回全图节点（未激活的节点分为 0），零分节点不是有效 Bayes 源：
        // 直接过滤，避免 softmax 给零分源非零权重后触发无关动作。
        if score <= 0.0 {
            continue;
        }
        if let Some(&is_abstract) = type_map.get(&id) {
            score_map
                .entry(id)
                .and_modify(|(best, _)| *best = best.max(score))
                .or_insert((score, is_abstract));
        }
    }
    score_map
        .into_iter()
        .map(|(id, (score, is_abstract))| (id, score, is_abstract))
        .collect()
}

fn softmax(logits: &[(MemoryId, f64)]) -> Vec<(MemoryId, f64)> {
    let max_x = logits
        .iter()
        .max_by(|&(_, v1), &(_, v2)| v1.partial_cmp(v2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(_, v)| v)
        .unwrap_or(0.0);

    let sum = logits
        .iter()
        .map(|&(_, x)| f64::exp(x - max_x))
        .sum::<f64>();

    logits
        .iter()
        .map(|&(id, x)| (id, f64::exp(x - max_x) / sum))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::MemoryLink;
    use soul_mem_core::memory_links::MemoryLinkType;
    use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
    use soul_mem_core::memory_links::sem_mem::SemMemLink;
    use soul_mem_core::memory_note::{
        MemoryNoteBuilder, MemoryType,
        proc_mem::{Action, ActionType, ProcMemory},
        sem_mem::{ConceptType, SemMemory},
    };
    use soul_mem_query::embedding::note::EmbeddedMemoryNote;
    use soul_mem_query::embedding::note::MemoryEmbedding;
    use soul_mem_query::embedding::note::MemoryEmbeddingVariant;

    /// 从席位结果里按 `MemoryId` 反查得分（跨三个席位，与类型无关）。
    fn seat_score(slots: &ActionTypeSlots, id: MemoryId) -> Option<f64> {
        slots
            .seated()
            .into_iter()
            .find(|(_, seat)| seat.id == id)
            .map(|(_, seat)| seat.score)
    }

    #[test]
    fn test_associate_with_action_config_defaults() {
        let config = AssociateWithActionConfig::default();
        assert_eq!(
            config.abstract_source_priority,
            default_abstract_source_priority()
        );
    }
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::sem::SemanticEmbedding;

    fn create_mock_working_memory_with_assoc_and_action()
    -> (WorkingMemory, MemoryId, MemoryId, MemoryId) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let action_id = MemoryId::new();

        let sem_link = SemMemLink::new("related".to_string(), 0.8);
        let link_type = MemoryLinkType::Sem(sem_link);
        let link1 = MemoryLink::new(id1, id2, link_type);

        let proc_link = ProcMemLink::TrigToAction(TrigToAction::new(0.5));
        let link_type2 = MemoryLinkType::Proc(proc_link);
        let link2 = MemoryLink::new(id2, action_id, link_type2);

        cluster.write(|c| {
            let note1 = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "Memory 1".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(id1)
            .mem_links(vec![link1])
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
            .id(id2)
            .mem_links(vec![link2])
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

            let action_mem_type = MemoryType::Procedure(ProcMemory::new(Action::new(
                "TestAction".to_string(),
                ActionType::new_speak(),
            )));
            let action_note = MemoryNoteBuilder::new(action_mem_type)
                .id(action_id)
                .build()
                .unwrap();
            let action_embedding =
                MemoryEmbedding::new(EmbeddingVec::zero(128), MemoryEmbeddingVariant::Procedure());
            c.add_single_node(EmbeddedMemoryNote {
                note: action_note,
                embedding: action_embedding,
            });
        });

        (wm, id1, id2, action_id)
    }

    #[test]
    fn test_retr_associate_with_action_basic() {
        let (wm, source_id, _, _) = create_mock_working_memory_with_assoc_and_action();
        let config = AssociateWithActionConfig::default();
        let request = config.into_request(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert!(!result.memory.is_empty() || !result.slots.is_empty());
    }

    #[test]
    fn test_retr_associate_with_action_empty_source() {
        let (wm, _, _, _) = create_mock_working_memory_with_assoc_and_action();
        let config = AssociateWithActionConfig::default();
        let request = config.into_request(Arc::new(wm), vec![]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert!(result.memory.is_empty());
        assert!(result.slots.is_empty());
    }

    #[test]
    fn test_retr_associate_with_action_slots_are_one_per_type() {
        // 一个具体情境源挂两条不同 ActionType 的 Proc 边：
        // 每个类型各占一席（Speak + Think），Skill 不产出，总席位 2。
        use chrono::{TimeZone, Utc};
        use soul_mem_core::memory_note::situation_mem::{Context, Environment, SpecificSituation};
        use soul_mem_query::embedding::Embeddable;
        use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;

        let model = BgeSmallZh::default_cpu().unwrap();
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let speak_id = MemoryId::new();
        let think_id = MemoryId::new();

        let spec_mem = SpecificSituation::new(
            "和对方在深夜聊了一整晚，分享了很多心里话".to_string(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Context::new(
                None,
                vec![],
                vec![],
                vec![],
                Environment {
                    atmosphere: "安静".to_string(),
                    tone: "温暖".to_string(),
                },
                vec![],
            ),
        );
        let source_note = MemoryNoteBuilder::new(MemoryType::Situation(
            SituationType::SpecificSituation(spec_mem),
        ))
        .id(source_id)
        .mem_links(vec![
            MemoryLink::new(
                source_id,
                speak_id,
                MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.9))),
            ),
            MemoryLink::new(
                source_id,
                think_id,
                MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.9))),
            ),
        ])
        .build()
        .unwrap();
        let source_emb = source_note.embed(&model).unwrap();

        cluster.write(|c| {
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_emb,
            });

            for (aid, action_type, name) in [
                (speak_id, ActionType::new_speak(), "SpeakProc"),
                (think_id, ActionType::new_think(), "ThinkProc"),
            ] {
                let note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new(name.into(), action_type),
                )))
                .id(aid)
                .build()
                .unwrap();
                let emb = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Procedure(),
                );
                c.add_single_node(EmbeddedMemoryNote {
                    note,
                    embedding: emb,
                });
            }
        });

        let config = AssociateWithActionConfig::default();
        let request = config.into_request(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert_eq!(
            result.slots.len(),
            2,
            "Speak 与 Think 各占一席，Skill 不产出"
        );
        assert_eq!(result.slots.speak.as_ref().map(|s| s.id), Some(speak_id));
        assert_eq!(result.slots.think.as_ref().map(|s| s.id), Some(think_id));
        assert!(result.slots.skill.is_none());
    }

    /// 构造含"具体情境→proc_specific"与"抽象情境→proc_abstract"两条 Proc 边的图。
    fn create_dual_source_wm() -> (WorkingMemory, MemoryId, MemoryId, MemoryId, MemoryId) {
        use chrono::{TimeZone, Utc};
        use soul_mem_core::memory_note::situation_mem::{
            AbstractSituation, Context, Environment, Event, SituationType, SpecificSituation,
        };
        use soul_mem_query::embedding::Embeddable;
        use soul_mem_query::embedding::EmbeddingVec;
        use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;

        let model = BgeSmallZh::default_cpu().unwrap();
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let spec_id = MemoryId::new();
        let abs_id = MemoryId::new();
        let proc_spec = MemoryId::new();
        let proc_abs = MemoryId::new();

        let spec_mem = SpecificSituation::new(
            "和对方在深夜聊了一整晚，分享了很多心里话".to_string(),
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Context::new(
                None,
                vec![],
                vec![],
                vec![],
                Environment {
                    atmosphere: "安静".to_string(),
                    tone: "温暖".to_string(),
                },
                vec![],
            ),
        );
        let spec_note = MemoryNoteBuilder::new(MemoryType::Situation(
            SituationType::SpecificSituation(spec_mem),
        ))
        .id(spec_id)
        .mem_links(vec![MemoryLink::new(
            spec_id,
            proc_spec,
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(1.0))),
        )])
        .build()
        .unwrap();
        let spec_emb = spec_note.embed(&model).unwrap();

        let abs_mem = AbstractSituation::Event(Event {
            action: "与人深夜谈心".to_string(),
            action_intensity: 0.5,
            initiator: "对方".to_string(),
            target: "我".to_string(),
        });
        let abs_note = MemoryNoteBuilder::new(MemoryType::Situation(
            SituationType::AbstractSituation(abs_mem),
        ))
        .id(abs_id)
        .mem_links(vec![MemoryLink::new(
            abs_id,
            proc_abs,
            MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(1.0))),
        )])
        .build()
        .unwrap();
        let abs_emb = abs_note.embed(&model).unwrap();

        cluster.write(|c| {
            c.add_single_node(EmbeddedMemoryNote {
                note: spec_note,
                embedding: spec_emb,
            });
            c.add_single_node(EmbeddedMemoryNote {
                note: abs_note,
                embedding: abs_emb,
            });
            // 两个 proc 刻意取不同类型（具体→Speak，抽象→Think）：
            // 每类型一席的贪心截断下，两者各自占席，分数才可直接比较。
            for (pid, name, action_type) in [
                (proc_spec, "proc_spec", ActionType::new_speak()),
                (proc_abs, "proc_abs", ActionType::new_think()),
            ] {
                let pnote = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new(name.to_string(), action_type),
                )))
                .id(pid)
                .build()
                .unwrap();
                let pemb = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Procedure(),
                );
                c.add_single_node(EmbeddedMemoryNote {
                    note: pnote,
                    embedding: pemb,
                });
            }
        });

        (wm, spec_id, abs_id, proc_spec, proc_abs)
    }

    #[test]
    fn test_dual_source_bayes_abstract_priority() {
        // 抽象源与具体源同权时，抽象源 ×2 优先 → 抽象触发的动作分应更高。
        let (wm, spec_id, abs_id, proc_spec, proc_abs) = create_dual_source_wm();
        let config = AssociateWithActionConfig::default();
        let request = config.into_request(Arc::new(wm), vec![(abs_id, 1.0), (spec_id, 1.0)]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        let abs_score = seat_score(&result.slots, proc_abs).expect("抽象情境触发的动作应被检出");
        let spec_score =
            seat_score(&result.slots, proc_spec).expect("具体情境触发的动作应参与（兜底源）");
        assert!(
            abs_score > spec_score,
            "抽象优先：抽象源加权后动作分应更高, abs={abs_score} spec={spec_score}"
        );
    }

    #[test]
    fn test_dual_source_bayes_specific_fallback_when_no_abstract() {
        // 抽象源为空（模式尚未巩固泛化）时，退化为仅具体源。
        let (wm, spec_id, _abs_id, proc_spec, proc_abs) = create_dual_source_wm();
        let config = AssociateWithActionConfig::default();
        let request = config.into_request(Arc::new(wm), vec![(spec_id, 1.0)]);
        let result = RetrAssociateWithAction {}.retrieve(request);

        assert!(
            result.slots.seated().iter().any(|(_, s)| s.id == proc_spec),
            "具体兜底应检出具体情境触发的动作"
        );
        assert!(
            !result.slots.seated().iter().any(|(_, s)| s.id == proc_abs),
            "无抽象源时不应检出抽象触发的动作, slots={:?}",
            result.slots
        );
    }

    #[test]
    fn test_dual_source_bayes_priority_one_is_neutral() {
        // priority=1.0 时抽象与具体同权 → 同权源触发分近似相等。
        let (wm, spec_id, abs_id, proc_spec, proc_abs) = create_dual_source_wm();
        let config = AssociateWithActionConfig::default();
        let request = config
            .into_request(Arc::new(wm), vec![(abs_id, 1.0), (spec_id, 1.0)])
            .with_abstract_source_priority(1.0);
        let result = RetrAssociateWithAction {}.retrieve(request);

        let abs_score = seat_score(&result.slots, proc_abs).unwrap_or(0.0);
        let spec_score = seat_score(&result.slots, proc_spec).unwrap_or(0.0);
        assert!(
            (abs_score - spec_score).abs() < 1e-6,
            "同权时抽象与具体动作分应相等, abs={abs_score} spec={spec_score}"
        );
    }

    #[test]
    fn test_softmax_function() {
        let input = vec![
            (MemoryId::new(), 1.0),
            (MemoryId::new(), 2.0),
            (MemoryId::new(), 3.0),
        ];
        let result = softmax(&input);
        let sum: f64 = result.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|(_, p)| *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn test_softmax_empty() {
        let input: Vec<(MemoryId, f64)> = vec![];
        let result = softmax(&input);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_extreme_values() {
        let huge = vec![(MemoryId::new(), 1e10), (MemoryId::new(), 1e10 + 1.0)];
        let result = softmax(&huge);
        let sum: f64 = result.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|(_, p)| *p >= 0.0 && *p <= 1.0));

        let tiny = vec![(MemoryId::new(), -1e10), (MemoryId::new(), -1e10)];
        let result2 = softmax(&tiny);
        let sum2: f64 = result2.iter().map(|(_, p)| p).sum();
        assert!((sum2 - 1.0).abs() < 1e-5);

        let uniform = vec![
            (MemoryId::new(), 5.0),
            (MemoryId::new(), 5.0),
            (MemoryId::new(), 5.0),
        ];
        let result3 = softmax(&uniform);
        let sum3: f64 = result3.iter().map(|(_, p)| p).sum();
        assert!((sum3 - 1.0).abs() < 1e-5);
        assert!(result3.iter().all(|(_, p)| (*p - 1.0 / 3.0).abs() < 1e-5));
    }
}
