use std::{collections::HashMap, sync::Arc};

use petgraph::{Direction::Outgoing, visit::EdgeRef};

use serde::Deserialize;

use crate::algo::retrieve::{RetrRequest, RetrStrategy};

use soul_mem_core::memory_links::{
    MemoryLinkType,
    proc_mem::{ProcMemLink, TrigToAction},
};
use soul_mem_core::memory_note::proc_mem::ActionType;
use soul_mem_core::memory_note::{MemoryId, MemoryType};
use soul_mem_runtime::cluster::memory_cluster::MemoryCluster;
use soul_mem_runtime::working_memory::WorkingMemory;

/// 动作席位：某个 `ActionType` 上被判出的**唯一**动作。
///
/// `score` 是贝叶斯推理的累积触发概率（`Σ prob × 源权重`），单位与
/// `TrigToAction.prob` 一致，量纲为"权重归一化后的相对概率"，非概率分布，
/// 上界不固定（源权重若经 softmax 归一化则 ≤ 各源 prob 的加权均值）。
#[derive(Debug, Clone, PartialEq)]
pub struct ActionSeat {
    pub id: MemoryId,
    pub score: f64,
}

/// 动作推理结果：按 [`ActionType`] 分成三个固定席位，每席**至多一个**动作。
///
/// 席位与类型一一对应，调用方无需再从记忆图反查节点类型；没有候选的类型
/// 对应 `None`（席位可空），因此刻意不提供 `ActionTypeSlots::new()`。
///
/// 已知性质：席位内的胜出者由贪心选择决定——得分只由
/// `TrigToAction.prob × 源权重` 决定，**与动作内容无关**；同类型内多个候选
/// 若共享同一批源边，分数可以完全相同，此时仅由 `MemoryId` 的 tiebreak 决定。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ActionTypeSlots {
    /// `ActionType::Speak`：语气 / 说话方式。
    pub speak: Option<ActionSeat>,
    /// `ActionType::Think`：思维习惯。
    ///
    /// `proc_none`（内容为"没有采取任何特定行动"）在图里也标记为 `Think`：
    /// 它在**语义上等同 null**，但类型上必须归入本席位，且照常参与贪心竞争——
    /// 它得分最高时本席位就返回 `proc_none` 本身，"是否当作空"由调用方解释。
    pub think: Option<ActionSeat>,
    /// `ActionType::Skill`：技能动作。
    ///
    /// `SkillRecord` 目前是空占位（无技能标识），组内没有可区分个体的信息，
    /// 且当前检索**不产出技能动作**（[`best_per_action_type`] 直接跳过该类型），
    /// 因此本席位恒为 `None`。预留席位是为了让"每类型一席"的结构在类型上完整。
    pub skill: Option<ActionSeat>,
}

impl ActionTypeSlots {
    /// 三个席位是否全空。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 已占用的席位数（0..=3）。
    pub fn len(&self) -> usize {
        [
            self.speak.is_some(),
            self.think.is_some(),
            self.skill.is_some(),
        ]
        .into_iter()
        .filter(|occupied| *occupied)
        .count()
    }

    /// 按类型取席位。
    pub fn get(&self, action_type: &ActionType) -> Option<&ActionSeat> {
        match action_type {
            ActionType::Speak => self.speak.as_ref(),
            ActionType::Think => self.think.as_ref(),
            ActionType::Skill(_) => self.skill.as_ref(),
        }
    }

    /// 按类型写入席位。
    pub fn set(&mut self, action_type: &ActionType, seat: Option<ActionSeat>) {
        match action_type {
            ActionType::Speak => self.speak = seat,
            ActionType::Think => self.think = seat,
            ActionType::Skill(_) => self.skill = seat,
        }
    }

    /// 已占用的席位，附带类型标注；空席位被丢弃，结果按分数降序
    /// （同分时按类型枚举顺序，保证输出确定）。
    pub fn seated(&self) -> Vec<(ActionType, ActionSeat)> {
        let mut seats: Vec<(ActionType, ActionSeat)> = ActionType::all()
            .into_iter()
            .filter_map(|action_type| {
                self.get(&action_type)
                    .cloned()
                    .map(|seat| (action_type, seat))
            })
            .collect();
        seats.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));
        seats
    }

    /// 三个席位本身（顺序：Speak / Think / Skill），无论是否为空。
    pub fn into_seats(self) -> [Option<ActionSeat>; 3] {
        [self.speak, self.think, self.skill]
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BayesActionConfig {}

impl BayesActionConfig {
    /// 构造请求。每类型固定一席（贪心取最高分），无可配置项：
    /// 席位数由 [`ActionTypeSlots`] 的类型结构决定，不再是可调 top-k。
    pub fn into_request(
        self,
        working_mem: Arc<WorkingMemory>,
        source: Vec<(MemoryId, f64)>,
    ) -> BayesActionRequest {
        BayesActionRequest {
            working_mem,
            source,
        }
    }
}

pub struct BayesActionRequest {
    pub working_mem: Arc<WorkingMemory>,
    pub source: Vec<(MemoryId, f64)>,
}

impl BayesActionRequest {
    pub fn new(working_mem: Arc<WorkingMemory>, source: Vec<(MemoryId, f64)>) -> Self {
        Self {
            working_mem,
            source,
        }
    }
}

pub struct RetrBayesAction;

impl RetrRequest for BayesActionRequest {}

impl RetrStrategy for RetrBayesAction {
    type Request = BayesActionRequest;
    type Return<'a> = ActionTypeSlots;

    #[hotpath::measure]
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let cluster = request.working_mem.memory_cluster();

        cluster.read_or_compute(|mem_cluster| {
            let mut possible_actions = get_possible_actions(mem_cluster, &request.source);

            // 概率累积与源顺序无关（+ 满足交换律），因此"贪心取每组最高分"的结果
            // 不受 `source` 排列影响；源内部则靠 `seated()` 的排序保证输出确定。
            request.source.iter().for_each(|&(id, weight)| {
                let idx = mem_cluster.get_mem_index(id);

                if let Some(idx) = idx {
                    let links = mem_cluster.graph().edges_directed(idx, Outgoing);

                    for link in links {
                        let neighbor_idx = link.target();

                        if let Some(embed_note) = mem_cluster.graph().node_weight(neighbor_idx) {
                            let note_id = embed_note.note().id();
                            let link_type = link.weight().link_type();

                            if let MemoryType::Procedure(_) = embed_note.note().mem_type()
                                && let MemoryLinkType::Proc(link_weight) = link_type
                            {
                                match link_weight {
                                    ProcMemLink::TrigToAction(TrigToAction { prob, .. }) => {
                                        if let Some(v) = possible_actions.get_mut(&note_id) {
                                            *v += prob * weight;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });

            best_per_action_type(mem_cluster, &possible_actions)
        })
    }
}

/// 按 `ActionType` 分席的**贪心**选择：每个类型只取分数最高的那一个候选。
///
/// - 分组键是节点的 `ActionType`（`Speak` / `Think` / `Skill`），结果至多 3 条；
/// - 空组产生空席位（`None`），不补齐、不填充；
/// - 同分时取 `MemoryId` 较小者（`MemoryId` 派生 `Ord`），保证同一张图上
///   结果逐位可复现——`HashMap` 的迭代顺序不参与决策；
/// - `ActionType::Skill` 当前直接跳过：`SkillRecord` 是空占位，无法在组内
///   区分个体，取谁都只是 tiebreak 的产物，因此不产出技能动作。
#[hotpath::measure]
fn best_per_action_type(
    cluster: &MemoryCluster,
    scores: &HashMap<MemoryId, f64>,
) -> ActionTypeSlots {
    let mut best: HashMap<ActionType, ActionSeat> = HashMap::new();

    for (id, score) in scores {
        let Some(idx) = cluster.get_mem_index(*id) else {
            continue;
        };
        let Some(note) = cluster.graph().node_weight(idx) else {
            continue;
        };
        let MemoryType::Procedure(proc_mem) = note.note().mem_type() else {
            continue;
        };

        let action_type = proc_mem.get_action().get_action_type();
        // 技能动作无个体标识，胜出者没有判别依据 → 本次不产出
        if matches!(action_type, ActionType::Skill(_)) {
            continue;
        }

        let better = match best.get(action_type) {
            None => true,
            Some(current) => {
                *score > current.score || (*score == current.score && *id < current.id)
            }
        };
        if better {
            best.insert(
                action_type.clone(),
                ActionSeat {
                    id: *id,
                    score: *score,
                },
            );
        }
    }

    let mut slots = ActionTypeSlots::default();
    for (action_type, seat) in best {
        slots.set(&action_type, Some(seat));
    }
    slots
}

#[hotpath::measure]
fn get_possible_actions(
    cluster: &MemoryCluster,
    source: &[(MemoryId, f64)],
) -> HashMap<MemoryId, f64> {
    source
        .iter()
        .filter_map(|&(id, _weight)| {
            let idx = cluster.get_mem_index(id)?;
            //同时检查邻居节点类型(Procedure)与链接类型(Proc)，
            //只有经Proc链接可达的动作节点才会指导行为，避免Sem/Situation链接
            //可达的Procedure节点以0.0分挤占席位
            let action_neighbors =
                cluster
                    .graph()
                    .edges_directed(idx, Outgoing)
                    .filter_map(|edge| {
                        let node_idx = edge.target();
                        if !matches!(edge.weight().link_type(), MemoryLinkType::Proc(_)) {
                            return None;
                        }
                        let note = cluster.graph().node_weight(node_idx)?;
                        match note.note().mem_type() {
                            MemoryType::Procedure(_) => Some((note.note().id(), 0.0)),
                            _ => None,
                        }
                    });
            Some(action_neighbors)
        })
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::MemoryLink;
    use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
    use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory, SkillRecord};
    use soul_mem_core::memory_note::{
        MemoryNoteBuilder, MemoryType,
        sem_mem::{ConceptType, SemMemory},
    };
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;

    #[test]
    fn test_bayes_action_config_defaults() {
        let _config = BayesActionConfig::default();
    }

    fn create_mock_working_memory_with_actions() -> (WorkingMemory, MemoryId, MemoryId) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let action_id = MemoryId::new();

        let proc_link = ProcMemLink::TrigToAction(TrigToAction::new(0.5));
        let link_type = MemoryLinkType::Proc(proc_link);
        let source_link = MemoryLink::new(source_id, action_id, link_type);

        cluster.write(|c| {
            let source_mem_type = MemoryType::Semantic(SemMemory {
                content: "Source Memory".to_string(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            });
            let source_note = MemoryNoteBuilder::new(source_mem_type)
                .id(source_id)
                .mem_links(vec![source_link])
                .build()
                .unwrap();
            let source_embedding = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_embedding,
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

        (wm, source_id, action_id)
    }

    #[test]
    fn test_retr_bayes_action_basic() {
        let (wm, source_id, action_id) = create_mock_working_memory_with_actions();
        // 权重取 0.8（而非 1.0），使 prob * weight 与 prob / weight 可区分
        let request = BayesActionRequest::new(Arc::new(wm), vec![(source_id, 0.8)]);
        let result = RetrBayesAction {}.retrieve(request);

        assert_eq!(result.len(), 1);
        let seat = result.speak.expect("Speak 席位应被占据");
        assert_eq!(seat.id, action_id);
        assert_eq!(seat.score, 0.4);
        assert_eq!(result.think, None);
        assert_eq!(result.skill, None);
    }

    #[test]
    fn test_retr_bayes_action_empty_source() {
        let (wm, _, _) = create_mock_working_memory_with_actions();
        let request = BayesActionRequest::new(Arc::new(wm), vec![]);
        let result = RetrBayesAction {}.retrieve(request);

        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_get_possible_actions() {
        let (wm, source_id, action_id) = create_mock_working_memory_with_actions();
        let cluster = wm.memory_cluster();
        let result = cluster.read_or_compute(|c| get_possible_actions(c, &[(source_id, 1.0)]));

        assert!(result.contains_key(&action_id));
    }

    /// 同一情境源 → 三个不同类型 `proc` 边，用于验证"每类型一席、互不挤占"。
    fn typed_source_wm() -> (WorkingMemory, MemoryId, Vec<(MemoryId, MemoryType)>) {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let types = vec![
            (
                MemoryId::new(),
                MemoryType::Procedure(ProcMemory::new(Action::new(
                    "Speak".into(),
                    ActionType::new_speak(),
                ))),
            ),
            (
                MemoryId::new(),
                MemoryType::Procedure(ProcMemory::new(Action::new(
                    "Think".into(),
                    ActionType::new_think(),
                ))),
            ),
            (
                MemoryId::new(),
                MemoryType::Procedure(ProcMemory::new(Action::new(
                    "Skill".into(),
                    ActionType::new_skill(SkillRecord {}),
                ))),
            ),
        ];

        cluster.write(|c| {
            let mut links = Vec::new();
            for (aid, _) in &types {
                links.push(MemoryLink::new(
                    source_id,
                    *aid,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(1.0))),
                ));
            }
            let source_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(links)
            .build()
            .unwrap();
            let source_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_emb,
            });

            for (aid, mem_type) in &types {
                let note = MemoryNoteBuilder::new(mem_type.clone())
                    .id(*aid)
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

        (wm, source_id, types)
    }

    #[test]
    fn test_slots_are_one_per_action_type() {
        // 三个类型各一个候选 → 三席齐满，Speak/Think 各有；Skill 恒空
        let (wm, source_id, types) = typed_source_wm();
        let result = RetrBayesAction {}.retrieve(BayesActionRequest::new(
            Arc::new(wm),
            vec![(source_id, 1.0)],
        ));

        assert_eq!(result.len(), 2, "Speak 与 Think 各占一席");
        // 非空这一侧必须被观测到：只断言 `len()` 时，`is_empty()` 恒返回 true 也能存活
        // （`test_retr_bayes_action_empty_source` 只覆盖了空态那一侧）。
        assert!(!result.is_empty(), "两席被占用时 is_empty() 必须为 false");
        assert_eq!(result.speak.as_ref().map(|s| s.id), Some(types[0].0));
        assert_eq!(result.think.as_ref().map(|s| s.id), Some(types[1].0));
        assert!(
            result.skill.is_none(),
            "Skill 动作不产出（SkillRecord 无标识）"
        );
        // 席位带类型标注：seated() 可反查类型
        let seated = result.seated();
        assert_eq!(seated.len(), 2);
        assert!(seated.iter().any(|(t, _)| matches!(t, ActionType::Speak)));
        assert!(seated.iter().any(|(t, _)| matches!(t, ActionType::Think)));
    }

    #[test]
    fn test_greedy_keeps_highest_score_per_type() {
        // 同类型（Speak）两个候选：0.7 与 0.3 → 只留 0.7
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let action_a = MemoryId::new();
        let action_b = MemoryId::new();

        cluster.write(|c| {
            let source_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(vec![
                MemoryLink::new(
                    source_id,
                    action_a,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.7))),
                ),
                MemoryLink::new(
                    source_id,
                    action_b,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.3))),
                ),
            ])
            .build()
            .unwrap();
            let source_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_emb,
            });

            for (aid, name) in [(action_a, "ActionA"), (action_b, "ActionB")] {
                let note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new(name.into(), ActionType::new_speak()),
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

        let result = RetrBayesAction {}.retrieve(BayesActionRequest::new(
            Arc::new(wm),
            vec![(source_id, 1.0)],
        ));

        assert_eq!(result.len(), 1);
        let seat = result.speak.expect("Speak 席位应被占据");
        assert_eq!(seat.id, action_a, "应保留分数更高的动作");
        assert!((seat.score - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_tie_break_is_deterministic() {
        // 同类型同分（0.5）两个候选 → tiebreak 取 MemoryId 较小者
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let mut candidates = vec![MemoryId::new(), MemoryId::new()];
        candidates.sort();

        cluster.write(|c| {
            let links: Vec<MemoryLink> = candidates
                .iter()
                .map(|aid| {
                    MemoryLink::new(
                        source_id,
                        *aid,
                        MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.5))),
                    )
                })
                .collect();
            let source_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(links)
            .build()
            .unwrap();
            let source_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_emb,
            });

            for aid in &candidates {
                let note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new("Tied".into(), ActionType::new_think()),
                )))
                .id(*aid)
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

        let result = RetrBayesAction {}.retrieve(BayesActionRequest::new(
            Arc::new(wm),
            vec![(source_id, 1.0)],
        ));

        assert_eq!(result.len(), 1);
        let seat = result.think.expect("Think 席位应被占据");
        assert_eq!(seat.id, candidates[0], "同分时应取 MemoryId 较小者");
    }

    /// 「一个情境源 + 若干**同类型**（Think）动作候选」的最小夹具。
    ///
    /// 只有同类型才会争同一个席位；候选分数等于 `TrigToAction` 权重。
    struct ThinkCandidates {
        wm: WorkingMemory,
        source_id: MemoryId,
    }

    /// 按 `(候选 id, 链接 `TrigToAction` 权重)` 造夹具；权重相同即同分。
    fn think_candidates(scored: &[(MemoryId, f64)]) -> ThinkCandidates {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();

        cluster.write(|c| {
            let links: Vec<MemoryLink> = scored
                .iter()
                .map(|(id, weight)| {
                    MemoryLink::new(
                        source_id,
                        *id,
                        MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(*weight))),
                    )
                })
                .collect();
            let source_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(links)
            .build()
            .unwrap();
            let source_emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note: source_note,
                embedding: source_emb,
            });

            for (id, _) in scored {
                let note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new("Candidate".into(), ActionType::new_think()),
                )))
                .id(*id)
                .build()
                .unwrap();
                c.add_single_node(EmbeddedMemoryNote {
                    note,
                    embedding: MemoryEmbedding::new(
                        EmbeddingVec::zero(128),
                        MemoryEmbeddingVariant::Procedure(),
                    ),
                });
            }
        });

        ThinkCandidates { wm, source_id }
    }

    /// 贪心比较与同分 tiebreak 的判定必须与 `scores` 的迭代顺序无关。
    ///
    /// `best_per_action_type` 遍历的是一张 `HashMap`，其迭代顺序每次运行都不同；单次断言
    /// 只能靠运气命中「顺序敏感」的变异体——例如把 `*score > current.score` 改成 `==` 后，
    /// 只有在最高分**恰好先被遍历到**时才不改变胜者，因此能否杀灭取决于当次哈希种子。
    /// 这里反复重建工作记忆（每轮都是全新的 `HashMap` 与新的哈希种子），把「靠运气通过」
    /// 的概率压到可忽略：两个场景各有 3 个候选，坏顺序恰好出现的概率约 1/3，64 轮后约
    /// 3^-64。
    #[test]
    fn test_greedy_and_tiebreak_are_order_independent() {
        const ROUNDS: usize = 64;

        for _ in 0..ROUNDS {
            // (1) 分数不同：分数最高者必须胜出，且与其 id 大小无关
            let mut distinct = [MemoryId::new(), MemoryId::new(), MemoryId::new()];
            distinct.sort();
            let fixture =
                think_candidates(&[(distinct[0], 0.3), (distinct[1], 0.5), (distinct[2], 0.7)]);
            let source_id = fixture.source_id;
            let winner = RetrBayesAction {}
                .retrieve(BayesActionRequest::new(
                    Arc::new(fixture.wm),
                    vec![(source_id, 1.0)],
                ))
                .think
                .expect("Think 席位应被占据")
                .id;
            assert_eq!(
                winner, distinct[2],
                "不同分时应取分数最高者，与遍历顺序无关"
            );

            // (2) 全同分：`MemoryId` 较小者必须胜出
            let mut tied = [MemoryId::new(), MemoryId::new(), MemoryId::new()];
            tied.sort();
            let fixture = think_candidates(&[(tied[0], 0.5), (tied[1], 0.5), (tied[2], 0.5)]);
            let source_id = fixture.source_id;
            let winner = RetrBayesAction {}
                .retrieve(BayesActionRequest::new(
                    Arc::new(fixture.wm),
                    vec![(source_id, 1.0)],
                ))
                .think
                .expect("Think 席位应被占据")
                .id;
            assert_eq!(
                winner, tied[0],
                "同分时应取 MemoryId 较小者，与遍历顺序无关"
            );
        }
    }

    #[test]
    fn test_multi_source_action_aggregation() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let src_a = MemoryId::new();
        let src_b = MemoryId::new();
        let action_id = MemoryId::new();

        cluster.write(|c| {
            for (sid, prob) in [(src_a, 0.6f64), (src_b, 0.4f64)] {
                let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                    content: "source".into(),
                    aliases: vec![],
                    concept_type: ConceptType::Entity,
                    description: String::new(),
                }))
                .id(sid)
                .mem_links(vec![MemoryLink::new(
                    sid,
                    action_id,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(prob))),
                )])
                .build()
                .unwrap();
                let emb = MemoryEmbedding::new(
                    EmbeddingVec::zero(128),
                    MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                        EmbeddingVec::zero(128),
                    )),
                );
                c.add_single_node(EmbeddedMemoryNote {
                    note,
                    embedding: emb,
                });
            }

            let action_note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                Action::new("AggregateAction".into(), ActionType::new_speak()),
            )))
            .id(action_id)
            .build()
            .unwrap();
            let action_emb =
                MemoryEmbedding::new(EmbeddingVec::zero(128), MemoryEmbeddingVariant::Procedure());
            c.add_single_node(EmbeddedMemoryNote {
                note: action_note,
                embedding: action_emb,
            });
        });

        let sources = vec![(src_a, 0.8), (src_b, 0.5)];
        let result = RetrBayesAction {}.retrieve(BayesActionRequest::new(Arc::new(wm), sources));

        assert_eq!(result.len(), 1);
        let seat = result.speak.expect("Speak 席位应被占据");
        assert_eq!(seat.id, action_id);
        // 多源聚合：0.6×0.8 + 0.4×0.5 = 0.68
        assert!((seat.score - 0.68).abs() < 1e-9, "实际 {}", seat.score);
    }

    #[test]
    fn test_bayes_action_prob_accuracy() {
        let wm = WorkingMemory::new(10);
        let cluster = wm.memory_cluster();
        let source_id = MemoryId::new();
        let action_id = MemoryId::new();

        cluster.write(|c| {
            let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
                content: "source".into(),
                aliases: vec![],
                concept_type: ConceptType::Entity,
                description: String::new(),
            }))
            .id(source_id)
            .mem_links(vec![MemoryLink::new(
                source_id,
                action_id,
                MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.5))),
            )])
            .build()
            .unwrap();
            let emb = MemoryEmbedding::new(
                EmbeddingVec::zero(128),
                MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                    EmbeddingVec::zero(128),
                )),
            );
            c.add_single_node(EmbeddedMemoryNote {
                note,
                embedding: emb,
            });

            let a_note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                Action::new("Test".into(), ActionType::new_speak()),
            )))
            .id(action_id)
            .build()
            .unwrap();
            let a_emb =
                MemoryEmbedding::new(EmbeddingVec::zero(128), MemoryEmbeddingVariant::Procedure());
            c.add_single_node(EmbeddedMemoryNote {
                note: a_note,
                embedding: a_emb,
            });
        });

        let request = BayesActionRequest::new(Arc::new(wm), vec![(source_id, 1.0)]);
        let result = RetrBayesAction {}.retrieve(request);

        let seat = result.speak.expect("Speak 席位应被占据");
        let expected = 0.5 * 1.0;
        assert!(
            (seat.score - expected).abs() < 1e-6,
            "Expected {expected}, got {}",
            seat.score
        );
    }
}
