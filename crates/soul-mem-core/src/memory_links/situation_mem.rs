use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SituationMemLink {
    AbstractToSpecific(AbstractToSpecific),
    /// 具体情境 → 抽象情境：PPR 从具体情境种子游走到抽象模式节点的反向边。
    /// 抽象模式节点由 PPR 检出后作为 Bayes 动作提取的优先源。
    SpecificToAbstract(SpecificToAbstract),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractToSpecific {}
impl AbstractToSpecific {
    pub fn new() -> Self {
        AbstractToSpecific {}
    }
}

/// 具体情境到抽象情境的关联（空结构，方向由边 from/to 表达）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecificToAbstract {}
impl SpecificToAbstract {
    pub fn new() -> Self {
        SpecificToAbstract {}
    }
}
