use std::fmt::Display;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::memory_links::proc_mem::ProcMemLink;
use crate::memory_links::sem_mem::SemMemLink;
use crate::memory_links::situation_mem::SituationMemLink;
use crate::memory_note::MemoryId;

pub mod proc_mem;
pub mod sem_mem;

pub mod situation_mem;
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize)]
pub struct LinkId(Uuid);
impl LinkId {
    pub fn new() -> Self {
        LinkId(Uuid::new_v4())
    }
}
impl Default for LinkId {
    fn default() -> Self {
        LinkId::new()
    }
}
impl Display for LinkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryLink {
    id: LinkId,
    from: MemoryId,
    to: MemoryId,
    pub intensity: f64,
    /// 遗忘缺失度（0.0 新鲜 ~ 1.0 完全遗忘）
    #[serde(default = "default_missing_degree")]
    pub missing_degree: f32,
    /// 缺失度最近一次计算的时间，用于增量更新
    #[serde(default = "default_last_forget_time")]
    last_forget_time: DateTime<Utc>,
    link_type: MemoryLinkType,
}

/// serde 默认：缺失度初始为 0
fn default_missing_degree() -> f32 {
    0.0
}

/// serde 默认：缺失度计算时间初始为当前
fn default_last_forget_time() -> DateTime<Utc> {
    Utc::now()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryLinkType {
    Proc(ProcMemLink),
    Sem(SemMemLink),
    Situation(SituationMemLink),
}

impl MemoryLink {
    pub fn new(from: MemoryId, to: MemoryId, link_type: MemoryLinkType) -> Self {
        MemoryLink {
            id: LinkId::default(),
            from,
            to,
            link_type,
            intensity: 1.0,
            missing_degree: 0.0,
            last_forget_time: Utc::now(),
        }
    }
    pub fn id(&self) -> LinkId {
        self.id
    }
    pub fn from(&self) -> MemoryId {
        self.from
    }
    pub fn to(&self) -> MemoryId {
        self.to
    }
    pub fn link_type(&self) -> &MemoryLinkType {
        &self.link_type
    }
    pub fn link_type_mut(&mut self) -> &mut MemoryLinkType {
        &mut self.link_type
    }
    pub fn missing_degree(&self) -> f32 {
        self.missing_degree
    }
    pub fn set_missing_degree(&mut self, missing_degree: f32) {
        self.missing_degree = missing_degree.clamp(0.0, 1.0);
    }
    pub fn last_forget_time(&self) -> DateTime<Utc> {
        self.last_forget_time
    }
    pub fn set_last_forget_time(&mut self, time: DateTime<Utc>) {
        self.last_forget_time = time;
    }
    pub fn into_tuple(self) -> (MemoryId, MemoryId, MemoryLinkType, f64) {
        (self.from, self.to, self.link_type, self.intensity)
    }
    pub fn from_tuple(
        from: MemoryId,
        to: MemoryId,
        link_type: MemoryLinkType,
        intensity: f64,
    ) -> Self {
        MemoryLink {
            id: LinkId::default(),
            from,
            to,
            link_type,
            intensity,
            missing_degree: 0.0,
            last_forget_time: Utc::now(),
        }
    }
    pub fn into_link_type(self) -> MemoryLinkType {
        self.link_type
    }
}

impl Default for MemoryLink {
    fn default() -> Self {
        Self {
            id: LinkId::default(),
            from: MemoryId::default(),
            to: MemoryId::default(),
            intensity: 1.0,
            missing_degree: 0.0,
            last_forget_time: Utc::now(),
            link_type: MemoryLinkType::Sem(SemMemLink::default()),
        }
    }
}
impl From<(MemoryId, MemoryId, MemoryLinkType, f64)> for MemoryLink {
    fn from(tuple: (MemoryId, MemoryId, MemoryLinkType, f64)) -> Self {
        MemoryLink::from_tuple(tuple.0, tuple.1, tuple.2, tuple.3)
    }
}
