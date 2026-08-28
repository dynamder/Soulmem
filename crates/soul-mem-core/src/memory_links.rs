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

/// `MemoryLink` 的构建器：允许覆盖 id / 权重 / 遗忘状态等持久化字段。
/// 必填项（端点 + 类型）在 `new` 时给出，其余可链式覆盖，风格与 `MemoryNoteBuilder` 一致。
pub struct MemoryLinkBuilder {
    id: Option<LinkId>,
    from: MemoryId,
    to: MemoryId,
    link_type: MemoryLinkType,
    intensity: Option<f64>,
    missing_degree: Option<f32>,
    last_forget_time: Option<DateTime<Utc>>,
}
impl MemoryLinkBuilder {
    pub fn new(from: MemoryId, to: MemoryId, link_type: MemoryLinkType) -> Self {
        Self {
            id: None,
            from,
            to,
            link_type,
            intensity: None,
            missing_degree: None,
            last_forget_time: None,
        }
    }
    pub fn id(mut self, id: LinkId) -> Self {
        self.id = Some(id);
        self
    }
    pub fn intensity(mut self, intensity: f64) -> Self {
        self.intensity = Some(intensity);
        self
    }
    pub fn missing_degree(mut self, missing_degree: f32) -> Self {
        self.missing_degree = Some(missing_degree);
        self
    }
    pub fn last_forget_time(mut self, time: DateTime<Utc>) -> Self {
        self.last_forget_time = Some(time);
        self
    }
    pub fn build(self) -> MemoryLink {
        MemoryLink {
            id: self.id.unwrap_or_default(),
            from: self.from,
            to: self.to,
            link_type: self.link_type,
            intensity: self.intensity.unwrap_or(1.0),
            // 与 set_missing_degree 保持一致：限制在 0.0~1.0
            missing_degree: self.missing_degree.unwrap_or(0.0).clamp(0.0, 1.0),
            last_forget_time: self.last_forget_time.unwrap_or_else(Utc::now),
        }
    }
}

impl From<(MemoryId, MemoryId, MemoryLinkType, f64)> for MemoryLink {
    fn from(tuple: (MemoryId, MemoryId, MemoryLinkType, f64)) -> Self {
        MemoryLink::from_tuple(tuple.0, tuple.1, tuple.2, tuple.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_id_default_is_new() {
        let a = LinkId::default();
        let b = LinkId::default();
        assert_ne!(a, b, "each default LinkId must be unique");
    }

    #[test]
    fn test_link_id_display() {
        let id = LinkId::default();
        let uuid = {
            // LinkId's inner Uuid is opaque; verify display is non-empty and stable
            let s1 = id.to_string();
            let s2 = id.to_string();
            assert_eq!(s1, s2);
            assert!(!s1.is_empty());
            s1
        };
        let parsed = uuid::Uuid::parse_str(&uuid).expect("display output must be a valid Uuid");
        let _ = parsed;
    }

    #[test]
    fn test_builder_full_chain() {
        let id = LinkId::new();
        let from = MemoryId::new();
        let to = MemoryId::new();
        let time = Utc::now() - chrono::Duration::hours(2);
        let link = MemoryLinkBuilder::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("relates".into(), 0.8)),
        )
        .id(id)
        .intensity(0.7)
        .missing_degree(0.3)
        .last_forget_time(time)
        .build();
        assert_eq!(link.id(), id);
        assert_eq!(link.from(), from);
        assert_eq!(link.to(), to);
        assert_eq!(link.intensity, 0.7);
        assert_eq!(link.missing_degree(), 0.3);
        assert_eq!(link.last_forget_time(), time);
    }

    #[test]
    fn test_builder_defaults() {
        let link = MemoryLinkBuilder::new(
            MemoryId::new(),
            MemoryId::new(),
            MemoryLinkType::Sem(SemMemLink::default()),
        )
        .build();
        assert_eq!(link.intensity, 1.0);
        assert_eq!(link.missing_degree(), 0.0);
        // 默认 id 是新建的（非零概率重复视为测试失败）
        assert_ne!(link.id(), LinkId::default());
    }

    #[test]
    fn test_builder_missing_degree_clamped() {
        let link = MemoryLinkBuilder::new(
            MemoryId::new(),
            MemoryId::new(),
            MemoryLinkType::Sem(SemMemLink::default()),
        )
        .missing_degree(1.7)
        .build();
        assert_eq!(link.missing_degree(), 1.0);
    }

    #[test]
    fn test_memory_link_from_to_roundtrip() {
        let from = MemoryId::default();
        let to = MemoryId::default();
        let link = MemoryLink::new(
            from,
            to,
            MemoryLinkType::Sem(SemMemLink::new("test".to_string(), 0.5)),
        );
        assert_eq!(link.from(), from);
        assert_eq!(link.to(), to);
        assert_eq!(link.id(), link.id());
    }

    #[test]
    fn test_memory_link_from_tuple() {
        let from = MemoryId::default();
        let to = MemoryId::default();
        let link_type = MemoryLinkType::Sem(SemMemLink::new("test".to_string(), 0.5));
        let link = MemoryLink::from_tuple(from, to, link_type.clone(), 2.5);
        assert_eq!(link.from(), from);
        assert_eq!(link.to(), to);
        assert_eq!(link.into_link_type(), link_type);
    }

    #[test]
    fn test_memory_link_into_tuple() {
        let from = MemoryId::default();
        let to = MemoryId::default();
        let link_type = MemoryLinkType::Sem(SemMemLink::new("test".to_string(), 0.5));
        let link = MemoryLink::from_tuple(from, to, link_type.clone(), 2.5);
        assert_eq!(link.into_tuple(), (from, to, link_type, 2.5));
    }
}
