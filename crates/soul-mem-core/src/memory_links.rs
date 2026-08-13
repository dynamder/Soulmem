use std::fmt::Display;

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
    link_type: MemoryLinkType,
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
        }
    }
    pub fn into_link_type(self) -> MemoryLinkType {
        self.link_type
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
