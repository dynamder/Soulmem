use std::fmt::Display;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::memory::{memory_links::proc_mem::ProcMemLink, memory_note::MemoryId};

pub mod proc_mem;
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
    link_type: MemoryLinkType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryLinkType {
    Proc(ProcMemLink),
}

impl MemoryLink {
    pub fn new(from: MemoryId, to: MemoryId, link_type: MemoryLinkType) -> Self {
        MemoryLink {
            id: LinkId::default(),
            from,
            to,
            link_type,
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
    pub fn into_tuple(self) -> (MemoryId, MemoryId, MemoryLinkType) {
        (self.from, self.to, self.link_type)
    }
    pub fn from_tuple(from: MemoryId, to: MemoryId, link_type: MemoryLinkType) -> Self {
        MemoryLink {
            id: LinkId::default(),
            from,
            to,
            link_type,
        }
    }
    pub fn into_link_type(self) -> MemoryLinkType {
        self.link_type
    }
}
impl From<(MemoryId, MemoryId, MemoryLinkType)> for MemoryLink {
    fn from(tuple: (MemoryId, MemoryId, MemoryLinkType)) -> Self {
        MemoryLink::from_tuple(tuple.0, tuple.1, tuple.2)
    }
}
