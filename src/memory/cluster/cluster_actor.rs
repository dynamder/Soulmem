use crate::memory::memory_note::{MemoryId, MemoryNote};
use tokio::sync::oneshot;

pub(super) enum ClusterActorMsg {
    //read operation
    GetNode(MemoryId, oneshot::Sender<Option<MemoryNote>>),
}
