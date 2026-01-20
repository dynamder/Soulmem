use crate::memory::{embedding::MemoryEmbedding, memory_note::MemoryId};
pub mod association;
pub mod cached_path;
pub mod deep_thought;
pub mod short_only;
pub mod similarity;

pub trait RetrStrategy {
    type Request: RetrRequest; //接受的查询参数类型
    fn retrieve(&self, request: Self::Request) -> Vec<MemoryId>; //TODO：返回类型还没想好，暂定Vec<MemoryId>，或许也可以考虑返回迭代器，看具体场景
}

pub trait RetrRequest {}
