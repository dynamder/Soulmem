use crate::memory::embedding::MemoryEmbedding;
pub mod association;
pub mod deep_thought;
pub mod short_only;
pub mod similarity;

pub trait RetrStrategy {
    type RetrRequest; //接受的查询参数类型
    fn retrieve(&self, request: Self::RetrRequest) -> Vec<String>; //TODO：返回类型还没想好，暂定Vec<String>，或许也可以考虑返回迭代器，看具体场景
}




