use super::RetrStrategy;
// 采用 LLM进行的Plan-on-Graph
pub struct RetrDeepThought {
    max_depth: usize,
}
pub struct DeepThoughtRequest {}
impl RetrStrategy for RetrDeepThought {
    type RetrRequest = DeepThoughtRequest;
    fn retrieve(&self, request: Self::RetrRequest) -> Vec<String> {
        todo!()
    }
}
