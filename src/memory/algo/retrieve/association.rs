use super::RetrStrategy;

//用PPR变种算法进行联想
pub struct RetrAssociation {
    max_results: usize,
}
pub struct AssociationRequest {}
impl RetrStrategy for RetrAssociation {
    type RetrRequest = AssociationRequest;
    fn retrieve(&self, request: Self::RetrRequest) -> Vec<String> {
        todo!()
    }
}
