use std::sync::Arc;

use crate::memory::{
    algo::retrieve::{
        RetrRequest, RetrStrategy,
        association::{AssociationRequest, RetrAssociation},
        bayesian_action::{BayesActionRequest, RetrBayesAction},
    },
    memory_note::MemoryId,
};

pub struct RetrAssociateWithAction;

pub struct AssociateWithActionRequest {
    association: AssociationRequest,
    action_top_k: usize,
}

impl AssociateWithActionRequest {
    pub fn new(association: AssociationRequest) -> Self {
        Self {
            association,
            action_top_k: 3,
        }
    }
    pub fn with_action_top_k(mut self, action_top_k: usize) -> Self {
        self.action_top_k = action_top_k;
        self
    }
}

impl RetrRequest for AssociateWithActionRequest {}

pub struct AssociateWithActionResult {
    pub memory: Vec<(MemoryId, f64)>,
    pub action: Vec<(MemoryId, f64)>,
}

impl RetrStrategy for RetrAssociateWithAction {
    type Request = AssociateWithActionRequest;
    type Return<'a> = AssociateWithActionResult;

    fn retrieve(&self, request: Self::Request) -> Self::Return<'_> {
        let working_mem = Arc::clone(&request.association.working_mem);
        let association_res = RetrAssociation {}.retrieve(request.association);
        let normalized_association_res = softmax(&association_res);

        let action_request = BayesActionRequest::new(working_mem, normalized_association_res)
            .with_top_k(request.action_top_k);

        let action_res = RetrBayesAction {}.retrieve(action_request);

        AssociateWithActionResult {
            memory: association_res,
            action: action_res,
        }
    }
}

fn softmax(logits: &[(MemoryId, f64)]) -> Vec<(MemoryId, f64)> {
    let max_x = logits
        .iter()
        .max_by(|&(_, v1), &(_, v2)| v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(_, v)| v)
        .unwrap_or(0.0);

    let sum = logits
        .iter()
        .map(|&(_, x)| f64::exp(x - max_x))
        .sum::<f64>();

    logits
        .iter()
        .map(|&(id, x)| (id, f64::exp(x - max_x) / sum))
        .collect()
}
