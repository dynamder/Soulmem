pub mod association;
pub mod bayes_action;
pub mod cached_path;
pub mod complex;
pub mod short_only;
pub mod similarity;

pub trait RetrStrategy: 'static {
    type Request: RetrRequest;
    type Return<'a>
    where
        Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_>;
}

pub trait RetrRequest {}

#[derive(serde::Deserialize)]
#[serde(tag = "type")]
pub enum RetrRequestConfig {
    Association(association::AssociationConfig),
    BayesAction(bayes_action::BayesActionConfig),
    AssociateWithAction(complex::assoc_with_action::AssociateWithActionConfig),
    ShortOnly(short_only::ShortOnlyConfig),
    Similarity(similarity::SimilarityConfig),
    CachedPath(cached_path::CachedPathConfig),
}
