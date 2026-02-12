use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcResult, EmbeddingVec, mean_pooling},
    query::retrieve::EventQueryUnit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct EventQueryUnitEmbedding {
    action: EmbeddingVec,
    initiator: Option<EmbeddingVec>,
    target: Option<EmbeddingVec>,
}
impl EventQueryUnitEmbedding {
    pub fn action(&self) -> &EmbeddingVec {
        &self.action
    }
    pub fn initiator(&self) -> Option<&EmbeddingVec> {
        self.initiator.as_ref()
    }
    pub fn target(&self) -> Option<&EmbeddingVec> {
        self.target.as_ref()
    }
    pub fn mean_pooling(vecs: &[EventQueryUnitEmbedding]) -> EmbeddingCalcResult<Option<Self>> {
        if vecs.is_empty() {
            return Ok(None);
        }
        let actions = vecs.iter().map(|vec| vec.action()).collect::<Vec<_>>();
        let initiators = vecs
            .iter()
            .filter_map(|vec| vec.initiator())
            .collect::<Vec<_>>();
        let targets = vecs
            .iter()
            .filter_map(|vec| vec.target())
            .collect::<Vec<_>>();

        let action_vec = mean_pooling(&actions)?;

        let initiator_vec = if initiators.is_empty() {
            None
        } else {
            Some(mean_pooling(&initiators)?)
        };

        let target_vec = if targets.is_empty() {
            None
        } else {
            Some(mean_pooling(&targets)?)
        };

        Ok(Some(Self {
            action: action_vec,
            initiator: initiator_vec,
            target: target_vec,
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedEventQueryUnit {
    pub embedding: EventQueryUnitEmbedding,
    pub query: EventQueryUnit,
}

impl Embeddable for EventQueryUnit {
    type EmbeddingGen = EventQueryUnitEmbedding;
    type EmbeddingFused = EmbedEventQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [action_vec] = model.infer_batch(&vec![self.action()])?.try_into().unwrap(); //SAFEUNWRAP: 此处长度必为1

        let initiator_batch_vec = self
            .initiator()
            .map(|initiator| model.infer_batch(&vec![initiator]))
            .transpose()?;

        let initiator_vec = initiator_batch_vec
            .map(|vec| vec.into_iter().next())
            .flatten();

        let target_batch_vec = self
            .target()
            .map(|target| model.infer_batch(&vec![target]))
            .transpose()?;

        let target_vec = target_batch_vec.map(|vec| vec.into_iter().next()).flatten();

        Ok(EventQueryUnitEmbedding {
            action: action_vec,
            initiator: initiator_vec,
            target: target_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedEventQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
