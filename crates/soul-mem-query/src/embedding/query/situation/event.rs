use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{mean_pooling, Embeddable, EmbeddingCalcResult, EmbeddingVec};
use crate::query::retrieve::EventQueryUnit;

#[derive(Debug, Clone, PartialEq)]
pub struct EventQueryUnitEmbedding {
    action: EmbeddingVec,
    initiator: Option<EmbeddingVec>,
    target: Option<EmbeddingVec>,
    pub blend_weights: BlendWeights,
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

    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
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
            blend_weights: BlendWeights::default(),
        }))
    }
}
#[cfg(test)]
impl EventQueryUnitEmbedding {
    pub(crate) fn test_new(
        action: EmbeddingVec,
        initiator: Option<EmbeddingVec>,
        target: Option<EmbeddingVec>,
        blend_weights: BlendWeights,
    ) -> Self {
        Self {
            action,
            initiator,
            target,
            blend_weights,
        }
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
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [action_vec] = model.infer_query_batch(&vec![self.action()])?.try_into().unwrap(); //SAFEUNWRAP: 此处长度必为1

        let initiator_batch_vec = self
            .initiator()
            .map(|initiator| model.infer_query_batch(&vec![initiator]))
            .transpose()?;

        let initiator_vec = initiator_batch_vec
            .map(|vec| vec.into_iter().next())
            .flatten();

        let target_batch_vec = self
            .target()
            .map(|target| model.infer_query_batch(&vec![target]))
            .transpose()?;

        let target_vec = target_batch_vec.map(|vec| vec.into_iter().next()).flatten();

        Ok(EventQueryUnitEmbedding {
            action: action_vec,
            initiator: initiator_vec,
            target: target_vec,
            blend_weights: BlendWeights::default(),
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedEventQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_query_unit_embedding_accessors() {
        let mut embedding = EventQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            Some(EmbeddingVec::new(vec![2.0])),
            Some(EmbeddingVec::new(vec![3.0])),
            BlendWeights::default(),
        );
        assert_eq!(embedding.action().shape(), 1);
        assert_eq!(embedding.initiator().unwrap().shape(), 1);
        assert_eq!(embedding.target().unwrap().shape(), 1);

        let mut bw = BlendWeights::default();
        bw.tag = 0.8;
        embedding.set_blend_weights(&bw);
        assert_eq!(embedding.blend_weights.tag, 0.8);
    }

    #[test]
    fn test_event_query_unit_embedding_optional_fields_none() {
        let embedding = EventQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            None,
            None,
            BlendWeights::default(),
        );
        assert!(embedding.initiator().is_none());
        assert!(embedding.target().is_none());
        assert_eq!(embedding.action().shape(), 1);
    }

    #[test]
    fn test_event_query_unit_mean_pooling() {
        let e1 = EventQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![1.0, 2.0]),
            Some(EmbeddingVec::new(vec![1.0, 2.0])),
            Some(EmbeddingVec::new(vec![1.0, 2.0])),
            BlendWeights::default(),
        );
        let e2 = EventQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![3.0, 4.0]),
            None,
            None,
            BlendWeights::default(),
        );
        let pooled = EventQueryUnitEmbedding::mean_pooling(&[e1, e2])
            .unwrap()
            .unwrap();
        assert_eq!(pooled.action().shape(), 2);
        // initiator/target 只出现在一个元素里 → 仍保留
        assert!(pooled.initiator().is_some());
        assert!(pooled.target().is_some());
    }

    #[test]
    fn test_event_query_unit_mean_pooling_empty() {
        assert!(EventQueryUnitEmbedding::mean_pooling(&[]).unwrap().is_none());
    }
}
