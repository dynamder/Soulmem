use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{mean_pooling, Embeddable, EmbeddingCalcResult, EmbeddingVec};
use crate::query::retrieve::ParticipantQueryUnit;

#[derive(Debug, Clone, PartialEq)]
pub struct ParticipantQueryUnitEmbedding {
    name: Option<EmbeddingVec>,
    role: Option<EmbeddingVec>,
    pub blend_weights: BlendWeights,
}
impl ParticipantQueryUnitEmbedding {
    pub fn name(&self) -> Option<&EmbeddingVec> {
        self.name.as_ref()
    }
    pub fn role(&self) -> Option<&EmbeddingVec> {
        self.role.as_ref()
    }
    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
    }
    pub fn mean_pooling(vecs: &[Self]) -> EmbeddingCalcResult<Option<Self>> {
        if vecs.is_empty() {
            return Ok(None);
        }
        let name_vecs = vecs.iter().filter_map(|p| p.name()).collect::<Vec<_>>();
        let role_vecs = vecs.iter().filter_map(|p| p.role()).collect::<Vec<_>>();

        let name_vec = if name_vecs.is_empty() {
            None
        } else {
            Some(mean_pooling(&name_vecs)?)
        };
        let role_vec = if role_vecs.is_empty() {
            None
        } else {
            Some(mean_pooling(&role_vecs)?)
        };

        Ok(Some(Self {
            name: name_vec,
            role: role_vec,
            blend_weights: BlendWeights::default(),
        }))
    }
}
#[cfg(test)]
impl ParticipantQueryUnitEmbedding {
    pub(crate) fn test_new(
        name: Option<EmbeddingVec>,
        role: Option<EmbeddingVec>,
        blend_weights: BlendWeights,
    ) -> Self {
        Self {
            name,
            role,
            blend_weights,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedParticipantQueryUnit {
    pub embedding: ParticipantQueryUnitEmbedding,
    pub query: ParticipantQueryUnit,
}

impl Embeddable for ParticipantQueryUnit {
    type EmbeddingGen = ParticipantQueryUnitEmbedding;
    type EmbeddingFused = EmbedParticipantQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let name_batch_vec = self
            .name()
            .map(|name| model.infer_query_batch(&vec![name]))
            .transpose()?;

        let name_vec = name_batch_vec.and_then(|vec| vec.into_iter().next());

        let role_batch_vec = self
            .role()
            .map(|role| model.infer_query_batch(&vec![role]))
            .transpose()?;

        let role_vec = role_batch_vec.and_then(|vec| vec.into_iter().next());

        Ok(ParticipantQueryUnitEmbedding {
            name: name_vec,
            role: role_vec,
            blend_weights: BlendWeights::default(),
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedParticipantQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_participant_query_unit_embedding_accessors() {
        let mut embedding = ParticipantQueryUnitEmbedding::test_new(
            Some(EmbeddingVec::new(vec![1.0])),
            Some(EmbeddingVec::new(vec![2.0])),
            BlendWeights::default(),
        );
        assert_eq!(embedding.name().unwrap().shape(), 1);
        assert_eq!(embedding.role().unwrap().shape(), 1);

        let mut bw = BlendWeights::default();
        bw.tag = 0.8;
        embedding.set_blend_weights(&bw);
        assert_eq!(embedding.blend_weights.tag, 0.8);
    }

    #[test]
    fn test_participant_query_unit_embedding_none() {
        let embedding = ParticipantQueryUnitEmbedding::test_new(None, None, BlendWeights::default());
        assert!(embedding.name().is_none());
        assert!(embedding.role().is_none());
    }

    #[test]
    fn test_participant_query_unit_mean_pooling() {
        let p1 = ParticipantQueryUnitEmbedding::test_new(
            Some(EmbeddingVec::new(vec![1.0, 2.0])),
            Some(EmbeddingVec::new(vec![1.0, 2.0])),
            BlendWeights::default(),
        );
        let p2 = ParticipantQueryUnitEmbedding::test_new(
            Some(EmbeddingVec::new(vec![3.0, 4.0])),
            None,
            BlendWeights::default(),
        );
        let pooled = ParticipantQueryUnitEmbedding::mean_pooling(&[p1, p2])
            .unwrap()
            .unwrap();
        assert_eq!(pooled.name().unwrap().shape(), 2);
        assert!(pooled.role().is_some());
    }

    #[test]
    fn test_participant_query_unit_mean_pooling_empty() {
        assert!(ParticipantQueryUnitEmbedding::mean_pooling(&[]).unwrap().is_none());
    }
}
