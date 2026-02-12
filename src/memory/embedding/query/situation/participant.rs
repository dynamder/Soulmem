use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcResult, EmbeddingVec, mean_pooling},
    query::retrieve::ParticipantQueryUnit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ParticipantQueryUnitEmbedding {
    name: Option<EmbeddingVec>,
    role: Option<EmbeddingVec>,
}
impl ParticipantQueryUnitEmbedding {
    pub fn name(&self) -> Option<&EmbeddingVec> {
        self.name.as_ref()
    }
    pub fn role(&self) -> Option<&EmbeddingVec> {
        self.role.as_ref()
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
        }))
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
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let name_batch_vec = self
            .name()
            .map(|name| model.infer_batch(&vec![name]))
            .transpose()?;

        let name_vec = name_batch_vec.map(|vec| vec.into_iter().next()).flatten();

        let role_batch_vec = self
            .role()
            .map(|role| model.infer_batch(&vec![role]))
            .transpose()?;

        let role_vec = role_batch_vec.map(|vec| vec.into_iter().next()).flatten();

        Ok(ParticipantQueryUnitEmbedding {
            name: name_vec,
            role: role_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedParticipantQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
