use crate::memory::{
    embedding::{Embeddable, EmbeddingVec, raw_linear_blend},
    memory_note::situation_mem::Participant,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SitParticipantEmbedding {
    name: EmbeddingVec,
    role: EmbeddingVec,
    fused: EmbeddingVec,
}
impl SitParticipantEmbedding {
    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
}
impl Embeddable for Participant {
    type EmbeddingGen = SitParticipantEmbedding;
    type EmbeddingFused = EmbeddedParticipant;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [name_vec, role_vec] = model
            .infer_batch(&vec![self.name.as_str(), self.role.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 可以确定此处的Vec长度为2
        let fused_vec = raw_linear_blend(&name_vec, &role_vec, 0.7).unwrap(); //SAFEUNWRAP: 此处两向量的维度必然相同
        Ok(SitParticipantEmbedding {
            name: name_vec,
            role: role_vec,
            fused: fused_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedParticipant {
            embedding: self.embed(model)?,
            participant: self,
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedParticipant {
    pub embedding: SitParticipantEmbedding,
    pub participant: Participant,
}
