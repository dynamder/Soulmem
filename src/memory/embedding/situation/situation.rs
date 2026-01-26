use crate::memory::{
    embedding::{Embeddable, EmbeddingVec},
    memory_note::situation_mem::Situation,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SitSituationEmbedding {
    atmosphere: EmbeddingVec,
    tone: EmbeddingVec,
}
impl SitSituationEmbedding {
    pub fn atmosphere(&self) -> &EmbeddingVec {
        &self.atmosphere
    }

    pub fn tone(&self) -> &EmbeddingVec {
        &self.tone
    }
}
impl Embeddable for Situation {
    type EmbeddingGen = SitSituationEmbedding;
    type EmbeddingFused = EmbeddedSitSituation;
    fn embed(
        &self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [atmosphere_vec, tone_vec] = model
            .infer_batch(&vec![self.atmosphere.as_str(), self.tone.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处返回的Vec长度必为2
        Ok(SitSituationEmbedding {
            atmosphere: atmosphere_vec,
            tone: tone_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedSitSituation {
            embedding: self.embed(model)?,
            situation: self,
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedSitSituation {
    pub embedding: SitSituationEmbedding,
    pub situation: Situation,
}
