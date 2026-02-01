use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcResult, EmbeddingVec, mean_pooling, raw_linear_blend},
    memory_note::situation_mem::Participant,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ParticipantEmbedding {
    name: EmbeddingVec,
    role: EmbeddingVec,
    fused: EmbeddingVec,
}
impl ParticipantEmbedding {
    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
    pub fn role(&self) -> &EmbeddingVec {
        &self.role
    }
    pub fn fused(&self) -> &EmbeddingVec {
        &self.fused
    }
    pub fn mean_pooling(vecs: &[Self]) -> EmbeddingCalcResult<Option<Self>> {
        if vecs.is_empty() {
            return Ok(None);
        }
        let names = vecs.iter().map(|p| p.name()).collect::<Vec<_>>();
        let name_vec = mean_pooling(&names)?;

        let roles = vecs.iter().map(|p| p.role()).collect::<Vec<_>>();
        let role_vec = mean_pooling(&roles)?;

        let fuses = vecs.iter().map(|p| p.fused()).collect::<Vec<_>>();
        let fused_vec = mean_pooling(&fuses)?;
        Ok(Some(ParticipantEmbedding {
            name: name_vec,
            role: role_vec,
            fused: fused_vec,
        }))
    }
}
impl Embeddable for Participant {
    type EmbeddingGen = ParticipantEmbedding;
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
        Ok(ParticipantEmbedding {
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
    pub embedding: ParticipantEmbedding,
    pub participant: Participant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding::embedding_model::bge::BgeSmallZh;

    #[test]
    fn test_embed() {
        let participant = Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        };
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedding = participant.embed(&model).unwrap();
        assert_eq!(embedding.name.len(), 512);
        assert_eq!(embedding.role.len(), 512);
        assert_eq!(embedding.fused.len(), 512);
    }
}
