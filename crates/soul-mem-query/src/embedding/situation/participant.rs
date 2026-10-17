use serde::{Deserialize, Serialize};

use crate::embedding::{
    Embeddable, EmbeddingCalcResult, EmbeddingVec, mean_pooling, raw_linear_blend,
};
use soul_mem_core::memory_note::situation_mem::Participant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipantEmbedding {
    pub name: EmbeddingVec,
    pub role: EmbeddingVec,
    pub fused: EmbeddingVec,
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
#[cfg(test)]
impl ParticipantEmbedding {
    pub(crate) fn test_new(name: EmbeddingVec, role: EmbeddingVec, fused: EmbeddingVec) -> Self {
        Self { name, role, fused }
    }
}
impl Embeddable for Participant {
    type EmbeddingGen = ParticipantEmbedding;
    type EmbeddingFused = EmbeddedParticipant;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let is_name_empty = self.name.trim().is_empty();
        let is_role_empty = self.role.trim().is_empty();

        let name_vec = if is_name_empty {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_with_chunk(self.name.as_str())?
        };

        let role_vec = if is_role_empty {
            EmbeddingVec::zero(model.dim())
        } else {
            model.infer_with_chunk(self.role.as_str())?
        };

        let fused_vec = match (is_name_empty, is_role_empty) {
            (true, true) => EmbeddingVec::zero(model.dim()),
            (false, true) => role_vec.clone(),
            (true, false) => name_vec.clone(),
            (false, false) => raw_linear_blend(&name_vec, &role_vec, 0.7).unwrap(), //SAFEUNWRAP: 此处两向量的维度必然相同
        };

        Ok(ParticipantEmbedding {
            name: name_vec,
            role: role_vec,
            fused: fused_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
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
    use crate::embedding::embedding_model::bge::BgeSmallZh;

    #[test]
    fn test_embed() {
        let participant = Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        };
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedding = participant.embed(&model).unwrap();
        assert_eq!(embedding.name.shape(), 512);
        assert_eq!(embedding.role.shape(), 512);
        assert_eq!(embedding.fused.shape(), 512);
    }

    #[test]
    fn test_participant_mean_pooling() {
        let model = BgeSmallZh::default_cpu().unwrap();

        let p1 = Participant {
            name: "张三".to_string(),
            role: "学生".to_string(),
        };
        let p2 = Participant {
            name: "李四".to_string(),
            role: "老师".to_string(),
        };
        let p3 = Participant {
            name: "王五".to_string(),
            role: "医生".to_string(),
        };

        let emb1 = p1.embed(&model).unwrap();
        let emb2 = p2.embed(&model).unwrap();
        let emb3 = p3.embed(&model).unwrap();

        let pooled = ParticipantEmbedding::mean_pooling(&[emb1, emb2, emb3])
            .unwrap()
            .unwrap();

        assert_eq!(pooled.name.shape(), 512);
        assert_eq!(pooled.role.shape(), 512);
        assert_eq!(pooled.fused.shape(), 512);
    }

    #[test]
    fn test_participant_mean_pooling_empty() {
        let result = ParticipantEmbedding::mean_pooling(&[]);
        assert!(result.unwrap().is_none());
    }
}
