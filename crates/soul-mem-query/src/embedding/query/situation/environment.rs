use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{Embeddable, EmbeddingVec};
use crate::query::retrieve::EnvironmentQueryUnit;

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentQueryUnitEmbedding {
    atmosphere: Option<EmbeddingVec>,
    tone: Option<EmbeddingVec>,
    pub blend_weights: BlendWeights,
}
impl EnvironmentQueryUnitEmbedding {
    /// 公开构造（外部/测试构造用；blend_weights 取默认值）。
    pub fn new(atmosphere: Option<EmbeddingVec>, tone: Option<EmbeddingVec>) -> Self {
        Self {
            atmosphere,
            tone,
            blend_weights: BlendWeights::default(),
        }
    }

    pub fn atmosphere(&self) -> Option<&EmbeddingVec> {
        self.atmosphere.as_ref()
    }
    pub fn tone(&self) -> Option<&EmbeddingVec> {
        self.tone.as_ref()
    }
    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
    }
    /// 解构取所有权：atmosphere 与 tone（移动而非克隆）。
    pub fn into_parts(self) -> (Option<EmbeddingVec>, Option<EmbeddingVec>) {
        let Self {
            atmosphere,
            tone,
            blend_weights: _,
        } = self;
        (atmosphere, tone)
    }
}
#[cfg(test)]
impl EnvironmentQueryUnitEmbedding {
    pub(crate) fn test_new(
        atmosphere: Option<EmbeddingVec>,
        tone: Option<EmbeddingVec>,
        blend_weights: BlendWeights,
    ) -> Self {
        Self {
            atmosphere,
            tone,
            blend_weights,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedEnvironmentQueryUnit {
    pub embedding: EnvironmentQueryUnitEmbedding,
    pub query: EnvironmentQueryUnit,
}

impl Embeddable for EnvironmentQueryUnit {
    type EmbeddingGen = EnvironmentQueryUnitEmbedding;
    type EmbeddingFused = EmbedEnvironmentQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let atmosphere_batch_vec = self
            .atmosphere()
            .map(|atmosphere| model.infer_query_batch(&vec![atmosphere]))
            .transpose()?;

        let atmosphere_vec = atmosphere_batch_vec.and_then(|vec| vec.into_iter().next());

        let tone_batch_vec = self
            .tone()
            .map(|tone| model.infer_query_batch(&vec![tone]))
            .transpose()?;

        let tone_vec = tone_batch_vec.and_then(|vec| vec.into_iter().next());

        Ok(EnvironmentQueryUnitEmbedding {
            atmosphere: atmosphere_vec,
            tone: tone_vec,
            blend_weights: BlendWeights::default(),
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedEnvironmentQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_query_unit_embedding_accessors() {
        let mut embedding = EnvironmentQueryUnitEmbedding::test_new(
            Some(EmbeddingVec::new(vec![1.0])),
            Some(EmbeddingVec::new(vec![2.0])),
            BlendWeights::default(),
        );
        assert_eq!(embedding.atmosphere().unwrap().shape(), 1);
        assert_eq!(embedding.tone().unwrap().shape(), 1);

        let mut bw = BlendWeights::default();
        bw.tag = 0.8;
        embedding.set_blend_weights(&bw);
        assert_eq!(embedding.blend_weights.tag, 0.8);
    }

    #[test]
    fn test_environment_query_unit_embedding_none() {
        let embedding = EnvironmentQueryUnitEmbedding::test_new(None, None, BlendWeights::default());
        assert!(embedding.atmosphere().is_none());
        assert!(embedding.tone().is_none());
    }

    #[test]
    fn test_into_parts_moves_fields() {
        let atmosphere = EmbeddingVec::new(vec![0.5, 0.5]);
        let tone = EmbeddingVec::new(vec![0.2, 0.8]);
        let env = EnvironmentQueryUnitEmbedding::new(Some(atmosphere.clone()), Some(tone.clone()));
        let (a, t) = env.into_parts();
        assert_eq!(a, Some(atmosphere), "atmosphere 应移动而非克隆");
        assert_eq!(t, Some(tone));
    }

    #[test]
    fn test_into_parts_none_fields() {
        let env = EnvironmentQueryUnitEmbedding::new(None, None);
        let (a, t) = env.into_parts();
        assert!(a.is_none());
        assert!(t.is_none());
    }
}
