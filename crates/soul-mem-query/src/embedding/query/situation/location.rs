use crate::embedding::blend_weights::BlendWeights;
use crate::embedding::{mean_pooling, Embeddable, EmbeddingCalcResult, EmbeddingVec};
use crate::query::retrieve::LocationQueryUnit;

#[derive(Debug, Clone, PartialEq)]
pub struct LocationQueryUnitEmbedding {
    name: EmbeddingVec,
    coordinates: Option<EmbeddingVec>,
    pub blend_weights: BlendWeights,
}
impl LocationQueryUnitEmbedding {
    /// 公开构造（外部/测试构造用；blend_weights 取默认值）。
    pub fn new(name: EmbeddingVec, coordinates: Option<EmbeddingVec>) -> Self {
        Self {
            name,
            coordinates,
            blend_weights: BlendWeights::default(),
        }
    }

    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
    pub fn coordinates(&self) -> Option<&EmbeddingVec> {
        self.coordinates.as_ref()
    }
    pub fn set_blend_weights(&mut self, bw: &BlendWeights) {
        self.blend_weights = bw.clone();
    }
    pub fn mean_pooling(vecs: &[Self]) -> EmbeddingCalcResult<Option<Self>> {
        if vecs.is_empty() {
            return Ok(None);
        }
        let names = vecs.iter().map(|vec| vec.name()).collect::<Vec<_>>();
        let coordinates = vecs
            .iter()
            .filter_map(|vec| vec.coordinates())
            .collect::<Vec<_>>();

        let name_embedding = mean_pooling(&names)?;
        let coordinate_embedding = if coordinates.is_empty() {
            None
        } else {
            Some(mean_pooling(&coordinates)?)
        };

        Ok(Some(LocationQueryUnitEmbedding {
            name: name_embedding,
            coordinates: coordinate_embedding,
            blend_weights: BlendWeights::default(),
        }))
    }
}
#[cfg(test)]
impl LocationQueryUnitEmbedding {
    pub(crate) fn test_new(
        name: EmbeddingVec,
        coordinates: Option<EmbeddingVec>,
        blend_weights: BlendWeights,
    ) -> Self {
        Self {
            name,
            coordinates,
            blend_weights,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedLocationQueryUnit {
    pub embedding: LocationQueryUnitEmbedding,
    pub query: LocationQueryUnit,
}

impl Embeddable for LocationQueryUnit {
    type EmbeddingGen = LocationQueryUnitEmbedding;
    type EmbeddingFused = EmbedLocationQueryUnit;
    fn embed(
        &self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [name_vec] = model.infer_query_batch(&vec![self.name()])?.try_into().unwrap(); //SAFEUNWRAP: 此处长度必为1

        let coordinates_batch_vec = self
            .coordinates()
            .map(|coord| model.infer_query_batch(&vec![coord]))
            .transpose()?;

        let coordinates_vec = coordinates_batch_vec.and_then(|vec| vec.into_iter().next());

        Ok(LocationQueryUnitEmbedding {
            name: name_vec,
            coordinates: coordinates_vec,
            blend_weights: BlendWeights::default(),
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::embedding::EmbeddingModel,
    ) -> crate::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedLocationQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_query_unit_embedding_accessors() {
        let mut embedding = LocationQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            Some(EmbeddingVec::new(vec![2.0])),
            BlendWeights::default(),
        );
        assert_eq!(embedding.name().shape(), 1);
        assert_eq!(embedding.coordinates().unwrap().shape(), 1);

        let mut bw = BlendWeights::default();
        bw.tag = 0.8;
        embedding.set_blend_weights(&bw);
        assert_eq!(embedding.blend_weights.tag, 0.8);
    }

    #[test]
    fn test_location_query_unit_embedding_without_coordinates() {
        let embedding = LocationQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![1.0]),
            None,
            BlendWeights::default(),
        );
        assert_eq!(embedding.name().shape(), 1);
        assert!(embedding.coordinates().is_none());
    }

    #[test]
    fn test_location_query_unit_mean_pooling() {
        let l1 = LocationQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![1.0, 2.0]),
            Some(EmbeddingVec::new(vec![1.0, 2.0])),
            BlendWeights::default(),
        );
        let l2 = LocationQueryUnitEmbedding::test_new(
            EmbeddingVec::new(vec![3.0, 4.0]),
            None,
            BlendWeights::default(),
        );
        let pooled = LocationQueryUnitEmbedding::mean_pooling(&[l1, l2])
            .unwrap()
            .unwrap();
        assert_eq!(pooled.name().shape(), 2);
        assert!(pooled.coordinates().is_some());
    }

    #[test]
    fn test_location_query_unit_mean_pooling_empty() {
        assert!(LocationQueryUnitEmbedding::mean_pooling(&[]).unwrap().is_none());
    }
}
