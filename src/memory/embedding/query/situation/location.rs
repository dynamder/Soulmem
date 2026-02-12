use crate::memory::{
    embedding::{Embeddable, EmbeddingCalcResult, EmbeddingVec, mean_pooling},
    query::retrieve::LocationQueryUnit,
};

#[derive(Debug, Clone, PartialEq)]
pub struct LocationQueryUnitEmbedding {
    name: EmbeddingVec,
    coordinates: Option<EmbeddingVec>,
}
impl LocationQueryUnitEmbedding {
    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
    pub fn coordinates(&self) -> Option<&EmbeddingVec> {
        self.coordinates.as_ref()
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
        }))
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
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingGen> {
        let [name_vec] = model.infer_batch(&vec![self.name()])?.try_into().unwrap(); //SAFEUNWRAP: 此处长度必为1

        let coordinates_batch_vec = self
            .coordinates()
            .map(|coord| model.infer_batch(&vec![coord]))
            .transpose()?;

        let coordinates_vec = coordinates_batch_vec
            .map(|vec| vec.into_iter().next())
            .flatten();

        Ok(LocationQueryUnitEmbedding {
            name: name_vec,
            coordinates: coordinates_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn crate::memory::embedding::EmbeddingModel,
    ) -> crate::memory::embedding::EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbedLocationQueryUnit {
            embedding: self.embed(model)?,
            query: self,
        })
    }
}
