use serde::{Deserialize, Serialize};

use crate::embedding::{Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec};
use soul_mem_core::memory_note::situation_mem::Location;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocationEmbedding {
    pub name: EmbeddingVec,
    pub coordinates: EmbeddingVec,
}
impl LocationEmbedding {
    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
    pub fn coordinates(&self) -> &EmbeddingVec {
        &self.coordinates
    }
}

#[cfg(test)]
impl LocationEmbedding {
    pub(crate) fn test_new(name: EmbeddingVec, coordinates: EmbeddingVec) -> Self {
        Self { name, coordinates }
    }
}

impl Embeddable for Location {
    type EmbeddingGen = LocationEmbedding;
    type EmbeddingFused = EmbeddedLocation;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        let is_name_empty = self.name.trim().is_empty();
        let is_coordinates_empty = self.coordinates.trim().is_empty();
        let (name_vec, coordinates_vec) = match (is_name_empty, is_coordinates_empty) {
            (true, true) => (
                EmbeddingVec::zero(model.dim()),
                EmbeddingVec::zero(model.dim()),
            ),
            (false, false) => {
                let [name_vec, coordinates_vec] = model
                    .infer_batch(&[self.name.as_str(), self.coordinates.as_str()])?
                    .try_into()
                    .unwrap(); //SAFEUNWRAP: 此处可以确定Vec的长度为2
                (name_vec, coordinates_vec)
            }
            (false, true) => {
                let name_vec = model.infer_with_chunk(self.name.as_str())?;
                (name_vec, EmbeddingVec::zero(model.dim()))
            }
            (true, false) => {
                let coordinates_vec = model.infer_with_chunk(self.coordinates.as_str())?;
                (EmbeddingVec::zero(model.dim()), coordinates_vec)
            }
        };

        Ok(LocationEmbedding {
            name: name_vec,
            coordinates: coordinates_vec,
        })
    }
    fn embed_and_fuse(
        self,
        model: &dyn EmbeddingModel,
    ) -> EmbeddingGenResult<Self::EmbeddingFused> {
        Ok(EmbeddedLocation {
            embedding: self.embed(model)?,
            location: self,
        })
    }
}

pub struct EmbeddedLocation {
    pub embedding: LocationEmbedding,
    pub location: Location,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::embedding_model::bge::BgeSmallZh;

    #[test]
    fn test_embed() {
        let location = Location {
            name: "北京".to_string(),
            coordinates: "亚洲，中国".to_string(),
        };
        let model = BgeSmallZh::default_cpu().unwrap();
        let embedding = location.embed(&model).unwrap();
        assert_eq!(embedding.name.shape(), 512);
        assert_eq!(embedding.coordinates.shape(), 512);
    }
}
