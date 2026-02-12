use crate::memory::{
    embedding::{Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec},
    memory_note::situation_mem::Location,
};

#[derive(Debug, Clone, PartialEq)]
pub struct LocationEmbedding {
    name: EmbeddingVec,
    coordinates: EmbeddingVec,
}
impl LocationEmbedding {
    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
    pub fn coordinates(&self) -> &EmbeddingVec {
        &self.coordinates
    }
}

impl Embeddable for Location {
    type EmbeddingGen = LocationEmbedding;
    type EmbeddingFused = EmbeddedLocation;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        let [name_vec, coordinates_vec] = model
            .infer_batch(&vec![self.name.as_str(), self.coordinates.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处可以确定Vec的长度为2

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
    use crate::memory::embedding::embedding_model::bge::BgeSmallZh;

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
