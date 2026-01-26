use crate::memory::{
    embedding::{Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec},
    memory_note::situation_mem::Location,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SitLocationEmbedding {
    name: EmbeddingVec,
    coordinates: EmbeddingVec,
}
impl SitLocationEmbedding {
    pub fn name(&self) -> &EmbeddingVec {
        &self.name
    }
    pub fn coordinates(&self) -> &EmbeddingVec {
        &self.coordinates
    }
}

impl Embeddable for Location {
    type EmbeddingGen = SitLocationEmbedding;
    type EmbeddingFused = EmbeddedLocation;
    fn embed(&self, model: &dyn EmbeddingModel) -> EmbeddingGenResult<Self::EmbeddingGen> {
        let [name_vec, coordinates_vec] = model
            .infer_batch(&vec![self.name.as_str(), self.coordinates.as_str()])?
            .try_into()
            .unwrap(); //SAFEUNWRAP: 此处可以确定Vec的长度为2

        Ok(SitLocationEmbedding {
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
    pub embedding: SitLocationEmbedding,
    pub location: Location,
}
