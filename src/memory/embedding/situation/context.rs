use crate::memory::embedding::{EmbeddingVec, situation::location::SitLocationEmbedding};

pub struct SitContextEmbedding {
    location_vec: SitLocationEmbedding,
    fused_participant_vec: EmbeddingVec,
    fused_emotion: EmbeddingVec,
    fused_sensory_data: EmbeddingVec,
}