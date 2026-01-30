use crate::memory::embedding::{EmbeddingVec, situation::location::LocationEmbedding};

pub struct SitContextEmbedding {
    location_vec: LocationEmbedding,
    fused_participant_vec: EmbeddingVec,
    fused_emotion: EmbeddingVec,
    fused_sensory_data: EmbeddingVec,
}
