use serde::{Deserialize, Serialize};

/// 语义记忆Link
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemMemLink {
    pub verb: String,
    pub intensity: f32,
    pub confidence: f32,
}

impl SemMemLink {
    pub fn new(verb: String, intensity: f32, confidence: f32) -> Self {
        Self {
            verb,
            intensity,
            confidence,
        }
    }
}
