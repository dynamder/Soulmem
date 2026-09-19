use serde::{Deserialize, Serialize};

/// 语义记忆Link
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemMemLink {
    pub verb: String,
    pub confidence: f32,
}

impl SemMemLink {
    pub fn new(verb: String, confidence: f32) -> Self {
        Self { verb, confidence }
    }
}

impl Default for SemMemLink {
    fn default() -> Self {
        Self {
            verb: String::new(),
            confidence: 1.0,
        }
    }
}
