use serde::{Deserialize, Serialize};

//语义记忆Link
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemMemLink {
    verb: String,
    intensity: f32,
    confidence: f32,
}

impl SemMemLink {
    pub fn new(verb: String, intensity: f32, confidence: f32) -> Self {
        SemMemLink {
            verb,
            intensity,
            confidence,
        }
    }
    //谓词
    pub fn get_verb(&self) -> &str {
        &self.verb
    }

    pub fn set_verb(&mut self, verb: String) {
        self.verb = verb;
    }

    //连接强度
    pub fn get_intensity(&self) -> f32 {
        self.intensity
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }

    //置信度
    pub fn get_confidence(&self) -> f32 {
        self.confidence
    }

    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = confidence;
    }
}
