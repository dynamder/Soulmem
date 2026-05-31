use chrono::{DateTime, Utc};
use jieba_rs::Jieba;
use parking_lot::RwLock as ParkRwLock;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::thread_rng;
use soul_mem_core::memory_note::MemoryId;

const MASK_WORD: &str = " [masked] ";

#[derive(Debug)]
pub struct DecayState {
    /// 缓存的衰落值，None 表示需要重新计算
    intensity: f32,
    /// 激活次数快照（用于检测是否需要重算）
    activation_count: u64,
    last_decay_time: DateTime<Utc>,
    pub decay_factor: f32,
    pub active_factor: f32,
    pub is_masked: bool,
}

impl DecayState {
    pub fn new(time: DateTime<Utc>) -> Self {
        Self {
            intensity: 1.0,
            activation_count: 0,
            last_decay_time: time,
            decay_factor: 1.0,
            active_factor: 1.0,
            is_masked: false,
        }
    }
    pub fn with_decay_factor(mut self, decay_factor: f32) -> Self {
        self.decay_factor = decay_factor;
        self
    }
    pub fn with_active_factor(mut self, active_factor: f32) -> Self {
        self.active_factor = active_factor;
        self
    }
    pub fn with_masked(mut self, is_masked: bool) -> Self {
        self.is_masked = is_masked;
        self
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
    pub fn activation_count(&self) -> u64 {
        self.activation_count
    }

    pub fn decay(&mut self, current_time: DateTime<Utc>) -> f32 {
        let old_intensity = self.intensity;
        let elapsed = (current_time - self.last_decay_time).num_hours() as f32;
        let factor = self.decay_factor / (1.0 + self.active_factor * self.activation_count as f32);
        self.intensity = self.intensity * (elapsed * factor).exp();
        self.last_decay_time = current_time;
        self.intensity - old_intensity
    }
    pub fn mask(&mut self, content: &mut String, jieba: &Jieba, intensity_diff: f32) {
        let mut words = jieba.cut(content, true);
        let n = (intensity_diff * words.len() as f32) as usize;
        let mut rng = thread_rng();
        if !self.is_masked {
            let indices: Vec<usize> = (0..words.len())
                .choose_multiple(&mut rng, n)
                .into_iter()
                .collect();
            for &i in &indices {
                words[i] = MASK_WORD;
            }
        } else {
            let indices: Vec<usize> = words
                .iter()
                .enumerate()
                .filter(|&(_, &s)| s != MASK_WORD)
                .map(|(i, _)| i)
                .choose_multiple(&mut rng, n)
                .into_iter()
                .collect();
            for &i in &indices {
                words[i] = MASK_WORD;
            }
            self.is_masked = true;
        }
        *content = words.concat();
    }
    pub fn forget(&mut self, current_time: DateTime<Utc>, content: &mut String, jieba: &Jieba) {
        let intensity_diff = self.decay(current_time);
        if self.intensity <= 0.9 {
            self.mask(content, jieba, intensity_diff)
        }
        self.activation_count += 1;
    }
}

impl Default for DecayState {
    fn default() -> Self {
        Self {
            is_masked: false,
            intensity: 1.0,
            activation_count: 0,
            last_decay_time: Utc::now(),
            decay_factor: 1.0,
            active_factor: 1.0,
        }
    }
}
