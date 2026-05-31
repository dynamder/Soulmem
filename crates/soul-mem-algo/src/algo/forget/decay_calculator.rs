use super::decay_state::DecayState;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use jieba_rs::Jieba;
use parking_lot::RwLock as ParkRwLock;
use soul_mem_core::memory_note::MemoryId;
use std::{io::SeekFrom::Current, sync::Arc};

const MASK_WORD: &str = " [masked] ";

pub struct DecayCalculator {
    states: DashMap<MemoryId, ParkRwLock<DecayState>>,
    jieba: Arc<Jieba>,
}

impl DecayCalculator {
    pub fn new() -> Self {
        let mut jieba = Jieba::new();
        _ = jieba.add_word(MASK_WORD, Some(114514), Some(""));
        Self {
            states: DashMap::new(),
            jieba: Arc::new(jieba),
        }
    }
    pub fn add_node(&self, memory_id: MemoryId) {
        self.states
            .insert(memory_id, ParkRwLock::new(DecayState::default()));
    }
    pub fn decay(&self, memory_id: MemoryId, current_time: DateTime<Utc>, content: &mut String) {
        if let Some(state) = self.states.get(&memory_id) {
            state.write().forget(current_time, content, &self.jieba);
        }
    }
}
