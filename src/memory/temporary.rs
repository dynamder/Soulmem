use std::collections::HashMap;
use chrono::DateTime;
use petgraph::graph::NodeIndex;
use crate::memory::share::NoteRecord;
//临时记忆，在对话过程中短期持续，根据被提及的次数，和共激活的记忆，决定是否转为长期，是否建立联系
//临时记忆的MemoryNote，links字段应当为空
//临时记忆自动关联到当前的焦点任务


#[allow(dead_code)]
#[derive(Debug,Clone)]
pub enum MemorySource {
    Dialogue(String), //来自对话
    Inference(String), //来自推理
    Observation(String) //来自观察
}
#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct TemporaryNoteRecord {
    pub index: NodeIndex,
    pub source: MemorySource,
    activation_count: u32,
    activation_history: HashMap<NodeIndex, u32>,
    create_timestamp: DateTime<chrono::Utc>,
    last_accessed: DateTime<chrono::Utc>,
}
impl TemporaryNoteRecord {
    #[allow(dead_code)]
    pub fn new(index: NodeIndex, source: MemorySource) -> Self {
        let current_time = chrono::Utc::now();
        Self {
            index,
            source,
            activation_count: 1,
            activation_history: HashMap::new(),
            create_timestamp: current_time,
            last_accessed: current_time,
        }
    }
}
impl NoteRecord for TemporaryNoteRecord {
    fn activation_count(&self) -> u32 {
        self.activation_count
    }
    fn record_activation(&mut self, indexes: &[NodeIndex]) {
        indexes.iter().for_each(|index| {
            if let Some(activations) = self.activation_history.get_mut(index) {
                *activations += 1;
            } else {
                self.activation_history.insert(*index, 1);
            }
        });
        self.activation_count += 1;
        self.last_accessed = chrono::Utc::now();
    }
    fn activation_history(&self) -> &HashMap<NodeIndex, u32> {
        &self.activation_history
    }
}
#[allow(unused)]
#[derive(Debug,Clone)]
pub struct TemporaryMemory {
    record_map: HashMap<NodeIndex, TemporaryNoteRecord>,
}
#[allow(unused)]
impl TemporaryMemory {
    pub fn new() -> Self {
        Self {
            record_map: HashMap::new(),
        }
    }
    pub fn add_temp_memory(&mut self, temp_record: TemporaryNoteRecord) {
        let index = temp_record.index;
        self.record_map.insert(index, temp_record);
    }
    pub fn get_temp_memory(&self, index: NodeIndex) -> Option<&TemporaryNoteRecord> {
        self.record_map.get(&index)
    }
    pub fn get_temp_memory_mut(&mut self, index: NodeIndex) -> Option<&mut TemporaryNoteRecord> {
        self.record_map.get_mut(&index)
    }
    pub fn remove_temp_memory(&mut self, index: NodeIndex) -> Option<TemporaryNoteRecord> {
        self.record_map.remove(&index)
    }
    pub fn get_all(&self) -> Vec<&TemporaryNoteRecord> {
        self.record_map.values().collect()
    }
    pub fn get_map(&self) -> &HashMap<NodeIndex, TemporaryNoteRecord> {
        &self.record_map
    }
    pub fn get_map_mut(&mut self) -> &mut HashMap<NodeIndex, TemporaryNoteRecord> {
        &mut self.record_map
    }
    pub fn contain(&self, index: NodeIndex) -> bool {
        self.record_map.contains_key(&index)
    }
    pub fn filter_temp_memory<F>(&self, filter: F) -> Vec<&TemporaryNoteRecord>
    where
        F: Fn(&TemporaryNoteRecord) -> bool {
            self.record_map.values().filter(|record| filter(record)).collect()
    }
}