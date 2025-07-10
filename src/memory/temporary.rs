use std::collections::HashMap;
use chrono::DateTime;
use petgraph::graph::NodeIndex;
use crate::memory::NodeRefId;
use crate::memory::share::NoteRecord;
//临时记忆，在对话过程中短期持续，根据被提及的次数，和共激活的记忆，决定是否转为长期，是否建立联系
//临时记忆的MemoryNote，links字段应当为空
//临时记忆自动关联到当前的焦点任务

///记忆来源，有不同的初始巩固权重
#[allow(dead_code)]
#[derive(Debug,Clone,PartialEq)]
pub enum MemorySource {
    Dialogue(String), //来自对话
    Inference(String), //来自推理
    Observation(String) //来自观察
}
#[allow(dead_code)]
impl MemorySource {
    pub fn as_dialogue(&self) -> Option<&str> {
        match self {
            MemorySource::Dialogue(s) => Some(s),
            _ => None
        }
    }
    pub fn as_inference(&self) -> Option<&str> {
        match self {
            MemorySource::Inference(s) => Some(s),
            _ => None
        }
    }
    pub fn as_observation(&self) -> Option<&str> {
        match self {
            MemorySource::Observation(s) => Some(s),
            _ => None
        }
    }
}
///临时记忆激活记录
#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct TemporaryNoteRecord {
    pub note_id: NodeRefId,
    pub source: MemorySource,
    activation_count: u32,
    activation_history: HashMap<NodeRefId, u32>,
    create_timestamp: DateTime<chrono::Utc>,
    last_accessed: DateTime<chrono::Utc>,
}
impl TemporaryNoteRecord {
    #[allow(dead_code)]
    pub fn new(note_id: NodeRefId, source: MemorySource) -> Self {
        let current_time = chrono::Utc::now();
        Self {
            note_id,
            source,
            activation_count: 1,
            activation_history: HashMap::new(),
            create_timestamp: current_time,
            last_accessed: current_time,
        }
    }
    #[allow(dead_code)]
    pub fn last_accessed_at(&self) -> &DateTime<chrono::Utc> {
        &self.last_accessed
    }
    #[allow(dead_code)]
    pub fn created_at(&self) -> &DateTime<chrono::Utc> {
        &self.create_timestamp
    }
}
impl NoteRecord for TemporaryNoteRecord {
    fn activation_count(&self) -> u32 {
        self.activation_count
    }
    fn record_activation(&mut self, indexes: &[NodeRefId]) {
        indexes.iter().for_each(|index| {
            if let Some(activations) = self.activation_history.get_mut(index) {
                *activations += 1;
            } else {
                self.activation_history.insert(index.clone(), 1);
            }
        });
        self.activation_count += 1;
        self.last_accessed = chrono::Utc::now();
    }
    fn activation_history(&self) -> &HashMap<NodeRefId, u32> {
        &self.activation_history
    }
}
#[allow(unused)]
#[derive(Debug,Clone)]
pub struct TemporaryMemory {
    record_map: HashMap<NodeRefId, TemporaryNoteRecord>,
}
#[allow(unused)]
impl TemporaryMemory {
    pub fn new() -> Self {
        Self {
            record_map: HashMap::new(),
        }
    }
    pub fn add_temp_memory(&mut self, temp_record: TemporaryNoteRecord) {
        let id = temp_record.note_id.clone();
        self.record_map.insert(id, temp_record);
    }
    pub fn get_temp_memory(&self, id: NodeRefId) -> Option<&TemporaryNoteRecord> {
        self.record_map.get(&id)
    }
    pub fn get_temp_memory_mut(&mut self, id: NodeRefId) -> Option<&mut TemporaryNoteRecord> {
        self.record_map.get_mut(&id)
    }
    pub fn remove_temp_memory(&mut self, id: NodeRefId) -> Option<TemporaryNoteRecord> {
        self.record_map.remove(&id)
    }
    pub fn get_all(&self) -> Vec<&TemporaryNoteRecord> {
        self.record_map.values().collect()
    }
    pub fn get_map(&self) -> &HashMap<NodeRefId, TemporaryNoteRecord> {
        &self.record_map
    }
    pub fn get_map_mut(&mut self) -> &mut HashMap<NodeRefId, TemporaryNoteRecord> {
        &mut self.record_map
    }
    pub fn contain(&self, id: NodeRefId) -> bool {
        self.record_map.contains_key(&id)
    }
    ///按给定条件函数筛选临时记忆
    pub fn filter_temp_memory<F>(&self, filter: F) -> Vec<&TemporaryNoteRecord>
    where
        F: Fn(&TemporaryNoteRecord) -> bool {
            self.record_map.values().filter(|record| filter(record)).collect()
    }
}

mod test {
    #[allow(unused)]
    use std::thread::sleep;
    #[allow(unused)]
    use super::*;
    #[test]
    fn test_note_record_new() {
        let rec = TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string()));
        assert_eq!(rec.activation_count(), 1);
        assert_eq!(rec.activation_history().len(), 0);
    }
    #[test]
    fn test_note_record_record_activation() { 
        let mut rec = TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string()));
        let created_at = rec.created_at().clone();
        sleep(std::time::Duration::from_millis(2000));
        rec.record_activation(&[NodeRefId::new("1"), NodeRefId::new("2")]);
        assert_eq!(rec.activation_count(), 2);
        assert_eq!(rec.activation_history().len(), 2);
        assert_ne!(rec.last_accessed_at(), &created_at)
    }
    #[test]
    fn test_add_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string())));
        assert!(mem.contain(NodeRefId::new("0")));
    }
    #[test]
    fn test_get_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string())));
        assert_eq!(mem.get_temp_memory(NodeRefId::new("0")).unwrap().source, MemorySource::Dialogue("test".to_string()));
    }
    #[test]
    fn test_get_all_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string())));
        assert_eq!(mem.get_all().len(), 1);
        assert_eq!(mem.get_all()[0].source, MemorySource::Dialogue("test".to_string()));
    }
    #[test]
    fn test_contain_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string())));
        assert!(mem.contain(NodeRefId::new("0")));
    }
    #[test]
    fn test_filter_none_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string())));
        assert_eq!(mem.filter_temp_memory(|_| false).len(), 0);
    }
    #[test]
    fn test_filter_all_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test".to_string())));
        assert_eq!(mem.filter_temp_memory(|_| true).len(), 1);
    }
    #[test]
    fn test_filter_some_temp_mem() {
        let mut mem = TemporaryMemory::new();
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("0"), MemorySource::Dialogue("test0".to_string())));
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("1"), MemorySource::Dialogue("test1".to_string())));
        mem.add_temp_memory(TemporaryNoteRecord::new(NodeRefId::new("2"), MemorySource::Dialogue("tst2".to_string())));
        assert_eq!(mem.filter_temp_memory(|record| record.source.as_dialogue().unwrap().to_string().contains("test")).len(), 2);
    }
}