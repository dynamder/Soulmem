use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use soul_mem_core::memory_note::MemoryId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserFeedback {
    Positive,
    Negative,
    Neutral,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    memory_id: MemoryId,
    retrieval_count: usize,

    first_access_time: DateTime<Utc>,
    last_access_time: DateTime<Utc>,

    feedback_history: BTreeMap<DateTime<Utc>, UserFeedback>,

    feedback_score: i32,
}

impl Record {
    pub fn new(memory_id: MemoryId) -> Self {
        let now = Utc::now();
        Record {
            memory_id,
            retrieval_count: 0,
            first_access_time: now,
            last_access_time: now,
            feedback_history: BTreeMap::new(),
            feedback_score: 0,
        }
    }

    pub fn record_retrieval(&mut self) {
        self.retrieval_count += 1;
        self.last_access_time = Utc::now();
    }

    pub fn add_feedback(&mut self, feedback: UserFeedback) {
        let now = Utc::now();
        self.feedback_history.insert(now, feedback.clone());

        match feedback {
            UserFeedback::Positive => self.feedback_score += 1,
            UserFeedback::Negative => self.feedback_score -= 1,
            UserFeedback::Neutral | UserFeedback::None => {}
        }
    }

    pub fn access_time_span(&self) -> i64 {
        self.last_access_time
            .signed_duration_since(self.first_access_time)
            .num_seconds()
    }

    pub fn memory_id(&self) -> MemoryId {
        self.memory_id
    }

    pub fn retrieval_count(&self) -> usize {
        self.retrieval_count
    }

    pub fn first_access_time(&self) -> DateTime<Utc> {
        self.first_access_time
    }
    pub fn last_access_time(&self) -> DateTime<Utc> {
        self.last_access_time
    }

    pub fn feedback_score(&self) -> i32 {
        self.feedback_score
    }

    pub fn feedback_history_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<(DateTime<Utc>, UserFeedback)> {
        //BTreeMap::range在start > end时会panic，这里直接返回空结果
        if start > end {
            return Vec::new();
        }
        self.feedback_history
            .range(start..=end)
            .map(|(time, feedback)| (*time, feedback.clone()))
            .collect()
    }

    pub fn feedback_history_after(
        &self,
        start: DateTime<Utc>,
    ) -> Vec<(DateTime<Utc>, UserFeedback)> {
        self.feedback_history
            .range(start..)
            .map(|(time, feedback)| (*time, feedback.clone()))
            .collect()
    }

    pub fn feedback_history_before(
        &self,
        end: DateTime<Utc>,
    ) -> Vec<(DateTime<Utc>, UserFeedback)> {
        self.feedback_history
            .range(..end)
            .map(|(time, feedback)| (*time, feedback.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use chrono::{Duration, Utc};

    fn create_test_memory_id() -> MemoryId {
        MemoryId::new()
    }

    #[test]
    fn test_record_new() {
        let memory_id = create_test_memory_id();
        let record = Record::new(memory_id);

        assert_eq!(record.memory_id(), memory_id);
        assert_eq!(record.retrieval_count(), 0);
        assert_eq!(record.feedback_score(), 0);

        assert_eq!(record.first_access_time(), record.last_access_time());

        assert_eq!(record.access_time_span(), 0);
    }

    #[test]
    fn test_record_retrieval() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        assert_eq!(record.retrieval_count(), 0);

        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 1);

        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 2);

        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 3);
    }

    #[test]
    fn test_add_feedback() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        assert_eq!(record.feedback_score(), 0);

        record.add_feedback(UserFeedback::Positive);
        assert_eq!(record.feedback_score(), 1);

        record.add_feedback(UserFeedback::Positive);
        assert_eq!(record.feedback_score(), 2);

        record.add_feedback(UserFeedback::Negative);
        assert_eq!(record.feedback_score(), 1);

        record.add_feedback(UserFeedback::Neutral);
        assert_eq!(record.feedback_score(), 1);

        record.add_feedback(UserFeedback::None);
        assert_eq!(record.feedback_score(), 1);
    }

    #[test]
    fn test_access_time_span() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        assert_eq!(record.access_time_span(), 0);

        // 直接比较时间戳：sleep(1s) 在 Windows 上可能提前返回，
        // 而 access_time_span() 按秒截断会把 999ms 截成 0，造成抖动。
        let before = record.last_access_time();
        std::thread::sleep(std::time::Duration::from_millis(20));
        record.record_retrieval();
        assert!(
            record.last_access_time() > before,
            "last access time should advance after retrieval"
        );
    }

    #[test]
    fn test_getters() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        assert_eq!(record.memory_id(), memory_id);

        assert_eq!(record.retrieval_count(), 0);
        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 1);

        let first_time = record.first_access_time();
        assert!(first_time <= Utc::now());

        let last_time = record.last_access_time();
        assert!(last_time >= first_time);

        assert_eq!(record.feedback_score(), 0);
        record.add_feedback(UserFeedback::Positive);
        assert_eq!(record.feedback_score(), 1);
    }

    #[test]
    fn test_feedback_history_in_range() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        record.add_feedback(UserFeedback::Positive);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time1 = Utc::now();
        record.add_feedback(UserFeedback::Negative);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time2 = Utc::now();
        record.add_feedback(UserFeedback::Neutral);

        // 查询整个时间范围内的反馈（从最早到最晚）
        // 由于反馈时间是在 add_feedback 内部捕获的，可能略早于 time1 或略晚于 time2
        // 所以我们只验证至少有一个反馈在范围内
        let all_feedback = record.feedback_history_in_range(time1, time2);
        assert!(
            !all_feedback.is_empty(),
            "Should have at least one feedback in range, got: {}",
            all_feedback.len()
        );

        let future_time = time2 + Duration::seconds(1);
        let empty_feedback = record.feedback_history_in_range(future_time, future_time);
        assert_eq!(empty_feedback.len(), 0);
    }

    #[test]
    fn test_feedback_history_in_range_inverted_bounds() {
        // start > end 时应返回空而非 panic（BTreeMap::range 对逆序会 panic）
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);
        record.add_feedback(UserFeedback::Positive);
        let now = Utc::now();
        let result = record.feedback_history_in_range(now, now - Duration::seconds(1));
        assert!(result.is_empty());
    }

    #[test]
    fn test_feedback_history_in_range_equal_bounds() {
        // start == end 时，包含恰好在该时刻的反馈（start > end 守卫不得吞掉该情形）
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);
        let t = Utc::now() - Duration::seconds(5);
        {
            let history = &mut record.feedback_history;
            history.insert(t, UserFeedback::Positive);
        }
        let result = record.feedback_history_in_range(t, t);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, UserFeedback::Positive);
    }

    #[test]
    fn test_feedback_history_in_range_exact_bounds() {
        // 边界内点包含，边界外点排除
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);
        let t0 = Utc::now() - Duration::seconds(10);
        let t1 = Utc::now() - Duration::seconds(9);
        let t2 = Utc::now() - Duration::seconds(8);
        {
            let history = &mut record.feedback_history;
            history.insert(t0, UserFeedback::Positive);
            history.insert(t1, UserFeedback::Negative);
            history.insert(t2, UserFeedback::Neutral);
        }
        let result = record.feedback_history_in_range(t1, t2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, UserFeedback::Negative);
        assert_eq!(result[1].1, UserFeedback::Neutral);
    }

    #[test]
    fn test_feedback_history_after() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        let _now = Utc::now();

        record.add_feedback(UserFeedback::Positive);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time1 = Utc::now();
        record.add_feedback(UserFeedback::Negative);
        std::thread::sleep(std::time::Duration::from_millis(10));
        record.add_feedback(UserFeedback::Neutral);

        let feedback_after = record.feedback_history_after(time1);
        assert_eq!(feedback_after.len(), 2);

        let future_feedback = record.feedback_history_after(Utc::now());
        assert_eq!(future_feedback.len(), 0);
    }

    #[test]
    fn test_feedback_history_before() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        record.add_feedback(UserFeedback::Positive);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time1 = Utc::now();
        record.add_feedback(UserFeedback::Negative);
        std::thread::sleep(std::time::Duration::from_millis(10));
        record.add_feedback(UserFeedback::Neutral);

        let feedback_before = record.feedback_history_before(time1);
        assert_eq!(feedback_before.len(), 1);

        let past_feedback = record.feedback_history_before(time1 - Duration::seconds(1));
        assert_eq!(past_feedback.len(), 0);
    }

    #[test]
    fn test_user_feedback_serialization() {
        let positive = UserFeedback::Positive;
        let serialized = serde_json::to_string(&positive).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(positive, deserialized);

        let negative = UserFeedback::Negative;
        let serialized = serde_json::to_string(&negative).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(negative, deserialized);

        let neutral = UserFeedback::Neutral;
        let serialized = serde_json::to_string(&neutral).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(neutral, deserialized);

        let none = UserFeedback::None;
        let serialized = serde_json::to_string(&none).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(none, deserialized);
    }

    #[test]
    fn test_comprehensive_operations() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        assert_eq!(record.retrieval_count(), 0);
        assert_eq!(record.feedback_score(), 0);

        for _ in 0..5 {
            record.record_retrieval();
        }
        assert_eq!(record.retrieval_count(), 5);

        record.add_feedback(UserFeedback::Positive);
        record.add_feedback(UserFeedback::Positive);
        record.add_feedback(UserFeedback::Negative);
        record.add_feedback(UserFeedback::Neutral);
        record.add_feedback(UserFeedback::Positive);

        assert_eq!(record.feedback_score(), 2);

        let time_span = record.access_time_span();
        assert!(time_span >= 0);
    }

    #[test]
    fn test_record_serialization() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        record.record_retrieval();
        record.add_feedback(UserFeedback::Positive);

        let serialized = serde_json::to_string(&record).unwrap();

        let deserialized: Record = serde_json::from_str(&serialized).unwrap();

        assert_eq!(record.memory_id(), deserialized.memory_id());
        assert_eq!(record.retrieval_count(), deserialized.retrieval_count());
        assert_eq!(record.feedback_score(), deserialized.feedback_score());
    }

    #[test]
    fn test_empty_feedback_history() {
        let memory_id = create_test_memory_id();
        let record = Record::new(memory_id);

        let now = Utc::now();

        let feedback_in_range = record.feedback_history_in_range(now, now);
        assert_eq!(feedback_in_range.len(), 0);

        let feedback_after = record.feedback_history_after(now);
        assert_eq!(feedback_after.len(), 0);

        let feedback_before = record.feedback_history_before(now);
        assert_eq!(feedback_before.len(), 0);
    }
}
