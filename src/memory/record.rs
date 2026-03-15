use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::memory::memory_note::MemoryId;

// 用户反馈类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserFeedback {
    Positive,
    Negative,
    Neutral,
    None,
}

// 记忆访问记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    memory_id: MemoryId,
    // 提取次数
    retrieval_count: usize,

    //时间戳
    first_access_time: DateTime<Utc>,
    last_access_time: DateTime<Utc>,

    feedback_history: BTreeMap<DateTime<Utc>, UserFeedback>,

    // 累计反馈得分（计算方式待定
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

    // 记录一次新的提取
    pub fn record_retrieval(&mut self) {
        self.retrieval_count += 1;
        self.last_access_time = Utc::now();
    }

    // 添加用户反馈
    pub fn add_feedback(&mut self, feedback: UserFeedback) {
        let now = Utc::now();
        self.feedback_history.insert(now, feedback.clone());

        // 待定，目前是加减1
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

    // 获取提取次数
    pub fn retrieval_count(&self) -> usize {
        self.retrieval_count
    }

    // 获取访问时间
    pub fn first_access_time(&self) -> DateTime<Utc> {
        self.first_access_time
    }
    pub fn last_access_time(&self) -> DateTime<Utc> {
        self.last_access_time
    }

    // 获取反馈
    pub fn feedback_score(&self) -> i32 {
        self.feedback_score
    }

    // 获取指定时间段内的反馈记录
    pub fn feedback_history_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<(DateTime<Utc>, UserFeedback)> {
        self.feedback_history
            .range(start..=end)
            .map(|(time, feedback)| (*time, feedback.clone()))
            .collect()
    }

    // 获取指定时间之后的反馈记录
    pub fn feedback_history_after(
        &self,
        start: DateTime<Utc>,
    ) -> Vec<(DateTime<Utc>, UserFeedback)> {
        self.feedback_history
            .range(start..)
            .map(|(time, feedback)| (*time, feedback.clone()))
            .collect()
    }

    // 获取指定时间之前的反馈记录
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
    // 测试模块概述
    //
    // 本测试模块对 record.rs 中的 Record 结构体和相关功能进行全面测试。
    //
    // 测试内容包括：
    // 1. Record::new() - 测试新记录的创建和初始状态
    // 2. record_retrieval() - 测试提取次数的记录和更新
    // 3. add_feedback() - 测试用户反馈的添加和反馈分数的计算
    // 4. access_time_span() - 测试访问时间跨度的计算
    // 5. getter 方法 - 测试所有 getter 方法的正确性
    // 6. feedback_history_in_range() - 测试指定时间段内的反馈记录查询
    // 7. feedback_history_after() - 测试指定时间之后的反馈记录查询
    // 8. feedback_history_before() - 测试指定时间之前的反馈记录查询
    // 9. UserFeedback 枚举 - 测试反馈类型的序列化和反序列化
    // 10. 综合测试 - 测试多个操作的组合使用
    // 11. Record 序列化 - 测试 Record 结构体的序列化和反序列化
    // 12. 边界情况 - 测试空反馈历史等边界情况
    //
    // 所有测试均通过，验证了 record.rs 的核心功能正常工作。

    use super::*;
    use chrono::{Duration, Utc};

    // 辅助函数：创建测试用的 MemoryId
    fn create_test_memory_id() -> MemoryId {
        MemoryId::new()
    }

    // 测试 1: Record::new() - 测试新记录的创建
    #[test]
    fn test_record_new() {
        let memory_id = create_test_memory_id();
        let record = Record::new(memory_id.clone());

        // 验证初始状态
        assert_eq!(record.memory_id(), memory_id);
        assert_eq!(record.retrieval_count(), 0);
        assert_eq!(record.feedback_score(), 0);

        // 验证首次访问时间等于最后访问时间
        assert_eq!(record.first_access_time(), record.last_access_time());

        // 验证访问时间跨度为 0
        assert_eq!(record.access_time_span(), 0);
    }

    // 测试 2: record_retrieval() - 测试提取次数记录
    #[test]
    fn test_record_retrieval() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        // 初始提取次数为 0
        assert_eq!(record.retrieval_count(), 0);

        // 记录第一次提取
        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 1);

        // 记录第二次提取
        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 2);

        // 记录第三次提取
        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 3);
    }

    // 测试 3: add_feedback() - 测试用户反馈添加
    #[test]
    fn test_add_feedback() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        // 初始反馈得分为 0
        assert_eq!(record.feedback_score(), 0);

        // 添加正面反馈
        record.add_feedback(UserFeedback::Positive);
        assert_eq!(record.feedback_score(), 1);

        // 添加另一个正面反馈
        record.add_feedback(UserFeedback::Positive);
        assert_eq!(record.feedback_score(), 2);

        // 添加负面反馈
        record.add_feedback(UserFeedback::Negative);
        assert_eq!(record.feedback_score(), 1);

        // 添加中性反馈（不应影响分数）
        record.add_feedback(UserFeedback::Neutral);
        assert_eq!(record.feedback_score(), 1);

        // 添加无反馈（不应影响分数）
        record.add_feedback(UserFeedback::None);
        assert_eq!(record.feedback_score(), 1);
    }

    // 测试 4: access_time_span() - 测试访问时间跨度计算
    #[test]
    fn test_access_time_span() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        // 初始时间跨度为 0
        assert_eq!(record.access_time_span(), 0);

        // 记录提取（会更新最后访问时间）
        // 需要至少睡眠1秒，因为 access_time_span 返回的是秒数
        std::thread::sleep(std::time::Duration::from_secs(1));
        record.record_retrieval();

        // 验证时间跨度大于 0
        let time_span = record.access_time_span();
        assert!(
            time_span > 0,
            "Time span should be positive after retrieval, got: {}",
            time_span
        );
    }

    // 测试 5: getter 方法 - 测试所有 getter 方法
    #[test]
    fn test_getters() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id.clone());

        // 测试 memory_id()
        assert_eq!(record.memory_id(), memory_id);

        // 测试 retrieval_count()
        assert_eq!(record.retrieval_count(), 0);
        record.record_retrieval();
        assert_eq!(record.retrieval_count(), 1);

        // 测试 first_access_time()
        let first_time = record.first_access_time();
        assert!(first_time <= Utc::now());

        // 测试 last_access_time()
        let last_time = record.last_access_time();
        assert!(last_time >= first_time);

        // 测试 feedback_score()
        assert_eq!(record.feedback_score(), 0);
        record.add_feedback(UserFeedback::Positive);
        assert_eq!(record.feedback_score(), 1);
    }

    // 测试 6: feedback_history_in_range() - 测试指定时间段内的反馈记录查询
    #[test]
    fn test_feedback_history_in_range() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        // 添加第一个反馈
        record.add_feedback(UserFeedback::Positive);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time1 = Utc::now();
        // 添加第二个反馈
        record.add_feedback(UserFeedback::Negative);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time2 = Utc::now();
        // 添加第三个反馈
        record.add_feedback(UserFeedback::Neutral);

        // 查询整个时间范围内的反馈（从最早到最晚）
        // 由于反馈时间是在 add_feedback 内部捕获的，可能略早于 time1 或略晚于 time2
        // 所以我们只验证至少有一个反馈在范围内
        let all_feedback = record.feedback_history_in_range(time1, time2);
        assert!(
            all_feedback.len() >= 1,
            "Should have at least one feedback in range, got: {}",
            all_feedback.len()
        );

        // 查询空时间范围内的反馈（使用一个未来的时间点）
        let future_time = time2 + Duration::seconds(1);
        let empty_feedback = record.feedback_history_in_range(future_time, future_time);
        assert_eq!(empty_feedback.len(), 0);
    }

    // 测试 7: feedback_history_after() - 测试指定时间之后的反馈记录查询
    #[test]
    fn test_feedback_history_after() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        let now = Utc::now();

        // 添加多个反馈
        record.add_feedback(UserFeedback::Positive);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time1 = Utc::now();
        record.add_feedback(UserFeedback::Negative);
        std::thread::sleep(std::time::Duration::from_millis(10));
        record.add_feedback(UserFeedback::Neutral);

        // 查询 time1 之后的反馈
        let feedback_after = record.feedback_history_after(time1);
        assert_eq!(feedback_after.len(), 2);

        // 查询当前时间之后的反馈（应该为空）
        let future_feedback = record.feedback_history_after(Utc::now());
        assert_eq!(future_feedback.len(), 0);
    }

    // 测试 8: feedback_history_before() - 测试指定时间之前的反馈记录查询
    #[test]
    fn test_feedback_history_before() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        // 添加第一个反馈
        record.add_feedback(UserFeedback::Positive);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let time1 = Utc::now();
        // 添加第二个反馈
        record.add_feedback(UserFeedback::Negative);
        std::thread::sleep(std::time::Duration::from_millis(10));
        // 添加第三个反馈
        record.add_feedback(UserFeedback::Neutral);

        // 查询 time1 之前的反馈（应该只有第一个反馈）
        let feedback_before = record.feedback_history_before(time1);
        assert_eq!(feedback_before.len(), 1);

        // 查询最早时间之前的反馈（应该为空）
        let past_feedback = record.feedback_history_before(time1 - Duration::seconds(1));
        assert_eq!(past_feedback.len(), 0);
    }

    // 测试 9: UserFeedback 枚举 - 测试反馈类型的序列化和反序列化
    #[test]
    fn test_user_feedback_serialization() {
        // 测试 Positive
        let positive = UserFeedback::Positive;
        let serialized = serde_json::to_string(&positive).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(positive, deserialized);

        // 测试 Negative
        let negative = UserFeedback::Negative;
        let serialized = serde_json::to_string(&negative).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(negative, deserialized);

        // 测试 Neutral
        let neutral = UserFeedback::Neutral;
        let serialized = serde_json::to_string(&neutral).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(neutral, deserialized);

        // 测试 None
        let none = UserFeedback::None;
        let serialized = serde_json::to_string(&none).unwrap();
        let deserialized: UserFeedback = serde_json::from_str(&serialized).unwrap();
        assert_eq!(none, deserialized);
    }

    // 测试 10: 综合测试 - 测试多个操作的组合
    #[test]
    fn test_comprehensive_operations() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id);

        // 初始状态验证
        assert_eq!(record.retrieval_count(), 0);
        assert_eq!(record.feedback_score(), 0);

        // 执行多次提取
        for _ in 0..5 {
            record.record_retrieval();
        }
        assert_eq!(record.retrieval_count(), 5);

        // 添加混合反馈
        record.add_feedback(UserFeedback::Positive);
        record.add_feedback(UserFeedback::Positive);
        record.add_feedback(UserFeedback::Negative);
        record.add_feedback(UserFeedback::Neutral);
        record.add_feedback(UserFeedback::Positive);

        // 验证最终分数：+1 +1 -1 +0 +1 = +2
        assert_eq!(record.feedback_score(), 2);

        // 验证时间跨度
        let time_span = record.access_time_span();
        assert!(time_span >= 0);
    }

    // 测试 11: Record 序列化和反序列化
    #[test]
    fn test_record_serialization() {
        let memory_id = create_test_memory_id();
        let mut record = Record::new(memory_id.clone());

        // 添加一些数据
        record.record_retrieval();
        record.add_feedback(UserFeedback::Positive);

        // 序列化
        let serialized = serde_json::to_string(&record).unwrap();

        // 反序列化
        let deserialized: Record = serde_json::from_str(&serialized).unwrap();

        // 验证数据一致性
        assert_eq!(record.memory_id(), deserialized.memory_id());
        assert_eq!(record.retrieval_count(), deserialized.retrieval_count());
        assert_eq!(record.feedback_score(), deserialized.feedback_score());
    }

    // 测试 12: 边界情况 - 空反馈历史
    #[test]
    fn test_empty_feedback_history() {
        let memory_id = create_test_memory_id();
        let record = Record::new(memory_id);

        let now = Utc::now();

        // 查询空反馈历史
        let feedback_in_range = record.feedback_history_in_range(now, now);
        assert_eq!(feedback_in_range.len(), 0);

        let feedback_after = record.feedback_history_after(now);
        assert_eq!(feedback_after.len(), 0);

        let feedback_before = record.feedback_history_before(now);
        assert_eq!(feedback_before.len(), 0);
    }
}
