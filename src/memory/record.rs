use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

    feedback_history: HashMap<DateTime<Utc>, UserFeedback>,

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
            feedback_history: HashMap::new(),
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

    pub fn feedback_history(&self) -> &HashMap<DateTime<Utc>, UserFeedback> {
        &self.feedback_history
    }
}
