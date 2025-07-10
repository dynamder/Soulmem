use std::collections::HashMap;
use ordered_float::OrderedFloat;
use rand::prelude::IteratorRandom;
use crate::memory::{probability, NodeRefId};
use crate::memory::working::DEFAULT_FOCUS;
use anyhow::Result;

///任务
#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct SoulTask {
    pub id: String, //任务ID
    pub summary: String, //任务描述摘要
    pub related_notes: Vec<NodeRefId>, //关联的记忆
    pub focus_prob: f32,//任务焦点概率分数
}
#[allow(dead_code)]
impl SoulTask {
    pub fn new(summary: String, related_notes: Vec<NodeRefId>) -> Self {
        SoulTask {
            id: uuid::Uuid::new_v4().to_string(),
            summary,
            related_notes,
            focus_prob: 0.0,
        }
    }
}
#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct SoulTaskSet { //挂起工作集
    tasks: HashMap<String,SoulTask>, //任务列表, (任务id，任务)
    focus: String, //当前聚焦任务
    inertia: f32 // 0 - 1
}
#[allow(dead_code)]
impl SoulTaskSet {
    pub fn new(inertia: f32) -> Result<Self> {
        if inertia <= 0.0 || inertia > 1.0 {
            return Err(anyhow::anyhow!("Inertia must be between 0 and 1"));
        }
        Ok(
            SoulTaskSet {
                tasks: HashMap::new(),
                focus: DEFAULT_FOCUS.to_string(),
                inertia
            }
        )
    }
    ///设置当前焦点任务
    pub fn set_focus(&mut self, focus: impl Into<String>) {
        self.focus = focus.into();
    }
    pub fn focus(&self) -> &String {
        &self.focus
    }

}
#[allow(dead_code)]
impl SoulTaskSet {

    ///用softmax归一化焦点概率分数
    pub fn focus_normalize(&mut self) {
        if self.tasks.is_empty() {
            return;
        }

        //异常值检测并修正
        let all_finite = self.tasks.values()
            .all(|task|task.focus_prob.is_finite());
        if !all_finite {
            // 将所有任务设置为相同的概率（0，即线性概率为1/len）
            let uniform_log = -(self.tasks.len() as f32).ln();
            for task in self.tasks.values_mut() {
                task.focus_prob = uniform_log;
            }
            return;
        }

        let (log_sum_exp,max_log) = probability::log_exp_sum(
            &self.tasks.values()
                .map(|task| task.focus_prob).collect::<Vec<_>>()
        );

        // 归一化所有概率
        for task in self.tasks.values_mut() {
            task.focus_prob -= max_log + log_sum_exp;
        }
    }
    ///添加新任务
    pub fn add_task(&mut self, task: SoulTask) {
        let initial_log_prob = if self.tasks.is_empty() {
            0.0 // log(1) = 0
        } else {
            -((self.tasks.len() + 1) as f32).ln()
        };
        let mut task = task;
        task.focus_prob = initial_log_prob;
        self.tasks.insert(task.id.clone(), task);

        self.focus_normalize();
    }
    pub fn get_task(&self, task_id: impl AsRef<str>) -> Option<&SoulTask> {
        self.tasks.get(task_id.as_ref())
    }
    pub fn get_task_mut(&mut self, task_id: impl AsRef<str>) -> Option<&mut SoulTask> {
        self.tasks.get_mut(task_id.as_ref())
    }
    pub fn remove_task(&mut self, task_id: impl AsRef<str>) -> Option<SoulTask> {
        let removed = self.tasks.remove(task_id.as_ref());
        self.focus_normalize();
        removed
    }
    ///获取全部的任务
    pub fn tasks(&self) -> impl Iterator<Item = &SoulTask> {
        self.tasks.values()
    }
    ///获取全部的任务描述
    pub fn task_summaries(&self) -> Vec<String> {
        self.tasks.values().map(|task| task.summary.clone()).collect()
    }
    ///设置记忆惯性系数
    pub fn set_inertia(&mut self, inertia: f32) -> Result<()> {
        if inertia <= 0.0 || inertia > 1.0 {
            return Err(anyhow::anyhow!("Inertia must be between 0 and 1"));
        }
        self.inertia = inertia;
        Ok(())
    }
    ///焦点迭代，根据当前焦点情况和新的相关任务序列选择新焦点
    pub fn shift_focus(&mut self, sorted_related_tasks: &[impl AsRef<str>]) -> String {
        // Handle empty task set,using the DEFAULT_FOCUS
        if self.tasks.is_empty() {
            self.focus = DEFAULT_FOCUS.to_string();
            return DEFAULT_FOCUS.to_string();
        }
        //if nothing related,don't shift the focus
        if sorted_related_tasks.is_empty() {
            return self.focus.clone();
        }

        //handle unexpected inertia
        let inertia_ln = if self.inertia > 0.0 {
            self.inertia.ln()
        }else {
            f32::NEG_INFINITY
        };

        // Step 1: Decay existing probabilities
        for task in self.tasks.values_mut() {
            task.focus_prob += inertia_ln;
        }

        // Step 2: Apply relevance boosts
        for (i, task_id) in sorted_related_tasks.iter().enumerate() {
            if let Some(task) = self.tasks.get_mut(task_id.as_ref()) {
                // Boost formula: (1 - inertia) * rank_weight
                // Rank weight = 1 / (1 + position_index)
                let rank_weight = 1.0 / (1.0 + i as f32);
                let boost = (1.0 - self.inertia) * rank_weight;

                // operate in logarithm space, and prevent overflow
                let boost_log = boost.ln();
                task.focus_prob = if task.focus_prob.is_finite() {
                    // 对数空间加法：log(exp(a) + exp(b)) = max + log(exp(a-max) + exp(b-max))
                    probability::log_exp_sum2(task.focus_prob, boost_log)
                } else {
                    boost_log
                };
            }
        }
        self.focus_normalize();

        // Step 3: Update focus to task with the highest probability
        self.focus = self.tasks.iter()
            .max_by_key(|(_, task)| OrderedFloat(task.focus_prob))
            .map(|(id, _)| id.clone())
            .unwrap_or(DEFAULT_FOCUS.to_string());



        self.focus.clone()
    }
    ///根据当前焦点情况，按比例采样直接关联记忆
    pub fn focus_sample(&self, sliding_window_size: usize, rng: &mut impl rand::RngCore) -> Vec<NodeRefId> {
        if self.tasks.is_empty() || sliding_window_size == 0 {
            return Vec::new();
        }
        // 概率有效性检查
        if self.tasks.values().any(|t| t.focus_prob.is_nan()) {
            return self.fallback_sample(sliding_window_size, rng);
        }

        let tasks = self.tasks.values().collect::<Vec<_>>();
        let expected_counts = self.allocate_samples(&tasks, sliding_window_size);
        self.perform_sampling(&tasks, &expected_counts, rng)
    }
    ///分配采样
    fn allocate_samples(&self, tasks: &[&SoulTask], sample_size: usize) -> Vec<usize> {
        // 1. 计算线性概率和总概率
        let linear_probs: Vec<f32> = tasks
            .iter()
            .map(|t| t.focus_prob.exp())
            .collect();

        let total_prob = linear_probs.iter().sum::<f32>().max(f32::MIN_POSITIVE);

        // 2. 计算预期样本数（保持比例）
        let expected_counts: Vec<f32> = linear_probs
            .iter()
            .map(|p| *p / total_prob * sample_size as f32)
            .collect();

        // 3. 分配样本
        let mut allocated = vec![0; tasks.len()];
        let mut remaining = sample_size;

        // 第一轮：分配整数部分（保持比例）
        for (i, count_f) in expected_counts.iter().enumerate() {
            let integer_part = count_f.floor() as usize;
            let max_possible = tasks[i].related_notes.len();
            let to_allocate = integer_part.min(max_possible).min(remaining);

            allocated[i] = to_allocate;
            remaining -= to_allocate;
        }

        // 第二轮：按小数部分分配（保持比例）
        if remaining > 0 {
            // 创建索引列表并按小数部分降序排序
            let mut indices: Vec<usize> = (0..tasks.len()).collect();
            indices.sort_by(|&a, &b| {
                // 比较小数部分
                let frac_a = expected_counts[a] - expected_counts[a].floor();
                let frac_b = expected_counts[b] - expected_counts[b].floor();
                frac_b.partial_cmp(&frac_a).unwrap_or(std::cmp::Ordering::Equal)
            });

            // 分配剩余样本（每个任务最多1个额外样本）
            for idx in indices {
                if remaining == 0 {
                    break;
                }

                // 检查是否还能采样（不超过节点上限）
                if allocated[idx] < tasks[idx].related_notes.len() {
                    allocated[idx] += 1;
                    remaining -= 1;
                }
            }
        }
        // 不再强制分配剩余样本 - 保持比例更重要
        allocated
    }
    /// 执行采样
    fn perform_sampling(
        &self,
        tasks: &[&SoulTask],
        counts: &[usize],
        rng: &mut impl rand::RngCore,
    ) -> Vec<NodeRefId> {
        let total_samples: usize = counts.iter().sum();
        let mut sampled = Vec::with_capacity(total_samples);

        for (i, &task) in tasks.iter().enumerate() {
            if counts[i] > 0 {
                sampled.extend(
                    task.related_notes
                        .iter()
                        .choose_multiple(rng, counts[i])
                        .into_iter()
                        .cloned(),
                );
            }
        }

        sampled
    }

    ///备用采样方法，当概率异常时
    fn fallback_sample(&self, sliding_window_size: usize, rng: &mut impl rand::RngCore) -> Vec<NodeRefId> {
        // 均匀采样所有相关节点
        let all_notes: Vec<_> = self.tasks.values()
            .flat_map(|t| &t.related_notes)
            .cloned()
            .collect();

        if all_notes.is_empty() {
            return Vec::new();
        }
        let len = all_notes.len();
        // 随机采样
        all_notes
            .into_iter()
            .choose_multiple(rng, sliding_window_size.min(len))
    }
    ///清理空任务
    pub fn clean_empty_tasks(&mut self) {
        self.tasks.retain(|_, task| {
            let retain = !task.related_notes.is_empty();
            if !retain && task.id == self.focus {
                // 如果删除的是当前焦点任务，重置焦点
                self.focus = DEFAULT_FOCUS.to_string();
            }
            retain
        });
        self.focus_normalize();
    }
}





#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng};
    use rand::rngs::StdRng;
    use tokio::task_local;

    #[test]
    fn test_new_with_valid_inertia() -> Result<()> {
        let set = SoulTaskSet::new(0.5)?;
        assert_eq!(set.inertia, 0.5);
        assert_eq!(set.tasks.len(), 0);
        assert_eq!(set.focus, DEFAULT_FOCUS.to_string());
        Ok(())
    }

    #[test]
    fn test_new_with_inertia_zero() {
        let result = SoulTaskSet::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_negative_inertia() {
        let result = SoulTaskSet::new(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_inertia_one() -> Result<()> {
        let set = SoulTaskSet::new(1.0)?;
        assert_eq!(set.inertia, 1.0);
        Ok(())
    }

    #[test]
    fn test_new_with_inertia_above_one() {
        let result = SoulTaskSet::new(1.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_focus_with_str() {
        let mut set = SoulTaskSet::new(0.5).unwrap();
        set.set_focus("task1");
        assert_eq!(set.focus, "task1");
    }

    #[test]
    fn test_set_focus_with_string() {
        let mut set = SoulTaskSet::new(0.5).unwrap();
        let s = String::from("task2");
        set.set_focus(s);
        assert_eq!(set.focus, "task2");
    }

    #[test]
    fn test_focus_returns_correct_reference() {
        let mut set = SoulTaskSet::new(0.5).unwrap();
        set.set_focus("task3");
        assert_eq!(*set.focus(), "task3");
    }

    // 创建测试用的简单任务
    fn create_test_task(id: &str, summary: &str) -> SoulTask {
        SoulTask {
            id: id.to_string(),
            summary: summary.to_string(),
            focus_prob: 0.0,
            related_notes: vec![],
        }
    }

    // 创建测试用的 RNG
    fn test_rng() -> StdRng {
        SeedableRng::from_seed([1; 32])
    }

    #[test]
    fn test_focus_normalize_empty() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        task_set.focus_normalize();
        assert!(task_set.tasks.is_empty());
    }

    #[test]
    fn test_focus_normalize_with_non_finite() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        task_set.tasks.insert("t1".to_string(), create_test_task("t1", "Test 1"));
        task_set.tasks.get_mut("t1").unwrap().focus_prob = f32::INFINITY;

        task_set.focus_normalize();

        assert_eq!(task_set.tasks["t1"].focus_prob, -1.0f32.ln());
    }

    #[test]
    fn test_add_task_first() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        let task = create_test_task("t1", "Test 1");
        task_set.add_task(task);

        assert_eq!(task_set.tasks.len(), 1);
        assert_eq!(task_set.tasks["t1"].focus_prob, 0.0);
    }

    #[test]
    fn test_remove_task() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        let task = create_test_task("t1", "Test 1");
        task_set.add_task(task);
        assert_eq!(task_set.tasks.len(), 1);

        let removed = task_set.remove_task("t1");
        assert!(removed.is_some());
        assert!(task_set.tasks.is_empty());
    }

    #[test]
    fn test_shift_focus_empty() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: DEFAULT_FOCUS.to_string(),
            inertia: 0.5,
        };

        let related_tasks: Vec<&str> = vec![];
        let new_focus = task_set.shift_focus(&related_tasks);

        assert_eq!(new_focus, DEFAULT_FOCUS.to_string());
        assert_eq!(task_set.focus, DEFAULT_FOCUS.to_string());
    }
    #[test]
    fn test_shift_focus_some() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: DEFAULT_FOCUS.to_string(),
            inertia: 0.01,
        };
        let task1 = SoulTask::new("test1".to_string(), vec![NodeRefId::from("test1")]);
        let task2 = SoulTask::new("test2".to_string(), vec![NodeRefId::from("test2")]);
        let task1_id = task1.id.clone();
        let task2_id = task2.id.clone();
        task_set.add_task(task1);
        task_set.add_task(task2);
        task_set.set_focus(&task1_id);
        task_set.shift_focus(&vec![task2_id.clone(),task1_id.clone()]);
        assert_eq!(task_set.focus, task2_id)
    }   

    #[test]
    fn test_focus_sample_empty() {
        let task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        let mut rng = test_rng();
        let samples = task_set.focus_sample(5, &mut rng);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_allocate_samples_even_distribution() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: DEFAULT_FOCUS.to_string(),
            inertia: 0.5,
        };
        let mut task1 = create_test_task("t1", "Task 1");
        let mut task2 = create_test_task("t2", "Task 2");
        task1.related_notes = vec![NodeRefId::from("111"),NodeRefId::from("222"),NodeRefId::from("333")];
        task2.related_notes = vec![NodeRefId::from("444"),NodeRefId::from("555"),NodeRefId::from("666")];

        // 创建两个任务，具有相同的概率
        let tasks = vec![
            &task1,
            &task2,
        ];

        // 修改它们的 focus_prob 为相同值
        let mut modified_tasks = tasks.clone();
        for task in &mut modified_tasks {
            unsafe {
                // 这里只是为了测试，实际应避免 unsafe
                let ptr = task as *const _ as *mut SoulTask;
                (*ptr).focus_prob = -0.693; // 设置为相同值
            }
        }
        task_set.focus_normalize();

        let sample_size = 5;
        let expected_counts = task_set.allocate_samples(&modified_tasks, sample_size);

        // 应该平均分配（2,3 或 3,2）
        assert_eq!(expected_counts.iter().sum::<usize>(), sample_size);
        assert_eq!(expected_counts[0].abs_diff(expected_counts[1]), 1)
    }

    #[test]
    fn test_perform_sampling() {
        let task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        // 创建一个带有相关笔记的任务
        let mut task = create_test_task("t1", "Task 1");
        task.related_notes = vec![
            NodeRefId::from("note1"),
            NodeRefId::from("note2"),
            NodeRefId::from("note3"),
        ];

        let tasks = vec![&task];
        let counts = vec![2];

        let mut rng = test_rng();
        let samples = task_set.perform_sampling(&tasks, &counts, &mut rng);

        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|s| s.as_str().starts_with("note")));
    }

    #[test]
    fn test_fallback_sample() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: "default".to_string(),
            inertia: 0.5,
        };

        // 创建一个带有相关笔记的任务
        let mut task = create_test_task("t1", "Task 1");
        task.related_notes = vec![
            NodeRefId::from("note1"),
            NodeRefId::from("note2"),
            NodeRefId::from("note3"),
        ];
        task_set.add_task(task);
            

        let mut rng = test_rng();
        let samples = task_set.fallback_sample(2, &mut rng);

        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|s| s.as_str().starts_with("note")));
    }

    #[test]
    fn test_clean_empty_tasks() {
        let mut task_set = SoulTaskSet {
            tasks: HashMap::new(),
            focus: DEFAULT_FOCUS.to_string(),
            inertia: 0.5,
        };

        // 添加两个任务，其中一个有笔记，一个没有
        let mut task1 = create_test_task("t1", "Task 1");
        task1.related_notes = vec![NodeRefId::from("note1")];

        let task2 = create_test_task("t2", "Task 2");

        task_set.tasks.insert("t1".to_string(), task1);
        task_set.tasks.insert("t2".to_string(), task2);
        task_set.focus = "t2".to_string();

        task_set.clean_empty_tasks();

        assert_eq!(task_set.tasks.len(), 1);
        assert_eq!(task_set.focus, DEFAULT_FOCUS.to_string());
    }
}
