use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;

use crate::memory::{GraphMemoryLink,MemoryNote};
use super::probability;
use anyhow::Result;
use ordered_float::OrderedFloat;
use rand::prelude::IteratorRandom;


const DEFAULT_FOCUS: &str = "发呆";

#[derive(Debug,Clone)]
pub struct RecordNewNode {
    pub index: NodeIndex,
    pub ref_count: u32 //被提及的次数
}
#[derive(Debug,Clone)]
pub struct SoulTask {
    pub id: String,
    pub summary: String,
    pub related_notes: Vec<NodeIndex>,
    pub focus_prob: f32,
}
#[derive(Debug,Clone)]
pub struct SoulTaskSet { //挂起工作集
    tasks: HashMap<String,SoulTask>, //.0 为对数概率值
    focus: String, //当前聚焦任务
    inertia: f32 // 0 - 1
}
impl SoulTaskSet {
    pub fn new(inertia: f32) -> Result<Self> {
        if inertia <= 0.0 || inertia > 1.0 {
            return Err(anyhow::anyhow!("Inertia must be between 0 and 1"));
        }
        Ok(
            SoulTaskSet {
                tasks: HashMap::new(),
                focus: String::default(),
                inertia
            }
        )
    }
    pub fn focus_normalize(&mut self) {
        if self.tasks.is_empty() {
            return;
        }

        //异常值检测并修正
        let all_finite = self.tasks.values()
            .all(|task|task.focus_prob.is_finite());
        if !all_finite {
            // 将所有任务设置为相同的概率（0，即线性概率为1/len）
            let uniform_log = -((self.tasks.len() as f32).ln());
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
    pub fn tasks(&self) -> impl Iterator<Item = &SoulTask> {
        self.tasks.values()
    }
    pub fn task_summaries(&self) -> Vec<String> {
        self.tasks.values().map(|task| task.summary.clone()).collect()
    }
    pub fn set_inertia(&mut self, inertia: f32) -> Result<()> {
        if inertia <= 0.0 || inertia > 1.0 {
            return Err(anyhow::anyhow!("Inertia must be between 0 and 1"));
        }
        self.inertia = inertia;
        Ok(())
    }
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
    pub fn focus_sample(&self, sliding_window_size: usize, rng: &mut impl rand::RngCore) -> Vec<NodeIndex> {
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
    fn perform_sampling(
        &self,
        tasks: &[&SoulTask],
        counts: &[usize],
        rng: &mut impl rand::RngCore,
    ) -> Vec<NodeIndex> {
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

    //备用方法，当概率异常时
    fn fallback_sample(&self, sliding_window_size: usize, rng: &mut impl rand::RngCore) -> Vec<NodeIndex> {
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
}

#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct WorkingMemory {
    graph: StableGraph<MemoryNote, GraphMemoryLink>,
    temporary: Vec<RecordNewNode>,
    task_set: SoulTaskSet,
}
impl WorkingMemory {
    pub fn new(task_inertia: f32) -> Result<WorkingMemory> {
        Ok(
            WorkingMemory {
                graph: StableGraph::new(),
                temporary: vec![],
                task_set: SoulTaskSet::new(task_inertia)?,
            }
        )
    }
    pub fn add_task(&mut self, task: SoulTask) {
        self.task_set.add_task(task);
    }
    pub fn merge_mem_graph(&mut self, mem: Vec<MemoryNote>) {
        todo!("merge_mem_graph")
    }
    pub fn add_temp_mem(&mut self, mem: MemoryNote) {
        todo!("add_temp_mem")
    }
    pub fn focus_context(&self) -> Vec<String> {
        todo!("focus_context")
    }
}