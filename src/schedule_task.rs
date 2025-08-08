use std::collections::HashMap;
use uuid::Uuid;
use anyhow::{anyhow, Result};
use std::time::Duration;
use tokio::time::Instant;

pub enum TaskType {
    Once(Option<Box<dyn FnOnce() -> Result<()> + Sync + Send>>),
    Repeat(RepeatTask),
}
pub struct RepeatTask {
    task: Box<dyn Fn() -> Result<()> + Sync + Send>,
    max_execution: Option<usize>,
    interval: Duration,
}
impl RepeatTask {
    pub fn new(task: impl Fn() -> Result<()> + Sync + Send + 'static, max_execution: Option<usize>, interval: Duration) -> Self {
        Self {
            task: Box::new(task),
            max_execution,
            interval
        }
    }
    pub fn get_task_fn(&self) -> &(dyn Fn() -> Result<()> + Send) {
        &self.task
    }
}
pub struct ScheduleTask {
    id: Uuid,
    task: TaskType,
    execution_count: usize,
    expires_at: Instant,
    created_at: Instant
}
impl ScheduleTask {
    pub fn new_once(task: impl FnOnce() -> Result<()> + Sync + Send + 'static, expires_at: Instant) -> Self {
        Self {
            id: Uuid::new_v4(),
            task: TaskType::Once(Some(Box::new(task))),
            execution_count: 0,
            expires_at,
            created_at: Instant::now()
        }
    }
    pub fn new_repeat(task: impl Fn() -> Result<()> + Sync + Send + 'static, max_execution: Option<usize>, expires_at: Instant, interval: Duration) -> Result<Self> {
        if let Some(max_execution) = max_execution {
            if max_execution == 0 {
                return Err(anyhow::anyhow!("max_execution must be greater than 0"));
            }
        }
        Ok(
            Self {
                id: Uuid::new_v4(),
                task: TaskType::Repeat(
                    RepeatTask {
                        task: Box::new(task),
                        max_execution,
                        interval
                    }
                ),
                execution_count: 0,
                expires_at,
                created_at: Instant::now()
            }
        )
    }
    pub fn execute(&mut self) -> Result<bool>{ // 返回是否继续执行，如果返回false，则任务结束，从时间轮剔除
        match &mut self.task {
            TaskType::Once(task) => {
                if let Some(task_fn) = task.take() {
                    task_fn()?;
                    self.execution_count += 1;
                }
                Ok(false)
            }
            TaskType::Repeat(repeat_task) => {
                repeat_task.get_task_fn()()?;
                self.execution_count += 1;
                let should_run_again = match repeat_task.max_execution {
                    Some(max) => self.execution_count < max,
                    None => true
                };
                if should_run_again {
                    let next_execution = self.created_at + repeat_task.interval * (self.execution_count as u32 + 1);
                    self.expires_at = next_execution;
                }
                Ok(should_run_again)
            }
        }
    }
}
pub struct SimpleTimeWheel {
    tick_duration: Duration,
    wheel_size: usize,
    current_slot: usize,
    slots: Vec<Vec<ScheduleTask>>,
    task_map: HashMap<Uuid, usize>,
    start_time: Option<Instant>
}
impl SimpleTimeWheel {
    pub fn new(tick_duration: Duration, wheel_size: usize) -> Self {
        Self {
            tick_duration,
            wheel_size,
            current_slot: 0,
            slots: {
                let mut vec = Vec::with_capacity(wheel_size);
                for _ in 0..wheel_size {
                    vec.push(Vec::new());
                }
                vec
            },
            task_map: HashMap::new(),
            start_time: None
        }
    }
    pub fn add_task(&mut self, task: ScheduleTask) -> Result<Uuid> {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }
        if task.expires_at <= Instant::now() {
            return Err(anyhow!("Task expiration time is in the past"));
        }

        if task.expires_at.duration_since(self.start_time.unwrap()) > self.max_delay() {
            return Err(anyhow!("Task delay exceeds maximum supported delay"));
        }

        let slot = self.calculate_slot(task.expires_at)?;
        
        let task_id = task.id;

        self.slots[slot].push(task);
        self.task_map.insert(task_id, slot);

        Ok(task_id)
    }
    pub fn task_count(&self) -> usize {
        self.task_map.len()
    }
    pub fn tick(&mut self) {
        self.current_slot = (self.current_slot + 1) % self.wheel_size;
    }
    pub fn get_ready_tasks(&mut self) -> Vec<ScheduleTask> {
        let slot = &mut self.slots[self.current_slot];
        std::mem::take(slot)
    }
    pub fn remove_task(&mut self, task_id: Uuid) -> Option<ScheduleTask> {
        let slot = self.task_map.remove(&task_id)?;
        let slot_tasks = &mut self.slots[slot];

        let index = slot_tasks.iter().position(|task| task.id == task_id)?;
        Some(slot_tasks.remove(index))
    }
    fn calculate_slot(&self, expires_at: Instant) -> Result<usize> {
        let start_time = self.start_time.ok_or(anyhow!("Time wheel not started"))?;
        let delay = expires_at.saturating_duration_since(start_time);
        let ticks = (delay.as_nanos() / self.tick_duration.as_nanos()) as usize;
        Ok((self.current_slot + ticks) % self.wheel_size)
    }

    fn max_delay(&self) -> Duration {
        self.tick_duration * (self.wheel_size as u32)
    }

}
