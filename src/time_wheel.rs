use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use anyhow::{anyhow, Result};
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use std::future::Future;

type BoxFuture = Box<dyn Future<Output = Result<()>> + Send>;

pub enum TaskType {
    Once(Option<Arc<dyn Fn() -> Pin<BoxFuture> + Send + Sync>>),
    Repeat(RepeatTask),
}
pub struct RepeatTask {
    task: Arc<dyn Fn() -> Pin<BoxFuture> + Send + Sync>,
    max_execution: Option<usize>,
    interval: Duration,
}
impl RepeatTask {
    pub fn new<F, Fut>(task: F, max_execution: Option<usize>, interval: Duration) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let boxed_task = move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            Box::pin(task())
        };
        Self {
            task: Arc::new(boxed_task),
            max_execution,
            interval
        }
    }
    pub fn get_task_fn(&self) -> &(dyn Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync) {
        &*self.task
    }
}
pub struct ErrorHandleConfig {
    pub max_retries: usize,
    pub retry_interval: Duration,
    pub on_failure: Option<Arc<dyn Fn() + Send + Sync>>
}
pub struct ScheduleTask {
    id: Uuid,
    task: TaskType,
    execution_count: usize,
    expires_at: Instant,
    created_at: Instant,
    last_executed_at: Option<Instant>,

    error_handle_config: Option<ErrorHandleConfig>,
}
impl ScheduleTask {
    pub fn new_once<F, Fut>(task: F, expires_at: Instant, error_handle_config: Option<ErrorHandleConfig>) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let boxed_task = move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            Box::pin(task())
        };
        Self {
            id: Uuid::new_v4(),
            task: TaskType::Once(Some(Arc::new(boxed_task))),
            execution_count: 0,
            expires_at,
            created_at: Instant::now(),
            last_executed_at: None,
            error_handle_config
        }
    }
    pub fn new_repeat<F, Fut>(
        task: F,
        max_execution: Option<usize>,
        expires_at: Instant,
        interval: Duration,
        error_handle_config: Option<ErrorHandleConfig>,
    ) -> Result<Self>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        if let Some(max_execution) = max_execution {
            if max_execution == 0 {
                return Err(anyhow::anyhow!("max_execution must be greater than 0"));
            }
        }

        let boxed_task = move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            Box::pin(task())
        };

        Ok(Self {
            id: Uuid::new_v4(),
            task: TaskType::Repeat(
                RepeatTask {
                    task: Arc::new(boxed_task),
                    max_execution,
                    interval
                }
            ),
            execution_count: 0,
            expires_at,
            created_at: Instant::now(),
            last_executed_at: None,
            error_handle_config
        })
    }
    pub async fn execute(&mut self) -> Result<bool>{ // 返回是否继续执行，如果返回false，则任务结束，从时间轮剔除
        let mut retry_count = 0;
        loop {
            let result = match &mut self.task {
                TaskType::Once(task) => {
                   if let Some(task_fn) = task.take() {
                       let res = task_fn().await;
                       if res.is_err() {
                           //Once任务失败，放回take拿出的所有权，以便重试可使用
                           self.task = TaskType::Once(Some(task_fn));
                       }
                       res
                   }else {
                       return Ok(false); //Once任务已经执行完毕
                   }
                }
                TaskType::Repeat(repeat_task) => {
                    repeat_task.get_task_fn()().await
                }
            };
            match result {
                Ok(_) => {
                    self.execution_count += 1;
                    self.last_executed_at = Some(Instant::now()); //此处

                    return if let TaskType::Repeat(repeat_task) = &self.task {
                        //更新Repeat任务的下次执行时间
                        let should_run_again = match repeat_task.max_execution {
                            Some(max) => self.execution_count < max,
                            None => true
                        };
                        if should_run_again {
                            self.expires_at = self.last_executed_at.unwrap() + repeat_task.interval; // ROBUST: 执行到此处时，必定为Some值
                        }
                        Ok(should_run_again)
                    } else {
                        Ok(false) //Once任务执行完毕
                    }
                }
                Err(e) => {
                    match &self.error_handle_config {
                        None => return Err(e),
                        Some(config) => {
                            if retry_count < config.max_retries {
                                retry_count += 1;
                                log::warn!("Task {} failed, retrying ({})", self.id, retry_count);
                                tokio::time::sleep(config.retry_interval).await;
                            } else {
                                log::error!("Task {} failed, giving up after {} retries, Err: {}", self.id, retry_count, e);
                                if let Some(failure_callback) = &config.on_failure {
                                    failure_callback();
                                }
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
    }
}
pub trait TimeWheel {
    fn tick(&mut self);
    fn add_task(&mut self, task: ScheduleTask) -> Result<Uuid>;
    fn remove_task(&mut self, task_id: Uuid) -> Option<ScheduleTask>;
    fn get_ready_tasks(&mut self) -> Vec<ScheduleTask>;
    fn task_count(&self) -> usize;
    fn get_tick_duration(&self) -> Duration;
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
    pub fn new(tick_duration: Duration, wheel_size: usize) -> Result<Self> {
        if wheel_size < 2 {
            return Err(anyhow::anyhow!("wheel_size must be greater than 1"));
        }
        if tick_duration < Duration::from_secs(1) {
            return Err(anyhow::anyhow!("tick_duration must be greater than 1 second"));
        }
        Ok(Self {
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
        })
    }
    fn calculate_slot(&self, expires_at: Instant) -> Result<usize> {
        let start_time = self.start_time.ok_or(anyhow!("Time wheel not started"))?;
        let delay = expires_at.saturating_duration_since(start_time);
        let ticks = (delay.as_millis() / self.tick_duration.as_millis()) as usize;
        Ok(ticks % self.wheel_size)
    }

    fn max_delay(&self) -> Duration {
        self.tick_duration.checked_mul(self.wheel_size as u32).unwrap_or(Duration::MAX)
    }

}
impl TimeWheel for SimpleTimeWheel {
    fn tick(&mut self) {
        self.current_slot = (self.current_slot + 1) % self.wheel_size;
    }
    fn add_task(&mut self, task: ScheduleTask) -> Result<Uuid> {
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
    fn remove_task(&mut self, task_id: Uuid) -> Option<ScheduleTask> {
        let slot = self.task_map.remove(&task_id)?;
        let slot_tasks = &mut self.slots[slot];

        let index = slot_tasks.iter().position(|task| task.id == task_id)?;
        Some(slot_tasks.remove(index))
    }
    fn get_ready_tasks(&mut self) -> Vec<ScheduleTask> {
        let slot = &mut self.slots[self.current_slot];
        std::mem::take(slot)
    }
    fn task_count(&self) -> usize {
        self.task_map.len()
    }
    fn get_tick_duration(&self) -> Duration {
        self.tick_duration
    }
}
pub struct TimeWheelRunner<T: TimeWheel + Send> {
    wheel: Arc<Mutex<T>>,
    tick_interval: Duration,
    cancel_token: CancellationToken,
}
impl<T: TimeWheel + Send + 'static> TimeWheelRunner<T> {
    pub fn new(wheel: T) -> Self {
        Self {
            tick_interval: wheel.get_tick_duration(),
            wheel: Arc::new(Mutex::new(wheel)),
            cancel_token: CancellationToken::new(),
        }
    }
    pub async fn run(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.tick_interval);
            loop {
                tokio::select! {
                    _ = self.cancel_token.cancelled() => {
                        log::info!("Time wheel runner received cancel signal, quitting.");
                        break;
                    }
                    _ = interval.tick() => {
                        let mut wheel = self.wheel.lock().await;
                        wheel.tick();
                        let tasks = wheel.get_ready_tasks();
                        drop(wheel);
                        for mut task in tasks {
                            tokio::spawn(async move {
                                if let Err(e) = task.execute().await {
                                    log::error!("Task execution error: {:?}", e);
                                }
                            });
                        }
                    }
                }
            }
        })
    }
    pub fn wheel(&self) -> Arc<Mutex<T>> {
        self.wheel.clone()
    }
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }
}

/* ------------------tests----------------------------------------------------*/
#[cfg(test)]
mod schedule_task_test {
    use super::*;
    use tokio::time::{Duration, Instant};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // 创建一个简单的成功任务
    fn success_task() -> impl Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            Box::pin(async { Ok(()) })
        }
    }

    // 创建一个失败任务
    fn failure_task() -> impl Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            Box::pin(async { Err(anyhow::anyhow!("Task failed")) })
        }
    }

    // 创建一个计数成功任务
    fn counting_success_task(counter: Arc<AtomicUsize>) -> impl Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            let counter = counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn test_new_once_creation() {
        // 测试正常创建一次性任务
        let expires_at = Instant::now() + Duration::from_secs(10);
        let task = ScheduleTask::new_once(success_task(), expires_at, None);

        // 验证任务属性
        assert_ne!(task.id, Uuid::nil()); // ID不应为空
        assert_eq!(task.execution_count, 0);
        assert_eq!(task.expires_at, expires_at);
        assert!(task.last_executed_at.is_none());
        assert!(task.error_handle_config.is_none());

        // 验证创建时间近似正确（在1秒内）
        let now = Instant::now();
        assert!(task.created_at <= now);
        assert!(now - task.created_at < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_new_repeat_creation_success() {
        // 测试正常创建重复任务
        let expires_at = Instant::now() + Duration::from_secs(10);
        let interval = Duration::from_secs(1);
        let task = ScheduleTask::new_repeat(
            success_task(),
            Some(5),
            expires_at,
            interval,
            None,
        ).unwrap();

        // 验证任务属性
        assert_ne!(task.id, Uuid::nil());
        assert_eq!(task.execution_count, 0);
        assert_eq!(task.expires_at, expires_at);
        assert!(task.last_executed_at.is_none());
        assert!(task.error_handle_config.is_none());

        // 验证创建时间近似正确
        let now = Instant::now();
        assert!(task.created_at <= now);
        assert!(now - task.created_at < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_new_repeat_creation_with_zero_max_execution() {
        // 测试max_execution为0时返回错误
        let expires_at = Instant::now() + Duration::from_secs(10);
        let interval = Duration::from_secs(1);
        let result = ScheduleTask::new_repeat(
            success_task(),
            Some(0),
            expires_at,
            interval,
            None,
        );

        // 验证返回错误
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_new_repeat_creation_without_max_execution() {
        // 测试不设置max_execution
        let expires_at = Instant::now() + Duration::from_secs(10);
        let interval = Duration::from_secs(1);
        let task = ScheduleTask::new_repeat(
            success_task(),
            None,
            expires_at,
            interval,
            None,
        ).unwrap();

        // 验证任务创建成功
        assert_ne!(task.id, Uuid::nil());
    }

    #[tokio::test]
    async fn test_execute_once_task_success() {
        // 测试一次性任务成功执行
        let expires_at = Instant::now() + Duration::from_secs(10);
        let mut task = ScheduleTask::new_once(success_task(), expires_at, None);

        // 执行任务
        let result = task.execute().await;

        // 验证执行结果
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false); // 一次性任务执行后应返回false
        assert_eq!(task.execution_count, 1);
        assert!(task.last_executed_at.is_some());
    }

    #[tokio::test]
    async fn test_execute_once_task_twice() {
        // 测试一次性任务执行两次（第二次应该直接返回false）
        let expires_at = Instant::now() + Duration::from_secs(10);
        let mut task = ScheduleTask::new_once(success_task(), expires_at, None);

        // 第一次执行
        let result1 = task.execute().await.unwrap();
        assert_eq!(result1, false);
        assert_eq!(task.execution_count, 1);

        // 第二次执行应该直接返回false，不增加执行计数
        let result2 = task.execute().await.unwrap();
        assert_eq!(result2, false);
        assert_eq!(task.execution_count, 1); // 计数不应增加
    }

    #[tokio::test]
    async fn test_execute_repeat_task_once() {
        // 测试重复任务执行一次
        let expires_at = Instant::now() + Duration::from_secs(10);
        let interval = Duration::from_secs(1);
        let mut task = ScheduleTask::new_repeat(
            success_task(),
            Some(3),
            expires_at,
            interval,
            None,
        ).unwrap();

        let original_expires_at = task.expires_at;
        let last_executed_at = task.last_executed_at;

        // 执行任务
        let result = task.execute().await.unwrap();

        // 验证执行结果
        assert_eq!(result, true); // 还有执行次数，应返回true
        assert_eq!(task.execution_count, 1);
        assert!(task.last_executed_at.is_some());
        assert_ne!(task.last_executed_at, last_executed_at);
        // 验证下次执行时间已更新
        assert_eq!(task.expires_at, task.last_executed_at.unwrap() + interval);
        assert_ne!(task.expires_at, original_expires_at);
    }

    #[tokio::test]
    async fn test_execute_repeat_task_until_max_execution() {
        // 测试重复任务执行到最大次数
        let expires_at = Instant::now() + Duration::from_secs(10);
        let interval = Duration::from_secs(1);
        let counter = Arc::new(AtomicUsize::new(0));
        let mut task = ScheduleTask::new_repeat(
            counting_success_task(counter.clone()),
            Some(3),
            expires_at,
            interval,
            None,
        ).unwrap();

        // 执行第一次
        assert_eq!(task.execute().await.unwrap(), true);
        assert_eq!(task.execution_count, 1);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // 执行第二次
        assert_eq!(task.execute().await.unwrap(), true);
        assert_eq!(task.execution_count, 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        // 执行第三次，应该返回false
        assert_eq!(task.execute().await.unwrap(), false);
        assert_eq!(task.execution_count, 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_execute_task_failure_without_error_config() {
        // 测试任务失败且无错误处理配置
        let expires_at = Instant::now() + Duration::from_secs(10);
        let mut task = ScheduleTask::new_once(failure_task(), expires_at, None);

        // 执行任务应该返回错误
        let result = task.execute().await;
        assert!(result.is_err());
        assert_eq!(task.execution_count, 0); // 失败不计数
    }

    #[tokio::test]
    async fn test_execute_task_failure_with_error_config_no_retry() {
        // 测试任务失败有错误处理配置但不重试
        let expires_at = Instant::now() + Duration::from_secs(10);
        let error_config = ErrorHandleConfig {
            max_retries: 0,
            retry_interval: Duration::from_millis(100),
            on_failure: None,
        };
        let mut task = ScheduleTask::new_once(failure_task(), expires_at, Some(error_config));

        // 执行任务应该返回错误
        let result = task.execute().await;
        assert!(result.is_err());
        assert_eq!(task.execution_count, 0);
    }

    #[tokio::test]
    async fn test_execute_task_failure_then_success_with_retry() {
        // 测试任务先失败后成功并重试
        let expires_at = Instant::now() + Duration::from_secs(10);
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let task_fn = move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            let call_count = call_count_clone.clone();
            Box::pin(async move {
                let count = call_count.fetch_add(1, Ordering::SeqCst);
                if count == 0 {
                    Err(anyhow::anyhow!("First attempt failed"))
                } else {
                    Ok(())
                }
            })
        };

        let error_config = ErrorHandleConfig {
            max_retries: 3,
            retry_interval: Duration::from_millis(100),
            on_failure: None,
        };
        let mut task = ScheduleTask::new_once(task_fn, expires_at, Some(error_config));

        // 执行任务应该成功（重试后）
        let result = task.execute().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false); // 一次性任务完成
        assert_eq!(task.execution_count, 1);
        assert_eq!(call_count.load(Ordering::SeqCst), 2); // 调用了两次
    }

    #[tokio::test]
    async fn test_execute_task_failure_exceed_max_retries() {
        // 测试任务失败且超过最大重试次数
        let expires_at = Instant::now() + Duration::from_secs(10);
        let error_config = ErrorHandleConfig {
            max_retries: 2,
            retry_interval: Duration::from_millis(100),
            on_failure: None,
        };
        let mut task = ScheduleTask::new_once(failure_task(), expires_at, Some(error_config));

        // 执行任务应该返回错误
        let result = task.execute().await;
        assert!(result.is_err());
        assert_eq!(task.execution_count, 0);
    }

    #[tokio::test]
    async fn test_execute_repeat_task_infinite_execution() {
        // 测试无限重复任务
        let expires_at = Instant::now() + Duration::from_secs(10);
        let interval = Duration::from_secs(1);
        let counter = Arc::new(AtomicUsize::new(0));
        let mut task = ScheduleTask::new_repeat(
            counting_success_task(counter.clone()),
            None, // 无最大执行次数限制
            expires_at,
            interval,
            None,
        ).unwrap();

        // 执行多次都应该返回true
        for i in 1..=5 {
            assert_eq!(task.execute().await.unwrap(), true);
            assert_eq!(task.execution_count, i);
            assert_eq!(counter.load(Ordering::SeqCst), i);
        }
    }
}

#[cfg(test)]
mod simple_time_wheel_tests {
    use super::*;
    use std::time::{Duration};
    use tokio::time::sleep;
    use tokio::time::Instant;
    use uuid::Uuid;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;


    // Helper function to create a simple successful task
    fn success_task() -> impl Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            Box::pin(async { Ok(()) })
        }
    }

    // Helper function to create a task that increments a counter
    fn counting_task(counter: Arc<AtomicUsize>) -> impl Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            let counter = counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    #[test]
    fn test_simple_time_wheel_new() {
        let tick_duration = Duration::from_secs(1);
        let wheel_size = 10;
        let wheel = SimpleTimeWheel::new(tick_duration, wheel_size).unwrap();

        assert_eq!(wheel.tick_duration, tick_duration);
        assert_eq!(wheel.wheel_size, wheel_size);
        assert_eq!(wheel.current_slot, 0);
        assert_eq!(wheel.slots.len(), wheel_size);
        assert!(wheel.task_map.is_empty());
        assert!(wheel.start_time.is_none());
    }

    #[test]
    fn test_simple_time_wheel_calculate_slot_not_started() {
        let wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        let expires_at = Instant::now();

        let result = wheel.calculate_slot(expires_at);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_time_wheel_calculate_slot() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        let expires_at = wheel.start_time.unwrap() + Duration::from_secs(5);
        let result = wheel.calculate_slot(expires_at);

        assert!(result.is_ok());
        // Slot should be (5 / 1) % 10 = 5 (基于起始位置0计算)
        assert_eq!(result.unwrap(), 5);
    }

    #[test]
    fn test_simple_time_wheel_max_delay() {
        let tick_duration = Duration::from_secs(2);
        let wheel_size = 5;
        let wheel = SimpleTimeWheel::new(tick_duration, wheel_size).unwrap();

        let expected_max_delay = tick_duration * wheel_size as u32;
        assert_eq!(wheel.max_delay(), expected_max_delay);
    }

    #[test]
    fn test_simple_time_wheel_tick() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 3).unwrap();

        assert_eq!(wheel.current_slot, 0);
        wheel.tick();
        assert_eq!(wheel.current_slot, 1);
        wheel.tick();
        assert_eq!(wheel.current_slot, 2);
        wheel.tick();
        assert_eq!(wheel.current_slot, 0); // Should wrap around
    }

    #[test]
    fn test_simple_time_wheel_add_task_wheel_not_started() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        let expires_at = Instant::now() + Duration::from_secs(5);
        let task = ScheduleTask::new_once(success_task(), expires_at, None);
        let task_id = task.id;

        let result = wheel.add_task(task);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), task_id);
        assert!(wheel.start_time.is_some()); // Should start the wheel
    }

    #[test]
    fn test_simple_time_wheel_add_task_expired_task() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        let expires_at = Instant::now() - Duration::from_secs(1); // In the past
        let task = ScheduleTask::new_once(success_task(), expires_at, None);

        let result = wheel.add_task(task);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_time_wheel_add_task_exceeds_max_delay() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        let max_delay = wheel.max_delay();
        let expires_at = wheel.start_time.unwrap() + max_delay + Duration::from_secs(1); // Exceeds max delay
        let task = ScheduleTask::new_once(success_task(), expires_at, None);

        let result = wheel.add_task(task);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_time_wheel_add_task_success() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        let expires_at = wheel.start_time.unwrap() + Duration::from_secs(5);
        let task = ScheduleTask::new_once(success_task(), expires_at, None);
        let task_id = task.id;

        let result = wheel.add_task(task);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), task_id);
        assert_eq!(wheel.task_count(), 1);
        assert!(wheel.task_map.contains_key(&task_id));
    }

    #[test]
    fn test_simple_time_wheel_remove_task_not_exists() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        let task_id = Uuid::new_v4();

        let result = wheel.remove_task(task_id);
        assert!(result.is_none());
    }

    #[test]
    fn test_simple_time_wheel_remove_task_success() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        let expires_at = wheel.start_time.unwrap() + Duration::from_secs(5);
        let task = ScheduleTask::new_once(success_task(), expires_at, None);
        let task_id = task.id;

        wheel.add_task(task).unwrap();
        assert_eq!(wheel.task_count(), 1);

        let result = wheel.remove_task(task_id);
        assert!(result.is_some());
        assert_eq!(wheel.task_count(), 0);
        assert!(!wheel.task_map.contains_key(&task_id));
    }

    #[test]
    fn test_simple_time_wheel_get_ready_tasks_empty() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();

        let tasks = wheel.get_ready_tasks();
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn test_simple_time_wheel_get_ready_tasks_with_tasks() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        // 为了确保任务被放在当前槽位，我们需要计算正确的过期时间

        let expires_at = wheel.start_time.unwrap() + Duration::from_micros(100);
        let task = ScheduleTask::new_once(success_task(), expires_at, None);
        let task_id = task.id;

        wheel.add_task(task).unwrap();
        assert_eq!(wheel.task_count(), 1);
        tokio::time::sleep(Duration::from_secs(5)).await;

        let tasks = wheel.get_ready_tasks();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, task_id);
        assert_eq!(wheel.task_count(), 1); // task_map still has the task
        assert!(wheel.slots[wheel.current_slot].is_empty()); // But slot is empty
    }

    #[test]
    fn test_simple_time_wheel_task_count() {
        let mut wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        wheel.start_time = Some(Instant::now());

        assert_eq!(wheel.task_count(), 0);

        let expires_at = wheel.start_time.unwrap() + Duration::from_secs(5);
        let task = ScheduleTask::new_once(success_task(), expires_at, None);
        wheel.add_task(task).unwrap();

        assert_eq!(wheel.task_count(), 1);
    }

    #[test]
    fn test_simple_time_wheel_get_tick_duration() {
        let tick_duration = Duration::from_secs(2);
        let wheel = SimpleTimeWheel::new(tick_duration, 10).unwrap();

        assert_eq!(wheel.get_tick_duration(), tick_duration);
    }

    #[tokio::test]
    async fn test_time_wheel_runner_new() {
        let wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        let runner = TimeWheelRunner::new(wheel);

        assert_eq!(runner.tick_interval, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_time_wheel_runner_cancel_token() {
        let wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        let runner = TimeWheelRunner::new(wheel);

        let token = runner.cancel_token();
        assert!(!token.is_cancelled());
    }

    #[tokio::test]
    async fn test_time_wheel_runner_executes_tasks() {
        // Create a counter to track task executions
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        // Create a task that increments the counter
        let task_fn = move || -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
            let counter = counter_clone.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        };

        // Create time wheel and runner
        let wheel = SimpleTimeWheel::new(Duration::from_secs(1), 10).unwrap();
        let runner = TimeWheelRunner::new(wheel);
        let cancel_token = runner.cancel_token();
        let wheel_clone = runner.wheel();

        // Start the runner in background
        let handle = runner.run().await;

        {
            let mut wheel_guard = wheel_clone.lock().await;
            let expires_at = Instant::now() + Duration::from_secs(5);
            let task = ScheduleTask::new_once(task_fn, expires_at, None);
            wheel_guard.add_task(task).unwrap();
        }

        // Wait a bit for the task to execute
        tokio::time::sleep(Duration::from_secs(5)).await;

        // Cancel the runner
        cancel_token.cancel();

        // Wait for the runner to finish
        let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;

        // Check that the task was executed
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
