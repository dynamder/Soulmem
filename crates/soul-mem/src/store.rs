//! 持久化抽象：`MemoryStore` trait + 快照 + 实现。
//!
//! 把「存到哪」抽象化：Demo 内置文件快照实现（`FileStore`），
//! 未来可新增 SurrealDB 等实现而不改上层（见 plan.md §2.6）。
//!
//! 说明：Rust 中带 `async fn` 的 trait 默认不可 `dyn`，故服务侧持有的是
//! 有穷枚举 `Store`，既保留可替换性又便于运行时分发。

mod file_store;
mod snapshot;

pub use file_store::FileStore;
pub use snapshot::{Snapshot, WindowEntryDto};

use crate::error::Result;

/// 内存状态快照的保存/加载接口。
///
/// 仅在本 crate 内部作为“实现契约”使用（不经 dyn 分发，auto-trait 约束无碍）。
#[allow(async_fn_in_trait)]
pub trait MemoryStore: Send + Sync {
    /// 保存一份完整快照；由实现方保证原子性与覆盖语义。
    async fn save(&self, snapshot: &Snapshot) -> Result<()>;

    /// 加载最近一份快照；若不存在返回 `Ok(None)`。
    async fn load(&self) -> Result<Option<Snapshot>>;
}

/// 不做任何持久化的空实现（等价于“不落盘”）。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopStore;

impl MemoryStore for NoopStore {
    async fn save(&self, _snapshot: &Snapshot) -> Result<()> {
        Ok(())
    }

    async fn load(&self) -> Result<Option<Snapshot>> {
        Ok(None)
    }
}

/// 有穷后端枚举：服务运行期直接持有（避免 dyn trait 对象）。
#[derive(Debug, Clone)]
pub enum Store {
    Noop(NoopStore),
    File(FileStore),
}

impl Default for Store {
    fn default() -> Self {
        Store::Noop(NoopStore)
    }
}

impl Store {
    /// 依据配置选择后端：有路径 → FileStore，否则 NoopStore。
    pub fn from_config(path: Option<std::path::PathBuf>) -> Self {
        match path {
            Some(path) => Store::File(FileStore::new(path)),
            None => Store::Noop(NoopStore),
        }
    }

    pub async fn save(&self, snapshot: &Snapshot) -> Result<()> {
        match self {
            Store::Noop(store) => store.save(snapshot).await,
            Store::File(store) => store.save(snapshot).await,
        }
    }

    pub async fn load(&self) -> Result<Option<Snapshot>> {
        match self {
            Store::Noop(store) => store.load().await,
            Store::File(store) => store.load().await,
        }
    }
}
