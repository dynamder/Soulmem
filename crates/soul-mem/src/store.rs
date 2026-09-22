//! 持久化：快照 + 后端实现。
//!
//! 把「存到哪」抽象化：Demo 内置文件快照实现（`FileStore`），
//! 服务侧持有有穷枚举 `Store`（`Noop`/`File`）做运行时分发。
//! 未来可新增 SurrealDB 等实现并扩展 `Store` 变体。

mod file_store;
mod snapshot;

pub use file_store::FileStore;
pub use snapshot::{Snapshot, WindowEntryDto};

use crate::error::Result;

/// 不做任何持久化的空实现（等价于“不落盘”）。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopStore;

impl NoopStore {
    pub async fn save(&self, _snapshot: &Snapshot) -> Result<()> {
        Ok(())
    }

    pub async fn load(&self) -> Result<Option<Snapshot>> {
        Ok(None)
    }
}

/// 有穷后端枚举：服务运行期直接持有。
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
