//! 工作记忆快照的可序列化视图。
//!
//! 快照用于持久化保存与启动恢复。带 schema 版本字段，便于未来格式迁移。
//!
//! 已知限制（Demo 期注明）：Record（活跃记录）的检索次数/反馈历史不随快照精确还原，
//! 恢复后按“重新 add_node 生成的新记录”处理；摘要仅保存文本，窗口 tag 状态近似还原。

use serde::{Deserialize, Serialize};
use soul_mem_query::embedding::note::EmbeddedMemoryNote;

/// 当前快照 schema 版本。结构变更时递增并在 `load` 处做迁移分派。
pub const SNAPSHOT_VERSION: u32 = 1;

/// 滑动窗口单条信息的持久化视图。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowEntryDto {
    /// "user" | "assistant"
    pub role: String,
    pub content: String,
    pub tagged: bool,
}

/// 完整工作记忆快照。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    pub summary: String,
    pub window: Vec<WindowEntryDto>,
    pub nodes: Vec<EmbeddedMemoryNote>,
}

impl Snapshot {
    pub fn new(
        summary: String,
        window: Vec<WindowEntryDto>,
        nodes: Vec<EmbeddedMemoryNote>,
    ) -> Self {
        Snapshot {
            version: SNAPSHOT_VERSION,
            summary,
            window,
            nodes,
        }
    }

    /// 校验版本可读。
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.version > SNAPSHOT_VERSION {
            return Err(crate::error::Error::InvalidArgument(format!(
                "snapshot version {} is newer than supported {}",
                self.version, SNAPSHOT_VERSION
            )));
        }
        Ok(())
    }
}
