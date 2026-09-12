//! 文件快照实现（Demo）：JSON + 原子写盘（临时文件 + 改名）。
//!
//! 原子写保证“写一半崩溃”不会损坏旧快照；JSON 便于人工检查与调试。

use crate::error::{Error, Result};
use crate::store::MemoryStore;
use crate::store::snapshot::Snapshot;
use std::path::{Path, PathBuf};

/// JSON 快照文件后缀。
const TMP_SUFFIX: &str = ".tmp";

#[derive(Debug, Clone)]
pub struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub fn new(path: PathBuf) -> Self {
        FileStore { path }
    }

    fn tmp_path(&self) -> PathBuf {
        let mut s = self.path.as_os_str().to_owned();
        s.push(TMP_SUFFIX);
        PathBuf::from(s)
    }
}

impl MemoryStore for FileStore {
    async fn save(&self, snapshot: &Snapshot) -> Result<()> {
        snapshot.validate()?;
        let json = serde_json::to_string_pretty(snapshot)
            .map_err(|e| Error::internal(format!("serialize snapshot: {e}")))?;

        let parent: Option<&Path> = self.path.parent().filter(|p| !p.as_os_str().is_empty());
        if let Some(dir) = parent {
            tokio::fs::create_dir_all(dir).await.map_err(|e| {
                Error::internal(format!("create snapshot dir {}: {e}", dir.display()))
            })?;
        }

        // 原子写：先写临时文件，再改名覆盖目标文件。
        let tmp = self.tmp_path();
        tokio::fs::write(&tmp, json.as_bytes())
            .await
            .map_err(|e| Error::internal(format!("write temp snapshot: {e}")))?;
        tokio::fs::rename(&tmp, &self.path)
            .await
            .map_err(|e| Error::internal(format!("commit snapshot rename: {e}")))?;
        Ok(())
    }

    async fn load(&self) -> Result<Option<Snapshot>> {
        let raw = match tokio::fs::read_to_string(&self.path).await {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::internal(format!("read snapshot: {e}"))),
        };
        let snapshot: Snapshot = serde_json::from_str(&raw)
            .map_err(|e| Error::InvalidArgument(format!("snapshot json malformed: {e}")))?;
        snapshot.validate()?;
        Ok(Some(snapshot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::snapshot::WindowEntryDto;

    #[tokio::test]
    async fn snapshot_roundtrip_and_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().join("snapshot.json"));

        let snap = Snapshot::new(
            "summary".to_string(),
            vec![WindowEntryDto {
                role: "user".to_string(),
                content: "hello".to_string(),
                tagged: false,
            }],
            Vec::new(),
        );
        store.save(&snap).await.unwrap();
        let loaded = store.load().await.unwrap().unwrap();
        assert_eq!(loaded, snap);

        // 覆盖语义：新快照替换旧快照，且旧临时文件不残留。
        let snap2 = Snapshot::new("new".to_string(), Vec::new(), Vec::new());
        store.save(&snap2).await.unwrap();
        let loaded2 = store.load().await.unwrap().unwrap();
        assert_eq!(loaded2.summary, "new");
        assert!(!store.tmp_path().exists());
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().join("does-not-exist.json"));
        assert!(store.load().await.unwrap().is_none());
    }
}
