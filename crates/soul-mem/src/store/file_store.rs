//! 文件快照实现（Demo）：JSON + 原子写盘（临时文件 + 改名）。
//!
//! 原子写保证“写一半崩溃”不会损坏旧快照；JSON 便于人工检查与调试。
//!
//! 并发安全：`save` 之间用内部互斥串行化，且临时文件名带唯一后缀，
//! 避免多个持久化调用（定时/控制信号/退出兜底）互相覆盖或 rename 半成品。

use crate::error::{Error, Result};
use crate::store::snapshot::Snapshot;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 临时文件后缀。
const TMP_SUFFIX: &str = ".tmp";

#[derive(Debug, Clone)]
pub struct FileStore {
    path: PathBuf,
    /// 串行化同一 FileStore 上的 save，避免并发写同一临时文件。
    save_lock: Arc<tokio::sync::Mutex<()>>,
}

impl FileStore {
    pub fn new(path: PathBuf) -> Self {
        FileStore {
            path,
            save_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// 唯一临时路径：`<path>.<uuid>.tmp`（避免并发使用同一临时文件）。
    fn tmp_path(&self) -> PathBuf {
        let mut s = self.path.as_os_str().to_owned();
        s.push(format!(".{}{TMP_SUFFIX}", uuid::Uuid::new_v4()));
        PathBuf::from(s)
    }
}

impl FileStore {
    pub async fn save(&self, snapshot: &Snapshot) -> Result<()> {
        // 串行化同一 store 的并发 save。
        let _guard = self.save_lock.lock().await;

        snapshot.validate()?;
        let json = serde_json::to_string_pretty(snapshot)
            .map_err(|e| Error::internal(format!("serialize snapshot: {e}")))?;

        let parent: Option<&Path> = self.path.parent().filter(|p| !p.as_os_str().is_empty());
        if let Some(dir) = parent {
            tokio::fs::create_dir_all(dir).await.map_err(|e| {
                Error::internal(format!("create snapshot dir {}: {e}", dir.display()))
            })?;
        }

        // 原子写：先写唯一临时文件，再改名覆盖目标文件。
        let tmp = self.tmp_path();
        if let Err(e) = tokio::fs::write(&tmp, json.as_bytes()).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::internal(format!("write temp snapshot: {e}")));
        }
        if let Err(e) = tokio::fs::rename(&tmp, &self.path).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::internal(format!("commit snapshot rename: {e}")));
        }
        Ok(())
    }

    pub async fn load(&self) -> Result<Option<Snapshot>> {
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

    fn snapshot(summary: &str) -> Snapshot {
        Snapshot::new(
            summary.to_string(),
            vec![WindowEntryDto {
                role: "user".to_string(),
                content: "hello".to_string(),
                tagged: false,
            }],
            Vec::new(),
        )
    }

    fn tmp_files_in(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(TMP_SUFFIX))
            .collect()
    }

    #[tokio::test]
    async fn snapshot_roundtrip_and_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().join("snapshot.json"));

        let snap = snapshot("summary");
        store.save(&snap).await.unwrap();
        assert_eq!(store.load().await.unwrap().unwrap(), snap);

        let snap2 = snapshot("new");
        store.save(&snap2).await.unwrap();
        assert_eq!(store.load().await.unwrap().unwrap().summary, "new");
        // 覆盖后无临时文件残留。
        assert!(tmp_files_in(dir.path()).is_empty());
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().join("does-not-exist.json"));
        assert!(store.load().await.unwrap().is_none());
    }

    /// 回归：并发 save 不得损坏快照，且最终可正常解析。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_save_is_consistent() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().join("snapshot.json"));

        let mut tasks = Vec::new();
        for i in 0..16 {
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                store.save(&snapshot(&format!("s{i}"))).await.unwrap();
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }

        let loaded = store.load().await.unwrap().unwrap();
        assert!(
            (0..16).any(|i| loaded.summary == format!("s{i}")),
            "loaded snapshot should equal one of the concurrent writes, got {:?}",
            loaded.summary
        );
        assert!(
            tmp_files_in(dir.path()).is_empty(),
            "no temp files should remain"
        );
    }

    /// save 到不可写位置（父路径是文件）应返回错误而非 panic。
    #[tokio::test]
    async fn save_error_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"i am a file").unwrap();
        let store = FileStore::new(blocker.join("child.json"));
        let err = store.save(&snapshot("x")).await.unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
    }
}
