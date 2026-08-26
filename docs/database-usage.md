# SoulMem 数据库快速使用

## 1. 修改配置

所有连接信息都在项目根目录的 `soulmem.env`：

```text
API_KEY=LLM密钥
API_BASE=LLM接口地址
MODEL=LLM模型名称

SURREAL_ENDPOINT=ws://127.0.0.1:8000
SURREAL_NAMESPACE=test
SURREAL_DATABASE=main
SURREAL_USERNAME=test
SURREAL_PASSWORD=test
SURREAL_PATH=rocksdb://soulmem.db
```

通常只需要修改这个文件。

## 2. 启动数据库

在项目根目录执行：

```bash
./start-database.sh
```

脚本会自动读取 `soulmem.env`。保持这个终端运行。

## 3. 在 Rust 中使用

```rust
use soul_mem_query::storage::{MemoryRepository, SurrealMemoryRepository};
use soul_mem_runtime::settings::AppSettings;

let settings = AppSettings::load()?;
let repository = SurrealMemoryRepository::new(settings.database);

repository.bootstrap().await?;
repository.upsert_note(&note).await?;
let saved = repository.get_note(note.id()).await?;
repository.delete_note(note.id()).await?;
```

常用方法：

- `upsert_note()`：保存节点
- `get_note()`：根据 ID 查询节点
- `upsert_link()`：保存节点之间的边
- `load_note()`：读取节点和它发出的边
- `delete_note()`：删除节点及相关边
- `query_similar_notes()`：查询 Top-K 相似节点

## 4. 查看和测试

在 Surrealist 中使用 `soulmem.env` 里的配置连接，然后执行：

```surql
SELECT * FROM memory_note;
SELECT * FROM memory_link;
```

运行 Rust 数据库测试：

```bash
cargo test -p soul-mem-query \
  storage::surreal::tests::test_repository_roundtrip \
  -- --ignored --nocapture
```

测试结果显示 `ok`，说明数据库基本流程正常。
