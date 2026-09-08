# Soul-Tune 使用指南

Soul-Tune 是 SoulMem 项目的测试框架。**GUI 前端**（Flutter）与 **headless CLI** 两种使用方式。

## GUI（推荐）

```
cd soul-tune-ui
flutter run -d windows
```

功能：运行检索测试 / 批量测试 / 对比测试 / 检视数据集 / 遗忘测试（含逐节点观测）/ 角色扮演测试。
交互设计见 `soul-tune-ui/UI_DESIGN.md`，架构见 `soul-tune-ui/README.md`。

## Headless CLI

```
cargo run -p soul-tune -- <inspect|run|playtest> ...
```

### 检视数据集

```
soul-tune inspect <graph.json|question.json>
```

打印图节点/连接或测试用例的结构化条目。

### 运行测试

```
soul-tune run <algo> <dataset> [--batch]
```

`algo`：
- `retrieve/embedding`（`re`）/ `retrieve/association`（`ra`）/ `retrieve/full`（`rf`）：
  直接模式——example_data（`graph.json`）全量载入工作记忆后跑检索；
- `retrieve/db`（`rd`）/ `retrieve/db/embedding`（`rde`）/ `retrieve/db/association`（`rda`）：
  数据库模式——example_data 先全量写入 mem 数据库（默认进程内 kv-mem），
  每个用例先经 DB 召回（HNSW 候选 + 一跳邻居）构建工作记忆子图，再跑同一条检索管线；
- `compare/db` / `compare/db/embedding` / `compare/db/association`：
  同管线「直接 vs 数据库」逐用例对比（两侧 Hit/MRR/Recall@3 与差量、提升/回退/持平计数）；
- `consolidate`（`c`）
- `forget`（`f`）/ `forget/mask`（`fm`）/ `forget/revise`（`fr`）

`--batch` 模式仅支持 retrieve：递归扫描目录下全部 `question.json` 并并发执行
（直接模式与 `retrieve/db/*` 数据库模式均可）。

### 角色扮演测试

```
soul-tune playtest <graph_dir> <dialogue_file>
```

需要环境变量 `SOUL_TUNE_CANDLE_MODEL_PATH`（自动拉起 llama-server）或
`SOUL_TUNE_LLAMA_URL`（直连已运行服务）。

## 数据集格式

测试数据集为 `question.json`（见 `fixtures/example_data/`）：

```json
{
  "name": "retr_basic",
  "description": "基础检索算法测试",
  "graph_path": "graph.json",
  "config": { "similarity_threshold": 0.7, "max_results": 10, "test_k_values": [1, 3, 5] },
  "test_cases": [ { "name": "...", "sub_queries": [...], "expected_combined_ranking": [...] } ]
}
```

`config` 额外可选字段：

- `db_candidate_k`：数据库模式（`retrieve/db/*`）下，DB 端每个槽位的 HNSW KNN 候选召回预算
  （精确重排与 top-k 截断在内存侧完成）。缺省用启发式 `max(2 * max_results, 20)`；
  也可在 GUI/API 参数中传 `db_candidate_k` 覆盖。预算越小，召回子图越接近真实 DB 路径
  （可能漏掉低相似度期望节点）；预算 ≥ 图节点数时 DB 路径与直接模式结果一致。
- 数据库模式默认使用进程内 kv-mem 内存库（无磁盘残留）；传参 `db_path=<目录>` 时改用
  磁盘 SurrealKv 验证持久化读回。

## 架构

- `engine/`：真实测试逻辑（套件/指标/LLM/playtest），headless 可运行
- `crates/soul-tune-api/`：FRB 桥接层（JSON-over-FRB 流式进度）
- `soul-tune-ui/`：Flutter GUI（纯渲染）
- TUI（ratatui）已在 GUI 达 parity 后移除；历史版本见 git 记录。
