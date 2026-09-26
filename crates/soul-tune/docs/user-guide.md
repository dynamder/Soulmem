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

### 检索子图深度审计

```
soul-tune depth-audit <question.json> [--depths 0,1,2,3] [--batch] [--verbose] [--json <path>]
```

**纯离线**：只读图与查询、在内存里跑 `DefaultPipeline`，**不碰数据库**。用途是判断
`prefetch_db` 的邻居扩展深度（`db_neighbor_depth`）够不够用。输出两部分：

- 期望节点距 **oracle 种子集**（全图相似度 top-k，即管线第一步）的无向最短跳数分布，
  分 0 跳 / 1 跳 / 2 跳 / ≥3 跳 / 跨组件不可达；并给出「直接模式召回到的期望节点里
  有多少位于 ≥2 跳」——即 **depth=1 的结构性损失上界**；
- 各跳数对照点（种子 0 跳 / 1 跳 / … / 全图）上的子图规模、覆盖率、通过用例数、
  期望命中数、Hit/MRR/Recall@3 与相对全图的丢失量。

用 oracle 种子而非真实 DB 候选，是为了把"深度"做成单变量：DB 候选还叠加了预算、
槽位 fan-out、HNSW 近似误差与字符串通道漏召，混在一起无法归因。

`--batch` 递归扫描目录下全部 `question.json`，逐数据集给一行摘要并附跨数据集汇总；
`--verbose` 追加逐用例明细；`--json <path>` 导出结构化结果（含各数据集的完整报告）。

### 依赖深度审计的等价性约束

`depth-audit` 的「全图」一行必须与 `retrieve/full` 的指标逐点一致，由
`engine::retrieve::depth_audit` 内的 `test_full_graph_matches_direct_suite` 锁定。
改了套件的合并/指标口径就要同步看那里。

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
  **注意该启发式与图规模无关**——大图上是否仍然够用需要单独验证（见 `depth-audit`）。
- `db_neighbor_depth`：数据库模式下 `prefetch_db` 的邻居扩展跳数（缺省 `1`）。
  `0` 表示关闭扩展（只把相似命中写进用例子图），用于观测"邻居扩展到底带来了什么"。
  由于工作记忆只物化被写入的节点，**图上距候选集超过该跳数的节点对该子图上的任何
  算法都不可达**，因此该值直接决定召回子图的可达范围。
- 数据库模式默认使用进程内 kv-mem 内存库（无磁盘残留）；传参 `db_path=<目录>` 时改用
  磁盘 SurrealKv 验证持久化读回。

## 架构

- `engine/`：真实测试逻辑（套件/指标/LLM/playtest），headless 可运行
- `crates/soul-tune-api/`：FRB 桥接层（JSON-over-FRB 流式进度）
- `soul-tune-ui/`：Flutter GUI（纯渲染）
- TUI（ratatui）已在 GUI 达 parity 后移除；历史版本见 git 记录。
