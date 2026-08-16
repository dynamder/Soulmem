# soul-tune

> 依据 `feature/test_framework` 分支工作区代码整理。soul-tune 是 SoulMem 的**测试/评测
> 框架**（TUI + headless CLI），description 为 "Test framework for SoulMem."。

## 1. 职责定位与依赖

`soul-tune` 是独立于运行时组件的评测工具，依赖全部四个 crate（core/algo/query/runtime），
提供：检索评测 suite、playtest 角色对话测试、遗忘 T1–T5 评测、巩固评测（stub）、批量对比
与 TUI 界面。

```toml
[dependencies]
ratatui = "0.30.1"          # TUI
ratatui-textarea = "0.9.1"
nucleo = "0.5.0"            # 模糊匹配
color-eyre = "0.6.5"        # 错误报告
reqwest = { version = "0.12", default-features = false, features = ["blocking", "json"] }
paw-rs = { version = "0.2", default-features = false, features = ["llamacpp"] }  # llama.cpp 绑定
jieba-rs = { workspace = true }
rayon / rand / chrono / serde_json ...

[features]
default = ["llamacpp"]
llamacpp = []
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers", "dep:tokenizers"]
```

LLM 后端三种：`LlamaServer`（本地 llama.cpp server HTTP）、`Candle`（candle 原生推理，
需 `candle` feature）、`Qwen3.5`（qwen35，封装自 candle）。

> ⚠️ 注意：`src/eval/`、`src/state/`（单数）、`src/tui/`、`src/metric.rs`、`src/reporter.rs`
> 是**孤儿源码**——未被 `main.rs` 的 `mod` 引用、不参与编译（旧版遗留：eval=旧 engine、
> state=states 旧名副本、tui=widgets 旧名副本、metric/reporter=更早的 trait 注册表骨架）。
> 活代码只走 `app/base/cmd/component/engine/states/widgets/utils`。

## 2. 模块结构

```text
src/
├── main.rs             # 入口：headless CLI（run/playtest）+ TUI app
├── base.rs             # AlgoType / RetrieveMode / TestReport 基础类型
├── cmd.rs              # 命令解析
├── app.rs / app/event_loop.rs   # TUI 应用与事件循环
├── component.rs        # TUI 组件
├── engine.rs           # 评测引擎模块树
│   ├── suite.rs        # TestSuite trait、ReportMetric、MetricFormat（KV/Chart）
│   ├── dataset.rs      # 数据集加载
│   ├── loader.rs       # 加载器
│   ├── batch.rs        # 批量运行 + ActionSummary 汇总
│   ├── compare.rs      # 对比报告（CompareCaseData/CompareAggregate/CompareReport）
│   ├── retrieve/       # 检索评测
│   │   ├── suite.rs    # RetrieveSuite（TestSuite 实现）
│   │   ├── data.rs     # RetrieveCaseData / RankingMetrics / ActionMetrics
│   │   ├── dataset.rs  # RetrQueryFileRaw / SubQuery / sweep 展开
│   │   └── batch.rs    # process_one_dataset
│   ├── playtest/       # 角色对话测试
│   │   ├── runner.rs   # PlayTestRunner（核心）
│   │   ├── trace.rs    # RetrievalTrace / QueryTrace / TracedNode / HitStage
│   │   └── repair.rs   # 修复逻辑
│   ├── metrics.rs / metrics/ranking.rs  # Recall@K / MRR / NDCG / HitRate
│   ├── llm/            # backend.rs / candle_llm.rs / llama_server.rs / qwen35.rs
│   ├── consolidate.rs  # ConsolidateSuite（stub，0 用例）
│   └── forget/         # 遗忘评测
│       ├── suite.rs    # ForgetSuite（T1–T5）
│       ├── data.rs     # ForgetFileRaw 数据
│       └── metric.rs   # 指标
├── states/             # TUI 状态机（main_menu/batch/compare/playtest/...）
├── widgets/            # TUI 控件（chart/drilldown/editable_table/...）
└── tui/                # TUI 组件（components/、wizard_page）
```

## 3. CLI 用法

```bash
soul-tune run <algo> <dataset> [--batch]
soul-tune playtest <graph_dir> <dialogue_file> ...
soul-tune            # 无参数进入 TUI
```

`<algo>` 可选值（main.rs）：

| 参数 | 别名 | 模式 |
|------|------|------|
| `retrieve` / `retrieve/embedding` | `r` / `re` | 仅向量相似度检索 |
| `retrieve/association` | `ra` | PPR 联想检索 |
| `retrieve/full` | `rf` | 完整管线（ShortOnly→Similarity→AssociateWithAction） |
| `consolidate` | — | 巩固（stub） |
| `forget` | `f` | 遗忘 T1–T5 评测（不支持 --batch） |

示例（README 亦引用）：

```bash
# 完整检索管线批量评测
cargo run -p soul-tune -- run retrieve/full fixtures/example_data --batch
# 遗忘评测
cargo run -p soul-tune -- run forget fixtures/forget/forget_ebbinghaus_smoke.json
# 角色对话测试
cargo run -p soul-tune -- playtest fixtures/graphs fixtures/example_data/dialogue.json
```

## 4. 评测套件

### 4.1 TestSuite 抽象（engine/suite.rs）

```rust,ignore
pub trait TestSuite {
    fn case_count(&self) -> usize;
    fn run_case(&self, index: usize) -> TestCaseOutcome;
    fn build_report(&self, outcomes, elapsed, total, passed, failed) -> SuiteReport;
}
// 指标：key_value_metric(...) / chart_metric(...)，ReportMetric { label, group, format }
```

### 4.2 检索评测（RetrieveSuite）

- 数据：Query JSON（`RetrQueryFileRaw`：name/graph_path/config/blend_sweep/test_cases/
  expected_*），图 JSON（`GraphNodeRaw[]`）。
- **三种评测模式**（`RetrieveMode`）：
  - `Embedding`：纯向量相似度；
  - `Association`：相似度 + PPR 混合（`EMBED_PPR_BLEND` ≈ 0.5 权重）；
  - `FullPipeline`：DefaultPipeline（含动作输出）。
- **合并排序**：`merge_by_priority`——分数主导 + priority 小偏移（0.05），多 query 按优先级
  加权合并。
- **判定**：must（`expected_combined_ranking` 必须命中）+ bonus（`bonus_combined_ranking`
  加分）拆分。
- **权重扫描**：`blend_sweep`（tag_sweep/pairs）→ `expand_sweep_pairs` 展开；
  默认权重 **tag=0.3 / variant=0.7**（以 `BlendWeights::default` 代码为准）。
- **抽象检出指标**：`has_expected_abstract` / `abstract_detected` / `abstract_direct_hit`
  （期望抽象节点是否在合并结果中 / 是否被相似度直接命中）。
- 指标：`RankingMetrics { recall_at, precision_at, mrr, ndcg_at, hit_rate }`、
  `ActionMetrics { action_hit_rate, action_recall_at, has_expected_actions }`。
- 批量：`run_batch`（4 worker + `AtomicUsize` + mpsc）扫描 `question_*.json`；
  仅统计带 `has_expected_actions` 真值的用例，`summarize_action_metrics` 汇总动作命中率。

### 4.3 Playtest（PlayTestRunner）

- 加载图目录（`load(graph_dir)`）+ 对话文件（`DialogueFile` / `ConversationEntry`）。
- `process_turn`：PAW 实体提取 → 两段式查询生成（含 `QUERY_VALIDATION_FLOOR` 0.35 兜底
  校验 + 空回退）→ 双管线检索（相似度 + PPR）→ LLM 生成回复。
- **动作三通道**：Speak / Think / 行为（独立 top-k 进入最终结果）。
- 追踪：`RetrievalTrace` / `QueryTrace` / `TracedNode` / `HitStage`（记录命中阶段）。
- 输出：`PlayTestResult` / `PlayTurnResult` / `PlayRunSnapshot`，日志写
  `%TEMP%/soul_tune_playtest_log.txt` 等；评测含盲测投票机制。

### 4.4 遗忘评测（ForgetSuite，T1–T5）

验证艾宾浩斯遗忘曲线的五个可观测推论（对应 [测试数据规范](../testing/测试数据规范.md)
Forget JSON）：

| 用例 | 验证点 |
|------|--------|
| T1 时间单调 | 缺失度随 t 单调不减 |
| T2 激活抑制 | 缺失度随 retrieval_count 单调不增 |
| T3 量级校准 | 缺失度接近期望值 |
| T4 分段行为 | 三个时间点分别落入 NoAction / MaskOnly / Revised 区间 |
| T5 节点效果 | 触发率 + 语义熵增 |

T1–T4 为纯函数计算（`compute_missing_degree` / `lazy_forget` + mock LLM 闭包），T5 使用
注入的虚拟时长；`transform_score`（图编辑距离评分）为预留、未启用。

### 4.5 对比评测（compare）

`build_compare_report`：对多组结果按 `(case_name, 权重)` 对齐两侧，生成 `CompareReport`
（aggregate + per-case 对比）。

## 5. 与文档的对应关系

- [测试数据规范](../testing/测试数据规范.md)：Graph/Query/Forget JSON 格式的权威定义，
  与 `suite.rs`/`forget/suite.rs`/`fixtures/` 一致。
- [历史报告](../testing/reports/README.md)：playtest 报告与检索轨迹报告均基于 soul-tune
  实测产出。
- 检索算法（PPR/相似度/Bayes）实现在 `soul-mem-algo`，soul-tune 只做编排与评测。
