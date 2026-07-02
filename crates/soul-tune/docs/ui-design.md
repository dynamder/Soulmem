# Soul-Tune 测试框架 UI 设计

## 状态机

```
Main ──(`:`)──▶ CommandMode ──(解析命令)──▶ SelectDataset ──▶ ConfigParams ──▶ TestRunning ──▶ TestResults
  ▲                    │  获取 algo_type       选择数据集        编辑参数       进度条       Summary|Detail
  │                    │
  └─── Esc ────────────┴───── Esc ──────────── Esc ─────────── Ctrl+C ────────┴─── Q ───┘
```

## App 结构体

```rust
pub struct App {
    pub state: AppState,
    pub cmd_registry: CmdRegistry,
    pub metric_registry: MetricRegistry,
    pub reporter_registry: ReporterRegistry,
}
```

## AppState 枚举

```rust
pub enum AppState {
    Main,
    CommandMode(CommandState),
    SelectDataset(DatasetState),
    ConfigParams(ParamState),
    TestRunning(RunningState),
    TestResults(ResultsState),
}
```

Transition: `None | To(AppState) | Quit`

## 各状态

### 1. MainState — 主界面

```
┌─ Soul-Tune · 记忆算法测试框架 ──────────────────────┐
│                                                     │
│  可用算法测试:                                       │
│    ● 检索 (retrieve)                                │
│    ● 巩固 (consolidate)                             │
│    ● 遗忘 (forget)                                  │
│                                                     │
│  输入 `:` 进入命令模式                               │
│  或按 `T` 快速开始测试向导                           │
│                                                     │
│─────────────────────────────────────────────────────│
│ [:]命令 [T]测试 [Q]退出                              │
└─────────────────────────────────────────────────────┘
```

按键: `:` → CommandMode | `T` → CommandMode("test ") | `Q` → Quit

### 2. CommandState — 命令模式

```
┌─ 命令模式 ──────────────────────────────────────────┐
│                                                     │
│  模糊匹配:                                          │
│    test retrieve                                    │
│    test consolidate                                 │
│    test forget                                      │
│  :t▌                                                │
│─────────────────────────────────────────────────────│
│ [Enter]执行 [Esc]取消 [Tab]补全                       │
└─────────────────────────────────────────────────────┘
```

```rust
pub struct CommandState {
    pub input: TextArea,
    pub suggestions: Vec<String>,
    pub selected_suggestion: usize,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,
}
```

按键: Enter → 解析 | Esc → Main | Tab → 补全 | ↑↓ → 切换建议/历史

### 3. DatasetState — 数据集选择

```
┌─ 选择数据集 · <算法名> ──────────────────────────────┐
│───────────────────────────┬──────────────────────────│
│  目录: /data/             │  预览                    │
│                           │                          │
│ ▶ file1.json             │  name: retr_basic        │
│   file2.json             │  entries: 50             │
│                           │                          │
│───────────────────────────┴──────────────────────────│
│  路径: /data/file1.json ▌                            │
│──────────────────────────────────────────────────────│
│ [↑↓]选择 [Tab]切换面板 [Enter]确认 [Esc]返回           │
└──────────────────────────────────────────────────────┘
```

```rust
pub struct DatasetState {
    pub algo_type: AlgoType,
    pub current_dir: PathBuf,
    pub entries: Vec<FileEntry>,
    pub selected: usize,
    pub path_input: TextArea,
    pub active_panel: Panel,
    pub preview_content: Option<String>,
}
```

### 4. ParamState — 参数配置

```
┌─ 参数配置 · <算法名> · <数据集> ──────────────────────┐
│                                                      │
│  Param          Value              Description        │
│  ────────────────────────────────────────────────── │
│ ▶ top_k      [ 10             ]  最大返回数          │
│   threshold  [ 0.7            ]  相似度阈值          │
│   damping    [ 0.85           ]  PPR 阻尼            │
│                                                      │
│──────────────────────────────────────────────────────│
│ [↑↓]选择 [Enter]编辑 [Ctrl+Enter]运行 [Esc]返回        │
└──────────────────────────────────────────────────────┘
```

```rust
pub struct ParamState {
    pub algo_type: AlgoType,
    pub dataset_path: PathBuf,
    pub rows: Vec<ParamRow>,
    pub selected: usize,
    pub editing: Option<usize>,
    pub scroll: usize,
}
pub struct ParamRow {
    pub name: String,
    pub value: String,
    pub description: String,
}
```

### 5. RunningState — 测试运行

```
┌─ ▶ 运行中 · <算法名> · <数据集> ───────────────────────┐
│                                                       │
│  ████████████████████░░░░░░░░░░░░  35/50 (70%)        │
│                                                       │
│  当前: 条目 #18                                       │
│  通过: 14    失败: 3    耗时: 1.2s                    │
│                                                       │
│───────────────────────────────────────────────────────│
│ [Ctrl+C]中止                                           │
└───────────────────────────────────────────────────────┘
```

```rust
pub struct RunningState {
    pub algo_type: AlgoType,
    pub dataset_path: PathBuf,
    pub total: usize,
    pub current: usize,
    pub passed: usize,
    pub failed: usize,
    pub elapsed: Duration,
    pub current_description: String,
}
```

### 6. ResultsState — 结果展示

```
┌─ ✓ 完成 · <算法> · <数据集> ── [Summary│Detail] ──────┐
│─────────────────────────┬──────────────────────────────│
│  性能                    │  ▲ 耗时分布                  │
│  ──────────────────     │  │ ██▄                       │
│  总耗时:    2.3s        │  │ ████                      │
│  平均:     46ms         │  └───────────────────────   │
│  最大:     180ms        │                              │
│                          │  ▲ 通过率趋势                │
│  准确率                  │  │   ╱                      │
│  ──────────────────     │  │ ╱                        │
│  通过率:    70%         │  │╱                         │
│  通过:      35          │  └───────────────────────   │
│  失败:      15          │                              │
│─────────────────────────┴──────────────────────────────│
│ [←→]切换指标组 [Tab]详情 [Q]返回                         │
└──────────────────────────────────────────────────────┘
```

Detail Tab:

```
┌─ ✓ 完成 · <算法> · <数据集> ── [Summary│Detail] ──────┐
│───────────────────────────────────────────────────────│
│  筛选: [全部 ▼]  搜索: [            ]                  │
│───────────────────────────────────────────────────────│
│  时间      级别   来源       消息                       │
│  ─────────────────────────────────────────────────── │
│  14:30:01  INFO   engine    测试开始                   │
│  14:30:01  INFO   entry[1]  ✓ 通过                    │
│  14:30:01  ERROR  entry[2]  ✗ 失败                    │
│───────────────────────────────────────────────────────│
│ [↑↓]滚动 [/]搜索 [F]筛选 [Q]返回                         │
└──────────────────────────────────────────────────────┘
```

```rust
pub struct ResultsState {
    pub algo_type: AlgoType,
    pub dataset_path: PathBuf,
    pub active_tab: ResultTab,
    pub kv_scroll: usize,
    pub metric_group_idx: usize,
    pub chart_scroll: usize,
    pub log_scroll: usize,
    pub log_filter: ReportLevel,
    pub log_search: String,
}
pub enum ResultTab { Summary, Detail }
```

## 组件列表

| 组件 | 文件 | 说明 |
|------|------|------|
| CommandBar | `tui/components/command_bar.rs` | 渲染 `:` + TextArea + 建议 |
| List | `tui/components/list.rs` | 可滚动列表，高亮选中 |
| KvTable | `tui/components/kv_table.rs` | 分组只读键值表 |
| EditableTable | `tui/components/editable_table.rs` | 可编辑表格 |
| Chart | `tui/components/chart.rs` | 封装 ratatui::widgets::Chart |
| TabBar | `tui/components/tab_bar.rs` | Tabs 切换 |
| StatusBar | `tui/components/status_bar.rs` | 底部快捷键提示 |
| Gauge | `tui/components/gauge.rs` | 进度条 |

## Trait 定义

### Metric

```rust
pub enum MetricDisplayKind { KeyValue, Chart { x_label, y_label } }
pub enum MetricData { Single(MetricValue), ChartPoints(Vec<(f64, f64)>) }
pub enum MetricValue { Int, Float, String, Duration, Percent, Bool }

pub trait Metric: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> &str;
    fn category(&self) -> &str;
    fn display_kind(&self) -> MetricDisplayKind;
    fn value(&self) -> MetricData;
}
pub struct MetricRegistry { metrics: Vec<Box<dyn Metric>> }
```

### Reporter

```rust
pub enum ReportLevel { Debug, Info, Warn, Error }
pub struct ReportEntry { pub time: String, pub level: ReportLevel, pub source: String, pub message: String }
pub trait TestReporter: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn entries(&self) -> Vec<ReportEntry>;
}
pub struct ReporterRegistry { reporters: Vec<Box<dyn TestReporter>> }
```

## SoulTuneEvent 扩展

```rust
pub enum SoulTuneEvent {
    CrossTerm(crossterm::event::Event),
    StartTest(AlgoType, Option<PathBuf>),
    TestComplete,
    Quit,
}
pub enum AlgoType { Retrieve, Consolidate, Forget }
```
