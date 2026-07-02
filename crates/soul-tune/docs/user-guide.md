# Soul-Tune 用户使用指南

## 概述

Soul-Tune 是 SoulMem 项目的 TUI 测试框架，用于可视化运行和展示记忆算法（检索、巩固、遗忘）的测试结果。

```
cargo run -p soul-tune
```

## 界面导航

### 1. 主界面

程序启动后显示主界面，列出可用算法类型。
按以下快捷键操作：

| 键 | 操作 |
|----|------|
| `:` | 进入命令模式 |
| `T` | 快速开始测试向导 |
| `Q` | 退出程序 |

### 2. 命令模式

按 `:` 进入命令模式，底部出现输入栏。

```
: test retrieve
```

| 键 | 操作 |
|----|------|
| `Enter` | 执行命令 |
| `Esc` | 取消，返回主界面 |
| `Tab` | 补全当前选中的建议 |
| `↑/↓` | 切换建议项 / 浏览历史命令 |

**内置命令：**

| 命令 | 别名 | 说明 |
|------|------|------|
| `test <algo>` | `t` | 开始算法测试。`algo` 为 `retrieve`、`consolidate` 或 `forget` |
| `help` | `h` | 显示帮助（预留） |
| `quit` | `q` | 退出程序 |

**补全提示：**
输入 `t` 时自动弹出补全建议列表，`Tab` 键可补全：
```
匹配命令:
  test — 运行算法测试
  test retrieve — 检索算法测试
  test consolidate — 巩固算法测试
  test forget — 遗忘算法测试
```

### 3. 选择数据集

执行 `test retrieve`（或 consolidate / forget）后进入数据集选择界面。

- 左侧显示当前目录下的 `.json` 文件和子目录
- 右侧显示选中文件的预览（名称、条目数等）
- 底部显示当前路径

| 键 | 操作 |
|----|------|
| `↑/↓` | 选择文件/目录 |
| `Enter` | 确认选择（文件）或进入目录（目录） |
| `Esc` | 返回主界面 |

### 4. 配置参数

选择数据集后进入参数配置界面。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 10 | 检索返回数量上限 |
| `threshold` | 0.7 | 相似度阈值 |
| `damping` | 0.85 | PPR 阻尼因子 |
| `iterations` | 20 | 迭代次数 |

| 键 | 操作 |
|----|------|
| `↑/↓` | 选择参数行 |
| `Enter` | 编辑当前值 |
| `Ctrl+Enter` | 开始测试 |
| `Esc` | 返回主界面 |

编辑状态下：

| 键 | 操作 |
|----|------|
| `Enter` | 确认编辑 |
| `Esc` | 取消编辑 |

### 5. 测试运行中

开始测试后显示进度界面，进度条实时更新。

- 模拟测试自动运行 100 个条目，约 1 秒完成
- 第 7 倍数条目标记为失败，其余通过

| 键 | 操作 |
|----|------|
| `Esc` / `Ctrl+C` | 中止测试，返回主界面 |

### 6. 测试结果

测试完成后显示结果界面，分为两个 Tab。

#### Summary 选项卡（默认）

左侧显示指标面板（1/3 宽度）：

```
性能
  总耗时: 1.0s
  条目总数: 100
准确率
  通过: 86
  失败: 14
  通过率: 86.0%
算法配置
  algo: retrieve
```

右侧显示相似度分布折线图（2/3 宽度），使用 ratatui 原生 Chart 组件渲染。

**图表特征：**
- X 轴：条目序号（自动生成等距标签）
- Y 轴：相似度值（自动分段显示）
- 折线图使用 Braille 点阵绘制

#### Detail 选项卡

逐条显示测试日志：

```
 条目   级别    相似度      消息
 [  1]  INFO    0.82      ✓ 通过
 [  2]  INFO    0.87      ✓ 通过
 [  7]  ERROR   ---       ✗ 失败 (预期≥3条, 返回2)
```

| 键 | 操作 |
|----|------|
| `←/→` | Summary 模式下切换指标组 |
| `Tab` | 切换 Summary / Detail |
| `↑/↓` | 滚动当前面板 |
| `F` | Detail 模式下切换日志筛选级别 |
| `/` | Detail 模式下搜索（预留） |
| `Q` | 返回主界面 |

## 数据集格式

测试数据集为 `.json` 文件，位于任意本地路径：

```json
{
  "name": "retr_basic",
  "description": "基础检索算法测试",
  "algo_type": "retrieve",
  "params": {
    "top_k": { "int": 10 }
  },
  "entries": [ ... ]
}
```

## 架构说明

### 状态机

```
Main → Command → SelectDataset → ConfigParams → TestRunning → TestResults
  ↑      ↑            ↑               ↑              ↑             │
  └──────┴────────────┴───────────────┴──────────────┴─────────────┘
                               Esc / Q
```

### 模块结构

```
src/
├── main.rs          # 程序入口
├── app.rs           # App 结构体 + 状态机 + 事件循环
├── base.rs          # AlgoType, Transition, TestConfig, TestReport
├── cmd.rs           # UserCmd, CmdRegistry (命令注册)
├── metric.rs        # Metric trait + MetricRegistry (指标注册)
├── reporter.rs      # TestReporter trait + ReporterRegistry (日志注册)
├── state/           # 各界面状态
│   ├── main.rs      # 主界面
│   ├── command.rs   # 命令模式
│   ├── dataset.rs   # 数据集选择
│   ├── params.rs    # 参数配置
│   ├── running.rs   # 测试运行 (含 mock tick)
│   └── results.rs   # 结果展示 (Summary + Detail Tab)
├── tui/
│   ├── wizard_page.rs  # 原有布局助手
│   └── components/     # 可复用 UI 组件
│       ├── command_bar.rs   # 命令输入条
│       ├── list.rs          # 可滚动列表
│       ├── kv_table.rs      # 键值对表格
│       ├── editable_table.rs # 可编辑表格
│       ├── chart.rs         # 图表封装
│       ├── tab_bar.rs       # Tab 切换
│       ├── status_bar.rs    # 底部状态栏
│       └── gauge.rs         # 进度条
└── utils/
    └── fuzzy.rs      # 模糊匹配 (nucleo)
```
