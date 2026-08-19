# Soul-Tune GUI 设计规范

Flutter 应用（`soul-tune-ui/`）+ Rust FRB 桥接（`crates/soul-tune-api/`）。
本文件是"交互设计 v2"的可执行版：**借鉴 TUI 已验证的信息架构，视觉做减法，
交互全面 GUI 化（直觉式，非命令式）**。

## 1. 设计原则

| 原则 | 说明 |
|---|---|
| 信息架构沿用 TUI | 指标分组 → 逐用例状态 → 钻取详情，这条观察路径已被验证有效 |
| 直觉式交互 | 卡片入口、表单、原生文件对话框、可排序表格、点击钻取；无命令输入 |
| 视觉减法 | Material 3 深色主题；单一强调色（靛蓝 `#3F51B5`）；灰阶层级；无渐变/阴影噪音 |
| 柔和状态色 | 通过/失败用 Material 300 级低饱和色（`#81C784`/`#E57373`）+ 10% 透明度底色徽章，避免刺眼对比（见 `theme.dart`） |
| 数字优先 | 指标值等宽字体放大；表格数字右对齐；状态以徽章呈现 |
| 纯渲染 | Flutter 零测试逻辑；所有数据组装在 Rust api 层完成 |

## 2. 页面结构与导航

```
Home ──► RunConfig（表单）──► Run（进度流）──► Results（总览/明细）──► CaseDetail（钻取）
  │
  ├──► BatchConfig ──► Batch（实时表格）──► Results（单数据集复用）
  ├──► CompareConfig ──► CompareRun（双阶段进度）──► CompareResults（Δ表格）──► CompareCaseDetail
  ├──► Inspect（文件选择 + 可折叠 JSON 树）
  ├──► ForgetConfig（模式+图）──► ForgetRun ──► ForgetResults（汇总/观测：逐节点 图原文/遮罩/LLM 回复）
  └──► PlaytestConfig（选角色图）──► PlaytestChat（逐轮对话：A embedding / B full 双回复 + 轨迹）
```

导航栈 + 返回按钮（非 Esc）；运行中页面提供红色"取消"按钮。

## 3. 每页规格

### HomePage
- 动作卡片（图标 + 标题 + 一句话说明）：运行检索测试 / 批量测试 可用；
  检视数据集 / 角色扮演测试 置灰标注"规划中"。

### RunConfigPage（单页表单，替代 TUI 向导多步状态）
1. **算法与模式**：`SegmentedButton`（embedding / association / full），tooltip 解释差异
2. **数据集**：`[选择文件…]`（file_picker，过滤 json）+ 路径输入框 + 预览卡
   （名称/描述/用例数/graph_path，来自 `dataset_meta_json`）+ 会话内"最近使用"chips
3. **参数**：top_k / threshold 数字输入 + helper 说明 + 恢复默认
4. **开始测试**：路径有效且元数据无错才可点

### RunPage
- 加载态：进度条 + 消息；运行态：粗进度条 + 大数字（通过/失败/耗时）+ 当前用例 + 取消
- 完成自动跳 ResultsPage；取消/错误回退并提示

### ResultsPage（GUI 化）
- 摘要区：**通过率环形图**（低饱和状态色，≥80% 绿 / 50-80% 琥珀 / <50% 红）+ 统计卡（通过/失败/耗时）
- `[总览 | 明细]` 分段切换
- 总览：**指标分组卡片网格**（组标题强调条 + 指标磁贴，图表指标内嵌迷你折线）
  + 逐用例状态条（柔和绿/红圆角柱）
- 明细：搜索框 + 全部/通过/失败筛选 + **列头点击排序**（用例/MRR/Hit/状态）
  + 斑马纹行 + **状态徽章**（低饱和底色胶囊），点击行 → 钻取页

### CaseDetailPage（钻取，卡片化）
- 分节卡片（强调条标题）：综合排序指标 / 各子查询 / 检索 vs 期望 / 动作指标 / 抽象检出
- 检索 vs 期望对比行：命中行**柔和绿底 + 对勾**，检索↔期望双向箭头
- 顶部柔和状态徽章（通过/失败）

### BatchPage
- 顶部总进度条 + 实时统计；数据集行实时插入（状态色：完成率≥80% 绿 / 失败红 / 错误橙）
- 完成后按通过率排序；点击行 → 复用 ResultsPage 钻取该数据集
- 运行中提供取消

## 4. 数据契约（Rust → Flutter）

全部 JSON-over-FRB（`Stream<String>`）。事件统一内部标签：

```
单跑: {"type":"loading"|"progress"|"done"|"error"|"cancelled", ...}
批量: {"type":"scanning"|"progress"|"dataset_done"|"done"|"error"|"cancelled", ...}
```

- 指标：`MetricEntry{label, group, kind:"key_value"|"chart", value?, x_label?, y_label?, datasets?}`
- 报告：`Report{algo, dataset_name, total, passed, failed, elapsed_secs, metrics[], detail_rows[], outcomes[]}`
- 用例数据：`outcomes[].data` = `RetrieveCaseData` 序列化（ranking/per-query/action/abstract）

## 5. 代码结构

```
lib/
├── main.dart                 # 应用壳（Material 3 深色主题）
├── src/
│   ├── rust/                 # codegen 生成（不入库）
│   ├── models.dart           # 全部 fromJson 模型
│   ├── bridge.dart           # FRB 封装：初始化 + 强类型事件流
│   ├── screens/              # home / run_config / run / results / case_detail / batch
│   └── widgets/              # metric_panel / mini_bar_chart / stat_card
```

## 6. 验收

- 同一数据集下 Flutter 与 `soul-tune run` 输出数值一致
- 取消可用；错误页面内提示；空态友好
