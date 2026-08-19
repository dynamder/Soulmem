# Soul-Tune GUI（Flutter + flutter_rust_bridge）

Soul-Tune 测试框架的 GUI 前端。Rust（`crates/soul-tune-api` + `crates/soul-tune` engine）负责真实测试与指标计算，Flutter 纯声明式渲染。

## 架构

```
soul-tune-ui/ (Flutter)  ──FRB(JSON)──►  crates/soul-tune-api/ (cdylib)
  纯渲染，无测试逻辑                      流式进度 + 报告 JSON
                                          │ 依赖
                                          ▼
                                       soul-tune engine/（真实测试）
```

- 数据契约：JSON-over-FRB，事件流见 `lib/src/models.dart`
- 交互设计：见 `UI_DESIGN.md`（卡片入口 / 表单 / 可排序表格 / 点击钻取）
- headless CLI（`cargo run -p soul-tune -- ...`）保留用于脚本/CI；TUI 已随 parity 移除

## 运行

```powershell
# 1. 生成/更新 FRB 绑定（api.rs 改动后需要）
cd soul-tune-ui
flutter_rust_bridge_codegen generate

# 2. 构建 Rust dll
cd ..
cargo build -p soul-tune-api

# 3. 运行（debug 构建会把 dll 放到 exe 旁；手动构建时需自行拷贝）
cd soul-tune-ui
flutter run -d windows
```

注意：Windows 构建需要系统开启"开发者模式"（插件 symlink 支持）。

## 当前功能

- 运行检索测试（embedding / association / full）：配置 → 实时进度 → 结果总览/明细 → 钻取
- 批量测试：目录扫描 → 并发执行 → 实时表格 → 单数据集钻取
- 对比测试：同数据集 embedding vs full pipeline，聚合卡 + 逐用例 Δ 表格 + 详情
- 检视数据集：question.json / graph.json 结构检视（可折叠 JSON 树）
- 遗忘测试：mask / revise / pipeline 三模式 + 逐用例观测页（节点列表、图原文、遮罩输入、LLM 原始回复）
- 角色扮演测试：选角色图 → 逐轮对话，embedding/full 双回复 + 可展开检索轨迹
- 数据集预览：选择 question.json 后展示名称/描述/用例数/图谱路径
- 取消：运行中红色"取消"按钮

## 待办（可选）

- 修复 playtest 查询生成相关的 3 个引擎测试（与迁移无关的既有断言问题）
- 数据集浏览器（原生目录浏览替代路径输入）
