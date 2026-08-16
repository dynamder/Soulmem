# 研究笔记

本部分收录 SoulMem 开发过程中的调研笔记、方案设计与性能报告。原始资料位于仓库
`developing_notes/`（该目录被 .gitignore 忽略、不参与版本控制），此处为纳入文档体系的
整理版。

## 笔记索引

| 笔记 | 主题 |
|------|------|
| [记忆算法概述](记忆算法概述.md) | 记忆算法三大类（检索/巩固/遗忘）的设计总览与 Soul-Retr 检索思路 |
| [WorkingMemory 并发安全与 API 重构](working_memory_fix.md) | 工作记忆滑动窗口的并发安全方案（2026-04） |
| [PPR 性能报告](ppr_performance_report.md) | Power Iteration 与 Forward Push 的性能对比实测 |
| ReMindRAG 笔记 | ReMindRAG（路径记忆 + 知识图谱搜索）调研（见 developing_notes/ReMindRAG_notes.md） |
| Agent Memory Survey 笔记 | 大模型 Agent 记忆机制综述（见 developing_notes/Agent Memory Survey notes.md） |
| Qwen 架构建议 | 程序性记忆子图的重构建议（TriggerContext/BehaviorPattern）（见 developing_notes/qwen的架构建议.md） |
| 重构笔记 | 第一次重构目标与 memory 模块重构方案（见 developing_notes/refactoring.md） |

> 注：`记忆算法概述.md` 与 `working_memory_fix.md`、`ppr_performance_report.md` 已直接纳入
> 本 mdBook；其余笔记体积较大或与主线文档重叠，保留在 `developing_notes/` 中供查阅。

## 参考资料（PDF，位于 developing_notes/）

| 资料 | 说明 |
|------|------|
| A-mem | Agentic RAG，本项目重要启发之一 |
| HippoRAG | 知识图谱 + PPR 联想检索，本项目检索算法重要启发 |
| PPR Introduction | PPR 算法综述（概念与常见计算算法） |
| EdgePush-PPR | 带边权图的 PPR 算法 |
| 人脑记忆机制与功能分类深度研究报告 | 脑认知科学参考 |
| 人类程序性记忆的联想机制深度研究报告 | 程序性记忆的神经科学基础 |
| Memory in the Age of AI Agents / Agent Memory Survey | 大模型记忆综述 |
| SoulMem 深度研究报告（×2） | 角色扮演大模型记忆系统的可行性与优化路径 |

## 分支概览

> 仓库存在大量 feature 分支，文档以当前活跃分支 `feature/test_framework`（含工作区未提交
> 修改）的代码状态为准。

| 分支 | 最新提交 | 主题 | 状态 |
|------|---------|------|------|
| `main` | 2026-03-29 | 基线（合并 devel/retrieve 的 PR #8） | 基线分支，已落后 |
| `dev` | 2026-07-18 | 集成分支：candle 0.11.0 升级、检索合入 | 活跃主干，已并入 test_framework |
| `feature/test_framework` | 2026-08-13 | 测试框架主线：playtest/eval、24 图、抽象 PPR 检出 | **活跃（当前分支）** |
| `feature/retrieve_algo` | 2026-08-13 | 检索算法：PPR 截断修复、字符串评分、insta 快照 | 已并入 test_framework |
| `feature/cluster_refactor` | 2026-04-04 | 记忆簇重构、RetrStrategy 关联类型 | 已完成（已并入），本地独有 |
| `feature/consolidate` | 2026-07-22 | 巩固 + 数据库 schema | **未合并**，疑似搁置（无 DB 路线取代） |
| `feature/forget` | 2026-04-18 | 遗忘 v1（遮罩 + Mask 字段） | **废弃**，被 newforge 取代 |
| `feature/newforge` | 2026-07-10 | 遗忘 v2：Ebbinghaus 衰减 + 遮罩 + LLM 修订 | **未合并**；算法以未跟踪形式存在于工作树 |
| `feat/abstract-ppr-detection` | 2026-08-13 | 抽象 PPR 检出试点 + 24 角色验证 | 未合并（内容并入 test_framework），本地独有 |
| `ci/setup` | 2026-08-09 | GitHub Actions CI + mutants 门禁 | **未合并**（独立 CI 分支） |
| `devel/embedding`、`devel/retrieve`、`devel/situation_mem`、`devel/sliding_window` | 2026-02~03 | 各功能开发 | 均已合并入 main |
| `docs/add_mem_algo` | 2025-12-04 | 记忆算法文档 | 已合并入 main，本地独有 |
| `alpha_deprecated` | 2025-10-08 | alpha 旧版存档 | 废弃存档 |
| `backup/feature/test_framework-pre-rewrite` | 2026-08-08 | 重写前快照 | 备份分支 |
| `ron-yc/WorkingMem` | 2026-03-13 | 个人工作记忆实验 | 废弃 |

> ⚠️ **重要**：遗忘算法实现（`soul-mem-algo/src/algo/forget/`、`soul-tune/src/engine/forget/`、
> `fixtures/forget/`）与 `docs/architecture/cluster.md`、本 mdBook `book/` 目前仅存在于
> **本地工作树（untracked）**，尚未进入任何 git 分支。建议尽快提交，避免工作树内容丢失。
