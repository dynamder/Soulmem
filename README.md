# SoulMem

[简体中文](README.md) | [English](README_en.md)

[![Project Status: WIP](https://img.shields.io/badge/Status-Active%20Development-orange)](https://github.com/dynamder/Soulmem)

[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

[![CI](https://github.com/dynamder/Soulmem/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamder/Soulmem/actions/workflows/ci.yml)

[![Mutants](https://github.com/dynamder/Soulmem/actions/workflows/mutants.yml/badge.svg)](https://github.com/dynamder/Soulmem/actions/workflows/mutants.yml)

SoulMem是一个专为角色扮演任务设计的记忆系统，它**旨在**使LLM的输出更拟人化成为可能，让模拟角色像人一样记住重要的、情感相关的、可驱动行为的事件，并建立关联。**它不旨在**精确无误地记忆事件的细节，或事实性知识。



**请注意！**：SoulMem是针对于**个人用户**，在**家用电脑**上运行的记忆系统，并非企业级解决方案。



## ✨ 核心特性

- ***记忆的整合与进化***：记忆会随着时间推移进行整合、概括，形成更高层次的认知。

- ***主动遗忘机制***：模拟人类的遗忘曲线，保留重要记忆，淡化琐碎细节。

- ***基于图谱的联想***：通过工作记忆子图实现记忆的主动联想。

- ***动态记忆更新***：支持在交互过程中实时添加和更新记忆。

- ***短期记忆抽象***：通过摘要机制处理短期上下文，防止信息过载。

## 🏗️设计哲学

SoulMem 的核心设计哲学是：***“一切特征和事件都属于记忆”***。

与传统角色扮演系统依赖静态的“角色卡”不同，SoulMem 认为角色的性格、口癖、行为习惯等都是长期记忆交互演化的结果。这种设计旨在更好地支持角色性格的动态演变，并保持高度的角色一致性。



## 📁 项目状态与架构

***当前状态：积极开发中***

*> 🚧 SoulMem 正在活跃开发中，尚未发布稳定版本。我们欢迎感兴趣的开发者关注、讨论甚至参与贡献！最新的架构设计和开发进展请参考* *`docs`* *目录。*

- ***重要通知***：项目已进行架构重构。`main` 分支为最新版本，旧的 alpha 版本代码可在 [`alpha_deprecated`](https://github.com/dynamder/SoulMem/tree/alpha_deprecated) 分支找到。

- ***当前架构***：请参阅 [`docs/architecture/orchestration.md`](docs/architecture/orchestration.md)——它带 ✅/🔲 标注，区分**已实现**与**规划中**，是与代码同步的架构说明。

- ***设计历史***：[`docs/architecture/beta_ver.md`](docs/architecture/beta_ver.md) 是 beta 阶段的**设计设想**（含未决问题与 `- [ ]` 待办），其中一部分已被实现取代，请勿当作现状。

- ***给 AI 代理与贡献者***：先读 [`AGENTS.md`](AGENTS.md)，其中有仓库地图、"从哪里开始读"索引、必须保持的不变量，以及不会报错但会静默出错的已知陷阱。

### 仓库结构

| 路径 | 说明 |
|---|---|
| `crates/soul-mem-core` | 纯数据模型（`MemoryNote` / `MemoryLink`），无内部依赖 |
| `crates/soul-mem-query` | 文本→向量嵌入、Query 类型、相似度与评分计算 |
| `crates/soul-mem-llm` | 全仓库唯一的 LLM 调用层（契约 / 传输 / 重试 / 流式 / trace） |
| `crates/soul-mem-runtime` | 工作记忆（滑动窗口、记忆簇、活跃记录）与 SurrealDB 仓储 |
| `crates/soul-mem-algo` | 检索策略、遗忘、巩固 |
| `crates/soul-tune` | 测试与基准框架（headless CLI + 库），**非运行时组件** |
| `crates/soul-tune-api` | Flutter Rust Bridge 桥接层 |
| `soul-tune-ui/` | Flutter GUI |
| `benches/` | criterion 基准（cosine / SIMD / PPR） |
| `fixtures/` | 测试数据集（角色图与对话） |
| `docs/` | 架构与规范文档（见 [`docs/README.md`](docs/README.md) 索引） |

## 🚀 快速开始

项目尚未发布稳定版本，但测试框架已经可用：

```bash
# 用 GUI（推荐，需要 Flutter 环境）
cd soul-tune-ui && flutter run -d windows

# 或用 headless CLI
cargo run -p soul-tune -- inspect fixtures/graphs/rust_small_zh.json
cargo run -p soul-tune -- run retrieve/full fixtures/example_data --batch
cargo run -p soul-tune -- playtest <graph_dir> <dialogue_file>
```

完整的命令说明、`algo` 取值表与数据集格式见 [`crates/soul-tune/docs/user-guide.md`](crates/soul-tune/docs/user-guide.md)。

> **注意**：部分测试与 playtest 需要下载嵌入模型（BGE）或本地 GGUF 模型。
> 离线环境下失败属于环境问题，不是构建坏了——详见 [`AGENTS.md`](AGENTS.md) §5。

## 🔧 开发与 CI

- 构建：`cargo build --workspace --all-targets`
- 测试：`cargo test --workspace`
- 格式与 lint：`cargo fmt --all --check`、`cargo clippy --workspace --all-targets -- -D warnings`
- 变异测试：`cargo mutants --workspace`（杀灭率 ≥90%，门禁脚本见 `scripts/mutants_gate.py`）
- push/PR 会触发 Windows、Ubuntu、macOS 三平台编译+测试；PR 还会按改动范围运行 mutants；每周（每隔一周）全量跑一次 mutants。



## 🤝 贡献

我们非常欢迎任何形式的贡献！无论是代码、文档、创意还是测试，都能帮助 SoulMem 成长。

1. Fork 本仓库
2. 创建您的功能分支
3. 提交您的更改
4. 推送到分支
5. 开启一个 Pull Request

请确保您的代码遵循项目已有的风格。

详细贡献指南请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。



## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。



## 🙏 致谢

感谢所有为这个项目提供想法和帮助的贡献者。
