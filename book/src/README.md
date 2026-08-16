# SoulMem 文档

[![Project Status: WIP](https://img.shields.io/badge/Status-Active%20Development-orange)](https://github.com/dynamder/SoulMem)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

SoulMem 是一个专为**角色扮演任务**设计的记忆系统，使 LLM 的输出更拟人化，让模拟角色像人一样
记住重要的、情感相关的、可驱动行为的事件并建立关联。

> ⚠️ SoulMem 处于**积极开发中**，尚未发布稳定版本。本项目文档以**本 mdBook** 为权威入口，
> 仓库根目录 `docs/` 下的旧文档将逐步迁移到这里。

## 文档地图

- **[架构](architecture/overview.md)**：总体架构、记忆模型、编排与数据流、集群设计
- **[算法](algorithm/retrieve.md)**：检索/联想（PPR）、遗忘、巩固、嵌入层
- **[Crate 参考](crates/soul-mem-core.md)**：五个 crate 的模块结构与公共 API
- **[测试与评测](testing/测试数据规范.md)**：测试数据规范、算法测试、历史评测报告
- **[研究笔记](research/README.md)**：开发过程笔记与调研资料索引

## 快速开始

当前处于早期开发阶段。`soul-tune`（评测框架 TUI）是主要的可运行入口：

```bash
# 运行检索评测 suite（需要先准备 fixtures 数据与本地 LLM server）
cargo run -p soul-tune -- run retrieve/full fixtures/example_data --batch
```

详细用法见 [soul-tune](crates/soul-tune.md)。

## 相关链接

- 源码：<https://github.com/dynamder/SoulMem>
- 开发状态：当前活跃分支为 `feature/test_framework`，见 [分支概览](research/README.md#分支概览)。
