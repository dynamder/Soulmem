# Summary

[序章：一个记性不好的角色扮演 AI](README.md)

# 第一部分 · 核心概念

- [三类记忆](concepts/memory-taxonomy.md)
- [记忆的生命周期](concepts/memory-lifecycle.md)
- [长期记忆、工作记忆与滑动窗口](concepts/memory-layers.md)

# 第二部分 · 设计旅程

- [一切的起点：从"聊天应用"到"赛博生命"](design/origin.md)
- [调研：A-mem 与 HippoRAG 给了我们什么](design/research.md)
- [为什么用图来组织记忆](design/why-graph.md)
- [为什么是三类记忆](design/memory-taxonomy.md)
- [为什么用 PPR 做联想](design/why-ppr.md)
- [技术选型：为什么是 Rust，为什么少用 LLM](design/tech-choices.md)
- [为什么必须自部署](design/self-host.md)
- [为什么遗忘用"遮罩"而不是删除](design/why-forgetting.md)

# 第三部分 · 系统概览

- [总体架构](architecture/overview.md)
- [记忆模型](architecture/memory-model.md)
- [编排与数据流](architecture/orchestration.md)
- [集群架构](architecture/cluster.md)
- [Beta ver. 设计历史（过时参考）](architecture/beta_ver-design-history.md)

# 第四部分 · 深入实现

- [检索与联想](algorithm/retrieve.md)
- [遗忘算法](algorithm/forget.md)
- [巩固算法](algorithm/consolidate.md)
- [嵌入层](algorithm/embedding.md)
- [soul-mem-core](crates/soul-mem-core.md)
- [soul-mem-algo](crates/soul-mem-algo.md)
- [soul-mem-query](crates/soul-mem-query.md)
- [soul-mem-runtime](crates/soul-mem-runtime.md)
- [soul-tune](crates/soul-tune.md)
  - [soul-tune 用户指南](crates/soul-tune-user-guide.md)
  - [soul-tune UI 设计](crates/soul-tune-ui-design.md)

# 第五部分 · 实践与贡献

- [测试数据规范](testing/测试数据规范.md)
- [算法测试](testing/算法测试.md)
- [历史报告](testing/reports/README.md)
  - [Playtest 检索效果测试报告](testing/reports/playtest检索效果测试报告.md)
  - [全量角色 playtest 验证报告 - 抽象 PPR 检出](testing/reports/全量角色playtest验证报告-抽象PPR检出.md)
  - [抽象 PPR 检出心智模型落地报告](testing/reports/抽象PPR检出心智模型落地报告.md)
  - [检索算法改进轨迹报告](testing/reports/检索算法改进轨迹报告.md)

# 附录

- [研究笔记索引](research/README.md)
- [记忆算法概述](research/记忆算法概述.md)
- [WorkingMemory 并发安全与 API 重构](research/working_memory_fix.md)
- [PPR 性能报告](research/ppr_performance_report.md)
