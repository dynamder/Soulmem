# docs/ 索引

本目录同时存放**规范（权威）**、**架构说明**与**一次性实验报告**。三类文档的权威性不同，
请按下面的分类取用——**不要从"哪个名字更像规范"去猜**。

> 另有 mdBook 形式的完整文档（`concepts/` `algorithm/` `crates/` `testing/`，含写作规范），
> 其源码保留在 `doc/book` 分支，**当前分支不含书源码**。书稿由维护者手工逐章撰写，正在推进中。

---

## 1. 规范类 —— 权威，写代码/数据前先看这里

| 文档 | 内容 |
|---|---|
| [`测试数据规范.md`](测试数据规范.md) | **测试数据集的唯一权威规范**：Graph JSON、Query JSON、BlendSweep 配置、完整示例。与 `fixtures/` 实际内容一致 |
| [`architecture/激发测试设计.md`](architecture/激发测试设计.md) | 激发测试（excitation）的接口契约、断言清单 E1~E6、延缓指标定义与质量门槛 |
| [`architecture/算法测试.md`](architecture/算法测试.md) | 记忆算法测试的流程定义与图谱变换操作 |

## 2. 架构类 —— 描述系统设计

| 文档 | 内容 | 状态 |
|---|---|---|
| [`architecture/orchestration.md`](architecture/orchestration.md) | 编排与数据流、crate 依赖方向、查询生命周期 | ✅ 与代码同步，带 ✅/🔲 实现标注 |
| [`architecture/llm-layer.md`](architecture/llm-layer.md) | LLM 调用层完整数据流（含 async-openai 侧函数名）、两层重试、分层原则 | ✅ 与代码同步；**代码里有 12 处引用它**，改了 `soul-mem-llm` 就要同步 |
| [`architecture/记忆算法概述-修订.md`](architecture/记忆算法概述-修订.md) | 检索/巩固/遗忘的算法思路与理论背景 | 设计叙述，含未定项 |
| [`architecture/测试数据格式.md`](architecture/测试数据格式.md) | **上游生产者**（`soul_scraper`）的输出格式，附到 `RetrTestCase` 的字段映射 | ⚠️ 不是仓库 fixture 的格式，权威规范见上表第 1 节 |
| [`architecture/beta_ver.md`](architecture/beta_ver.md) | beta 阶段的设计设想（三大子图、工作记忆、PPR 变种、状态机） | ⚠️ **设计历史**，含未决问题与 `- [ ]` 待办；部分已被实现取代 |

**面向 AI 代理与贡献者的入口是仓库根目录的 [`AGENTS.md`](../AGENTS.md)** —— 那里有仓库地图、
"从哪里开始读"索引、必须保持的不变量，以及不会报错但会静默出错的已知陷阱。

## 3. 一次性报告类 —— 记录某次实验，**不是当前状态的描述**

这些文档保存的是特定时间点、特定数据上的测量结果。数字与结论对当时的 commit 有效，
**不要当作当前行为**。

| 文档 | 内容 |
|---|---|
| [`检索算法改进轨迹报告.md`](检索算法改进轨迹报告.md) | 逐 commit 的检索算法改进实测（含"embcache 不随边失效"等横切发现，仍值得一读） |
| [`playtest检索效果测试报告.md`](playtest检索效果测试报告.md) | Playtest 检索效果与角色三级谱系效果 |
| [`全量角色playtest验证报告-抽象PPR检出.md`](全量角色playtest验证报告-抽象PPR检出.md) | 24 角色全量 playtest 验证 |
| [`抽象PPR检出心智模型落地报告.md`](抽象PPR检出心智模型落地报告.md) | 抽象经 PPR 检出的实现与试点验证 |
| [`tests/forget-test-report.md`](tests/forget-test-report.md) | 遗忘功能集成测试报告 v1 |
| [`tests/forget-test-report-v2.md`](tests/forget-test-report-v2.md) | v2 |
| [`tests/forget-test-report-v3.md`](tests/forget-test-report-v3.md) | v3（最新，含 LLM 生成记忆与"删除记忆的泄漏现象"） |

> 三个 `forget-test-report*` 是**递进版本**而非并列结论，取最新一份为准。

---

## 维护提示

新增文档时请归入上面对应的一节。若某份"规范"被取代，**在同一提交里更新本索引并标注取代关系**，
不要把两份互相矛盾的规范并列留在目录里——这个目录曾经在"测试数据格式/规范"上出现过
两份命名相近、字段互不相同的文档，导致读者无法判断哪份权威。
