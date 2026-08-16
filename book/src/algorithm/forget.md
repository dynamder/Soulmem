# 遗忘算法

> 本文档基于 `soul-mem-algo/src/algo/forget/` 当前代码（`feature/test_framework` 分支新增
> 模块，尚未提交）。实现细节见 [soul-mem-algo](../crates/soul-mem-algo.md)。

## 1. 设计：惰性遗忘 + 遮罩

遗忘算法模拟人类遗忘曲线，采用**惰性遗忘**策略——不主动删除记忆，而是按遗忘曲线计算
**缺失度**，对记忆文本进行**遮罩**（将部分内容替换为 `[masked]`），缺失度较高时调用 LLM
尝试**修订**（基于遮罩文本猜补）。该机制同时为测试框架提供了可量化的观测维度
（原图 vs 遗忘后图的变换，见 [算法测试](../testing/算法测试.md)）。

## 2. 艾宾浩斯衰减曲线

```text
R(t) = e^(-t/τ)
τ = adjusted_half_life / ln(2)
adjusted_half_life = base_half_life_hours × (1 + active_factor × min(retrieval_count, cap))
```

- `t`：自创建到现在的**小时数**。
- **半衰期**：R=0.5 时恰为 `adjusted_half_life`。
- **激活减缓遗忘**：`retrieval_count` 越大半衰期越长（封顶 50 次）——经常被回忆的记忆
  遗忘更慢，符合"复习"直觉。
- 默认参数：`base_half_life_hours = 24.0`、`active_factor = 0.1`、`max_activation_cap = 50`。
- `missing_degree = 1 - R(t)`（0~1，越大忘得越多）。
- 边的遗忘：`edge_decay_intensity = 原边强度 × 源节点衰减`（边跟随源节点遗忘）。

## 3. 惰性遗忘主流程（lazy_forget）

```text
输入：MemoryNote + 当前时间 + jieba + llm_call 闭包
  ↓
① 类型过滤：仅 SpecificSituation 与 Semantic 可遗忘
  （Procedure、AbstractSituation → NoAction）
  ↓
② 计算缺失度 md（默认 24.0/0.1/50）
  ↓
③ 分支：
   md < 0.05  ──→ NoAction（几乎没忘，不动）
   md < 0.15  ──→ MaskOnly（只遮罩，不调 LLM）
   md ≥ 0.15  ──→ 调 LLM 猜补（Revised）；失败则降级 MaskOnly
```

- 摘要文本来源：SpecificSituation → `narrative`；Semantic → `content`。
- LLM prompt：system 为固定英文（"reconstructs partially masked memories"），user 为遮罩文本。

## 4. 文本遮罩（mask_text）

1. `jieba.cut(text, true)`（开启 HMM）分词。
2. 遮罩词数 `n = round(missing_degree × total)`，上限 total。
3. **确定性随机**选词：种子 = `hash(text) ^ (degree.to_bits() × 114514)`，StdRng 洗牌取
   前 n 个下标，替换为 `" [masked] "`。
4. 同文本同缺失度输出恒等（可复现，利于测试）。

## 5. 语义字段对齐（align_sem_fields）

仅对 Semantic 记忆：调用 LLM 输出 `Aliases:` / `Description:` / `ConceptType:` 三行，
与原文比对后更新字段：

- 文意一致则保留原值。
- 缺失度 < 0.6 时**禁止 aliases 增长**（防 LLM 幻觉膨胀记忆内容）。
- description 非空才写回；concept_type 有解析结果才写回。

## 6. 测试框架对接

- 遗忘评测由 `soul-tune run forget fixtures/forget/*.json` 驱动（soul-tune 层）。
- 测试数据规范中 Forget JSON 的 T1–T5 用例与代码常量一一对应：
  - config `{ base_half_life_hours: 24.0, active_factor: 0.1, max_activation_cap: 50 }`
    与代码常量完全一致；
  - T4 的三段时间点分别落入 `NoAction`（md<0.05）/ `MaskOnly`（<0.15）/ `Revised`（≥0.15）
    三段，与 `MASK_THRESHOLD` / `REVISE_THRESHOLD` 一致。

## 7. 与旧文档的差异

| 旧设计（记忆算法概述-修订.md） | 实际实现 |
|--------|---------|
| 记忆微元 + 透明度 + 微元概括性描述 + 层叠包含关系的精细遮罩模型 | 整段文本 jieba 分词 + 确定性随机遮罩，无微元/透明度分层 |
| 连接强度按曲线衰减、低于阈值删除连接、无连接的节点删除 | 仅 `edge_decay_intensity` 计算函数；**无删边/删孤立节点实现** |
| 遗忘范围（未明确） | 仅 SpecificSituation 与 Semantic；Procedure/Abstract 不遗忘 |
