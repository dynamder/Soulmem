# 检索与联想

> 本文档基于 `soul-mem-algo/src/algo/retrieve/` 当前代码（`feature/test_framework` 分支）。
> 实现细节见 [soul-mem-algo](../crates/soul-mem-algo.md)。

## 1. 检索策略抽象

所有检索策略实现 `RetrStrategy` trait：

```rust,ignore
pub trait RetrStrategy: 'static {
    type Request: RetrRequest;
    type Return<'a> where Self: 'a;
    fn retrieve(&self, request: Self::Request) -> Self::Return<'_>;
}
```

| 策略 | 职责 | 返回 |
|------|------|------|
| `RetrShortOnly` | 短期记忆：滑动窗口 + 摘要 | `(Arc<[Information]>, Arc<str>)` |
| `RetrSimilarity` | 向量相似度 top-k（rayon 并行） | `Vec<(MemoryId, f32)>` |
| `RetrAssociation` | 多源 PPR 联想扩散 | `Vec<(MemoryId, f64)>` |
| `RetrBayesAction` | 贝叶斯动作推理 | `Vec<(MemoryId, f64)>` |
| `RetrAssociateWithAction` | PPR 结果 + 双源 Bayes 组合 | `AssociateWithActionResult { memory, action }` |
| `RetrDefaultPipeline` | 三段式默认管线 | `DefaultPipelineResult` |

## 2. 默认管线（DefaultPipeline）

三段式，严格按序：

```text
① ShortOnly ── 窗口内容 + 摘要
      ↓
② Similarity ── query 向量与 Cluster 节点余弦相似度（兜底分过滤 + top-k）→ PPR 源种子
      ↓
③ AssociateWithAction ── PPR 联想扩散 + softmax 归一化 + 贝叶斯动作推理 topK
      ↓
DefaultPipelineResult { association, action, short_history, short_mem, priority }
```

- ① 提取滑动窗口消息与摘要（短期记忆，无条件进入最终上下文）。
- ② 以相似度结果作为 PPR 的源节点（种子）。
- ③ PPR 扩散 → 只保留 Situation 节点（抽象源权重 ×2 优先）→ softmax → Bayes 动作推理。
- 最后 `merge_note_scores`：相似度结果与 PPR 结果同 id 取更高分，降序截断 10 条。

## 3. PPR 联想（核心机制）

目标方程：`ppr_s = damping × P × ppr_s + (1 - damping) × personalized_vec`

实现为 `weighted_ppr_fp`（EdgePush/Forward-Push 风格）：

- **多源**：相似度命中的多个节点共同作为个性化向量源。
- **带边权**：边权由动态权重函数计算——综合 `intensity`（连接强度）、`confidence`
  （语义置信度）与类型偏好（Semantic/Situation 通道，默认 Situation 优先）。
- **程序性记忆节点不参与联想**：目标为 Procedure 的边权重为 0，动作检出交给 Bayes 推理。
- 无出度节点与源节点建立"虚拟连接"均分残差。
- 参数：damping 0.65、残差阈值 1e-5；`damping == 1.0` 被 assert 拒绝（防无限循环 DoS）。

PPR 的联想语义：模拟记忆联想——从当前检索命中的记忆出发，沿图游走扩散到相关记忆，
多跳可达（"钟离假死" → "米哈游"这类梗联想）。

## 4. 动作推理（BayesAction）

- 源：PPR 结果 + 相似度种子（合并、只留 Situation、softmax 归一化）。
- 动作候选：源节点的出边中，`边类型 = Proc` 且目标为 `Procedure` 的邻居。
- 分数：`Σ prob × weight`（多源累加，prob 来自 `TrigToAction` 边的转移概率）。
- 语义记忆 → 动作的路径**不触发**动作（仅作自我认知补充，不进入动作提示词）。

## 5. 与旧文档/报告的演进对照

- **基线**：纯向量相似 top-k（beta_ver.md 的 baseline 思路）→ 当前为
  相似度种子 + PPR 扩散 + Bayes 动作的组合。
- **阈值语义变化**：早期 `SITUATION_SIMILARITY_THRESHOLD = 0.5`（情境专用阈值）已被
  **"兜底分 floor + 必取 top-k"** 取代（`similarity_threshold` 默认 0.35 作为最低兜底分，
  达到即参与 top-k，避免绝对阈值饿死查询）。
- **抽象检出**：抽象情境节点经 PPR 检出（`SpecificToAbstract` 反向边），作为 Bayes 动作
  提取的优先源，抽象源权重默认 ×2。详见 [历史报告](../testing/reports/README.md)。

## 6. 未来规划（代码 TODO）

- 数据库向量相似结果接入并混合（当前无数据库通道）。
- `RetrRequestConfig` 尚无生产调用方（仅定义 + serde 反序列化）。
