# 嵌入层

> 本文档基于 `soul-mem-query/src/embedding/` 当前代码（`feature/test_framework` 分支）。
> 实现细节见 [soul-mem-query](../crates/soul-mem-query.md)。

## 1. 职责

嵌入层负责：嵌入模型接入、记忆节点/查询的向量化、查询构建与相似度计算。它是检索管线的
**向量通道**（与字符串通道、tag 通道共同构成相似度评分）。

## 2. 模型接入

| 模型 | 池化 | 维度 | 分块上限 | 查询指令 |
|------|------|------|---------|---------|
| BGE `bge-small-zh-v1.5` | CLS | 512 | 200 字符 | `QUERY_INSTRUCTION` 前置到每个分块 |
| Qwen3 `Qwen3-Embedding-0.6B` | — | 1024（F32） | 6000 字符 | 无覆写（同 passage 侧） |

- 加载：`embed_anything`（内部基于 candle）本地加载，BGE 为进程级 `OnceLock` 单例。
- 查询侧与 passage 侧分离：`EmbeddingModel` trait 提供 `infer_query_batch` 等查询侧方法，
  BGE 覆写为前置查询指令（非对称训练用法），Qwen3 使用默认（对称）。

## 3. 权重体系（BlendWeights）

`BlendWeights` 集中定义全部 16 个可调权重，经 `set_blend_weights` 递归传播到查询嵌入：

- `tag` = 0.3、`variant` = 0.7（两个顶层通道权重，默认）
- `string_blend_alpha` = 0.6（字符串通道混合系数）
- 各结构化字段（location/participant/emotion/environment/event）子权重

评测框架通过 `blend_sweep`（`tag_sweep` / `pairs`）批量扫描权重组合，见
[测试数据规范](../testing/测试数据规范.md)。

## 4. 相似度计算（compute_fused）

```text
score = max(embedding_score, alpha × embedding_score + (1 - alpha) × string_score)
```

- `embedding_score`：余弦相似度（结构化查询按字段 max 融合）。
- `string_score`：Jaro-Winkler / 归一化 Levenshtein 字符串距离（仅对精确标识符、
  Semantic content/aliases、抽象情境结构化字段生效；具体情境恒为 0）。
- **字符串通道只加分不拉低**（`max` 语义）。
- 抽象情境查询有 `fused_self()` 叙事回退（无结构化输入时用叙事文本嵌入）。

## 5. 数据流

```text
记忆节点 ──embed──▶ EmbeddedMemoryNote { note, embedding } ──入图──▶ MemoryCluster
查询     ──embed──▶ EmbeddedMemoryRetrieveQuery ──compute_fused──▶ 相似度 top-k（soul-mem-algo）
```

## 6. 与旧文档的差异

- beta_ver.md 只提"向量搜索"；当前实现新增**字符串通道**、**tag 通道**与抽象情境叙事回退。
- 程序性记忆嵌入侧仍为空占位（`Procedure()`）。
