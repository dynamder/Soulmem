# AGENTS.md — soul-mem-query

嵌入与评分层：把记忆与查询变成向量，并计算相似度分数。

## 导航

- 先读 crate 根模块文档：有「从哪里开始读」索引和两条跨 crate 契约。
- 两个子域：`embedding`（领域对象 → 向量）与 `query`（查询类型 + 评分）。
- 生产依赖只有 `soul-mem-core`；被 runtime / algo / soul-tune / soul-tune-api / benches 反向依赖（全为生产依赖）。

## 风格

- 唯一评分入口是 `query::compute` 的 `compute_fused`，不要在别处新开评分路径。
- `Embeddable` 的两个方法分工是刻意的：`embed` 借用、`embed_and_fuse` 取所有权。

## 坑

| 现象 | 所在模块 |
|---|---|
| `time_span` 被接受、可存储、可取回，**评分从未实现**；embedding 侧连字段都没有。无测试断言时间生效 | `query::retrieve`、`query::compute` |
| 分数契约声称 [0,1]，但裸 cosine 可为负；Situation 路径用 max 折叠把负值夹到 0 → 「负相关」与「无信号」不可区分 | `query::compute` |
| 变体不匹配静默返回 `Ok(0.0)` | `query::compute` |
| `into_parts` / `mean_pooling` 会把 `blend_weights` 静默重置为默认 | `embedding::vec`、`embedding::query` |
| `string_blend_alpha` 存在查询里，但 `compute_fused` 从参数取；只有 algo 侧调用方手工接上 | `query::compute` |
| `is_zero()` 对空向量返回 true | `embedding::vec` |
| `normalize()` 对零向量产出 NaN 而非 Err | `embedding::vec` |
| 空输入两套语义：`mean_pooling(&[])` 返回 Ok(空向量)，模型层返回 Err | `embedding::vec` vs `embedding::embedding_model` |
| 权重的注释是**旧值**，以 `embedding::blend_weights` 的实际默认值为准 | `query::compute` |
| `EmbbdedSituationQueryUnit` 拼写错误已成公开 API，勿顺手改名 | `embedding::query::situation` |

## 流程

- **查询 / 段落不对称**：`EmbeddingModel` 的 `infer_query_*` 默认实现**等于**非 query 版本。需要指令前缀的模型必须覆写——忘了覆写不报错，只是检索质量下降。
- **空 tag → 零向量是跨 crate 契约**：runtime 侧据此跳过该向量列的写入。改动会静默改变检索结果。
- `dim()` 与 `max_input_token()` 是硬编码常量，换模型必须同步改。
- 生产路径有 2 处无 `//SAFEUNWRAP:` 的 panic（BGE 模型初始化、一处 `unreachable!`）。
- 大量测试用真实模型构造、会下载权重；BGE 的模型测试未标 `#[ignore]`，Qwen3 的有。离线跑测试用 `--skip test_bge_small_zh_cpu --skip test_qwen3_embedding_600m_cpu`。
- `time_span` 意图未定——不要擅自实现评分或删字段，先问维护者。
