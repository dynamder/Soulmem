# AGENTS.md — soul-mem-algo

记忆算法层：检索、遗忘、巩固。

## 导航

- 先读 crate 根模块文档：三子域表 + 「改这一层之前要知道的」。
- 检索管线 `RetrDefaultPipeline`（`retrieve::complex`）串联三步：ShortOnly → Similarity → AssociateWithAction。
- 巩固生成入口 `consolidate::generate`；遗忘重建入口 `forget::llm_completion`。
- LLM 调用链见 [`../../docs/architecture/llm-layer.md`](../../docs/architecture/llm-layer.md)。

## 风格

- 算法常数要写清**单位、含义、取值范围**，照 `forget::decay_calculator` 的**条目注释**风格（公式、值域、失败模式、算例）。
- 反例：检索配置的若干裸值字段（`damping_factor`、`residue_threshold`、`top_k`）无注释、无单位、无值域，别学。
- 需要 LLM 的算法入口统一接收 `&LlmEngine`；本层**没有**自己的重试，重试全在 `soul-mem-llm`。

## 坑

- `naive_ppr` 传 `damping_factor = 1.0` 会 **panic**，不是返回 Err。
- 打分 / 合并逻辑分散在三处，NaN 策略不一致（一处 `total_cmp`，别处 `partial_cmp().unwrap_or(Equal)`）。
- 动作结果**不是** top-k，而是按 ActionType 每类一席；Skill 席位恒空。
- 本 crate 唯一一处 `//SAFEUNWRAP:` 在 `common::ppr`。

## 流程

- 排除项只写在 workspace 根的 `.cargo/mutants.toml`；`#[mutants::skip]` 全仓库 0 命中。
- 加排除项要写原因；「行为可观测但没有测试」应补测试，而不是加排除项。
- `exclude_re` 不作用于结构体字面量字段变异体（cargo-mutants 实测）——这类等价变异体另想办法。
