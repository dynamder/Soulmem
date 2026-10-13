# AGENTS.md — soul-mem-llm

全仓库**唯一的 LLM 调用层**，也是唯一真正的叶子 crate（生产与 dev 都无内部依赖）。

## 导航

- 先读 crate 根模块文档：有「从哪里开始读」索引、分层原则、两层重试说明。
- 顶层契约（`Task` / `Completion` / `StreamEvent` / `ChatBackend`）在 `backend` 模块。
- 调用方唯一需要认识的入口是 `engine` 模块的 `LlmEngine`。
- OpenAI-compatible 的一切在 `oai_comp` 子树；其中 `oai_comp::wire` 是唯一出现 provider 字段名的地方。
- 数据流与「新增一个后端要做什么」见 [`../../docs/architecture/llm-layer.md`](../../docs/architecture/llm-layer.md)。

## 风格

- 顶层语义类型里**不出现 wire 字段**。判断规则：描述「provider 管它叫什么」→ 归 `oai_comp`。
- 语义类型只描述「我们要什么 / 得到了什么」，与 provider 的字段名无关。

## 坑

- 请求实际发出点**不在本 crate**，在 async-openai 的 `ReqwestService::call`——源码里 grep 不到，别白找。
- `oai_comp` 的文件职责表少列了一个文件（`e2e_tests`）。
- `transport` 模块刻意**不装** tower timeout 层；整体 / 首字节时限由后端用 `tokio::time::timeout` 包在响应体读取之外。
- 观测上下文只在同一 task 内可见；流式下 async-openai 会 spawn，逐 chunk 处理**不得**依赖它。
- 本 crate 没有 `//SAFEUNWRAP:` 标记、没有 TODO、没有 `#[allow]`——新增这些就是退步。

## 流程

改代码前确认没破坏这些不变量：

1. 一次语义调用**恰好一次** `on_end` 上报，包括被提前丢弃的流。
2. 已产出内容后的流中断**绝不重放**——否则摘要 / 记忆写入会出现重复文本（静默损坏）。
3. 请求体里 `max_tokens` 与 `max_completion_tokens` **不同时出现**。
4. 日志与观测事件里**不放提示词正文、不放密钥**；配置的 `Debug` 只打印 `has_api_key`。
5. **不臆造数据**：`Usage` 拿不到就留 `None`；`reasoning` 不伪造；`stop` 不确定就是 `Unknown`。
6. 改了调用链就同步 [`../../docs/architecture/llm-layer.md`](../../docs/architecture/llm-layer.md)。
