# LLM 调用层（`soul-mem-llm`）

> **一句话**：语义在链的上游，wire 细节全部关在 `oai_comp/` 子树里，而 **HTTP 请求由 `async-openai` 在我们提供的 tower 栈的最内层发出**——我们自己的代码里没有 `reqwest` 的发送调用。

本文面向"需要改动或调试 LLM 调用链"的人。目标是让你从任意一个疑问出发，三步之内落到具体文件与函数。

---

## 0. 从哪里开始读

| 你想知道 | 去看 |
|---|---|
| 调用方怎么写一次 LLM 请求 | `Task::system_user(..)` + `LlmEngine::complete(..)`（`src/backend.rs` / `src/engine.rs`） |
| 请求"要什么"的语义类型长什么样 | `src/backend.rs`（`Task` / `Sampling` / `Hints` / `Completion` / `StreamEvent`） |
| 一个语义请求怎么变成 JSON | `OaiCompatBackend::encode`（`src/oai_comp/backend.rs`） |
| JSON 响应怎么变成 `Completion` | `OaiCompatBackend::decode`（同上） |
| 出错的 HTTP 状态怎么变成语义错误 | `classify_error`（同上） |
| **请求到底在哪发出** | 见 [§2](#2-完整数据流非流式)；真正的点在 `async-openai/src/executor.rs` 的 `ReqwestService::call` |
| 重试是怎么做的、等多久 | `JitterRetryPolicy`（`src/oai_comp/transport.rs`，内层）+ `LlmEngine` 的整调用循环（`src/engine.rs`，外层） |
| 为什么会有两层重试 | 见 [§5](#5-两个关键分界决定了整个分层) |
| trace 里有哪几个事件 | `src/observer.rs` |
| SSE chunk 怎么变成增量事件 | `ChunkState::next_event` / `map_chunks`（`src/oai_comp/backend.rs`） |
| LLM 输出里的 JSON 怎么宽容取出 | `src/json.rs` |
| 重试次数怎么和"某一次调用"对上 | `src/ctx.rs`（task-local） |
| 本地 llama-server 的进程是谁管的 | `soul-tune/src/engine/llm/llama_server.rs`（不在本 crate） |

---

## 1. 模块地图与职责边界

```
crates/soul-mem-llm/src/
├─ lib.rs              ← 本 crate 的入口与分层原则（先读它）
├─ backend.rs          ← ★ 顶层唯一契约：trait ChatBackend + 语义类型
├─ engine.rs           ← 门面：观测生命周期 + 整调用重试
├─ error.rs            ← 语义错误分类 LlmError / LlmErrorKind
├─ observer.rs         ← 三个观测事件 + JSONL 落盘
├─ json.rs             ← 宽容抽取（剥 think / 剥围栏 / 平衡括号 / 反序列化）
├─ ctx.rs              ← task-local 调用上下文（crate 内部）
├─ mock.rs             ← 脚本化测试后端（仅 #[cfg(test)]）
└─ oai_comp.rs
   └─ oai_comp/
      ├─ backend.rs    ← 编码 / 解码 / 错误映射（OpenAI-compatible）
      ├─ wire.rs       ← ★ 私有 wire 类型：唯一出现 provider 字段名的地方
      ├─ provider.rs   ← impl async_openai::config::Config（URL / 鉴权头 / 额外头）
      ├─ transport.rs  ← ★ tower 栈：重试策略 + HTTP client 构造
      └─ e2e_tests.rs  ← 自建回环 HTTP 服务器的端到端测试
```

### 职责边界（改代码前先对号入座）

| 关注点 | 归谁 | 规则 |
|---|---|---|
| 语义类型（`Task`/`Completion`/`StreamEvent`） | `backend.rs` | **不许出现 wire 字段**：不能有 `max_tokens` vs `max_completion_tokens` 之分、`response_format`、`chat_template_kwargs`、`finish_reason` 原文、SSE delta 形状 |
| wire 字段名与请求体形状 | `oai_comp/wire.rs` | provider 差异只能体现在这里 + `config.rs` |
| 错误"类别" | `error.rs` | 类别是稳定的公共口径 |
| 错误"映射" | `oai_comp/backend.rs` | 只有后端知道自己的 HTTP status 与错误体形状 |
| 重试 / 超时 / 限流 | `oai_comp/transport.rs` + `engine.rs` | 见 §5 的分界 |
| 观测事件 | `observer.rs`（产生点：`engine.rs` 与 `transport.rs`） | 事件里不放正文、不放密钥 |
| 文本抽取 | `json.rs` | 与 provider、与任务都无关的纯文本处理 |

> 判断规则：**如果一个类型描述的是"我们要什么"或"我们得到了什么"，它属于 `backend.rs`；如果它描述的是"provider 管它叫什么"，它属于 `oai_comp/`。**

---

## 2. 完整数据流（非流式）

下图每一跳都标了文件与函数名。标注 `async-openai:` 的跳在依赖库里，其余都在本 crate。

```
调用方（soul-mem-algo / soul-mem-runtime / soul-tune）
 │
 │  Task::system_user(system, user).with_sampling(..).with_hints(..)
 ▼
[engine.rs] LlmEngine::complete(task)
 │   · ctx::CallCtx::new + 进入 task-local        → ctx.rs
 │   · observer.on_start(CallStart)               → observer.rs
 │   · 整调用重试循环（超时 / body 读取失败 / 流中断）
 ▼  ChatBackend::complete(&Task)                    ← trait 定义在 backend.rs
 ▼
[oai_comp/backend.rs] OaiCompatBackend::complete(task)
 │   · encode(&task, false) → OaiChatRequest      ← 语义 → wire
 │   · tokio::time::timeout(总时限, pending)      ← 包住整次调用（含 body 读取）
 ▼  client.chat().create_byot::<OaiChatRequest, OaiResponse>(request)
 │
 ├─ async-openai: Chat::create_byot                  ← ★ 由 #[byot] 属性宏在编译期生成
 │    展开后等价于：self.client.post("/chat/completions", request, &req_opts)
 ├─ async-openai: Client::post                       (client.rs)
 │     · serde_json::to_vec(我们的 OaiChatRequest) → Bytes   ← 请求体只序列化一次
 │     · url     = config.url(path)          ┐ 二者来自我们实现的 Config
 │     · headers = config.headers()          ┘ → oai_comp/provider.rs
 ├─ async-openai: HttpRequestFactory                 ← 可重放的请求工厂
 ├─ async-openai: Client::execute → execute_raw → execute_response
 ├─ async-openai: self.executor.execute(factory)
 ├─ async-openai: TowerExecutor::execute → service.oneshot(factory)
 │
 ▼  ★ 我们在 transport.rs 装配的 tower 栈 ★
 │   RetryLayer(JitterRetryPolicy)
 │     · 429 / 5xx / 连接失败 → 重试
 │     · 有 Retry-After 就照它等（夹在上限内），否则指数退避 + 抖动
 │     · 每次重试：clone_request → factory.build() 重建 reqwest::Request
 │     · 上报：ctx.note_inner_retry(..) → observer.on_retry(..)
 ▼
 ├─ async-openai: ReqwestService::call
 └─► ★★★ client.execute(request).await ★★★          ← HTTP 请求真正在此发出
 │
 │  ── 响应回流 ──
 ├─ async-openai: execute_response：检查 status，失败 → read_error_response
 ├─ async-openai: read_response：response.bytes()    ← ★ body 在这里读，**在 tower 之上**
 ├─ async-openai: serde_json::from_slice::<OaiResponse>
 ▼
[oai_comp/backend.rs] OaiCompatBackend::decode(response)
 │   · OaiResponse → Completion（text / reasoning / stop / usage）
 │   · 正文为空但 reasoning_content 非空 → 兜底并记 warn（不丢内容）
 ▼
[engine.rs] 收尾
     · err 上的 retry_after 从 ctx 补回（预算耗尽后拿不到响应头）
     · observer.on_end(CallEnd)：耗时 / token / 重试次数 / 错误类别
```

## 3. 流式的差别

```
[oai_comp/backend.rs] OaiCompatBackend::stream(task)
 ▼  client.chat().create_stream_byot::<OaiChatRequest, OaiChunk>(request)
 ├─ async-openai: post_stream → execute_stream → execute_response
 │     ← ★ tower 栈只跑到"拿到响应头"为止；这就是 first_byte_timeout 的边界
 ├─ async-openai: stream_mapped_raw_events（native 分支）
 │     · tokio::spawn + mpsc：**SSE 的读取发生在一个独立任务里**
 │     · eventsource_stream 逐条解析 `data: {...}`；`data: [DONE]` → 流结束
 ▼  每个 chunk 用我们的 OaiChunk 反序列化
[oai_comp/backend.rs] map_chunks → ChunkState::next_event
 │     · 跳过空 chunk（只带 usage / 只带 finish_reason / 空 delta）
 │     · 产出语义事件：Delta / ReasoningDelta / Done
 │     · 已产出内容后再出错 → StreamInterrupted{partial}，**不可重放**
 │     · 没有任何内容也没有 finish_reason → StreamInterrupted（可重试）
 ▼
[engine.rs] ObservedStream
     · 累加 text_chars；Done 即收尾；被提前丢弃 → Cancelled
     · 保证"恰好一次" on_end
```

**注意**：SSE 的读取在 async-openai spawn 出来的任务里，而逐 chunk 映射发生在消费方 poll 的 task 里。所以 `ctx`（task-local）在流式场景下只在"开流"阶段有效——这正是"流式中断不可重放"实现在 `ChunkState` 而不是依赖 `ctx` 的原因。

---

## 4. 两种后端的形态

| 后端 | 位置 | 编码/解码 | 同步/异步 |
|---|---|---|---|
| OpenAI-compatible（远程 API 与本地 llama-server 共用） | `oai_comp/` | 本 crate 私有 wire 类型 | async（`ChatBackend`） |
| 进程内 candle | `soul-tune/src/engine/llm/candle_llm.rs` | 自渲染提示词 | 同步（实现 `soul-tune` 的 `LlmBackend` 外壳） |

`soul-tune` 的 `LlmBackend` **不是第二套 LLM 实现**，而是同步外壳：它的各测试套件与 FRB 入口都是同步的，所以用一个全局运行时把异步引擎桥接过来（`block_on`，内部由 `soul-mem-llm` 负责一切网络细节）。进程管理（探活 / 拉起 llama-server / 关闭）也留在 `soul-tune/src/engine/llm/llama_server.rs`。

---

## 5. 两个关键分界（决定了整个分层）

### 分界一：响应体读取发生在 tower 之上

`async-openai` 的结构是 `executor.execute(factory)`（tower 栈）返回响应头之后，才由 `read_response` 读 body。结论：

- 传输层重试**看不到**：整体超时、读到一半断连、流在产出前就断了；
- 因此需要一个"整调用重试"层（`engine.rs`），它重发整个请求；
- 内层重试仍**不可省略**：它是唯一能读到 `Retry-After` 响应头的位置（`OpenAIError::ApiError` 只带 status 与错误体，没有头）。

单次语义调用的最大请求数 = `(1 + 传输层重试) × (1 + 整调用重试)`，两者都会出现在 trace 里。

### 分界二：请求体一次序列化，请求逐次重建

`build_request_factory_with_json` 把我们的请求类型 `serde_json::to_vec` 成 `Bytes` 一次，而工厂闭包在**每次尝试时重建 `reqwest::Request`**（`reqwest::Request` 不可 `Clone`）。这是重试可行的前提。

---

## 6. 三个"找不到代码"的常见原因

1. **`create_byot` 在 async-openai 源码里 grep 不到**：它由 `#[crate::byot(..)]` 属性宏在编译期生成（源码里只有属性标注）。rust-analyzer 能展开属性宏，"转到定义"通常可用；纯文本搜索不行。
2. **本 crate 里没有 `reqwest` 的发送调用**：只有 `transport.rs::build_http_client`（**构造** client：`connect_timeout`、`no_proxy`）与类型引用（`reqwest::Response`、`RETRY_AFTER` 常量）。发送点是 async-openai 的 `ReqwestService`。
3. **`soul-tune` 里仍有 `reqwest`，但都不是 LLM 调用**：那是 `/health` 探活（`resolver::probe_health` 与 `llama_server::load` 的启动轮询），属于进程管理。

---

## 7. async-openai 侧调用链（版本 0.41.3）

行号仅作参考，**升级版本后必须重新核对**；稳定的锚点是函数名。

| 函数 | 位置（0.41.3） | 作用 |
|---|---|---|
| `Chat::create_byot` / `create_stream_byot` | `chat.rs`（宏生成） | 把我们的请求类型交给 `Client::post` / `post_stream` |
| `Client::post` | `client.rs:527` | 反序列化响应体 |
| `Client::post_stream` | `client.rs:669` | 返回 SSE 流 |
| `Client::build_request_factory_with_json` | `client.rs:393` | 序列化请求体 + 组 URL/头 |
| `Client::build_request_parts` | `client.rs:342` | `config.url()` / `config.headers()` / `config.query()` |
| `Client::execute_response` | `client.rs:631` | 调 executor，检查 status |
| `Client::read_response` | `client.rs:730` | 读 body（**在 tower 之上**） |
| `Client::read_error_response` | `client.rs:736` | 解析错误体（5xx 保留 status） |
| `TowerExecutor::execute` | `executor.rs:184` | `service.oneshot(factory)` |
| `ReqwestService::call` | `executor.rs:127` | **★ 请求发出点** |
| `stream_mapped_raw_events`（native） | `client.rs:827` | spawn 任务读 SSE → mpsc |
| `middleware::retry::should_retry` | `middleware/retry/mod.rs` | 上游分类；我们的策略在 `Err` 分支复用它 |

依赖的特性开关：`chat-completion`、`byot`、`middleware`（见 `crates/soul-mem-llm/Cargo.toml`）。`middleware` 是拿到 `with_http_service` / `ReqwestService` / `should_retry` 的前提。

---

## 8. 新增一个后端要做什么

1. 在 `backend.rs` 里**不要**加东西——除非某个语义概念对**所有**后端都成立。
2. 实现 `ChatBackend`（`info` / `complete` / `stream`），自己负责：请求编码、响应/chunk 解析、错误映射成 `LlmError`（含 `Retry-After`）。
3. 不需要实现：重试、超时、并发、trace——这些由 `LlmEngine` 与传输层统一负责。
4. 如果它自己就是 HTTP 的，把 tower 栈照 `oai_comp/transport.rs` 装一遍，并复用上游的 `should_retry` 保持分类一致（`transport.rs` 的测试里有一条"与上游逐状态一致"的断言）。
5. 加一个脚本化 / 回环的测试，至少覆盖：成功、可重试失败后成功、不可重试失败、超时、空结果、流中断。

---

## 9. 改动后的检查清单

```bash
cargo fmt --all -- --check
cargo clippy --offline --workspace --all-targets -- -D warnings
cargo check --offline --workspace --all-targets
cargo test --offline -p soul-mem-llm        # 离线，无需真机
cargo test --offline -p soul-mem-runtime -p soul-mem-core
```

必须保持的不变量：

- 顶层语义类型里**不出现 wire 字段**（§1 的规则）；
- 一次语义调用**恰好一次** `on_end` 上报（含被提前丢弃的流）；
- 已产出内容后的流中断**绝不重放**（否则调用方会看到重复文本，在摘要/记忆写入场景下是静默损坏）；
- `ServerError` / `BadRequest` 等类别的归类不因重构而改变（`error.rs` 的测试锁死了可重试/可降级集合）；
- 请求体里 `max_tokens` 与 `max_completion_tokens` **不同时出现**（`wire.rs` 的测试锁死）。
