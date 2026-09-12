# SoulMem 顶层服务封装（`crates/soul-mem`）实现计划

> 本文档描述对 `crates/soul-mem`（soulmen workspace 中的顶层封装 crate）的实现计划：
> 只列出**新增/修改哪些文件、各自负责什么、放在什么位置、按什么约定组织**，不展开代码细节。
>
> **通信口径**：与其他设备的通信**仅**依赖 **zenoh 的订阅/发布（pub/sub）**；协议由
> `proto/soul_mem.proto` 定义、protobuf 编码。**不使用 gRPC**。
> 请求-应答用「请求主题 + 按请求 id 的应答主题」在 pub/sub 之上模拟；服务发现/心跳使用 zenoh liveliness。

---

## 1. 背景与目标

`Soulmem` 按 workspace 拆分为多个只负责单一职责的库 crate：

| crate | 职责 | 状态 |
|---|---|---|
| `soul-mem-core` | 纯数据模型（MemoryNote / MemoryLink） | ✅ 已实现 |
| `soul-mem-query` | Embedding 生成 / Query 类型 / 相似度 | ✅ 已实现 |
| `soul-mem-runtime` | WorkingMemory / SlidingWindow / Cluster / Record / LLM 摘要 | ✅ 已实现 |
| `soul-mem-algo` | 检索策略 RetrStrategy / DefaultPipeline 编排 | ✅ 已实现 |
| `soul-tune` / `soul-tune-api` | CLI 基准测试框架（TUI / FRB） | ✅ 已实现 |

`crates/soul-mem` 是最上层封装 crate，目标：

1. **串联** 既有四个运行时 crate（core / query / runtime / algo），对外表现为一个完整服务；
2. 提供 **zenoh 订阅/发布**（唯一的对外通信方式，protobuf 载荷）用于与其他设备通信；
3. 承载**服务自身状态维护流程**的编排入口（持久化、巩固、遗忘等定时任务 + 控制信号）；
4. 采用与其他 crate 一致的 **workspace 布局与命名/工程约定**（见 §2.1）。

> Demo 定位：先在**本机/局域网两台设备**上把「信息增量 → 检索 → 结果返回」主链路经 zenoh 跑通；
> 下层尚未提供的算法（巩固、SurrealDB 持久化、遗忘衰减等）只做**调度骨架与接口边界**，不越界实现。
>
> **单设备可完整模拟**：运行 `mock-device`（见 §3.1-H）在同机通过**回环(loopback)**扮演"外部设备"。
> 回环只验证功能与协议正确性，真实跨设备网络因素（IP/防火墙/丢包/多跳 router）仍需真机验证。
>
> **自动化测试不依赖网络**：见 §3.1-I。

---

## 2. 关键设计决策

### 2.1 crate 定位、依赖方向与文件命名约定

- `soul-mem` 是最上层 crate，依赖 `soul-mem-core`、`soul-mem-query`、`soul-mem-runtime`、`soul-mem-algo`（单向）。
- 同时提供 **lib** 与两个 **bin**：服务端 `src/main.rs`、模拟器 `src/bin/mock_device/main.rs`。
- 包名 `soul-mem`（`publish = false`）；`autobins = false`，显式声明两个 bin：
  - `[[bin]] name = "soul-mem" path = "src/main.rs"`
  - `[[bin]] name = "mock-device" path = "src/bin/mock_device/main.rs"`
  保证 `cargo run -p soul-mem` 与 `cargo run -p soul-mem --bin mock-device` 均可用。
- 工程元信息（version `0.1.0`、edition `2024`、license `MIT`）、`/target` 的 `.gitignore`、`src/` 组织方式，全部仿照现有 crate。
- **模块文件命名（Rust 2024 edition，不使用 `mod.rs`）**：目录型模块用同名 `.rs` 模块根文件（如 `service.rs`）+ 同名目录（`service/`）放子模块；无子模块则单文件（`config.rs`、`error.rs`）。命名统一小写蛇形。

### 2.2 对外通信：仅 zenoh 订阅/发布

| zenoh 机制 | 用途 | 说明 |
|---|---|---|
| **pub/sub（主题）** | 单向推式输入、事件广播 | 外部设备把信息增量发布到 `ingest` 主题；服务可把记忆变更广播到 `events` 主题（预留）。 |
| **pub/sub（模拟请求-应答）** | ping/ingest/retrieve/read/write/feedback/control | 请求方发布 `RequestEnvelope` 到 `request` 主题；服务处理后把 `ReplyEnvelope` 发布到 `reply/<request_id>`；请求方订阅该应答主题并按 `request_id` 匹配。 |
| **liveliness** | 服务发现 / 心跳 | 服务上线声明 liveliness token（`liveliness/<device_id>`），其他设备可订阅上下线。 |

- 主题（Key Expression）在 `zenoh/keys.rs` 集中定义。
- 所有载荷统一为 `.proto` 定义、protobuf 二进制编码，单一 schema，无 gRPC。

### 2.3 单一事实源：`.proto` 协议

- `proto/soul_mem.proto` 是收发协议的唯一事实源：所有消息/枚举/信封（含 `MemoryNote` 全部嵌套类型、Query 各单元、`RequestEnvelope`/`ReplyEnvelope`）。
- `build.rs` 用 **prost** 生成 Rust 消息类型（`wire::pb`）；`protoc` 走 vendored，任何机器无需本机安装。
- `wire::convert` 负责 protobuf ⇄ 内部领域模型的双向转换与校验（id/时间/枚举/嵌套结构）。
- 信封：`RequestEnvelope { request_id, op, payload(bytes) }`、`ReplyEnvelope { request_id, payload(bytes), error }`；`payload` 为各操作消息的 protobuf 字节。

### 2.4 并发 / 状态模型

- 服务进程内共享一份工作记忆：`WorkingMemorySlot` 持有 `Arc<WorkingMemory>`；只读操作取 `Arc` 克隆，写操作（add_node/record_retrieval/feedback）用 `Arc::get_mut` 迁移式独占访问（非唯一时让出重试）。
- zenoh handler 只做协议编解码与参数校验，业务都收敛到 service 编排层（单入口）。
- `WorkingState`：service 读取 `Idle` 作为后台巩固/遗忘任务的**门控条件**；Demo 未主动做 `Working` 迁移。
- 对外路径具备：入参校验 → 明确错误码（§2.7）→ 结构化日志 → 请求-应答超时（5s）。

### 2.5 后台任务与"控制信号"

- 后台调度器（tokio interval）按配置间隔注册：**持久化 Persist / 巩固 Consolidate / 遗忘 Forget**；间隔为 0 表示停用；单任务失败被隔离并记日志。
- `Control`（经 zenoh 请求-应答触发）强制触发任务——对应"控制信号（强制触发定时任务）🔲"。
- 任务体与调度解耦（`background` 各一个 `run_once`）：
  - `persist` 调用 `service.persist()`（不门控 Idle）；
  - `consolidate`/`forget` 经 `Idle` 门控后调用 `service.control`，因下层算法未就绪返回 `Unimplemented`，不伪造结果。

### 2.6 持久化抽象

- `MemoryStore` trait（save/load 快照）+ 后端枚举 `Store::{Noop, File}`，`Store::from_config` 依据是否配置路径选择。
- 快照 `Snapshot`：`version`、`summary`、`window`（`WindowEntryDto`）、`nodes`（`EmbeddedMemoryNote`，链接含在节点内）；`SNAPSHOT_VERSION` + `validate()`。
- Demo 内置 `FileStore`（JSON 快照，原子写盘 + 启动恢复），**不引入 SurrealDB**。
- 已知限制：`Record`（活跃记录）的检索次数/反馈历史不随快照精确还原；恢复时按 `add_node` 生成新记录处理。

### 2.7 鲁棒性要求

- 统一 `Error` + `ErrorCode`（invalid_argument/not_found/already_exists/unavailable/failed_precondition/internal/unimplemented），映射为对外稳定错误码。
- 所有网络入参做长度/枚举/格式校验，非法输入返回明确错误而非 panic。
- 配置解析：`SOUL_MEM_*` 数值项**未设置或空白 → 默认值**；显式 `0` 合法（用于停用定时任务）；非法值在启动即失败。
- 后台任务互相隔离；请求-应答带超时；提供 **graceful shutdown**（持久化 → 停后台 → 下线 zenoh）。

---

## 3. 总体目录结构

```
crates/soul-mem/
├─ Cargo.toml          # 包描述：名称/版本/edition/依赖 + features + 显式 [[bin]]
├─ .gitignore          # 忽略 /target
├─ build.rs            # prost-build：proto → Rust 消息类型（vendored protoc）
├─ proto/
│  └─ soul_mem.proto   # 收发协议（唯一事实源）
├─ src/
│  ├─ lib.rs           # 库根：pub mod 汇总 + 顶层再导出
│  ├─ main.rs          # 服务端入口（薄，bin name = soul-mem）
│  ├─ server.rs        # 运行时装配：服务上下文+zenoh+后台 → 优雅退出
│  ├─ config.rs        # 类型化配置 + 默认值 + 校验 + SOUL_MEM_* 解析
│  ├─ error.rs         # 统一错误类型与对外错误码
│  ├─ wire.rs          # wire 模块根（pb 生成代码 + op 常量）
│  ├─ service.rs       # Service 编排层模块根（门面 + WorkingMemorySlot + Hash 模型）
│  ├─ background.rs    # 后台任务模块根（run_background/BackgroundRuntime）
│  ├─ store.rs         # 持久化模块根（MemoryStore trait + Store 枚举）
│  ├─ zenoh.rs         # zenoh 模块根（唯一对外通道）
│  ├─ bin/
│  │  └─ mock_device/  # 单机"外部设备"模拟器（bin name = mock-device）
│  │     ├─ main.rs    #   入口（薄）：--zenoh-prefix → ZenohClient → 场景
│  │     └─ scenario.rs#   演示场景编排
│  ├─ wire/
│  │  └─ convert.rs    # protobuf ⇄ 内部领域模型转换与校验
│  ├─ service/
│  │  ├─ ingest.rs     # 增量 → 压窗/摘要
│  │  ├─ retrieve.rs   # 多 query → 检索/按 Note 合并/topK/取全文
│  │  ├─ note_ops.rs   # 按 id 读写 Note、反馈
│  │  └─ control.rs    # 控制信号 → 强制触发后台任务
│  ├─ background/
│  │  ├─ scheduler.rs  # 定时调度 + 失败隔离
│  │  ├─ consolidate.rs# 巩固挂接点（Unimplemented）
│  │  ├─ persist.rs    # 定时持久化
│  │  └─ forget.rs     # 遗忘挂接点（Unimplemented）
│  ├─ store/
│  │  ├─ snapshot.rs   # 快照结构 + SNAPSHOT_VERSION
│  │  └─ file_store.rs # JSON 原子写盘实现
│  └─ zenoh/
│     ├─ keys.rs       # Key Expression 集中定义
│     ├─ liveliness.rs # 服务发现/心跳
│     ├─ request.rs    # 订阅 request → 调 service → 发布 reply
│     ├─ pubsub.rs     # 订阅 ingest（推式输入）
│     ├─ publish.rs    # 事件广播（预留，EventNotice）
│     └─ client.rs     # 设备端 ZenohClient
└─ tests/
   └─ offline.rs       # 离线集成测试（不依赖网络）
```

### 3.1 逐文件说明（作用 + 为什么）

#### A. 工程与外壳
- `Cargo.toml`：包元信息、依赖、features、显式两个 bin（`autobins=false`）。为什么：`--bin mock-device` 名称稳定可用。
- `.gitignore`：忽略 `/target`。
- `build.rs`：prost-build 生成 `wire::pb`。为什么：协议与代码不漂移；vendored protoc 免安装。
- `proto/soul_mem.proto`：收发协议唯一事实源。

#### B. `src/` 顶层单文件模块
- `lib.rs`：`pub mod` 汇总与再导出；`#[cfg(feature="zenoh")] pub mod zenoh;`。
- `main.rs`：服务端入口（薄）：`Config::from_env` → `RunningServer::run`。
- `server.rs`：`RunningServer::start/run/shutdown`：装配 zenoh + 后台；优雅退出顺序「持久化 → 停后台 → 下线 zenoh」。
- `config.rs`：`Config` + `EmbeddingMode`；`parse_u64_env`/`parse_f32_env`（空值→默认，显式 0 合法）；`validate()`；含单元测试。
- `error.rs`：`Error` + `ErrorCode` + `Result`。
- 模块根：`wire.rs`/`service.rs`/`background.rs`/`store.rs`/`zenoh.rs`（Rust 2024：根文件声明子模块）。

#### C. `wire/`（协议层）
- `wire.rs`：`pub mod pb { include!(OUT_DIR/soulmem.rs) }` + `op` 常量。
- `convert.rs`：protobuf ⇄ 内部模型转换（`note_to_proto`/`note_from_proto`/`query_from_proto`/`role_to_str`/`feedback_from_proto`/`time_from_string`/`parse_memory_id` 等）与校验。

#### D. `service/`（唯一业务入口）
- `ingest.rs`：`ingest(pb::IngestRequest) -> pb::Ack`；校验（非空/长度/角色）+ 压入滑动窗口。
- `retrieve.rs`：`retrieve(pb::RetrieveRequest) -> pb::RetrieveResponse`；query 转换 → embed → `DefaultPipeline` → 合并/topK/取全文；记录命中。
- `note_ops.rs`：`read_note`/`write_note`/`feedback`；写入 upsert 并刷新 embedding。
- `control.rs`：`control(pb::Control) -> pb::ControlResponse`；Persist 真实，Consolidate/Forget 返回 `Unimplemented`。
- `service.rs`：`SoulMemService` 门面（`from_config`、`ping`、`persist`、`snapshot`、`device_id`、`is_idle`）、`ServiceCore`、`WorkingMemorySlot`、`HashEmbeddingModel`。

#### E. `background/`
- `scheduler.rs`：`run_background`/`BackgroundRuntime`；tokio interval；巩固/遗忘 Idle 门控；失败隔离。
- `consolidate.rs`/`forget.rs`：`run_once` 调 `service.control(...)`（占位，Unimplemented）。
- `persist.rs`：`run_once` 调 `service.persist()`。

#### F. `store/`
- `store.rs`：`MemoryStore` trait（`#[allow(async_fn_in_trait)]`）+ `NoopStore` + `Store` 枚举（`Noop`/`File`）+ `Store::from_config`。
- `snapshot.rs`：`Snapshot`/`WindowEntryDto`/`SNAPSHOT_VERSION`/`validate()`。
- `file_store.rs`：`FileStore`：JSON 原子写盘（临时文件+改名）、缺文件返回 `None`；含单元测试。

#### G. `zenoh/`（唯一对外通道）
- `keys.rs`：`request`/`reply/<id>`/`ingest`/`events`/`liveliness/<id>`/`liveliness/*`。
- `liveliness.rs`：`announce` 声明 token（退出随 runtime 释放下线）。
- `request.rs`：订阅 `request`：解码 `RequestEnvelope` → 按 op 解码内层消息 → 调 service → 编码 `ReplyEnvelope` 发布到 `reply/<id>`。
- `pubsub.rs`：订阅 `ingest`：解码 `IngestRequest` 转交 service。
- `publish.rs`：`publish_event` 发布 `EventNotice`（预留）。
- `client.rs`：`ZenohClient`：`open/call/...`；请求-应答（订阅应答主题 + `request_id` 匹配 + 5s 超时）；`observe_liveliness` 返回样本数（best-effort）。

#### H. `bin/mock_device`（单机模拟器）
- `main.rs`：解析 `--zenoh-prefix` → 打开 `ZenohClient` → 跑场景；bin 名 `mock-device`。
- `scenario.rs`：ping→ingest→write→retrieve→read→feedback→control(persist)→liveliness 观察；断言与日志。

#### I. 测试（离线，不依赖网络）
- `tests/offline.rs`：直接驱动 `SoulMemService` 完成 ingest→write→retrieve→read→feedback→control(persist)，并经 FileStore 验证「落盘 → 重启恢复」。
  为什么：自动化测试不得依赖网络；网络能力由 `mock-device` 手动演示。

### 3.2 总体设计理由
1. **单一业务入口**：所有通道与任务只调 `service` 门面，并发访问/错误处理只在一处。
2. **仅用 pub/sub**：请求-应答以主题 + request_id 模拟，不引入第二种传输与 schema。
3. **模块根 + 同名子目录**：满足 Rust 2024（无 `mod.rs`），与既有 crate 一致。
4. **调度与实现解耦**：`scheduler` 管"何时跑、怎么隔离"，占位任务先占接口位。
5. **持久化抽象化**：`store` 定义契约，未来切 SurrealDB = 新增实现。
6. **薄进程入口**：`main`/`server` 只管装配与启停，业务全在 lib 内可测。
7. **命名 = 领域能力**：顶层模块名即能力分类，目录树可直接映射模块。
8. **客户端逻辑可复用**：`zenoh/client.rs` 供 mock-device 与后续设备端复用。
9. **演示与实现分离**：`mock_device` 独立于服务端，演示逻辑不进核心业务。

### 3.3 英文专有名词速查
| 术语 | 概念 |
|---|---|
| crate / workspace | Rust 编译单元 / Cargo 工作区。 |
| lib / bin | 库 / 可执行程序。 |
| edition / features | Rust 版本 / Cargo 特性开关。 |
| DTO | 传输对象；本 crate 中即 protobuf 生成的消息类型。 |
| proto / protobuf | 结构化消息定义格式（`.proto`）。 |
| prost / prost-build | Rust protobuf 编解码库与代码生成器。 |
| protoc（vendored） | protobuf 官方编译器，随依赖自带。 |
| 消息信封（envelope） | 含 request_id/op/payload 的外层消息，内层为具体操作 protobuf 字节。 |
| pub/sub | 发布/订阅。 |
| 请求-应答 | 在 pub/sub 上模拟"发请求拿结果"。 |
| request_id | 请求唯一标识，用于匹配应答。 |
| Key Expression | zenoh 主题表达式。 |
| liveliness | zenoh 存活/发现机制。 |
| session | zenoh 会话句柄。 |
| loopback | 本机回环。 |
| mock / mock-device | 模拟"外部设备"的可执行程序。 |
| embedding / LLM client / DefaultPipeline | 见 soul-mem-query / runtime / algo。 |
| MemoryNote / MemoryLink / Record / Information | 既有数据概念，见 orchestration.md 与 core/runtime。 |
| top-K / cluster / Idle 门控 | 取前 K 名 / 记忆簇 / 仅空闲运行后台任务。 |
| tokio interval / trait | 定时器 / Rust 接口。 |
| snapshot / FileStore / 原子写盘 | 状态副本 / 文件实现 / 临时文件+改名。 |
| graceful shutdown | 优雅退出。 |
| schema | 数据结构版式定义。 |

---

## 4. 新增文件清单与职责

### 4.1 包工程文件
| 文件 | 类型 | 职责 |
|---|---|---|
| `Cargo.toml` | 新增 | 包名 `soul-mem`（`publish=false`）；`autobins=false` + 显式 `[[bin]]`：`soul-mem`(src/main.rs)、`mock-device`(src/bin/mock_device/main.rs)；features：`zenoh`、`file-store`（默认开启）。 |
| `.gitignore` | 新增 | `/target`。 |
| `build.rs` | 新增 | `prost-build` 编译 `proto/soul_mem.proto` 到 `OUT_DIR`；vendored protoc。 |
| `proto/soul_mem.proto` | 新增 | 收发协议唯一事实源。 |
| `tests/offline.rs` | 新增 | 离线集成测试（不依赖网络）。 |

### 4.2 第三方依赖
| 依赖 | 用途 |
|---|---|
| `zenoh` | 订阅/发布、liveliness（唯一对外通信）。 |
| `prost` | protobuf 编解码。 |
| `serde`/`serde_json`/`chrono`/`uuid`/`thiserror`/`anyhow`/`tokio`/`petgraph` | 与 workspace 对齐（`serde_json` 仅用于持久化快照）。 |
| `log` + `env_logger` | 日志。 |
| （build）`prost-build`、`protoc-bin-vendored` | 代码生成、自带 protoc。 |
| （dev）`tempfile` | 快照文件测试。 |

### 4.3 `src/` 源码模块
| 文件/目录 | 职责 |
|---|---|
| `src/lib.rs` | 顶层模块声明与再导出。 |
| `src/main.rs` | 服务端入口（bin `soul-mem`）。 |
| `src/server.rs` | `RunningServer` 装配与优雅退出。 |
| `src/config.rs` | `Config`/`EmbeddingMode` + `parse_u64_env`/`parse_f32_env`（空值→默认、显式 0 合法）+ `validate` + 单元测试。 |
| `src/error.rs` | `Error` + `ErrorCode` + `Result`。 |
| `src/wire.rs` | `pb`（prost 生成）+ `op` 常量。 |
| `src/wire/convert.rs` | protobuf ⇄ 内部模型转换与校验。 |
| `src/service.rs` | `SoulMemService`/`ServiceCore`/`WorkingMemorySlot`/`HashEmbeddingModel`；对外方法 `ping/ingest/retrieve/read_note/write_note/feedback/control/persist`。 |
| `src/service/ingest.rs` | ingest 实现。 |
| `src/service/retrieve.rs` | retrieve 实现。 |
| `src/service/note_ops.rs` | read/write/feedback 实现。 |
| `src/service/control.rs` | control 实现。 |
| `src/background.rs` | `run_background`/`BackgroundRuntime`。 |
| `src/background/scheduler.rs` | 定时调度与失败隔离。 |
| `src/background/consolidate.rs` | 巩固占位。 |
| `src/background/persist.rs` | 定时持久化。 |
| `src/background/forget.rs` | 遗忘占位。 |
| `src/store.rs` | `MemoryStore`/`Store::{Noop,File}`/`from_config`。 |
| `src/store/snapshot.rs` | `Snapshot`/`WindowEntryDto`/`SNAPSHOT_VERSION`。 |
| `src/store/file_store.rs` | `FileStore` JSON 原子写盘/恢复。 |
| `src/zenoh.rs` | `ZenohRuntime` 创建 session、liveliness、request/ingest 订阅。 |
| `src/zenoh/keys.rs` | Key Expression 集中定义。 |
| `src/zenoh/liveliness.rs` | 服务发现/心跳。 |
| `src/zenoh/request.rs` | 请求-应答服务端。 |
| `src/zenoh/pubsub.rs` | ingest 订阅。 |
| `src/zenoh/publish.rs` | 事件广播（预留）。 |
| `src/zenoh/client.rs` | 设备端 `ZenohClient`（含 `observe_liveliness -> usize`）。 |
| `src/bin/mock_device/main.rs` | 模拟器入口（bin `mock-device`）。 |
| `src/bin/mock_device/scenario.rs` | 演示场景。 |

---

## 5. 修改文件清单

| 文件（相对仓库根） | 类型 | 修改内容 |
|---|---|---|
| `Cargo.toml`（根 workspace） | 修改 | `[workspace].members` 增加 `"crates/soul-mem"`；`[workspace.dependencies]` 增加 `zenoh`、`prost`、`log`。 |

> 其余 crate 一律不改；`soul-mem` 仅通过四个运行时 crate 的公开 API 组装。

---

## 6. 与 `orchestration.md` 的对照

| orchestration 规划点 | 落实位置 |
|---|---|
| Service 编排层 | `service.rs`、`service/*`、`server.rs` |
| 服务输入（query 集合 + 信息增量） | zenoh `request`/`ingest` → `service/ingest.rs`、`service/retrieve.rs` |
| 输出 MemoryNote 集合 | `service/retrieve.rs` + `wire/convert.rs` |
| 多 query 合并/topK/取内容（🔲） | `service/retrieve.rs`（结构化输出；自然语言模板延后） |
| 控制信号强制触发（🔲） | zenoh `request`(op=control) → `service/control.rs` → `background/scheduler.rs` |
| 定时 Idle 巩固（🔲） | `background/consolidate.rs`（占位） |
| 定时/优雅退出持久化（🔲） | `background/persist.rs` + `store/*` |
| SurrealDb（🔲） | `store` trait 抽象预留 |
| 遗忘机制（🔲） | `background/forget.rs` 占位 |
| 按 id 读写 MemoryNote（🔲） | `service/note_ops.rs` + zenoh `request`(op=read/write) |
| liveliness/heartbeat（🔲） | `zenoh/liveliness.rs` |

---

## 7. Demo 验收场景

> 单机运行：终端 A `cargo run -p soul-mem`；终端 B `cargo run -p soul-mem --bin mock-device`。
> 可选：终端 A 设置 `$env:SOUL_MEM_STORE_PATH="...\snapshot.json"` 以验证落盘/重启恢复。

1. 设备 A 启动服务（zenoh + FileStore），liveliness 上线。
2. 设备 B 经 zenoh 请求-应答：ping → ingest → write → retrieve → read → feedback → control(persist)；经 ingest 主题单向发布增量；观察 liveliness（best-effort）。
3. 优雅退出设备 A：持久化 → 下线 → 退出码 0；重启后快照恢复。
4. 验证点：主链路正确、非法入参被拒、日志可观测、可重复。

---

## 8. 实施阶段

- **M0 工程骨架**：Cargo/`.gitignore`/`build.rs`/proto/根 workspace、`config/error/lib.rs`，编译通过。
- **M1 核心编排（无网络）**：`wire`(pb+convert) + `service` + `store`(FileStore) + 单测。
- **M2 zenoh 通道**：`zenoh/*` + `server.rs` 装配与优雅退出。
- **M3 后台与占位任务**：`background` + 控制信号接线。
- **M4 mock-device 与离线测试**：`bin/mock_device` + `tests/offline.rs`；网络演示手动完成。

---

## 9. 决策记录与待确认

**已按当前代码确定**：
1. 包名 `soul-mem`；bin 名 `soul-mem`/`mock-device`（显式 `[[bin]]`）。
2. zenoh 部署：`Config::default()`（P2P/scouting，免 router）。
3. 请求-应答：`reply/<request_id>` + 5s 超时。
4. 配置：环境变量 `SOUL_MEM_*`；数值空值→默认，显式 `0` 停用定时任务。
5. 日志：`log` + `env_logger`。
6. SurrealDB：本期不引入。
7. mock-device：默认连接已运行的服务，不自动拉起。

**仍待确认/后续**：
- 自然语言模板输出（本期返回结构化 MemoryNote 集合）。
- 真机双设备验收（回环无法覆盖真实网络因素）。
