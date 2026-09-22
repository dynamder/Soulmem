# SoulMem `feature/orchestration` 审阅报告

| | |
|---|---|
| **审阅对象** | `origin/feature/orchestration` @ `c539c4e` |
| **提交规模** | 38 文件 / +6005 / −91；新增 crate `crates/soul-mem`（31 个 `.rs`、约 2.9k 行，外加 `proto` 351 行与设计文档 605 行） |
| **与基线的关系** | 基于 `origin/dev`，恰好 **1 个提交、0 落后**（相对 `origin/main` 则领先 163、落后 1） |
| **审阅方式** | 静态通读全部新增代码 + **实际编译与门禁验证**（见 §2）；未做运行时网络联调 |
| **状态** | 结论待与作者讨论；§6 列出需要团队决策的事项 |

---

## 1. 结论摘要（只读这一节也够）

### 1.1 这份提交做对的部分

先说清楚，因为它决定了下面所有讨论的基调：

- **工程约定全部对齐**，而且是"自愿"对齐：全 crate 无 `mod.rs`，`service.rs` + `service/`、`zenoh.rs` + `zenoh/`、`store.rs` + `store/`、`wire.rs` + `wire/`、`background.rs` + `background/` 一一对应。需要说明的是，`AGENTS.md`、`.editorconfig`、`scripts/check_layout.py` 这些约束**在 `dev` 分支上并不存在**（`dev` 的 `scripts/` 只有 `mutants_gate.py`），作者是在没有约束的情况下自己遵守的。
- **格式与编码干净**：`rustfmt --edition 2024 --check` 对 31 个文件全部通过；全树 370 个文本文件 **0 个非法 UTF-8**。
- **`plan.md` 质量很高**（365 行）：逐文件说明职责与理由，明确标注哪些是占位、哪些是已知限制，且实测与实现一致。
- **单一业务入口真的做到了**：`zenoh/request.rs:65-97` 的 `dispatch` 是 7 个分支的纯 op → 方法映射，零业务逻辑；`pubsub.rs:28` 直接调 `service.ingest`。**没有任何传输层里的业务判断。**
- **库本身没有全局可变状态**（只有 `main.rs` 的 `env_logger` `Once`），因此多实例天然可行。

### 1.2 需要处理的问题，按性质分三类

| 类别 | 条目 | 为什么单独列出 |
|---|---|---|
| **A. 会真丢数据的缺陷**（4 条，与本次重构解耦） | §3.1 `write_note` 清零行为历史／§3.2 并发 `persist` 可损坏快照／§3.3 落盘失败仍退出码 0／§3.4 退出顺序 | 这些不是设计取舍，是正确性问题。建议**先修这批，再谈重构** |
| **B. 架构层面的修正方向**（维护者提出 7 条，见 §4） | 领域 API／op 枚举化／zenoh 用 queryable／gRPC／薄封装分层／类型签名／YAGNI | 方向都成立；其中若干条附带了需要收紧的地方 |
| **C. 流程与门禁** | 单提交粒度、测试覆盖不足、CI 门禁当前是红的 | 决定"能不能合并"，不决定代码质量 |

### 1.3 三个需要团队（而非个人）决定的事项

1. **zenoh 主题命名约定**——目前是自创的一套。既然本团队的组件互通用 zenoh，**主题命名就是所有组件的公共接口**，应该有跨组件约定而不是各写一套。**维护者将与作者就此专门讨论**（见 §6.1）。
2. **gRPC 与 zenoh 并存的需求确认**——见 §4.4，这不是实现缺陷。
3. **对外 gRPC 的 health 探针是否认证**——见 §6.2。

---

## 2. 编译与门禁验证（实测结果）

为保证结论可信，本次审阅实际执行了构建，而不是只做静态判断。

| 检查项 | 命令 | 结果 |
|---|---|---|
| 单元 + 集成测试 | `cargo test -p soul-mem --locked` | ✅ **exit 0，8/8 通过**（1m38s；lib + 两个 bin + 集成测试**全部链接成功**） |
| 新代码 clippy | `cargo clippy -p soul-mem --all-targets --no-deps -- -D warnings` | ✅ **exit 0，零告警** |
| 全 workspace 类型检查 | `cargo check --workspace --all-targets --locked` | ✅ **exit 0** |
| 全 workspace clippy（CI 门禁） | `cargo clippy --workspace --all-targets -- -D warnings` | ❌ **exit 101**，但 3 条错误**全在 `soul-mem-core`**（见下） |
| 格式 | `rustfmt --edition 2024 --check`（31 文件） | ✅ exit 0 |
| 全树编码 | 370 个文本文件 | ✅ 0 个非法 UTF-8 |

通过的 8 个测试：`config::tests` 5 个 + `store::file_store::tests` 2 个 + 集成测试 `offline_flow_and_restart_restore`。

### 2.1 关于 clippy 门禁失败的重要澄清

`cargo clippy --workspace --all-targets -- -D warnings`（`CONTRIBUTING.md` 与 CI 明确要求的门禁）**当前是失败的**，但：

```
error: you should consider adding a `Default` implementation for `SpecificToAbstract`
  --> crates\soul-mem-core\src\memory_links\situation_mem.rs:29:5
error: this `impl` can be derived
  --> crates\soul-mem-core\src\memory_note\situation_mem.rs:165:1
error: this `impl` can be derived
  --> crates\soul-mem-core\src\memory_note\situation_mem.rs:194:1
```

**三条错误全部落在 `soul-mem-core`，而本提交一行都没改这个 crate**（`git diff origin/dev...feature/orchestration -- crates/soul-mem-core/` 输出为空）。`new_without_default` 与 `derivable_impls` 都是纯局部的代码形态 lint，不可能由新 crate 引入。

结论：**这是 `dev` 分支自己的门禁问题，不是本提交引入的**。但它意味着"这份代码要能合并，仓库自己得先是绿的"——建议先单独修掉，不要混进本次改动。

### 2.2 未验证项（诚实声明）

- **`cargo mutants`（≥90% 杀灭率门禁）未实跑**（耗时过长）。按测试数量推断会失败：新 crate 约 2.9k 行生产代码只有 8 个测试函数，而 `wire/convert.rs` 613 行**零测试**。
- **`cargo build --workspace --all-targets` 的全量链接未跑**（只跑了 `check`，不链接；`soul-mem` 自身的链接由 `cargo test -p soul-mem` 覆盖）。
- **未做双设备网络联调**，zenoh 的跨进程行为均为静态推断。
- 运行环境：rustc/cargo 1.97.1，Windows，通过本地代理下载依赖（`zenoh` 1.10.0 与 `protoc-bin-vendored` 需联网获取）。

### 2.3 顺带发现：`dev` 上有一个文件的中文注释已损坏（与本提交无关）

全树扫描中，**只有一个源码文件**命中"重编码损坏"（字节合法 UTF-8，所以任何字节级检查都发现不了）：

`crates/soul-mem-runtime/src/cluster/memory_cluster.rs` —— 内含 8 个注音符号/片假名码点（`\u3125`–`\u3128`、`\u30e4`…），是 UTF-8 被按 GBK 解读后再存成 UTF-8 的典型痕迹。两个值得注意的后果：

- 该文件里那条 `//SAFEUNWRAP:` 的**说明文字本身已损坏**，按规范等于没有说明；
- 注释末尾出现 `?`，说明**已有字节永久丢失**。

这正是 `AGENTS.md` §7 警告的那种不可逆损坏。**它在 `dev` 上，不是本提交引入的**，建议单独处理。

（另有 8 个 `fixtures/example_data/**` 文件命中同类特征，但那是萌娘百科正文里的合法日文，已排除。）

---

## 3. 必修：会真丢数据的缺陷（与重构解耦）

建议这四条**独立于架构重构先行修复**，否则它们会被淹没在架构讨论里。

### 3.1 `write_note` 的 upsert 会静默清空这条记忆的行为历史

`service/note_ops.rs:41-53` 用 `remove_node` + `add_node` 实现 upsert：

```rust
let existed = wm.memory_cluster().read_or_compute(|c| c.contains_node(id));
if existed { let _ = wm.remove_node(id); }
wm.add_node(EmbeddedMemoryNote { note: note_clone, embedding });
```

而 `soul-mem-runtime/src/working_memory.rs:79-85`：

```rust
pub fn remove_node(&mut self, node_id: MemoryId) -> Option<EmbeddedMemoryNote> {
    self.records.remove(&node_id);        // ← Record 被一起删除
    ...
}
```

`add_node` 随后用 `entry().or_insert_with(|| Record::new(node_id))` 建一条**全零**记录。

**后果**：改写一条记忆的正文（哪怕只改一个错别字）会**清零它的检索计数与全部正/负反馈**——而这些正是巩固与遗忘要消费的行为信号。`note_ops.rs` 的文档注释只说"刷新 embedding"，未提及这一点。

**建议**：不要用 remove + add 实现 upsert。`MemoryCluster` 已有 `refresh_node(&MemoryId)` 与 `get_node_mut`，应走"原地替换 + 刷新边"的路径，保留 `Record`。

### 3.2 `FileStore` 的"原子写"在并发下不原子

`store/file_store.rs:23-27` 的临时文件名是**固定**的 `<path>.tmp`，而 `service.persist()` 至少有 3 个并发调用方：

- 后台定时任务（`background/persist.rs`）
- 控制信号（`service/control.rs` 的 `ControlPersist`，经 zenoh 请求，可并发）
- 退出兜底（`server.rs:64`）

**尤其确定的是第三条**：`server.rs:64` 的兜底持久化发生在 `bg.shutdown()`（69-71 行）**之前**，所以"正在跑的周期持久化"与"退出持久化"**结构性必然可能并发**。

两条路径写同一个 `<path>.tmp`：A 写完 tmp 后 B 截断重写，A 的 `rename` 会把被截断的半成品挪成正式快照 → 磁盘上是残缺 JSON；B 的 `rename` 报 `NotFound`。重启时 `store.load()` 解析失败（`file_store.rs:60-61`），`RunningServer::start` 用 `?` 传播（`server.rs:39`）→ **服务根本起不来**，而旧快照已被覆盖，无法回退。

**建议**：① 临时文件名加唯一后缀（如 `<path>.<uuid>.tmp`），或在 `MemoryStore`/`Store` 层为 `save` 加互斥；② 把 `bg.shutdown()` 放在兜底 persist **之前**；③ 补一条"并发写同一 `FileStore`"的回归测试。

### 3.3 最终持久化失败仍然以退出码 0 结束

`server.rs:64-66` 只记日志；`shutdown()` 返回 `()`；`main.rs:19-22` 仅在 `run` 返回 `Err` 时才 `exit(1)`：

```rust
if let Err(e) = self.service.persist().await {
    log::error!("final persist failed: {e}");     // ← 之后照常 Ok(())
}
```

**后果**：运维层面"最后一次没落盘、数据已丢"与"干净退出"**完全无法区分**。这与 `AGENTS.md` §4.6「不假装正常结束」直接冲突。

**建议**：`shutdown()` 返回 `Result`，把兜底持久化失败作为非零退出码上报。

### 3.4 优雅退出顺序：先取快照、后停入口

`server.rs:26-29 / 62-80` 的顺序是 **持久化 → 停后台 → 下线 zenoh**，而 `zenoh.shutdown()`（`zenoh.rs:62-65`）才是 `task.abort()` 的地方。

**中间没有任何静默期**：此刻仍可能有 `write_note` / `feedback` 正在 `dispatch` 里执行，其变更在快照**之后**才落到内存，随后任务被 abort、进程退出——这次写入既没进快照、也没有第二次持久化；同时**调用方因中途 abort 收不到应答**，无法知道操作是否生效，重试就可能重复。

需要说明的是：这个顺序与 `plan.md` §2.7 写的完全一致，**即计划本身在这里就是错的**，不是实现偏离了计划。

**建议**：改为"静默入口 → 等在途请求完成 → 兜底 persist → 关闭 session"；若保留现顺序，至少 `zenoh.shutdown()` 之后再补一次 persist。**注意：加上 gRPC 后有两个入口，这个丢数据窗口会翻倍，所以本条在双传输方案下是必修项。**

---

## 4. 架构层面的修正方向

以下 7 条由维护者提出。逐条给出**评估结论**与**落地要点**。

### 4.1 编写符合 Rust lib 习惯的接口，不经网络栈，可内联

**评估：成立，且是最重要的一条。但只改签名不够，必须同时给出注入点。**

现状：`SoulMemService` 的公开方法**全部收发 protobuf 类型**（`service.rs:6` 自己写明）：

```rust
pub async fn ingest(&self, req: pb::IngestRequest) -> Result<pb::Ack>
pub async fn retrieve(&self, req: pb::RetrieveRequest) -> Result<pb::RetrieveResponse>
```

于是"把一句话塞进滑动窗口"需要手写 `pb::InfoDelta { role: pb::MessageRole::RoleUser as i32, content }`。全 crate **没有任何 `From`/`TryFrom` 在领域类型与 pb 之间搭桥**，也没有 `ingest_text(&str)` 这类薄包装。

**但更关键的是缺少注入点**——没有它，这个 lib 对内联使用者等于"只能用假 embedding 跑通的封闭服务"：

```rust
fn build_model(config: &Config) -> Result<Arc<dyn EmbeddingModel + Send + Sync>>   // 私有
match config.embedding_mode {
    EmbeddingMode::Hash => Ok(Arc::new(HashEmbeddingModel::new(64))),
    EmbeddingMode::Bge  => Err(Error::Unimplemented(..)),        // 真实模型未接
}
```

`EmbeddingMode` 只有 `Hash`/`Bge`、`build_model` 私有、`ServiceCore` 是 `pub(crate)`、`from_config` 是唯一构造入口。同类问题还有两个：

| 想替换的东西 | 现状 | 阻塞点 |
|---|---|---|
| embedding 模型 | 字段是 `Arc<dyn EmbeddingModel>` | 构造私有，无注入入口 |
| LLM | `Arc<LlmClient>` | 由 config 内部构造，无注入入口 |
| 持久化 | `MemoryStore` trait 是 `pub` | 服务接收的是**闭合枚举** `Store::{Noop, File}`，实现了 trait 也塞不进去（`store.rs:44-58`） |

**落地要点**：① 门面改为领域类型签名；② 加 builder 提供 `.embedding(...)` / `.llm(...)` / `.store(...)` 三个注入点；③ `Store` 从闭合枚举改为可注入（trait object 或泛型）；④ `lib.rs` 加根级再导出（现在 8 个 `pub mod` 之外没有任何 `pub use`，`soul_mem::SoulMemService` 不存在，必须写 `soul_mem::service::SoulMemService`）。

### 4.2 op 操作使用枚举而非常量字符串

**评估：成立，而且可以有比"换成枚举"更强的结果。**

现状 `wire.rs:17-25` 是 7 个字符串常量，`RequestEnvelope { request_id: String, op: String, payload: Vec<u8> }` —— 内层消息**类型在类型系统里不存在**，且 `other => Err(InvalidArgument)`（`request.rs:95`）意味着 **op 写错不会编译报错**。

**落地要点**：因为已决定改用 per-op selector（见 4.3），`op` 字符串可以**整个消失**——操作由主题承载，编译期完整性由"每个 selector 一个处理函数"提供，`RequestEnvelope`/`ReplyEnvelope`/`request_id` 一并删除。这比"换成一个 `enum Op`"更彻底。

### 4.3 zenoh 的请求-应答改用 query API

**评估：成立，且与设计文档的原始意图一致。**

`orchestration.md:7` 写的正是"通过 zenoh 的 pub/sub，**query**，liveliness api"；而 `plan.md` §2.2 主动选择在 pub/sub 上模拟。实测确认：`declare_queryable` / `get_async` / `.reply(` / `session.get` 在**整个 crate 里零匹配**。

现状是手搓的：`reply/<request_id>` 主题 + 客户端 5s 超时 + `request_id` 匹配循环（`client.rs:52-101` + `request.rs:40-59`）。

**⚠️ 精确的改法是：只把"请求-应答"换成 queryable，不要全量替换。**

| 机制 | 归属 |
|---|---|
| `queryable` | ping / ingest（带 ack 的那条）/ retrieve / feedback / control |
| `pub/sub` | `<prefix>/ingest` 单向推式输入（本来就是 pub/sub 语义，**保持不变**） |
| `liveliness` | 服务发现/心跳（**保持不变**，原生能力用对了） |

**收益（不只是"更地道"）**：

1. **消灭每次 RPC 都 declare 一个订阅者**——现在 `client.rs:53-58` 每个请求都新建 `reply/<uuid>` 订阅，返回时释放；高频调用下是持续的 declare/undeclare 抖动。
2. **消灭手工应答过滤**——`client.rs:83-85` 的 `if reply.request_id != request_id { continue }` 之所以存在，就是因为主题对所有人可写。
3. **顺带消掉一个安全项**——不再有公开可写的 `reply/<id>` 主题，"同网段节点伪造应答"的攻击面基本消失，`request_id` 也不再需要作为 key 的一部分去校验。
4. **得到更地道的主题设计**——queryable 需要 selector，于是自然变成**按操作分主题**（`<prefix>/req/retrieve` 等），而不是"一个 request 主题 + op 字符串"。
5. **信封整个消失**——selector 已表达操作，zenoh 已负责应答路由，`RequestEnvelope` / `ReplyEnvelope` / `request_id` 都不再需要。

**要诚实的一点**：queryable **不是认证**。P2P 局域网里其他节点仍能声明同 selector 的 queryable 或直接发 query。它缩小攻击面，但不构成信任边界——文档里需写明。

### 4.4 添加 gRPC 接口

**评估：需求成立，但需要先把问题定性说清楚——这不是本提交的实现缺陷。**

原因：**本提交实现的是 `plan.md`，而 `plan.md` 抬头明写"不使用 gRPC"**，§2.2 重申"仅 zenoh 订阅/发布"。作者没有漏做，他做的是另一份文档。真正的问题是**同一个提交里两份设计文档对传输契约互相矛盾，而实现只覆盖了其中一份**。

维护者已确认需求：本提交所属组件是团队"数字生命计划"的一环，**团队内部组件走 zenoh，对外集成与可替换性走 gRPC，两者并存，都不是可选项。**

有两个代码侧的观察支持这个方向：

- **`ErrorCode` 的 7 个变体正好是 gRPC canonical code 的精确子集**（`InvalidArgument` / `NotFound` / `AlreadyExists` / `Unavailable` / `FailedPrecondition` / `Unimplemented` / `Internal`，名字逐字一致）。这说明错误模型本来就是按对外 RPC 设计的，`ErrorPayload` 只是把它临时塞进 pub/sub 载荷的容器——加 gRPC 时这块几乎即插即用。
- **proto 里连 `service` 块都没有**，所以消息可复用，但服务定义要从零写。

**落地要点**：① proto 加 `service` 定义（与消息同文件，**一份 proto**，见 §5.1）；② `tonic` 与 `prost` 大版本必须对齐（现为 `prost = "0.14"`）；③ `build.rs` 用 `CARGO_FEATURE_GRPC` 决定是否运行 `tonic-build`（protoc 已由 `protoc-bin-vendored` 提供，无需新增）；④ 新增监听地址配置；⑤ **退出顺序必须按 §3.4 修好**（双入口让丢数据窗口翻倍）；⑥ **必须补一个非 Rust 的 gRPC 客户端示例**（`grpcurl` 或 Python），否则"方便外部集成"这个理由从未被验证过，而那是加 gRPC 的全部意义。

### 4.5 zenoh 与 gRPC 薄封装 Rust lib 接口，避免重复代码

**评估：完全成立，这是整套修正的枢纽。**

现状比预期好：**zenoh 现在已经是薄封装**（`dispatch` 是纯映射，`pubsub` 直接调 service）。所以这条的工作量主要在"让 lib 说领域语言"，而不是"把逻辑从适配器搬回去"。

建议固化成 4 条可执行规则，否则"薄"会在几个月后重新变厚：

1. **单一转换点**：pb ⇄ 领域只存在于 `wire/convert.rs`；`service` 永不 `use crate::wire`。
2. **适配器零业务**：`transport/*` 只能 解码 → 调 service → 编码。需要新行为就扩 service，不在适配器里写分支。
3. **校验分层**：格式/枚举/id 合法性属于适配器（wire 关注点）；语义约束（窗口容量、相似度阈值、空 query 拒绝）属于 service。
4. **返回值语义由领域定，传输再翻译**：例如 `read_note` 在领域层返回 `Result<Option<MemoryNote>>`（Rust 习惯），由 gRPC 适配器把 `None` 映射成 `NOT_FOUND`、zenoh 适配器映射成对应的 `Error` 载荷。**这正是"薄封装"的含义——传输关心的事留在传输里。**

**同时要明确一点**：要避免的是**重复实现**，不是重复契约。两个适配器共享同一组领域方法，但暴露的能力面可以不同（见 §5.3）。

### 4.6 善用类型签名，阅读签名即可理解用途、输入与输出

**评估：成立。但最大的收益不在签名措辞，而在消掉"隐形的锁持有期"。**

签名只是其中一半。现在真正让读者跟不住数据路径的是这段（`service/retrieve.rs`）：

```
:39  let wm = self.wm_arc().await;   // 取只读 Arc
:77  drop(wm);                       // ← 纯粹为了释放才写的
:82  self.with_wm(...).await?;       // 再取独占
:99  let wm = self.wm_arc().await;   // 又取一次
:110 drop(wm);                       // ← 又为了释放
```

两个 `drop(wm)` 对业务零信息量，却是**承重的**（不 drop 会让 `with_wm` 的 `Arc::get_mut` 永久失败而自旋）。签名里完全看不出"本函数跨 await 持有共享借用"。

**建议改为作用域化访问器**：

```rust
self.memory.read(|wm| pipeline.run(wm)).await;            // 只读，同一临界区内取 hits + summary
self.memory.write(|wm| wm.record_retrieval(id)).await?;   // 独占
```

一次解决三件事：① 读者从签名即知访问模式；② **顺带修掉"跨时刻拼装"**（现在 `hits` 在 `drop` 前取、`summary`/`short_history` 在重新获取后取，两者不是同一时刻的快照）；③ 写者不再因一个只读 `Arc` 克隆而满核自旋。

**⚠️ 但不可一刀切**：`ingest` 必须在临界区内 `await` 一次 LLM 摘要调用（`push(...).await`），同步闭包覆盖不了。规则应是"**能用同步闭包就用；只有必须 await 的地方才允许持借用跨 await，并在签名或注释里显式标注**"。

签名层面另有两个现成的小问题：`ControlResponse.ok` 是**恒为 true、永不为 false 的字段**（失败走 `Err`），属于签名谎报信息；`retrieve` 的返回类型**没有任何东西表达顺序**，而实际顺序是 `MemoryId`（UUID）字典序——签名要表达输出，前提是先定下顺序契约（见 §6.3）。

### 4.7 YAGNI：精简、避免过度抽象，但保证健壮性

**评估：成立，但它与 4.4 存在明显张力，且当前已有"过度抽象"需要删除而非保留。**

**（a）与 gRPC 的张力要写明。** 为个人家用场景引入 `tonic + hyper + tower` 是本次改动里最大的一笔非必需依赖。维护者已确认 gRPC 是需求，那么它就是需求而非 YAGNI 问题——但报告里必须把这个取舍明写，否则"YAGNI"与"必须加 gRPC"并列会显得标准不一致。

**（b）现存应当删除的过度抽象**：

- **`MemoryStore` trait + 闭合 `Store` 枚举 = 只有成本没有收益**。trait 的存在暗示可替换，但服务接收枚举，替换不了。二选一：做成真正可注入（4.1 需要），或删掉 trait 只留 `FileStore`/`NoopStore`。
- **`background/consolidate.rs` + `forget.rs` 共 40 行，函数体只是"过 Idle 门控 → 调 `control(Unimplemented)`"**。与其三个近乎相同的 `run_once`，不如让 scheduler 持一张任务表 `[(TaskKind, Interval)]` + 一个 `dispatch(kind)`：代码更少，且"一共有哪些定时任务"在一个地方看得全。
- **`wire.rs:14` 的 `pub use pb::*` 与 `pub mod pb` 并存**：同一批类型两条路径。
- **两个 `pub fn spawn_*` 在私有模块里**（`zenoh.rs:14-15` 的 `mod request;` / `mod pubsub;`），从 crate 外不可达却标了 `pub`。

**（c）"保证健壮性"这半句对应 §3 的四条必修缺陷**，它们不是重构的附带品。

---

## 5. 目标架构

### 5.1 契约：一份 proto，只写必要的消息类型

已确定**只用一份 proto**，理由是"即使在内部组件，也需要保证可维护性——不能实现一动、所有接口都要动"。这个理由对内部组件同样成立。

关键纪律：**富模型留在 Rust 侧，线上只走窄契约。** 于是当前 proto 里 55 个 message/enum 中，绝大多数（只为 1:1 镜像 `MemoryNote` 内部形状而存在）可以整批删除。

```protobuf
syntax = "proto3";
package soulmem.v1;

// ---------- 通用 ----------
enum Role { ROLE_UNSPECIFIED = 0; ROLE_USER = 1; ROLE_ASSISTANT = 2; }

message InfoDelta { Role role = 1; string content = 2; }

message Error {                 // ErrorCode 的 7 个值走这里；与 gRPC status 逐字对应
  string code = 1;              // invalid_argument / not_found / ... / internal
  string message = 2;
}

// ---------- 输入 ----------
message PingRequest {}

message IngestRequest { repeated InfoDelta deltas = 1; }

message RetrieveRequest {
  string query = 1;             // 自由文本：不再需要 5 层嵌套的查询结构
  uint32 limit = 2;             // 0 = 服务端默认
}

message FeedbackRequest { string note_id = 1; FeedbackKind kind = 2; }
enum FeedbackKind {
  FEEDBACK_UNSPECIFIED = 0; FEEDBACK_POSITIVE = 1;
  FEEDBACK_NEGATIVE = 2;    FEEDBACK_NEUTRAL = 3;
}

message ControlRequest { ControlKind kind = 1; }
enum ControlKind {
  CONTROL_UNSPECIFIED = 0; CONTROL_PERSIST = 1;
  CONTROL_CONSOLIDATE = 2; CONTROL_FORGET = 3;
}

// ---------- 输出 ----------
message Memory {                // 契约自己的类型，不是 MemoryNote 的镜像
  string id = 1;                // 仅作引用凭据（feedback 用）
  string content = 2;           // 记忆内容文本，可直接喂 LLM
  double score = 3;             // 相关度
  repeated string tags = 4;
}

message Retrieval {
  repeated Memory memories = 1;  // 按 score 降序
  string summary = 2;            // 工作记忆当前摘要
  repeated InfoDelta recent = 3; // 短期上下文（滑动窗口）
}

message Ack { string message = 1; }

// ---------- 一份 proto 同时定义 gRPC 服务 ----------
service SoulMem {
  rpc Ping    (PingRequest)     returns (Ack);
  rpc Ingest  (IngestRequest)   returns (Ack);
  rpc Retrieve(RetrieveRequest) returns (Retrieval);
  rpc Feedback(FeedbackRequest) returns (Ack);
  rpc Control (ControlRequest)  returns (Ack);
}
```

zenoh 侧（per-op selector + queryable，**无信封、无 request_id**）：

```
queryable   <prefix>/req/ping      payload = PingRequest      reply = Ack
            <prefix>/req/ingest    payload = IngestRequest    reply = Ack
            <prefix>/req/retrieve  payload = RetrieveRequest  reply = Retrieval
            <prefix>/req/feedback  payload = FeedbackRequest  reply = Ack
            <prefix>/req/control   payload = ControlRequest   reply = Ack
pub/sub     <prefix>/ingest        payload = IngestRequest    （单向，无应答）
liveliness  <prefix>/liveliness
```

**这套收缩顺带消灭的已报缺陷**（这是"只写必要消息类型"最有力的支撑）：

| 缺陷 | 为什么消失 |
|---|---|
| 6 处"缺失子消息 → 凭空造默认值"（`convert.rs:396/401/436/508/600/609`） | 全在 `*_from_proto`（从线上**接受**富模型的方向）。窄契约不再接受富模型 → 这些函数整个不存在 |
| `parse_time_or_now` 把空时间戳伪造成 `Utc::now()`（`convert.rs:44-50`，用于 235-237 三处） | 同上，`note_from_proto` 消失 |
| `MemoryLink.id` 在 wire 往返被随机重建（proto 无 id 字段 + `from_tuple` 用 `LinkId::default()`） | 同上，链接不再上线 |
| `time_span` 解析错误被 `if let Ok` 吞掉（`convert.rs:168`） | 查询结构简化成 `string query` 后不再存在 |
| `concept_from_proto` / `action_kind_from_proto` 把未指定落到具体语义（`convert.rs:296-301 / 526-532`） | 同上 |
| `wire/convert.rs` 613 行、约 40 个对称转换函数 | 收缩为约 5 个**单向投影** |

**代价（一项，且必须做）**：`MemoryNote → Memory.content` 这段"把内部记忆投影成文本"目前在仓库里**不存在**——它正是 `orchestration.md:217` 标 🔲 的「提取记忆内容，按照模板填充为自然语言」。窄契约把这条从"可选优化"变成**契约要求**。建议最小实现：按 `MemoryType` 分支取各自的描述字段拼接，**不要**引入模板引擎。

同理，`RetrieveRequest.query: String` → 领域查询需要一个"文本 → `MemoryRetrieveQuery`"的构造（最小版：`text → SemanticQueryUnit { concept_identifier: text }`，与 mock-device 现在的用法一致）。按 4.5 的分层规则，这属于**语义**，应落在 service 侧。

### 5.2 read/write 的归属（已定）

**v1 线上契约不含 `read_note` / `write_note`**，它们只保留在 Rust 领域 API 上（供内联消费者、测试、工具使用）。

理由：富模型不能上线，而"按 id 读写一条 `MemoryNote`"本质就是富模型操作。`orchestration.md` 本来就把"按 id 读写"标为 🔲，与路线图一致。若将来外部确实需要，再**专门设计一个窄的 note 表示**——并且必须明确声明它是**投影、不是镜像**。

⚠️ **不要走"给 read/write 造一个扁平 note"这条路**：写入会静默丢掉 links 与情境/程序性结构，等于制造一个新的静默数据损失点（与 §3.1 同类，而且更隐蔽）。

### 5.3 两个传输共享实现，但不共享能力面

| 能力 | fabric（zenoh） | 对外（gRPC） |
|---|---|---|
| ping | ✓ | ✓（并建议额外提供标准 `grpc.health.v1`，让外部工具链直接可用） |
| ingest（带 ack） | ✓ | ✓ |
| ingest（单向推式） | ✓ | — |
| retrieve / feedback / control | ✓ | ✓ |
| liveliness | ✓ | —（外部靠 host:port 寻址） |
| read/write 富 note | 仅 Rust 领域 API | — |

### 5.4 建议的模块结构

沿用仓库既有约定（`xxx.rs` + `xxx/`、无 `mod.rs`、标识符英文、注释中文）：

```
crates/soul-mem/
├─ Cargo.toml          features: default = ["zenoh", "grpc", "file-store"]
├─ build.rs            prost-build；CARGO_FEATURE_GRPC 时加 tonic-build
├─ proto/soul_mem.proto  消息 + service（一份）
└─ src/
   ├─ lib.rs           仅 pub mod + 根级再导出（SoulMemService / Config / Error）
   ├─ config.rs        配置 + 校验（新增 grpc addr / tls / token；密钥用 secrecy）
   ├─ error.rs         Error + ErrorCode；附 ErrorCode ⇄ gRPC status 映射
   ├─ memory.rs        ★替代 WorkingMemorySlot：read / write 作用域访问器
   ├─ model.rs         ★embedding 注入点 + HashEmbeddingModel（从 service 移出）
   ├─ service.rs       ★领域门面：只收领域类型，永不 use crate::wire
   │  ├─ service/ingest.rs / retrieve.rs / note_ops.rs / control.rs
   ├─ store.rs         ★可注入的 MemoryStore（替换闭合枚举）
   │  ├─ store/snapshot.rs / file_store.rs
   ├─ background.rs    ★任务表调度（删除 consolidate/forget 占位文件）
   ├─ wire.rs          pb 生成 + 领域投影；不再 re-export pb
   │  └─ wire/convert.rs   ★唯一的 pb ⇄ 领域 转换点
   ├─ transport.rs     ★传输适配层根
   │  ├─ transport/zenoh.rs      queryable + pubsub + liveliness
   │  │  └─ transport/zenoh/keys.rs / client.rs
   │  └─ transport/grpc.rs       tonic 服务实现 + TLS/token 拦截器
   ├─ server.rs        daemon 装配 + 优雅退出（按 §3.4 修正顺序）
   ├─ main.rs          bin soul-mem
   └─ bin/mock_device/ bin mock-device
└─ tests/
   ├─ offline.rs       领域 API 直驱（不经网络）
   ├─ convert.rs       ★投影保真
   └─ transport.rs     ★协议/错误码端到端
```

**关于 `transport/` 这一层**：严格按 YAGNI，两个模块不值得加一层目录。仍推荐它，理由是它把"两个适配器对等且都薄"这个不变量**在目录结构上显式化**了，且 gRPC 会长出子模块。若倾向扁平的 `zenoh/` + `grpc/` 并列，也完全站得住——这是判断题。

### 5.5 建议写进 `lib.rs` 模块文档的不变量

```
R1 依赖单向：service ⊥ wire；wire 只依赖领域类型 + pb；transport 依赖 service + wire
R2 单一转换点：pb ⇄ 领域 只存在于 wire/convert.rs
R3 适配器零业务：transport/* 只做 解码 → 调 service → 编码
R4 校验分层：格式/枚举/id → 适配器；语义约束 → service
R5 领域 API ⊇ 线上契约；线上契约只含必要消息类型
R6 YAGNI 边界：不为 2 个适配器引入 trait / 插件注册 / 泛型传输抽象
R7 错误码全程不丢：Error → ErrorCode → {Error 载荷 | gRPC status} → 客户端 Error，用测试锁定
```

### 5.6 device_id 的删除（已定）

`device_id` 删除。理由：zenoh 靠 key expression 识别资源，对外靠 host:port 寻址，一个带内设备标识是多余的。

**这个删除是净收益**——它一次修掉三个问题：

| 原问题 | 删除后 |
|---|---|
| 身份每次重启都变（`default_device_id()` 取 uuid 低 32 位，且 `Snapshot` 不存它）→ fabric 里其他组件看到"陌生设备上线" | 消失：实例标识改为 `zenoh_key_prefix`（配置值，跨重启稳定） |
| 32 位截断 UUID 的碰撞风险 | 消失 |
| `device_id` 含 `/` 让 `<prefix>/liveliness/*` 静默匹配不到（"对团队隐身"） | 消失（校验对象转为 prefix） |

连带改动：`Config.device_id` 与 `default_device_id()`、`Config::validate` 的非空校验（改为校验 prefix）、`SoulMemService::device_id()`、`PingResponse.device_id`、`ZenohClient::open` 的 `device_id` 参数、liveliness token key 改为 `<prefix>/liveliness`，以及 `plan.md` §2.2 与 `orchestration.md` 的对应描述。

⚠️ **必须同时确立一条部署不变量：「一个 `zenoh_key_prefix` = 一个 SoulMem 实例」。** 因为 prefix 现在是唯一命名空间，两个实例共用 prefix 时同一 selector 会出现两个 responder——**一次 ingest 可能被执行两次**。这是正确性约束，不是命名偏好。建议写进文档作为部署约束，不引入新概念。

### 5.7 events 广播的删除（已定）

删除 `zenoh/publish.rs`（25 行）、`keys.rs:44` 的 `events()`、proto 里的 `EventNotice`，以及 `publish.rs:14` 的 `#[allow(dead_code)]`——**那个 allow 本来就是"预留但没做"的痕迹**。

实测确认它确实是死代码：`publish_event` 全 crate 零调用点，`mod publish;` 私有且无 `pub use`。

将来恢复的入口建议写明：它属于 **fabric 能力面**（不是对外契约），恢复时应在 `transport/zenoh/` 下新增，并在 `transport/grpc/` 侧决定是否用 server-streaming 提供同一能力。

---

## 6. 需要团队决策 / 讨论的事项

### 6.1 zenoh 主题命名约定（**维护者将与作者专门讨论**）

目前是自己发明的一套：`<prefix>/request`、`/ingest`、`/reply/<id>`、`/liveliness/<device_id>`、`/events`，`prefix` 默认 `soulmem`。

既然本团队的组件互通用 zenoh，**主题命名就是所有组件的公共接口**，应该有跨组件约定（例如 `<fabric>/<component>/<resource>`），而不是每个组件各写一套。团队目前**尚无**这份约定，维护者将与作者讨论后确定。

改成 per-op selector 正是引入这套约定的最佳时机。约定确定后，版本信息也可以放在 selector 里（如 `<prefix>/v1/req/retrieve`）。

### 6.2 对外 gRPC 的 health 探针是否认证

标准做法是 `grpc.health.v1.Health` 不认证（否则负载均衡/探针用不了），但这意味着探针端口对外可达。需要团队决定：health 不认证但只暴露最小信息，还是把 health 放独立端口，或与其他 RPC 一同要求 token。

### 6.3 检索结果的排序契约（**已定为：按 score 降序 + limit 截断**）

现状：`service/retrieve.rs:42` 用 `BTreeMap<MemoryId, (f64, MemoryNote)>` 累积，91 行 `by_id.into_values()` 直接产出结果。`MemoryId` 是 `Uuid` 的 newtype 且实现 `Ord`，所以**返回顺序是 UUID 字典序，即随机**；同时跨 query 合并用的是 `max` 而非 `orchestration.md:217` 说的"按优先级加权平均"，`pq.priority` 在合并阶段完全没被使用；也没有 top-K 截断。

已定为：**按 score 降序 + `limit` 截断**，并在返回类型与文档里体现。需要补的实现：排序、截断、以及对"多 query 合并"策略的明确（加权平均 vs 取最大）——后者建议要么实现加权平均，要么在文档里写明当前为取最大。

### 6.4 对外 gRPC 的认证方案

已确认对外需要认证。建议基线：**TLS + bearer token**（mTLS 可选）。两个必须一并处理的点：

- **密钥卫生**：`Config` 目前派生 `Debug` 且含 `llm_api_key` 明文——加 gRPC token 后就是两个密钥。建议用 `secrecy::SecretString`（`soul-mem-runtime` 已在用该 crate），`Debug` 只输出 `has_*`，token 比较用恒定时间比较。这与 `AGENTS.md` §4.5 点名的先例（`OaiCompatConfig` 的 `Debug` 只打印 `has_api_key`）一致。
- **运营成本要知情**：TLS 意味着证书管理（自签 + 本地 CA 或 pinning），token 意味着分发问题。这是"对外集成"必然要付的账。

配置项大致：`SOUL_MEM_GRPC_ADDR`、`SOUL_MEM_GRPC_TLS_CERT`、`SOUL_MEM_GRPC_TLS_KEY`、`SOUL_MEM_GRPC_CLIENT_CA`（mTLS）、`SOUL_MEM_GRPC_TOKEN`。

### 6.5 其他未决项

- **`mock-device` 的定位**：它现在纯粹是 zenoh 客户端。对外契约落地后，是保留它作 fabric 演示，还是补一个 gRPC 演示（或两者）？
- **协议版本字段**：已决定"允许破坏性变更、不留兼容层"，因此 v1 不加版本字段；将来上线后再谈。

---

## 7. 迁移路径

```
P0  绿门禁 + 安全网（先做，否则后面所有改动都没有保险）
    - 修 soul-mem-core 的 3 条 clippy（§2.1）
    - 修 §3 的四条必修缺陷，并补对应回归测试：
      · convert 投影保真
      · 并发写同一 FileStore
      · shutdown 路径（落盘成功/失败、在途请求）
    - 跑一次 cargo mutants 建立基线

P1  领域 API（4.1 / 4.6 / 4.7b）
    - service 去 pb、加三个注入点、read/write 作用域访问器
    - 命名结果结构体、定下排序契约
    - 根级再导出
    ※ 必须最先做：P2–P4 都是在它定义的类型上做改名与接线

P2  契约与分层（4.2 / 4.5 / §5.1 / §5.2）
    - proto 收缩为窄契约（一份）+ service 定义
    - 建 transport/ 目录，pb 转换下沉到 wire
    - 错误码全程贯通（R7）+ 相应测试

P3  zenoh 改造（4.3 / §5.6 / §5.7）
    - per-op selector + queryable；删信封与 request_id
    - 删 device_id、删 events

P4  gRPC（4.4 / §6.4）
    - feature 门控 tonic；TLS + token 拦截器
    - grpcurl/Python 客户端示例（验证"方便外部集成"这个理由）

P5  文档收敛（§8）
    - 收敛两份同名 orchestration.md；修正 ✅/🔲 标记；补代码路径图
```

**执行顺序的两个要点**：P1 必须最先（否则改两遍）；P0 不能跳过（现在 CI 门禁本身就是红的，且现有 8 个测试覆盖约 2.9k 行，重构会作废仅有的覆盖）。

---

## 8. 文档问题（建议与 §5 的架构改动一并处理）

### 8.1 两份同名文档

`crates/soul-mem/orchestration.md` 与 `docs/architecture/orchestration.md` **同名**，且前者"各 Crate 依赖与交互关系"整节是从后者**逐字复制**的——复制时就已经过时：两边都只列 5 个 crate，漏掉 `soul-mem-llm`、`soul-tune-api`，也没把自己（`soul-mem`）列进去。

`AGENTS.md` §6 明确要求文档有权威性分级（"规范类 / 架构类 / 一次性报告类"），一个 crate 内再放一份同名架构文档会绕开这个分级。建议：crate 内那份收敛为指向 `docs/architecture/orchestration.md` 的指针，或至少删除复制来的过时段落。

### 8.2 ✅/🔲 标记两个方向都有错

同一提交里两份文档对传输契约也互相矛盾（`orchestration.md:7` 声称使用 zenoh **query** api 与 **grpc**；`plan.md` 抬头写"不使用 gRPC"、§2.2 用 pub/sub 模拟）。

已核实的标记错误：

| 文档所述 | 实际 |
|---|---|
| 输出（MemoryNote 集合）🔲 | 已实现（`retrieve` 返回结构化结果） |
| 按 id 读写 MemoryNote 🔲 | 已实现（op=READ/WRITE） |
| liveliness/heartbeat 🔲 | 已实现（`announce` + `observe_liveliness`） |
| 使用 zenoh query api | 代码里零 query 原语 |
| 使用 grpc | proto 无 `service` 块，plan.md 明确否决 |
| "query 与信息增量同时存在时先压入再检索" ✅ | **不存在这样的入口**：`retrieve` 不含 deltas、`ingest` 不检索，且两者分属 request 与 pub/sub 两条通道 |

最后一条尤其需要修正：它被标为 ✅，但 proto 里没有任何承载两者的消息，因此"先压入 → 等摘要完成 → 再检索"在接口上不可表达（客户端只能走 request 通道连发两次并逐个 await，文档从未这样说明）。**要么补一个组合入口，要么把 ✅ 改成 🔲 并写明"由调用方分两步发起"。**

### 8.3 缺失"代码路径"文档

现有文档是按**文件**或**概念**组织的：`plan.md` §3.1 是逐文件职责表（回答"每个文件干什么"，很完整），`orchestration.md` 的时序图是概念层的（Client / Svc / SlidingWindow / LLM / DefaultPipeline / Cluster / WorkingMemory / DB / BG）。**没有任何一份文档对应真实代码路径**：

```
zenoh 订阅 → request.rs:dispatch → service.retrieve → convert::query_from_proto
  → DefaultPipeline → convert::note_to_proto → ReplyEnvelope
```

读者必须自己从零构建"文档流程 ↔ 代码流程"的映射，再叠加 8.2 的标记错误，映射就建不起来了。**这是"难以跟踪实际数据路径"最主要的成因。** 建议在 `lib.rs` 模块文档或 crate 内 README 补一张"请求从进入到返回经过哪些函数"的图，成本很低、收益很高。

---

## 9. 流程与门禁（决定"能否合并"）

`CONTRIBUTING.md`（**此文件在 `dev` 上存在，作者本应遵守**）要求：

| 要求 | 现状 |
|---|---|
| 保持小步提交，一个提交只做一件事 | ❌ 6000 行 / 38 文件 / 一个新 crate 压在**单个提交**里 |
| 新增业务逻辑必须配套单元测试 | ❌ 新 crate 约 2.9k 行生产代码只有 8 个测试函数；`wire/convert.rs` 613 行零测试 |
| `cargo mutants` 杀灭率 ≥90% | ⚠️ **未实跑**；按测试数量推断会失败。注意 CI 的 `mutants-pr` 触发路径是 `crates/**`、`Cargo.toml`、`Cargo.lock`，本次三者全中 |
| `cargo clippy --workspace --all-targets -- -D warnings` | ❌ 当前失败（3 条错误在 `soul-mem-core`，与本提交无关，见 §2.1） |

**同时要肯定的是**：`AGENTS.md`、`.editorconfig`、`scripts/check_layout.py`（模块布局门禁）**在 `dev` 上都不存在**，作者是在没有约束的情况下自己遵守了布局约定与格式约定的。真正需要改进的是**提交粒度与测试覆盖**这两条，而不是"不守规矩"。

---

## 附录 A：关键证据索引

| 主题 | 位置 |
|---|---|
| 领域门面收发 protobuf | `src/service.rs:6`、`src/service/*.rs` |
| embedding 不可注入 | `src/service.rs:244-251`（`build_model`）、`src/config.rs:10-16` |
| `Store` 是闭合枚举 | `src/store.rs:44-58` |
| op 字符串常量 | `src/wire.rs:17-25`、`src/zenoh/request.rs:95` |
| 手搓请求-应答 | `src/zenoh/request.rs:40-59`、`src/zenoh/client.rs:52-101` |
| 每次 RPC 新建订阅者 | `src/zenoh/client.rs:53-58` |
| 应答 id 手工过滤 | `src/zenoh/client.rs:83-85` |
| 错误码被压成 Internal | `src/zenoh/client.rs:86-88` |
| `write_note` upsert | `src/service/note_ops.rs:41-53` + `soul-mem-runtime/src/working_memory.rs:79-85` |
| 固定 tmp 名 | `src/store/file_store.rs:23-27` |
| 退出顺序 | `src/server.rs:62-80`、`src/zenoh.rs:62-65` |
| 落盘失败仍 Ok | `src/server.rs:64-66`、`src/main.rs:19-22` |
| 隐形的锁持有期 | `src/service/retrieve.rs:39/77/82/99/110`、`src/service/ingest.rs:34/42` |
| 检索不排序 | `src/service/retrieve.rs:42/91` |
| 伪造时间戳 | `src/wire/convert.rs:44-50`、`235-237` |
| 6 处造默认值 | `src/wire/convert.rs:396/401/436/508-511/600/609` |
| LinkId 往返丢失 | `proto/soul_mem.proto:244-256` + `soul-mem-core/src/memory_links.rs:115` |
| 死代码 events | `src/zenoh/publish.rs:15`、`src/zenoh/keys.rs:44`、`proto:327` |
| 不可达的 pub fn | `src/zenoh.rs:14-15` + `request.rs:16` / `pubsub.rs:11` |
| clippy 门禁错误 | `soul-mem-core/src/memory_links/situation_mem.rs:29`、`memory_note/situation_mem.rs:165/194` |
| 损坏的注释文件 | `soul-mem-runtime/src/cluster/memory_cluster.rs`（`dev` 既有） |

## 附录 B：复现方式

```powershell
# 建立审阅工作区（不影响主工作区）
cd D:\Soul-Plan\SoulMem
git fetch origin
wt switch feature/orchestration        # → D:\Soul-Plan\.worktrees\SoulMem\feature-orchestration

# 复现本次验证
cargo test -p soul-mem --locked
cargo clippy -p soul-mem --all-targets --no-deps -- -D warnings
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets -- -D warnings   # 预期失败（soul-mem-core）
```

> 提示：`zenoh` 与 `protoc-bin-vendored` 不在本地 cargo 缓存中，首次构建需联网获取。

---

*本报告基于静态通读与实测构建。所有结论均附 `file:line` 证据；未能验证的部分已在 §2.2 明确声明。*
