# AGENTS.md — SoulMem 协作须知

面向在本仓库工作的 AI 代理与人类贡献者。**先读这一页**，再决定改哪里。

> 本文只描述**当前状态与约定**。修复历史属于 git log，不写在这里——过时的"曾有问题"叙述会误导读者的上下文。

---

## 1. 项目是什么

SoulMem 是**角色扮演用的记忆系统**，本质是一个"向量检索 + 知识图谱"的复合 RAG 系统。

- **目标**：让 LLM 扮演的角色像人一样记住重要的、情感相关的、能驱动行为的事件，并建立联想。**不**追求精确记忆细节与事实性知识。
- **目标环境**：个人用户的家用电脑。不是企业级方案——不要为高并发/多租户做设计。
- **非商业化**：本项目是个人自用 + 公开源码的项目，不做商业化运营。这条会影响若干判断（§6 里 fixtures 的许可结论就依赖它），不要顺手改掉。
- **技术栈**：Rust（edition 2024）+ SurrealDB（向量 + 图 + 时间序列一体，嵌入式运行）+ async-openai。GUI 是 Flutter（`soul-tune-ui/`），经 FRB 桥接。
- **核心原则**：**能不用 LLM 就不用 LLM**。LLM 调用是秒级延迟且要花钱，只在复杂整合/抽取时使用，并保证提示词精简。

---

## 2. 仓库地图

| 路径 | 角色 |
|---|---|
| `crates/soul-mem-core` | 纯数据模型（`MemoryNote` / `MemoryLink`）。**无任何内部依赖**，是其余 crate 的基础 |
| `crates/soul-mem-query` | 文本→向量嵌入（BGE / Qwen3）、Query 类型、相似度与评分计算 |
| `crates/soul-mem-llm` | **叶子 crate，不依赖任何内部 crate**。全仓库唯一的 LLM 调用入口：语义契约、OpenAI 兼容传输、重试/超时、流式、错误分类、JSONL trace |
| `crates/soul-mem-runtime` | 工作记忆（滑动窗口、记忆簇、活跃记录）、SurrealDB 仓储层 |
| `crates/soul-mem-algo` | 检索策略（`DefaultPipeline`）、遗忘、巩固；**算法入口需要 LLM 时统一接收 `&LlmEngine`** |
| `crates/soul-tune` | 测试与基准框架（headless CLI + lib）。**不是运行时组件**。lib 目标供 `soul-tune-api` 复用 |
| `crates/soul-tune-api` | FRB 桥接层（JSON-over-FRB）。不含 UI 逻辑 |
| `soul-tune-ui/` | Flutter GUI（纯渲染） |
| `benches/` | criterion 基准（cosine / cosine_simd / ppr）+ `e2e_profile` |
| `fixtures/` | 测试数据（角色图与对话集） |
| `docs/` | 架构与规范文档（见 §6） |
| `scripts/check_encoding.py` | 编码门禁（CI 中执行） |
| `scripts/check_layout.py` | 模块布局门禁（CI 中执行） |

依赖方向：`core → query/runtime/llm → algo → soul-tune → soul-tune-api`。
`runtime → algo` 仅存在于 **dev-dependencies**，生产依赖图中没有反向边。

---

## 3. 从哪里开始读

| 你想知道 | 去看 |
|---|---|
| 整体架构与数据流 | `docs/architecture/orchestration.md`（带 ✅/🔲 实现标注） |
| **LLM 调用链的任何问题** | `crates/soul-mem-llm/src/lib.rs` 的模块文档，再读 `docs/architecture/llm-layer.md` |
| 检索管线三步 | `docs/architecture/orchestration.md` + `crates/soul-mem-algo/src/algo/retrieve/complex/default_pipeline.rs` |
| 遗忘算法与参数含义 | `crates/soul-mem-algo/src/algo/forget/decay_calculator.rs` 的模块注释（**本仓库常数文档化的模板**） |
| 遗忘测试套件的文件划分 | `crates/soul-tune/src/engine/forget.rs` 的模块文档（含文件布局表） |
| 测试数据格式 | `docs/测试数据规范.md`（权威），`crates/soul-tune/docs/user-guide.md`（CLI 用法） |
| 怎么跑一次实验 | `crates/soul-tune/docs/user-guide.md` |

---

## 4. 必须保持的不变量

改代码前先确认没有破坏这些。前四条有测试锁定：

1. **顶层语义类型里不出现 wire 字段**。`Task` / `Completion` / `StreamEvent` 只描述"我们要什么/得到了什么"；provider 字段名只允许出现在 `oai_comp/wire.rs`。判断规则：描述"provider 管它叫什么"→ 归 `oai_comp/`。
2. **一次语义调用恰好一次 `on_end` 上报**，包括被提前丢弃的流。
3. **已产出内容后的流中断绝不重放**——否则摘要/记忆写入场景会出现重复文本（静默损坏）。
4. **请求体里 `max_tokens` 与 `max_completion_tokens` 不同时出现**。
5. **日志与观测事件里不放提示词正文、不放密钥**。`OaiCompatConfig` 的 `Debug` 只打印 `has_api_key`。
6. **不臆造数据**：`Usage` 拿不到就留 `None` 而不估算；`reasoning` 不伪造；`stop` 不确定就是 `Unknown`，不假装正常结束。
7. **生产代码不用 `unwrap()`/`expect()`/`panic!()`**，除非紧邻一行 `//SAFEUNWRAP: <该不变量为什么成立>` 说明。全仓库生产路径的 unwrap 都用这个标记逐一说明——**新增的也必须带，不要放宽这个标准**。

---

## 5. 构建、测试与门禁

```bash
cargo build --workspace --all-targets
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
python3 scripts/check_encoding.py         # 编码门禁
python3 scripts/check_layout.py           # 模块布局门禁
cargo mutants --workspace                 # 杀灭率 ≥90%，门禁见 scripts/mutants_gate.py
```

**离线/沙箱注意**：

- 部分测试依赖**下载嵌入模型**（`BgeSmallZh::default_cpu()`），CI 靠 `actions/cache` 缓存 `~/.cache/huggingface` 才能过。冷启动离线环境下这类测试报 `BGE model init failed: ... (os error 5)`——那是**环境问题，不是逻辑回归**。
- `.cargo/mutants.toml` 用 `--skip` 排除了两个模型测试。
- 若构建卡住（下载依赖/模型被沙箱阻断），**停下来向用户申请提权**，不要试图绕过。

**注意 CI 覆盖不到的地方**：§8 列出的问题当前 CI 全部绿灯。

---

## 6. 文档现状（动手前必读）

仓库里存在**三个文档来源**，且并不同步：

| 来源 | 状态 |
|---|---|
| `docs/`（当前分支） | 架构与规范。**先看 [`docs/README.md`](docs/README.md) 索引**——它按权威性把"规范类 / 架构类 / 一次性报告类"分开 |
| `origin/doc/book` 分支上的 mdBook（`book/src/**`） | **最完整**的文档体系，含 `concepts/` `algorithm/` `crates/` `testing/` 与写作规范 |
| `crates/soul-tune/docs/` | CLI 用户指南 |

> [!important]
> **`book/` 的书稿由维护者手工撰写，正在逐章推进。**
> **不要批量改写、重排或"优化"书正文**——那不是可以自动化的区域。需要文档改动时改 `docs/` 或 crate 内注释。
> 书源码保留在 `doc/book` 分支，当前分支**不含**书源码（磁盘上的 `book/book/` 只是 HTML 构建产物）。

**取用文档时的三条规则**：

- **测试数据集**以 [`docs/测试数据规范.md`](docs/测试数据规范.md) 为准。
  [`docs/architecture/测试数据格式.md`](docs/architecture/测试数据格式.md) 描述的是**上游生产者**（`soul_scraper`）的格式，命名与字段都不同。
- **当前架构**看 [`docs/architecture/orchestration.md`](docs/architecture/orchestration.md)。
  [`docs/architecture/beta_ver.md`](docs/architecture/beta_ver.md) 是**设计历史**（含未决问题与 `- [ ]` 待办），部分已被实现取代，不要当作现状。
- **`fixtures/example_data/` 里的数据是萌娘百科文本的演绎作品**，按其上游许可（CC BY-NC-SA 3.0 CN）提供。本项目**非商业化**，因此上游的 NC 条款不构成冲突；但演绎作品须继续以同协议提供，且转载须给出原页面 URL 署名。新增或改写 fixture 之前，先读 [`docs/测试数据规范.md`](docs/测试数据规范.md) 第六节。

---

## 7. 代码约定

- **模块目录一律用 `xxx.rs` + `xxx/`，禁止 `mod.rs`。** `xxx.rs` 里的 `mod foo;` 会解析到 `xxx/foo.rs`，因此拆分子模块无需 `mod.rs`；已有 `xxx/mod.rs` 时重命名为同级 `xxx.rs` 即可，不必改代码。此条由 `scripts/check_layout.py` 强制（`patches/` 下的 vendored 代码豁免）。
- **标识符一律英文/ASCII；注释与文档用中文。** 这个分工是刻意的，不要"统一"成一种语言。
- **模块级 `//!` 文档**是期望做法。`crates/soul-mem-llm/src/lib.rs` 是最佳范例（含"从哪里开始读"索引与分层原则）；新增模块请照此写。
- **新增/修改的公开项要有文档注释**；算法常数要写清单位、含义与取值范围（照 `decay_calculator.rs` 的风格）。
- **TODO 必须可执行**：写清"要做什么、为什么现在不做"。写下之前先 `grep` 确认它真没做。
- **不要让函数返回宽元组**：调用方只能靠位置记忆，而且 cargo-mutants 对"整体替换返回值"会按元素做组合爆炸，让杀灭率指标被单个函数淹掉。用命名结构体——字段名顺带就是文档。
- **`.cargo/mutants.toml` 的每个排除项都要写原因**。另注意两个坑：`exclude_re` **不作用于结构体字面量字段变异体**（cargo-mutants 27.1.0 实测）；"行为可观测但没有测试覆盖"应当补测试，而不是加排除项。
- **编码由 `.editorconfig` 约束为 UTF-8**：中文注释被 GBK 保存会变成乱码且部分字节永久丢失（不可逆）。提交前跑 `python3 scripts/check_encoding.py`。
- **行尾不一致是已知现状**：`core.autocrlf=true` 而索引里部分 blob 本身含 CRLF，工作区因此 CRLF/LF 混杂。`.gitattributes` 目前只统一 `*.md`。**要统一 `.rs` 请单独提一个只改行尾的提交**，不要混在功能改动里。

---

## 8. 已知陷阱：不报错，但会静默出错

改动相关代码时请一并核实，或至少不要加剧：

| 陷阱 | 位置 |
|---|---|
| `run_batch` 的 `params_json` 被导出给 Dart 却完全忽略（`run_suite` / `run_compare` 认这个参数，batch 不认） | `crates/soul-tune-api/src/api.rs` |
| 数据集显示名恒为字符串 `"question.json"`（用 `file_name()`，而所有数据集都叫这个名字；引擎侧用的是父目录名） | `crates/soul-tune-api/src/api.rs` |
| `time_span` 被查询 API 接受并存储，然后被评分逻辑静默丢弃 | `crates/soul-mem-query/src/query/retrieve.rs` |
| SurrealDB schema 全为 `IF NOT EXISTS` 且无迁移；所有 DB 测试用内存库 → 磁盘上的旧库永远保持旧 schema | `crates/soul-mem-runtime/src/storage/surreal/schema.surql` |
| `.embcache` 只按版本号失效、**不随图边变化重建** → 改图后跑实验会读到旧数据 | `crates/soul-tune/src/engine/loader.rs` |
| 场景分派用魔法负数 `elapsed_hours == -10/-11/-12/-13/-14`；typo 会静默跑错实验 | `crates/soul-tune/src/engine/forget/pipeline.rs` |
| `inspect` 吞掉所有 IO/解析错误并返回成功（退出码 0） | `crates/soul-tune/src/engine/inspect.rs` |
| CLI 无法传 `--param`，但文档让你用 `db_path=<目录>` | `crates/soul-tune/src/main.rs` |
| `#![allow(dead_code)]` 在两个 crate 级放行，掩盖真实死代码；`dead_code` 警告因此局部失效 | `crates/soul-tune/src/main.rs`、`crates/soul-tune-api/src/lib.rs` |
| `tests_playtest_mock.rs` 定义了 `MockLlm` 却从未传给 `PlayTestRunner`，并未真正覆盖 playtest | `crates/soul-tune/src/tests_playtest_mock.rs` |

**本地配置不再排除任何目录**：`.git/info/exclude`（**不随仓库分发**）曾排除 `fixtures/` 与 `soul-tune-ui/`，目的是在同一个工作副本里切到 `doc/book` 分支时不把这两处显示成成片未追踪。代价是本机新增的这两个目录下的文件在 `git status` 里**完全不可见**，会被静默漏提交——`soul-tune-ui/` 在 `dev` 上是已跟踪目录，风险尤其大。现在改为给 `doc/book` 一个独立工作区（`git worktree add ../SoulMem-book doc/book`），排除项已清空。实测这两个目录的构建产物已被根 `.gitignore` 与 `soul-tune-ui/.gitignore` 完整覆盖，移除排除项后未追踪文件数为 0。

---

## 9. 改动后的检查清单

```bash
cargo fmt --all -- --check
cargo clippy --offline --workspace --all-targets -- -D warnings
cargo check --offline --workspace --all-targets
python3 scripts/check_encoding.py
python3 scripts/check_layout.py
cargo test --offline -p <改动的 crate>      # 注意 §5 的模型依赖
```

- [ ] 没有破坏 §4 的任一条不变量
- [ ] 生产路径没有新增无 `//SAFEUNWRAP:` 说明的 `unwrap`/`expect`/`panic!`
- [ ] 新增/修改的公开项有文档；新增模块有 `//!` 头
- [ ] 模块目录是 `xxx.rs` + `xxx/`，没有引入 `mod.rs`
- [ ] 若动了算法常数，说明了单位/含义/取值范围
- [ ] 若给 `.cargo/mutants.toml` 加了排除项，写了原因
- [ ] 若动了 `soul-mem-llm` 的调用链，`docs/architecture/llm-layer.md` 同步更新（代码里有 12 处引用它）
