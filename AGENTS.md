# AGENTS.md — SoulMem 协作须知

面向在本仓库工作的 AI 代理。这是**操作手册**，不是说明书；架构与设计叙述见 `docs/`。

## 导航

| 你要找 | 读 |
|---|---|
| 某个 crate 的边界与易错点 | 该 crate 根目录的 `AGENTS.md`（`soul-mem-llm` / `soul-mem-algo` / `soul-mem-runtime` / `soul-mem-query`） |
| 文档权威分类（规范 / 架构 / 一次性报告） | [`docs/README.md`](docs/README.md) |
| 分支模型、提交信息、门禁、文档规范 | [`docs/dev-convention.md`](docs/dev-convention.md) |
| 测试数据格式与许可 | [`docs/测试数据规范.md`](docs/测试数据规范.md) |
| 门禁脚本各自管什么 | [`scripts/AGENTS.md`](scripts/AGENTS.md) |

目录清单：

- `soul-mem-core` — 纯数据模型，无内部依赖
- `soul-mem-query` — 嵌入与评分
- `soul-mem-llm` — 唯一 LLM 调用层（叶子 crate）
- `soul-mem-runtime` — 工作记忆 + SurrealDB 仓储
- `soul-mem-algo` — 检索 / 遗忘 / 巩固
- `soul-tune` / `soul-tune-api` — 测试框架 + FRB 桥接（**即将被完全重写，不要在它上面投入**）
- `soul-tune-ui/` — Flutter GUI（纯渲染）
- `benches/` — criterion 基准与 e2e_profile
- `fixtures/` — 测试数据（许可见下方「流程」）

## 风格

- 标识符用英文 / ASCII；注释与文档用中文。
- 模块目录一律 `xxx.rs` + `xxx/`，禁止 `mod.rs`。
- 模块级 `//!` 文档是期望做法；新增公开项要有文档注释。
- 算法常数要写清单位、含义、取值范围。
- 生产代码不用 `unwrap()` / `expect()` / `panic!()`，除非紧邻一行 `//SAFEUNWRAP:` 说明原因。
- 提交信息用 Conventional Commits，规则见 [`docs/dev-convention.md`](docs/dev-convention.md) 的「提交信息」一节。

## 坑

- `soul-mem-runtime` 对 `soul-mem-algo` 的依赖**只在 dev-dependencies**——那不是循环依赖。
- `soul-mem-core` 的渲染契约注释声称「对外以 zenoh / gRPC 服务形式提供」，但该服务层只存在于 `feature/orchestration` 分支，`dev` 上没有。别去找它。
- 行尾 CRLF/LF 混杂是已知现状，`.gitattributes` 目前只统一 `*.md`；要统一 `.rs` 请单独提一个只改行尾的提交。

## 流程

- 完整门禁清单见 [`docs/dev-convention.md`](docs/dev-convention.md) 的「门禁」一节。快速版：
  - `cargo check --offline --workspace --all-targets`
  - `cargo test --offline -p <crate>`
  - `cargo fmt --all --check`；`cargo clippy --workspace --all-targets -- -D warnings`
- 部分测试会下载嵌入模型；离线环境报 `BGE model init failed` 是**环境问题，不是逻辑回归**。
- 构建 / 下载被沙箱阻断卡住时，**停下来申请提权**，不要绕过。
- `doc/book` 分支的书稿由维护者手工撰写，**禁止批量改写或重排**。
- 禁止向 `.git/info/exclude` 添加排除项；处理差异大的分支用 `git worktree`。
- 改动 `fixtures/` 前先读 [`docs/测试数据规范.md`](docs/测试数据规范.md) 的许可说明。
- 工程规范类改动（CI、门禁脚本、`.editorconfig`、`.gitattributes`、`AGENTS.md`、文档索引）单独开 PR。
