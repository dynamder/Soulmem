# 贡献指南

感谢你愿意为 SoulMem 贡献代码、文档或想法！

## 分支模型

本仓库有两条长期分支。下面的规则是硬性的，它们来自一次真实的教训——`main` 曾经在无人察觉的
情况下落后 `dev` 162 个提交，两条线看起来像两个不同的项目。

| 分支 | 角色 | 规则 |
|---|---|---|
| `main` | 面向用户的发布线 | **只接受来自 `dev` 的 release PR，禁止直接提交。** 任意时刻 `main` 都必须能构建、能发布 |
| `dev` | 唯一的集成分支 | 普通 PR 的目标分支 |
| 主题分支 | 具体工作 | `feat/` `fix/` `chore/` `docs/` 前缀，从 `dev` 切出 |

主题分支的寿命上限：**≤ 1 周，且 ≤ 20 个提交**。超过任一条，说明它应该被拆成多个 PR。

**工程规范类改动必须单独开 PR。** CI 配置、门禁脚本、`.editorconfig`、`.gitattributes`、
`AGENTS.md`、文档索引这类改动不要和功能改动混在同一个分支里——一旦混装，它就只能作为一个
不可审、也不可单独 revert 的巨块落地。

## 开发流程

1. Fork 本仓库并创建自己的功能分支。
2. 本地完成开发后，运行以下检查并确保全部通过：
   - `cargo build --workspace --all-targets`
   - `cargo test --workspace`
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo mutants --workspace`（杀灭率 ≥90%）
3. 推送到你的分支，开启 Pull Request 到 `dev` 分支。

## 发布流程

1. 开 release PR：`dev` → `main`，标题用 `release: vX.Y.Z`，正文写本次变更摘要。
2. 合并后用 annotated tag 标记发布点（`git tag -a vX.Y.Z -m "..."`），并写 release notes。
3. **合并后立刻把 `main` 合回 `dev`**：`git checkout dev && git merge --no-ff origin/main`。
   这一步让下一次 release PR 变成 `--ff-only`，是防止两条线再次漂移的关键。
4. 若 `main` 落后 `dev` 超过 30 个提交，`Branch drift` workflow 会自动开 issue 提醒。

## 文档分支 `doc/book`

`doc/book` 存放 mdBook 书稿，它与 `dev` 的内容差异很大。**不要用同一个工作副本切分支去写书**：
切过去之后 `fixtures/`、`soul-tune-ui/` 会整片显示为未追踪，历史上正是这个原因让人往
`.git/info/exclude` 里塞了两个排除项——而那个文件只在本机生效、不随仓库分发，代价是本机新增的
文件在 `git status` 里完全不可见，会被静默漏提交。

用独立工作区，让两个分支各占一个目录：

```bash
git worktree add ../SoulMem-book doc/book
```

## CI 说明

- push 到 `main` / `dev`，以及所有 Pull Request，会在 Windows、Ubuntu、macOS 三平台执行编译与测试。
- PR 还会按本次改动范围运行 cargo-mutants，杀灭率低于 90% 会直接失败。
- 每周（每隔一周）自动执行一次全量 mutants，并上传报告 artifact。
- 依赖安全由 Dependabot（自动更新 PR）和 cargo-deny（安全公告 + 许可证）把关。

## 测试要求

- 新增业务逻辑必须配套单元测试；算法类改动请确保现有 mutants 排除项之外没有新的存活变异。
- 需要下载大模型（如 Qwen3-0.6B）或调用真实 LLM API 的测试，请用 `#[ignore = "..."]` 标注，避免阻塞 CI。
- 如果确实需要新增 mutants 排除项，请在 `.cargo/mutants.toml` 中写明原因。
- 测试数据（`fixtures/`）的**入库分层、数据来源与许可**见
  [`docs/测试数据规范.md`](docs/测试数据规范.md) 第六节——新增数据之前请先读那一节。

## 提交规范

- 提交信息建议遵循 `type(scope): 描述` 的格式，例如 `fix(retrieve): 修复PPR传播截断`。
- 保持小步提交，一个提交只做一件事。
