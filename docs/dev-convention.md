# dev 约定

SoulMem 协作规范的唯一权威出处。[`CONTRIBUTING.md`](../CONTRIBUTING.md) 为简版入口；冲突以本文件为准。

**适用范围**：新提交与新 PR。已合并的历史不追溯、不重写。

---

## 1. 分支模型

| 分支 | 规则 |
|---|---|
| `main` | 发布线。只接受来自 `dev` 的 release PR；禁止直接提交 |
| `dev` | 唯一的集成分支。所有普通 PR 的目标 |
| 主题分支 | 自 `dev` 切出。前缀 `feat/` `fix/` `chore/` `docs/` `refactor/` `perf/` |

- 主题分支寿命 ≤ 1 周，且提交数 ≤ 20；超出则拆分。
- 分支保护（`main` 与 `dev` 相同，`enforce_admins = true`）：§4 的 5 个必需检查；`main` 另需 1 个 approve。
- 工程规范类改动单独开 PR：CI 配置、门禁脚本、`.editorconfig`、`.gitattributes`、`AGENTS.md`、本文件、文档索引。

---

## 2. 提交信息

Conventional Commits。格式参考 [Cheatsheet](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)。

```
<type>(<scope>): <description>

<可选 body>

<可选 footer>
```

| 部分 | 规则 |
|---|---|
| `type` | 小写。白名单：`feat` `fix` `refactor` `perf` `style` `test` `docs` `build` `ci` `chore` `revert` |
| `scope` | 可选。填受影响模块或 crate；单个值；不用 issue 号 |
| `description` | 必填。祈使/陈述式；不加句号；英文小写开头 |
| `body` | 可选。动机与行为差异；子弹列表 3~5 条；不写段落 |
| `footer` | 可选。`Closes #123`；破坏性变更必须写 `BREAKING CHANGE: <影响>` |
| 破坏性变更 | `!` 置于 `:` 之前，如 `feat(api)!: ...` |

- 一个提交只做一件事。
- 门禁仅校验 `type` / `scope` / `description`（§7）；语气、篇幅与正文结构由 review 把关。

```bash
python3 scripts/check_commit_messages.py origin/dev..HEAD
```

---

## 3. 推进流程

1. 开 issue（推荐）：bug 用 `bug_report`，需求用 `feature_request`。blank issue 已禁用。
2. 自 `dev` 切主题分支：`git switch -c feat/xxx origin/dev`。
3. 本地通过 §4 的全部门禁。
4. 开 PR 到 `dev`；按 `.github/PULL_REQUEST_TEMPLATE.md` 填写，关联 issue 写 `Closes #N`。
5. 5 个必需检查全绿后合并。
6. 合并方式为 merge commit；禁止 squash。
7. 删除主题分支。

---

## 4. 门禁

| 检查 | 内容 |
|---|---|
| `Quality (fmt + clippy)` | 编码、模块布局、提交信息 → `cargo fmt --check` → `clippy -D warnings` |
| `Test (ubuntu / windows / macos)` | `cargo build --all-targets` + `cargo test --workspace` |
| `Security (cargo-deny)` | RustSec advisory + 许可证合规 |

纯文本门禁不需要工具链，提交前本地执行：

```bash
python3 scripts/check_encoding.py
python3 scripts/check_layout.py
python3 scripts/check_commit_messages.py <范围>
```

`Mutants (PR diff)` 为非必需检查：按改动范围运行 cargo-mutants，杀灭率 <90% 失败。

`Security (cargo-deny)` 失败而本次改动未触及依赖时，属上游 advisory 漂移；处理步骤见 `Deny drift` workflow 所开 issue。不得向 `deny.toml` 的 `ignore` 添加条目消红。

---

## 5. 发布流程

触发（任一）：

1. 定期：每两周一次，与全量 mutants 同周节奏（周一，ISO 奇数周）。
2. 版本驱动：该版本功能开发完成时。

步骤：

1. 开 release PR（`dev` → `main`），标题 `release: vX.Y.Z`，正文写变更摘要。
2. 合并条件：1 个 approve + 5 个必需检查全绿。`main` 不接受其他 PR。
3. 合并后打 annotated tag：`git tag -a vX.Y.Z -m "..."`，并写 release notes。
4. 立即把 `main` 合回 `dev`：`--no-ff` 合并提交，经 PR 完成。
5. 兜底：`main` 落后 `dev` 超过 30 个提交时，`Branch drift` workflow 开 issue。

版本号：`0.x` 阶段破坏性变更升 minor，其余升 patch；`1.0.0` 起按 SemVer。

---

## 6. 文档规范

### 6.1 落点

| 内容 | 落点 | 权威性 |
|---|---|---|
| 规范 | `docs/` 根 | 权威 |
| 架构说明 | `docs/architecture/` | 描述当前实现 |
| 一次性报告 | `docs/` 或 `docs/tests/` | 仅对当时的 commit 与数据集有效 |
| crate 用法 | `crates/<crate>/docs/` | 该 crate 的权威 |
| 成书章节 | `doc/book` 分支的 `book/src/**` | 见 6.5 |

- 同一事实只允许一个权威出处；别处用相对链接引用，不复制。
- 一次性报告必须注明测量时间点（或 commit）、数据集、环境，并声明其非当前状态。

### 6.2 命名

- 文件名可用中文（既有先例）；不改名既有文档。
- 迭代报告用 `-v2` / `-v3` 后缀，并在新版本注明取代关系。
- 目录索引统一为 `README.md`。
- 非 ASCII 文件名：解析 `git diff` 的工具须设 `-c core.quotepath=false`（见 CI 的 mutants 步骤）。

### 6.3 索引维护

- 新增、重命名、删除 `docs/` 下的文档，必须在同一提交更新 [`docs/README.md`](README.md)。
- 规范被取代时，在同一提交标注取代关系；不并列两套相互矛盾的规范。
- 本条无门禁，由 review 把关。

### 6.4 编码与格式

- 标识符 ASCII；注释与文档用中文。
- 文件编码 UTF-8：`.editorconfig` 约束，`scripts/check_encoding.py` 强制。
- `*.md` 行尾 LF（`.gitattributes`）；`*.rs` / `*.toml` 不强制，统一须单独提交。
- markdown：每文件一个 H1 且与索引名一致；标题不跳级；相对链接；代码块标语言；表格仅用于结构化对照。
- `*.md` 的行尾双空格为硬换行，故 `.editorconfig` 对 `*.md` 关闭 `trim_trailing_whitespace`。

### 6.5 禁止改动的区域

- `doc/book` 分支的 `book/src/**`：由维护者手工撰写，禁止批量改写、重排或优化。其写作规范见该分支的 `book/src/contributing/writing-style.md`。
- 忽略规则不得覆盖源码：

  ```gitignore
  /book/book/     # 正确：仅忽略 HTML 输出目录
  /book/          # 错误：连带忽略 book/book.toml 与 book/src/**
  ```

  `__pycache__/` 与 `*.pyc` 必须保持忽略（`scripts/` 下为 Python 门禁脚本）。
- 禁止向 `.git/info/exclude` 添加排除项（仅本机生效、不随仓库分发）；并行处理差异较大的分支用 `git worktree`。

---

## 7. 门禁覆盖现状

| 规则 | 状态 | 实现 |
|---|---|---|
| UTF-8 编码（6.4） | 强制 | `scripts/check_encoding.py`（CI `Quality`） |
| 模块布局 | 强制 | `scripts/check_layout.py`（CI `Quality`） |
| 提交信息格式（§2） | 强制 | `scripts/check_commit_messages.py`（CI `Quality`；仅目标为 `dev` 的 PR） |
| 文档索引（6.3） | review | — |
| mutants 杀灭率 ≥90% | 非必需检查 | `scripts/mutants_gate.py`（CI `Mutants`） |
| 标题层级、死链、术语统一 | 待接入 | 计划 mdbook-linkcheck / markdownlint |

变更门禁时同步更新本表。

---

## 8. PR 自检清单

- [ ] 分支自 `dev` 切出，前缀正确，寿命与提交数符合 §1
- [ ] 提交信息符合 §2
- [ ] `check_encoding.py`、`check_layout.py`、`check_commit_messages.py` 通过
- [ ] `cargo fmt --all --check`、`cargo clippy --workspace --all-targets -- -D warnings` 通过
- [ ] 关联 issue 已写 `Closes #N`
- [ ] 新增、重命名或删除的文档已登记 `docs/README.md`
- [ ] 未改动 `doc/book` 书稿；未向 `.git/info/exclude` 添加排除项
- [ ] 改动 `soul-mem-llm` 调用链时已同步 `docs/architecture/llm-layer.md`
