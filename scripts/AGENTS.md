# AGENTS.md — scripts/

门禁与 CI 辅助脚本的协作须知。

## 导航

| 脚本 | 管什么 | 规则出处 |
|---|---|---|
| `check_encoding.py` | 源文件 UTF-8 编码 | 风格（中文注释被 GBK 保存会不可逆损坏） |
| `check_layout.py` | 禁止 `mod.rs` | 风格 |
| `check_commit_messages.py` | Conventional Commits 首行 | [`../docs/dev-convention.md`](../docs/dev-convention.md) |
| `classify_changes.py` | 纯文档改动分类（据此跳过 Test 重步骤） | — |
| `mutants_gate.py` | 杀灭率 ≥90% | [`../docs/dev-convention.md`](../docs/dev-convention.md) |

## 风格

- 门禁脚本统一退出码：0 = 干净，1 = 有发现，2 = 用法或 IO 错误。
- 输出统一 `PASS (...)` / `FAILED: ...` + Fix 指引（照 `check_layout.py` 的样式）。

## 流程

- **`scripts/` 下不得存在未接入 CI 的门禁**——新写门禁必须同一提交接进 workflow，或在脚本头标注「未接入」。
- 纯文本门禁（编码 / 布局 / 提交信息）不需要工具链，提交前本地执行。
- 门禁清单以 `.github/workflows/*.yml` + 本目录为唯一事实源；[`../docs/dev-convention.md`](../docs/dev-convention.md) 的「门禁」表是其镜像。
