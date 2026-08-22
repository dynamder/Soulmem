# 贡献指南

感谢你愿意为 SoulMem 贡献代码、文档或想法！

## 开发流程

1. Fork 本仓库并创建自己的功能分支。
2. 本地完成开发后，运行以下检查并确保全部通过：
   - `cargo build --workspace --all-targets`
   - `cargo test --workspace`
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo mutants --workspace`（杀灭率 ≥90%）
3. 推送到你的分支，开启 Pull Request 到 `dev` 分支。

## CI 说明

- push / PR 会在 Windows、Ubuntu、macOS 三平台执行编译与测试。
- PR 还会按本次改动范围运行 cargo-mutants，杀灭率低于 90% 会直接失败。
- 每周（每隔一周）自动执行一次全量 mutants，并上传报告 artifact。
- 依赖安全由 Dependabot（自动更新 PR）和 cargo-deny（安全公告 + 许可证）把关。

## 测试要求

- 新增业务逻辑必须配套单元测试；算法类改动请确保现有 mutants 排除项之外没有新的存活变异。
- 需要下载大模型（如 Qwen3-0.6B）或调用真实 LLM API 的测试，请用 `#[ignore = "..."]` 标注，避免阻塞 CI。
- 如果确实需要新增 mutants 排除项，请在 `.cargo/mutants.toml` 中写明原因。

## 提交规范

- 提交信息建议遵循 `type(scope): 描述` 的格式，例如 `fix(retrieve): 修复PPR传播截断`。
- 保持小步提交，一个提交只做一件事。
