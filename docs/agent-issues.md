# agent 自动提交 issue

审阅 agent 把发现的问题提交到本仓库 issue 的通道规范。实现在
[`.github/workflows/file-agent-issue.yml`](../.github/workflows/file-agent-issue.yml)，
与 [`dev-convention.md`](dev-convention.md) §3 的推进流程衔接。

## 为什么 issue 的作者不是维护者

agent 用的是维护者的凭据（`gh` 的 keyring 凭据或 `GH_TOKEN`），让它直接跑 `gh issue create`，
署名的必然是维护者。而 agent 的结论尚未核实——由维护者署名发出去，读者会默认它已经被维护者
确认过，方向正好反了：**越是不确定的内容，越不该借用维护者的信任度。**

所以 agent 一律走 `File agent issue` workflow：issue 的作者是 `github-actions[bot]`，平台自带
的机器人徽章直接告诉读者这是自动化产物，不代表维护者判断。

## 调用

workflow 有两条建单路径，二选一。

### 短报告：内联正文

```bash
gh workflow run file-agent-issue.yml \
  -f title="retrieve: 边失效后 embcache 未同步" \
  -f body="$(cat report.md)"
```

`workflow_dispatch` 的 inputs 总量上限 65,535 字符（[官方文档](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#providing-inputs)），
折成中文约两万字，一般够用。

### 长报告：先推到分支，再按路径读

```bash
git switch -c agent-reports/2026-05-14-embcache
mkdir -p .agent-reports
cp report.md .agent-reports/2026-05-14-embcache.md
git add .agent-reports
git commit -m "docs(agent-reports): add embcache review"
git push -u origin agent-reports/2026-05-14-embcache

gh workflow run file-agent-issue.yml \
  -f title="retrieve: 边失效后 embcache 未同步" \
  -f report_ref=agent-reports/2026-05-14-embcache \
  -f report_path=.agent-reports/2026-05-14-embcache.md
```

报告会被完整复制进 issue 正文，所以归档分支删掉也不影响 issue 的可读性；留着则多一份未删改的
原始报告。这种归档分支不进 PR，`agent-reports/` 前缀也不在
[`dev-convention.md`](dev-convention.md) §1 的主题分支白名单里——它是引用，不是待合并的分支。

> 归档分支的提交作者是推送者，也就是维护者。介意的话 commit 时加
> `--author="SoulMem Review Agent <noreply@soulmem.invalid>"` 覆盖署名。注意那只是一个未经
> 验证的作者字符串，不构成身份；真正的身份是 issue 上的 `github-actions[bot]`。

## 行为

| 项 | 行为 |
|---|---|
| issue 作者 | `github-actions[bot]` |
| 正文开头 | 固定插入一行声明：结论未经维护者核实 |
| 正文结尾 | 固定插入提交该 issue 的 run 链接，供溯源 |
| 默认标签 | `agent-report` |
| 已存在同标题的开放 issue | 拒绝建单，并以该 issue 号失败，不静默跳过 |
| 结果输出 | 日志与 job summary 里都有 `ISSUE_URL=...` |

`gh workflow run` 是异步的，拿 issue 号要等 run 结束：

```bash
run_id=$(gh run list --workflow file-agent-issue.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$run_id"
gh run view "$run_id" --log | grep ISSUE_URL
```

## 标签语义

- `agent-report` —— agent 自动提交，**未经核实**。看到它就不要把 issue 描述当既定事实。
- 核实之后由维护者补 `bug` / `enhancement` / `documentation`，或直接关闭。

标签必须在建单前已存在：workflow 不隐式创建标签，免得一次建单顺手改了仓库的其他状态。任何
不存在的标签都会让 `gh issue create` 直接失败。

## 已知约束

1. **workflow 必须在默认分支上**：`workflow_dispatch` 只在 workflow 文件已存在于默认分支
   （`main`）后可触发。本仓库的 `main` 只接受来自 `dev` 的 PR，所以本 workflow 要随
   `dev → main` 的同步 PR 合上去之后才能真正建单；在那之前 `gh workflow run` 会报找不到
   workflow。
2. **不触发后续 workflow**：`GITHUB_TOKEN` 产生的事件不会再启动 workflow，所以针对 agent
   issue 的自动处理不能放在另一个 `on: issues` 的 workflow 里，必须放进本 workflow 的 job。
3. **触发者仍有记录**：`workflow_dispatch` 需要仓库写权限。换成 bot 署名的只是 issue 的
   作者，run 里仍然记着是谁触发的。
