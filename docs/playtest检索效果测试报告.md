# Playtest 检索效果测试报告

> 日期：2026-08-04
> 测试对象：`soul-tune` playtest（headless CLI）与基础 retrieve
> 日志目录：`%TEMP%/soul_tune_playtest_*.txt`、`soul_tune_retrieve_log.txt`、`soul_tune_llm_output.txt`

---

## 1. 背景

此前 playtest 检索存在严重问题：**几乎无有效检索**。排查确认根因是 `RawVariant` 的
`#[serde(untagged)]` 枚举顺序缺陷——LLM 按提示词输出的 `{"Semantic": [{"concept_identifier": ...}]}`
会被误解析成**概念为空的单单元**（`SemanticSingle` 贪婪吞掉包裹对象），导致空概念 → 零相似度 →
被 0.3 阈值过滤 → 检索为空。

本次测试在以下修复后执行：
1. `RawVariant` 重构：包裹形态用结构体正确解析（`Semantic { Semantic: [...] }` / `Situation { Situation: [...] }`），
   `RawSemUnit`/`RawSitUnit` 加 `deny_unknown_fields` 防止贪婪错配。
2. 新增 **Situation 查询支持**：`RawSitUnit` 系列类型 + LLM 提示词示例，playtest 首次可生成情境查询。
3. **Priority 加权合并**：移植 suite 的 `merge_by_priority`，同节点跨查询累加 `priority × score`。
4. `DialogueFile.config` 应用到 headless playtest（与 TUI 对齐）。
5. `DialogueFile.role` 字段：headless CLI 可设置自身角色。

---

## 2. 测试环境

| 项 | 值 |
|----|----|
| 平台 | Windows，Rust debug 构建 |
| Chat LLM | Qwen3.5-4B-Q6_K.gguf（llama-server 子进程） |
| 嵌入模型 | BgeSmallZh（hf-hub 自动下载 + 镜像回退） |
| 图数据 | `fixtures/example_data/`（萌娘百科角色图） |
| CLI | `soul-tune playtest <graph_dir> <dialogue_file>` |

---

## 3. 测试矩阵（9 档）

同一批对话、仅切换 `role`，构成**宽泛角色 → NPC（靠 role 描述建立交集）→ 具体角色**三级谱系。

| 角色 | L1 宽泛 | L2 NPC+交集 | L3 具体角色 |
|------|---------|-------------|-------------|
| 博丽灵梦 | `神社的常客` | 常客 + 受托调查异变 | `雾雨魔理沙`，老朋友 + 调查异变 |
| 格蕾修 | `一起生活的同伴` | 同伴 + 启明城壁画任务 | `华（符华姐姐）`，长辈 + 壁画任务 |
| 花火 | `列车上的朋友` | 朋友 + 假面舞会筹备 | `开拓者「星」`，列车乘客 + 假面舞会 |

---

## 4. 基础 retrieve 基线

以格蕾修 `question.json`（30 用例，embedding 模式）验证：

```
共 30 用例，通过 30
全部 MRR=1.0000 / Hit=1.00
```

**结论：基础检索在 example 数据上保持满分，修复未破坏既有评测链路。**

---

## 5. 检索管线健康度

| 检查项 | 结果 |
|--------|------|
| 空查询 `[]` | 格蕾修/花火全部非空；灵梦 L1/L2 寒暄轮偶发空数组（LLM 自决，见 §7） |
| 空 concept_identifier / 空 tag | 0 处（修复生效，无回归） |
| NaN / Inf 分数 | 0 处 |
| trace=None（无检索） | 0 次（有效查询轮次） |
| 命中量 | sim 6–8 / PPR 8 / Situation 查询带 action 3 命中，稳定 |

---

## 6. Role 三级谱系效果

### 6.1 检索层级随 role 具体化而提升

以灵梦第 3 轮（"神社附近可疑身影"）为例：

| 层级 | 检索模式 | 代表命中 |
|------|----------|----------|
| L1 | Semantic 概念层 | sem_self、符卡规则、大结界、魔理沙、爱丽丝 |
| L2 | Situation 事件层 | sit_urban_legend_incident（都市传说异变）、sit_suika_stay、sit_drunken_hanami |
| L3 | Situation 事件层（分数更高） | **sit_marisa_disappear**（魔理沙被怨灵附身消失）、sit_red_mist_incident、sit_party_* |

### 6.2 L3 解锁角色专属记忆

- **灵梦 → 魔理沙**：激活 `sit_marisa_disappear`；回复直接称呼"魔理沙"——
  "喂喂，魔理沙，你该不会又看到什么吓人的妖怪了吧？"
- **格蕾修 → 华**：回复称呼"华姐姐"——"华姐姐，我画的是把星星都聚成暖光的样子"；
  命中 `sem_hua`、`sit_awaken_by_hua`（被符华唤醒）。
- **花火 → 开拓者星**：激活 `sit_pam_sms`（花火假扮帕姆给开拓者发短信）；回复称呼"开拓者"——
  "别急嘛开拓者，舞会的请柬已经像雪花一样塞进大家的行李里了"。

### 6.3 观察小结

1. **宽泛角色（L1）**：检索停留在概念层，寒暄轮常返回空查询，对话缺乏记忆锚点。
2. **NPC+交集（L2）**：role 中的场景/任务描述让查询进入事件记忆层，情境感明显增强。
3. **具体角色（L3）**：检索命中角色专属记忆，且角色关系真正参与对话（称呼、口吻、共享事件），
   是三者中体验与检索质量最优的一档。

---

## 7. 发现的问题与建议

### 7.1 缺陷：空内容节点注入上下文（未修复，建议优先处理）

**现象**：`format_nodes()` 向 LLM 上下文注入空行。
实测：格蕾修 `proc_none Action score=2.72 |（空）`；花火 `sit_bored_event ... |（空）`。

**根因**：`engine/playtest/runner.rs::load()` 构建 `NodeSummary.primary` 时：

```rust
MemoryType::Procedure(_)                    => primary = ""          // 丢弃 action.content
MemoryType::Situation(_)  // AbstractSituation => primary = ""      // 丢弃 Event.action
```

但图中这些节点实际有内容：
- `proc_none`：Procedure，动作 = "平时没有采取任何特定行动"
- `sit_bored_event` / `sit_chaos_event`：AbstractSituation，Event.action = "感到无聊" / "制造混乱"

**影响**：浪费上下文、丢失记忆信息、提示词中出现裸 `- [流程] ` / `- [情境] ` 噪声行。

**建议**：`runner.rs` 中 Procedure 分支取 `action.content`，AbstractSituation 分支取 `Event.action`
（纯数据提取改进，不涉及检索逻辑）。

### 7.2 观察：分数尺度失衡（设计副作用）

Semantic 查询 priority 7–10，合并分约 10.6；Situation 记忆约 2.4。上下文被语义概念节点主导，
情境记忆沉底。来自 priority 加权合并（与 suite 一致），非 bug，但使最终上下文偏语义化，
是否需归一化或分层截断值得产品层权衡。

### 7.3 观察：LLM 偶发返回空查询数组 `[]`

灵梦 L1/L2 寒暄轮（"神社有活动吗"）返回 `[]`，该轮无检索、纯靠 LLM 常识回复。
非管线 bug，但无兜底（如空查询时回退到系统提示词内记忆）。若希望寒暄轮也有记忆支撑，
可考虑空查询兜底策略。

---

## 8. 结论

1. **此前"playtest 几乎无检索"的问题已解决**：查询非空、概念完整、命中稳定、Situation 端到端可用。
2. **基础 retrieve 回归通过**：30/30，MRR=1.0。
3. **Role 三级谱系成立**：宽泛 → NPC → 具体角色的检索深度与互动质量逐级提升，具体角色激活专属记忆。
4. **遗留 1 个数据提取缺陷**（空内容节点，§7.1）与 2 个产品层权衡点（§7.2、§7.3）。

---

## 附录：测试数据文件

```
fixtures/daily_dialogues/
├── reimu_daily_l1_broad.json
├── reimu_daily_l2_npc.json
├── reimu_daily_l3_marisa.json
├── geluoxiu_daily_l1_broad.json
├── geluoxiu_daily_l2_npc.json
├── geluoxiu_peer_daily.json        # L3 华/符华
├── huohua_daily_l1_broad.json
├── huohua_daily_l2_npc.json
└── huohua_daily_l3_trailblazer.json
```
