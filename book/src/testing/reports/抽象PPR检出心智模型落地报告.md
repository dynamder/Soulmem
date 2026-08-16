# 检索算法改进：抽象经 PPR 检出心智模型落地报告

> 日期：2026-08-11
> 分支：`feat/abstract-ppr-detection`（SoulMem + soul_scraper）
> 状态：两角色试点验证通过，全量生成进行中

## 1. 背景与心智模型

先前的检索测试暴露了两类问题：抽象情境节点只能被查询文本"直接命中"（相似度），
过程性记忆（proc）检出依赖具体情境直连边，行为倾向不稳定。

据此提出并落地了新的心智模型：

- **specific = 经历，abstract = 从经历提炼出的模式**。
- 纯 AI 对话场景下，抽象模式只与"对方是谁、对方说了什么"相关。
- 查询生成只产出两部分：**实体概念（Semantic）** 与 **环境氛围（Situation）**，
  相似度命中集中在 sem 与 specific 节点。
- **抽象情境由 PPR 检出**：经"具体情境 → 抽象情境"边（`SpecificToAbstract`），
  从具体情境种子游走到抽象模式——这是模式匹配的实现路径。
- **Bayes 双源提取 proc**：抽象源优先（权重 ×2），具体源兜底（未巩固泛化的模式）。
- **hint 不再需要**：氛围与事件只从对话上下文提取，不注入记忆片段。

## 2. 实现改动

### SoulMem
- `SituationMemLink` 新增 `SpecificToAbstract` 变体（soul-mem-core）。
- Bayes 源改为（相似度种子 ∪ PPR 结果）中的 abstract+specific；
  抽象源权重 ×2（抽象优先），抽象源为空退化为仅具体源；Semantic 不触发行为
  （soul-mem-algo `assoc_with_action.rs`）。
- 查询生成移除 hint 链路；runner 维护最近 6 轮对话（含助手回复）注入提示词；
  两段式查询：Semantic 实体概念 / Situation narrative+environment(+event)，
  氛围只从对话上下文提取（soul-tune `runner.rs`）。
- 评测新增"抽象检出率 / 抽象直接命中率"指标（soul-tune suite + batch 输出）。

### soul_scraper
- `SituationMemLink` 同步新增 `SpecificToAbstract`。
- 节点提取提示词：抽象字段泛化为"可复现的一类模式"（避免查询直接命中）。
- 边生成提示词：Sit→Sit 双向；每个抽象情境应有 `SpecificToAbstract` 入边
  （缺失为数据完整性警告，不阻塞）。
- 边生成流程修复：窄触发补充新建抽象节点后重跑边生成（有界 3 轮）。
- 新工具：
  - `mirror_sit_edges`：确定性镜像现有 abstract→specific 边。
  - `link_abstract_specific`：每图一次 LLM 调用，语义补齐缺失的抽象↔具体双向边。
  - `regenerate_questions`：两段式查询 question.json 重生成（复用批量管线）。

## 3. 试点验证（格蕾修 / 桑多涅）

对两个性格差异很大的角色做了"全量重生成 → 链接 → playtest"试点。

### 图重生成效果

| 项 | 格蕾修（安静画家） | 桑多涅（傲娇机械师） |
|---|---|---|
| 节点 | 76（Sem 51 / Sit 17 / Proc 8） | 53（Sem 21 / Sit 24 / Proc 8） |
| 边 | 153（Sem 124 / Proc 18 / Sit 11） | 117（Sem 76 / Proc 18 / Sit 23） |
| 结构 | 合法（5 组件） | 合法（连通） |
| 抽象链接 | 7/8 双向 | 9/9 双向 |
| proc 特色 | 沉默 Speak / 借色模仿 Speak / 光剑 Skill | 傲娇口癖 Speak / 茶会 Think / 日记天气 Think / 浮游剑 Skill |

抽象字段确实泛化：例如格蕾修"接触他人后沾染对方的颜色，言行模仿对方"、
桑多涅"被亲近的人戳穿口是心非""熬夜进行机械研究"。

### playtest proc 检出

**格蕾修（3 轮）**：说话风格/思维习惯每轮注入，且随上下文切换
（借色模仿 ↔ 沉默寡言）；电影话题正确触发学习类 proc。

**桑多涅（8 轮）**：说话风格每轮为傲娇口癖；思维习惯随话题切换——
咖啡邀请→茶会（2.150）、实验室/熬夜→研究沉迷、茶会邀请→茶会（0.477）。
回复全程"哼/哈？/谁稀罕"，行为倾向真实进入上下文。

### playtest abstract 检出（PPR 证据）

playtest 日志新增逐查询 ppr 节点输出（id + stage），可区分"直接命中"与"纯 PPR 检出"：

- 格蕾修：`sit_abs_painting_location:Ppr`、`sit_abs_borrow_color_event:Ppr`、
  `sit_abstract_unspeakable_moment:Ppr`（相似度未命中、靠边游走发现）。
- 桑多涅：`sit_abs_participants_close_friends:Ppr`、`sit_abs_location_lab:Ppr`；
  第 8 轮茶会邀请 → `sit_abs_event_tea_party` + `sit_tea_party_memory` → Bayes → `proc_tea_party`。

## 4. 回归数据（suite full，旧 question.json 口径）

在"镜像边 + 定向链接"数据上（未全量重生成节点的 24 图）：

| 指标 | 基线 | 镜像后 | 定向链接后 |
|---|---|---|---|
| 通过率 | 705/722（97.6%） | 703/722（97.4%） | 703/722（97.4%） |
| 动作 Hit（proc 检出率） | 81.5% | 85.2% | 85.2% |
| Recall@3 | 0.584 | 0.727 | 0.729 |

抽象检出率按图在 33%~100% 之间。

## 5. 遗留问题

1. **question.json 重生成被 DeepSeek 大输出阻塞**：questioner 单次生成 45+ 条查询，
   流式响应卡住（85 分钟无进展）。工具 `regenerate_questions` 已就绪，待解决生成负载后执行。
2. **抽象直接命中率仍偏高**：未全量重生成节点的图里抽象字段仍较具体；
   全量重生成（节点新提示词）后直接命中率应下降，PPR 检出占比上升。
3. **81 个抽象节点无对应具体经历**（数据完整性缺口）：宽泛触发模式缺少具体实例，
   在补齐角色经历前无法被 PPR 检出。

## 6. 复现方法

```powershell
# 单图全量重生成（节点+边，新提示词）
cd D:\Soul-Plan\SoulFlasher\soul_scraper
$env:SOUL_SCRAPER_KEY = "<key>"
cargo run --bin soul_scraper -- --model deepseek-v4-flash --api-base https://api.deepseek.com `
  --extract "<图目录>\scrape.md" -o "<图目录>\graph.json"

# 补齐抽象↔具体双向边
cargo run --bin link_abstract_specific -- --model deepseek-v4-flash `
  --api-base https://api.deepseek.com --out-dir "D:\Soul-Plan\SoulMem\fixtures\example_data"

# 清空嵌入缓存（边变更后必须）
Get-ChildItem fixtures -Recurse -Filter *.embcache | Remove-Item -Force

# playtest
cargo run -p soul-tune -- playtest "<图目录>" "<对话文件>"
```

## 附录：哥伦比娅晨间对话 playtest 真实运行记录

> 图：桑多涅（全量重生成后）；对话：`fixtures/daily_dialogues/sandone_columbina_morning.json`；
> 角色：哥伦比娅，愚人众执行官；模型：Qwen3.5-4B-Q6_K；运行日志：`soul_tune_playtest_log.txt`。

### 第 1 轮

**用户**：早上好，桑多涅

**生成的查询**（4 条 Semantic）：
1. `"那个总是叫我'桑多涅'的执行官"`（priority 10）
2. `"哥伦比娅"` + "十一执行官第七席与执行官第十一席，平时用代号相称"（9）
3. `"桑多涅"` + "哥伦比娅私下对我的昵称，听起来很亲切但让我很在意"（8）
4. `"木偶"` + "愚人众官方赋予我的代号，对应我的发条构造"（7）

**检索到的记忆**（FullPipeline 合并，stage + 原文）：
- `sem_columbia` [Similarity 0.8508]：哥伦比娅
- `sem_self` [Similarity 0.6991]：小玛丽安·吉约丹
- `sem_fatui_harbingers` [Similarity 0.5270]：愚人众十一执行官
- `sem_catherine_network` [Similarity 0.4852]：凯瑟琳网络
- `sem_dottore` [Similarity 0.4041]：多托雷
- `sem_alan` [Similarity 0.3916]：阿兰·吉约丹
- `sem_rosalyne` [Similarity 0.3892]：罗莎琳·克鲁兹希卡·洛厄法特
- `sem_arlecchino` [Similarity 0.3899]：阿蕾奇诺
- `sem_tsaritsa` [Similarity 0.3774]：冰之女皇
- `sem_tea_party` [**Ppr** 0.0852]：茶会四人组（纯 PPR 联想检出）

PPR 节点另含 `sem_engineering_thought:Ppr`、`sem_soul_second_birth:Ppr`、`sem_moon_force:Ppr`。

**动作通道**：
- 说话风格：傲娇口癖（与亲近的人交流时说话带刺、口是心非，常用『哼』『哈？』开头……）（1.400）
- 思维习惯：茶会（定期举办茶会，准备点心，邀请熟悉的同僚参加）（2.150）

**输出**：
- Embedding：早啊，哥伦比娅，别这么客气，叫代号就行。
- FullPipeline：哼，一大早发这种无聊的消息，你的脑子被发条卡住了吗？

### 第 2 轮

**用户**：我怕你睡过头，来叫你去开会，不行吗？

**生成的查询**（4 条，含两段式 Situation）：
1. Semantic：`"哥伦比娅"` + "那个总是用完美无缺的声音说话的大人"（8）
2. Situation：narrative "哥伦比娅发消息说要开会，担心我睡过头"，environment atmosphere=互相调侃/tone=轻松（7）
3. Situation：narrative "我通常会在深夜研究发条或举办茶会，很少早起"，environment atmosphere=自嘲/tone=傲娇（6）
4. Semantic：`"十一位大人"` + "包括哥伦比娅在内的所有执行官成员"（4）

**检索到的记忆**（FullPipeline 合并，stage + 原文）：
- `sem_columbia` [Similarity 0.7692]：哥伦比娅
- `sit_columbia_clockwork_back` [Similarity 0.5881]：哥伦比娅因为好奇我背后的发条，问了一堆问题。我跟她说：『你那么好奇的话，我给你背后安一个试试得了。』她还真背了几天发条生活。自那之后，我就更关心她了。
- `sit_columbia_singing` [Similarity 0.5638]：哥伦比娅在至冬总部的时候，总爱在我的房间外面唱歌。我一开始觉得烦，后来习惯了。她甚至会在我睡觉时跑来唱歌，直到我把她赶走。
- `sit_qiyue_night` [Similarity 0.5243]：祈月之夜，哥伦比娅离开前转身的瞬间，我才小声说了一句她走了，结果被派蒙听到——天哪，那是我最丢人的时刻。
- `sit_abs_event_tea_party` [Similarity 0.5168]：Event「与友人举办茶会」（抽象模式）
- `sit_tea_party_memory` [Similarity 0.5081]：我定期举办的茶会上，哥伦比娅总是抢走最大的一块点心，阿蕾奇诺严肃地喝茶，罗莎琳还在时也常来。「队长」偶尔会带「公子」参加。「公鸡」和「富人」？哼，那种人在茶会上除了吵架还能干什么。
- `sem_fatui_harbingers` [Similarity 0.5216]：愚人众十一执行官
- `sit_abstract_boredom` [Similarity 0.4570]：Environment「生活单调重复，让人感到无聊和缺乏新鲜感 / 懒散、烦闷」（抽象模式）
- `sit_abs_event_late_research` [Similarity 0.4538]：Event「熬夜进行机械研究」（抽象模式）
- `sem_self` [Similarity 0.4238]：小玛丽安·吉约丹

PPR 检出的抽象：`sit_abs_participants_close_friends:Ppr`（Participant「亲近的友人，无条件包容我的口是心非」）、
`sit_abs_event_teasing:Ppr`（Event「被亲近的人戳穿口是心非」）——纯 PPR 联想发现，相似度未直接命中。

**动作通道**：
- 说话风格：傲娇口癖（0.156）
- 思维习惯：茶会（0.422）

**输出**：
- Embedding：哼，谁要你这种笨蛋叫醒啊，我自己有节奏感，再敢打扰我休息我就把你扔出去。
- FullPipeline：哼，谁稀罕你这种大嗓门叫我啊！不过既然来了，就赶紧去开会吧，别挡路。

**观察**：第 2 轮检索到了三条哥伦比娅专属记忆（背后的发条、唱歌、祈月之夜），
但小模型回复只用了"开会/睡过头"的即时语境，未把往事织进台词——属生成侧未吃透记忆，检索侧无漏检。
