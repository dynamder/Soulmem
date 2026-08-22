# SoulMem Orchestration

## 总述

SoulMem Orchestration主要指在SoulMem各个功能单元编写完成的情况下，将他们串联起来以形成按预期工作的流程的过程。

对外部，SoulMem以服务形式提供，通过zenoh的pub/sub，query，liveliness api以及通过grpc 🔲。服务的输入是query集合和当前信息增量（均为可选字段），输出是检索到的MemoryNote集合，其余组件均用于维护SoulMem自身的状态（巩固，遗忘等)



此外，服务的输入还有一组控制信号，用于强制触发一些定时任务 🔲



流程描述如下：

### 图例

色彩按**角色/类型**区分，实线 = 已实现，虚线 = 设计规划中（尚未实现）：

| 角色类型 | 填充色 | 已实现（实线） | 规划中（虚线） |
|---------|--------|:---:|:---:|
| 输入 | 蓝色 | ✅ | 🔲 |
| 输出 | 紫色 | — | 🔲 |
| 结构实体 | 绿色 | ✅ | 🔲 |
| 算法流程 | 橙色 | ✅ | 🔲 |

```mermaid
graph TD
    classDef inputImpl fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a;
    classDef inputPlan fill:#dbeafe,stroke:#60a5fa,stroke-width:2px,stroke-dasharray:6 4,color:#1e3a8a;
    classDef outputPlan fill:#ede9fe,stroke:#a78bfa,stroke-width:2px,stroke-dasharray:6 4,color:#4c1d95;
    classDef structImpl fill:#d4edda,stroke:#16a34a,stroke-width:2px,color:#14532d;
    classDef structPlan fill:#d4edda,stroke:#86efac,stroke-width:2px,stroke-dasharray:6 4,color:#14532d;
    classDef algoImpl fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e;
    classDef algoPlan fill:#fef3c7,stroke:#fbbf24,stroke-width:2px,stroke-dasharray:6 4,color:#92400e;

    subgraph "输入"
        Input1["Query 集合"]
        Input2["当前信息增量"]
        Ctrl["控制信号（强制触发定时任务）"]
    end

    subgraph "算法流程：检索管线 DefaultPipeline"
        direction TB
        S1["① ShortOnly<br/>提取窗口信息 + 摘要"]
        S2["② Similarity<br/>query 向量与 Cluster 节点余弦相似度，阈值过滤"]
        S3["③ AssociateWithAction<br/>PPR 联想扩散 + softmax 归一化 + 贝叶斯动作推理 topK"]
        S1 --> S2 --> S3
        Res["DefaultPipelineResult<br/>{ association, action, short_history, short_mem, priority }"]
        S3 --> Res
        S1 -. 窗口和摘要 .-> Res
    end

    subgraph "结构实体：工作记忆"
        SW["SlidingWindow 滑动窗口"]
        Sum["Summary 摘要"]
        Cluster["MemoryCluster 记忆簇"]
        Record["Record 活跃记录"]
        WM["WorkingState Idle/Working 状态机"]
    end

    subgraph "结构实体：外部存储"
        DB["SurrealDb 数据库"]
    end

    subgraph "算法流程：定时任务及被动流程"
        Cons["巩固算法 Consolidation"]
        Persist["持久化"]
        Forget["遗忘机制 ✅<br/>Ebbinghaus 衰减 + 遮罩 + LLM 修订"]
        Mask["遗忘遮罩 ✅<br/>jieba 分词确定性遮罩"]
    end

    subgraph "输出"
        Out["检索输出（MemoryNote 集合）"]
    end

    Input2 --> SW
    SW --> Sum
    Input1 --> S1
    Input1 --> S2
    SW --> S1
    Cluster --> S2
    Res --> Out
    Res --> Record

    WM -. 状态迁移 .-> SW
    WM -. 状态迁移 .-> Cluster

    Record -. 巩固时读取活跃节点 .-> Cons
    Sum -. Idle 且摘要非空 .-> Cons
    Cons -. 生成新 MemoryNote + 拓扑链接 .-> Cluster
    Cluster -. 定时 / 优雅退出 .-> Persist
    Persist --> DB
    Cons -. 新节点创建时 .-> Mask
    Mask -. 定时衰减权重 .-> DB
    Out -. 命中被遮盖内容 .-> Forget

    Ctrl -. 强制触发 .-> Cons
    Ctrl -. 强制触发 .-> Persist
    Ctrl -. 强制触发 .-> Forget

    class Input1,Input2 inputImpl;
    class Ctrl inputPlan;
    class SW,Sum,Cluster,Record,WM structImpl;
    class DB structPlan;
    class S1,S2,S3,Res algoImpl;
    class Cons,Persist,Forget,Mask algoPlan;
    class Out outputPlan;
```

## 各 Crate 依赖与交互关系

代码按以下 5 个 crate 分层组织，箭头表示依赖方向（被依赖方 → 依赖方）。

```mermaid
graph TD
    classDef core fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a;
    classDef layer fill:#f1f5f9,stroke:#475569,stroke-width:2px,color:#1e293b;

    core["soul-mem-core<br/>MemoryNote / MemoryLink 数据模型<br/>✅ 已实现"]
    query["soul-mem-query<br/>Embedding 生成 / Query 类型 / 相似度计算<br/>✅ 已实现"]
    runtime["soul-mem-runtime<br/>WorkingMemory / SlidingWindow / Cluster / Record / LLM 摘要<br/>✅ 已实现"]
    algo["soul-mem-algo<br/>检索策略 RetrStrategy / DefaultPipeline 编排<br/>✅ 已实现"]
    tune["soul-tune<br/>CLI 基准测试框架（TUI）<br/>✅ 已实现"]

    core --> query
    core --> runtime
    query --> runtime
    core --> algo
    query --> algo
    runtime --> algo
    core --> tune
    query --> tune
    runtime --> tune
    algo --> tune

    class core core;
    class query,runtime,algo,tune layer;
```

- `soul-mem-core`：纯数据模型，无任何内部依赖，是其余 crate 的基础。
- `soul-mem-query`：依赖 core，负责文本→向量嵌入（BGE/Qwen3 模型）与查询类型定义。
- `soul-mem-runtime`：依赖 core、query，维护工作记忆（滑动窗口、记忆簇、活跃记录）并封装 LLM 摘要调用。
- `soul-mem-algo`：依赖 core、query、runtime，实现全部检索策略，其中 `RetrDefaultPipeline` 串联三步构成完整检索管线。
- `soul-tune`：依赖全部 crate，是用于基准测试的命令行工具，非运行时组件。

> 注：`soul-mem-runtime` 对 `soul-mem-algo` 的依赖仅存在于 `dev-dependencies`（测试用），生产依赖图中不存在反向依赖。

## 查询请求完整生命周期

一次完整检索请求从外部输入到最终输出的时序如下。颜色含义与主图一致（见上文图例），橙色矩形内的内容为规划中。

```mermaid
sequenceDiagram
    autonumber
    actor Client as 外部调用方
    participant Svc as Service 编排层
    participant SW as SlidingWindow 滑动窗口
    participant LLM as LLM 摘要模型
    participant P as DefaultPipeline 检索管线
    participant CL as MemoryCluster 记忆簇
    participant WM as WorkingMemory / Record
    participant DB as SurrealDb 外部存储
    participant BG as 后台定时任务

    Note over Client,Svc: 请求 = query 集合 + 当前信息增量（均为可选）+ 控制信号
    Client->>Svc: query[], infoDelta?, priority

    alt 存在信息增量
        Svc->>SW: push(infoDelta, role)
        SW->>SW: auto_tag（每 capacity 次标记一条）
        alt 窗口超容量弹出被标记信息
            SW->>LLM: summarize（旧摘要 + 窗口 + 被弹出信息）
            LLM-->>SW: 更新 Summary
        end
    end

    Svc->>P: retrieve(query, priority, working_mem)
    P->>SW: ① ShortOnly：提取窗口内容 + 摘要
    SW-->>P: (short_history, short_mem)
    P->>CL: ② Similarity：query 向量余弦相似度检索
    CL-->>P: top-N (MemoryId, score)
    P->>P: ③a Association：PPR 联想扩散（以 top-N 为源节点）
    P->>P: ③b softmax 归一化
    P->>P: ③c BayesAction：动作概率推理 topK
    P-->>Svc: DefaultPipelineResult { association, action, short_history, short_mem, priority }
    Svc->>WM: record_retrieval / add_feedback 更新活跃记录

    rect rgb(254, 243, 199)
    Note over Svc: 🔲 多 query 按优先级加权合并
    Note over Svc: 🔲 逐 MemoryNote 归一化 → top-K → 提取内容 → 模板填充为自然语言
    end
    Svc-->>Client: MemoryNote 集合 / 自然语言输出

    rect rgb(254, 243, 199)
    Note over BG,DB: 🔲 定时任务及被动流程（规划中）
    Note over BG: WorkingState = Idle 时定时触发
    BG->>SW: 读取摘要
    alt 摘要非空
        BG->>BG: 巩固算法：摘要 + 滑动窗口 → 新 MemoryNote
        BG->>WM: 读取活跃 Record（检索中被激活的节点）
        BG->>BG: 活跃 MemoryNote 与新节点建立拓扑链接
        BG->>CL: 新节点入簇
        BG->>DB: 持久化写入
    end
    Note over BG: 新节点创建 → 生成遗忘遮罩
    Note over BG: 定时衰减数据库中的遮罩权重
    Note over BG: 检索结果命中被遮盖内容 → 遗忘补全
    end
```

## 行为流程说明

> **状态更新（2026-08）**：遗忘机制已实现（`soul-mem-algo/src/algo/forget/`：
> `ebbinghaus_decay` 衰减曲线、`mask_text` 文本遮罩、`lazy_forget` 三档
> NoAction/MaskOnly/Revised），并由 `soul-tune run forget` 评测。巩固/持久化仍为 🔲。

当没有query输入时，信息增量被压入滑动窗口，之后可能会触发滑动窗口的summary机制，并生成新的摘要 ✅

当只有query时，走检索算法的DefaultPipeline，得到DefaultPipelineResult，一份query对应一个DefaultPipelineResult ✅，后续根据query的优先级，以每一个MemoryNote为单位，将分数加权平均，取top-k并提取记忆内容，按照模板填充为自然语言，输出 🔲

当query和信息增量同时存在时，先执行信息增量压入，summary完成后执行检索算法 ✅

当工作记忆状态为Idle时，每隔一段时间，如果摘要不为空，执行巩固算法，它根据摘要和滑动窗口生成新的MemoryNote并建立与Cluster的拓扑链接 🔲

活跃记录（Record）追踪检索中被激活的MemoryNote；巩固时，被标记为活跃的MemoryNote将作为新生成节点的候选拓扑链接目标 🔲

当工作记忆状态为Idle时每隔一段时间，或服务优雅退出时，将工作记忆节点写入数据库持久化 🔲

每当新节点被巩固算法创建时，生成遗忘遮罩 🔲

每隔一段时间，衰减数据库中遗忘遮罩权重 🔲

每当含有“被遮盖”的文本的MemoryNote作为检索算法的最终结果时，调用遗忘补全 🔲

## 外部接口

除了上述的服务输入外，还提供额外的接口（均规划中 🔲）

- 对指定id的MemoryNote读写
- 控制信道，强制触发上述的定时任务
- liveliness/heartbeat

