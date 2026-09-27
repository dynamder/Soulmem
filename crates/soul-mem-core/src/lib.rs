//! SoulMem 的数据模型层。
//!
//! 本 crate **不依赖任何内部 crate**，是其余 crate 的基础。这里只有数据结构与
//! 构造/访问逻辑——没有 IO、没有 LLM、没有任何算法。
//!
//! # 两个核心类型
//!
//! | 类型 | 含义 | 位置 |
//! |---|---|---|
//! | [`MemoryNote`](memory_note::MemoryNote) | 记忆节点：标签、类型特定内容、出边 | [`memory_note`] |
//! | [`MemoryLink`](memory_links::MemoryLink) | 记忆之间的有向边：关系类型、连接强度、遗忘状态 | [`memory_links`] |
//!
//! 二者都是"公共外壳 + 类型特定载荷"的形态：节点带一个
//! [`MemoryType`](memory_note::MemoryType)（Situation / Semantic / Procedure），
//! 边带一个 [`MemoryLinkType`](memory_links::MemoryLinkType)。
//! 类型特定字段分别在 `memory_note/{situation_mem,sem_mem,proc_mem}.rs` 与
//! `memory_links/{situation_mem,sem_mem,proc_mem}.rs` 下。
//!
//! # 遗忘状态是节点与边共用的
//!
//! `missing_degree`（0.0 新鲜 ~ 1.0 完全遗忘）与 `last_forget_time` 同时存在于节点和边上，
//! 并且都用 `#[serde(default = "...")]` 声明——**这样老数据缺字段时仍能反序列化**
//! （按"新鲜"处理）。改动这两个字段时请保留该属性，否则已持久化的记忆会读不出来。
//!
//! # 已知不一致（改动时留意，但不要顺手扩散）
//!
//! - **封装程度不同**：`MemoryNote::set_missing_degree` 会做 `clamp(0.0, 1.0)`，
//!   而 `MemoryLink::missing_degree` 是 `pub` 字段，可以直接写入越界值绕过该不变量。
//! - **访问器风格不统一**：一部分是 `get_xxx()`（且返回 `&Option<T>` / `&Vec<T>`），
//!   另一部分是裸名词方法（`id()` / `tags()`，返回 `&[T]`）。**新代码请用后者。**
//! - `default_missing_degree` / `default_last_forget_time` 在两个模块里各有一份。

pub mod memory_links;
pub mod memory_note;
