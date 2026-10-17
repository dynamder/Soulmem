//! SoulMem 的数据模型层；不依赖任何内部 crate，无 IO、LLM 或算法。
//!
//! `MemoryNote` / `MemoryLink` 是公共外壳 + 类型特定载荷，字段分别在
//! `memory_note/` 与 `memory_links/` 下。
//!
//! # 改动前注意
//!
//! - `missing_degree` / `last_forget_time` 的 `#[serde(default)]` 保证老数据可反序列化，不要删。
//! - `MemoryNote::set_missing_degree` 会 clamp，但 `MemoryLink::missing_degree` 是 pub 字段，可不一致。
//! - 新代码用裸名词访问器（`id()` / `tags()`），不要再用 `get_xxx()`。
//! - 对外形态由 [`render`] 定义；内部类型不要直接序列化出去。

pub mod memory_links;
pub mod memory_note;
pub mod render;
