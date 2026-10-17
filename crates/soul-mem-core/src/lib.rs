//! SoulMem 的数据模型层；不依赖任何内部 crate，无 IO、LLM 或算法。
//!
//! `MemoryNote` / `MemoryLink` 是公共外壳 + 类型特定载荷，字段分别在
//! `memory_note/` 与 `memory_links/` 下。

pub mod memory_links;
pub mod memory_note;
pub mod render;
