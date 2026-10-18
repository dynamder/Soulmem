//! soul-tune 库目标：暴露 headless 测试核心（engine + base）。
//!
//! 二进制目标（main.rs）是 headless CLI，直接复用本库的 `engine` / `base`，
//! 不再自己声明一份 mod 树：同一批源文件被两个 crate root 各声明一次，
//! 会让 cargo-mutants 把 engine/** 下的变异体生成并测试两遍。
//! 该库目标供 crates/soul-tune-api（FRB 桥接层）复用真实测试逻辑。

pub mod base;
pub mod engine;
