//! soul-tune 库目标：暴露 headless 测试核心（engine + base）。
//!
//! 二进制目标（main.rs）保持独立的 `mod` 声明与 headless CLI，二者互不影响。
//! 该库目标供 crates/soul-tune-api（FRB 桥接层）复用真实测试逻辑。

pub mod base;
pub mod engine;
