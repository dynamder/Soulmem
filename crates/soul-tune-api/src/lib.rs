// cdylib/staticlib 无 Rust 下游：pub 项仅供 FFI 导出与 Dart 端消费（rustc 无法
// 感知外部消费者），dead_code 按 crate 视角判定属误报，统一放行；
// not_unsafe_ptr_arg_deref 来自 FRB 生成代码（frb_generated.rs）的宏展开。
#![allow(dead_code, clippy::not_unsafe_ptr_arg_deref)]

//! soul-tune-api：FRB 桥接层（JSON-over-FRB）。
//!
//! 架构约定：
//! - 所有跨边界数据均为 JSON 字符串：FRB 只传递 `String` 与 `Stream<String>`，
//!   Dart 侧用 fromJson 模型解析，Rust 侧无需为 FRB 生成任何 mirror 类型。
//! - 测试执行在 std 线程中运行（engine 为同步 API），进度通过 `StreamSink<String>`
//!   推流；事件统一为内部标签枚举（见 `api::RunEvent` / `api::BatchEvent`）。
//! - 取消：全局 `AtomicBool` 标志，运行循环在用例之间检查。
//! - 本 crate 不包含任何 UI 逻辑；数据组装（downcast 序列化、报告 JSON）也在此完成，
//!   保证 Flutter 侧纯渲染。

pub mod api;
pub mod frb_generated; // 由 `flutter_rust_bridge_codegen generate` 生成
