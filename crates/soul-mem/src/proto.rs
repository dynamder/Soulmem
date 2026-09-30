//! protobuf 生成代码。
//!
//! 由 `build.rs` 从 `proto/soulmem.proto` 生成；消息在 [`v1`]，服务端 stub
//! 亦由 tonic 生成（`soul_mem_server`）。本 crate 只做服务端，不生成客户端 stub。
//!
//! 本模块只做原样导出，不做任何加工——协议相关逻辑一律放在 `transport` 的转换层。
#![allow(clippy::all, clippy::pedantic)]

pub mod v1 {
    tonic::include_proto!("soulmem.v1");
}
