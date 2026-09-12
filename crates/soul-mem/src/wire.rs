//! wire：**收发协议层**。
//!
//! 协议由 `proto/soul_mem.proto` 定义，`build.rs` 用 prost 生成 Rust 消息类型（`pb`），
//! zenoh 载荷即这些消息的 protobuf 二进制编码。`convert` 负责 protobuf 消息与
//! 内部领域模型（core/query/runtime）的双向转换与校验。

pub mod convert;

/// prost 生成的消息类型（来自 `proto/soul_mem.proto`，包 `soulmem`）。
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/soulmem.rs"));
}

pub use pb::*;

/// 请求操作名常量（`RequestEnvelope.op`），服务端/客户端共用，避免魔法字符串。
pub mod op {
    pub const PING: &str = "ping";
    pub const INGEST: &str = "ingest";
    pub const RETRIEVE: &str = "retrieve";
    pub const READ: &str = "read";
    pub const WRITE: &str = "write";
    pub const FEEDBACK: &str = "feedback";
    pub const CONTROL: &str = "control";
}
