//! 传输适配层：zenoh pub/sub 与 gRPC 共用同一份 proto，核心与传输解耦。
//!
//! | 模块 | 职责 |
//! |---|---|
//! | [`convert`] | proto 消息 <-> 领域 DTO |
//! | [`dispatch`] | 把一条 `Request` 分派到 [`crate::service::ServiceHandle`] 并组装 `Reply` |
//! | [`zenoh`] | `<prefix>/req` 订阅、`<prefix>/resp` 与 `<prefix>/event` 发布 |
//! | [`grpc`] | `SoulMem::Exchange` + `SoulMem::Subscribe` |

pub mod convert;
pub mod dispatch;
pub mod grpc;
pub mod zenoh;

pub use grpc::GrpcService;
pub use zenoh::ZenohService;
