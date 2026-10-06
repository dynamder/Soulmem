//! 请求分派：一条 proto `Request` -> 核心 -> 一条 proto `Reply`。
//!
//! prost 的 `Message::{encode_to_vec, decode}` 完成。zenoh 与 gRPC 两个适配器共用本函数，
//! 保证两条传输的业务行为一致。

use super::convert;
use crate::proto::v1;
use crate::service::ServiceHandle;

/// 分派一条请求并返回响应。
///
/// 请求转换失败（未知控制信号）与服务内部失败都以 `Reply { ok: false, error }` 返回；
/// 服务内部失败仍保留 `state`/`accepted`（见 [`convert::reply`]），只有服务循环不可用时
/// 才退化为 [`convert::reply_error`]。
pub async fn dispatch(handle: &ServiceHandle, request: v1::Request) -> v1::Reply {
    let request_id = request.request_id.clone();
    let service_request = match convert::service_request(request) {
        Ok(service_request) => service_request,
        Err(error) => return convert::reply_error(request_id, error.to_string()),
    };
    match handle.exchange(service_request).await {
        Ok(response) => convert::reply(request_id, response),
        Err(error) => convert::reply_error(request_id, error.to_string()),
    }
}
