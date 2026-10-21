//! 请求分派：一条 proto `Request` -> 核心 -> 一条 proto `Reply`。
//!
//! prost 的 `Message::{encode_to_vec, decode}` 完成。zenoh 与 gRPC 两个适配器共用本函数，
//! 保证两条传输的业务行为一致。

use super::convert;
use crate::proto::v1;
use crate::service::{ServiceHandle, ServiceResponse};

/// 分派一条原子请求并返回响应。
///
/// proto 的 body 只能是一种操作；服务端顺序执行。请求转换失败（未知控制信号）与
/// 服务内部失败都以 `Reply { ok: false, error }` 返回。
pub async fn dispatch(handle: &ServiceHandle, request: v1::Request) -> v1::Reply {
    let request_id = request.request_id.clone();
    let service_request = match convert::service_request(request) {
        Ok(Some(request)) => request,
        Ok(None) => return snapshot_reply(handle, request_id).await,
        Err(error) => return convert::reply_error(request_id, error.to_string()),
    };

    match handle.send(service_request).await {
        Ok(response) => reply_for(request_id, response),
        Err(error) => {
            let state = handle.snapshot().await.ok();
            convert::reply(convert::ReplyData {
                request_id,
                error: Some(error.to_string()),
                output: None,
                memories: Vec::new(),
                state,
                accepted: 0,
            })
        }
    }
}

async fn snapshot_reply(handle: &ServiceHandle, request_id: String) -> v1::Reply {
    let state = handle.snapshot().await.ok();
    convert::reply(convert::ReplyData {
        request_id,
        error: None,
        output: None,
        memories: Vec::new(),
        state,
        accepted: 0,
    })
}

fn reply_for(request_id: String, response: ServiceResponse) -> v1::Reply {
    let (accepted, output, memories, state) = match response {
        ServiceResponse::Ingested(state) => (1, None, Vec::new(), Some(state)),
        ServiceResponse::Retrieved(retrieval) => (
            0,
            Some(retrieval.output),
            retrieval.memories,
            Some(retrieval.state),
        ),
        ServiceResponse::Controlled(state) => (0, None, Vec::new(), Some(state)),
    };
    convert::reply(convert::ReplyData {
        request_id,
        error: None,
        output,
        memories,
        state,
        accepted,
    })
}
