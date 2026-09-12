//! liveliness：服务发现/心跳。
//!
//! 服务上线时 declare liveliness token（由 zenoh 维持存活，设备可订阅上下线），
//! token 在运行时期间持有，运行结束/掉线自动消失——无需自研注册表。

use crate::error::{Error, Result};
use zenoh::Session;
use zenoh::liveliness::LivelinessToken;

/// 以设备身份声明 liveliness token。
pub async fn announce(session: &Session, key: &str) -> Result<LivelinessToken> {
    session
        .liveliness()
        .declare_token(key)
        .await
        .map_err(|e| Error::internal(format!("declare liveliness token {key}: {e}")))
}
