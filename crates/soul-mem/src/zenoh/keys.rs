//! Key Expression 常量与语义（唯一主题命名处）。
//!
//! 通信仅使用 zenoh 的订阅/发布：
//! - 请求-应答：请求方发布到 `request`，服务应答到 `reply/<request_id>`；
//! - 单向输入：外部设备发布到 `ingest`；
//! - 事件：服务/设备发布到 `events`；
//! - 存活：liveliness token 位于 `liveliness/<device_id>`。
//!
//! 集中于此可避免字符串散落拼错；本模块即“协议目录”。

/// zenoh 侧对外 key 的集中构造。
#[derive(Debug, Clone)]
pub struct Keys {
    prefix: String,
}

impl Keys {
    pub fn new(prefix: impl Into<String>) -> Self {
        Keys {
            prefix: prefix.into(),
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// 请求主题（服务订阅，客户端发布）。
    pub fn request(&self) -> String {
        format!("{}/request", self.prefix)
    }

    /// 某请求 id 的应答主题（客户端订阅，服务发布）。
    pub fn reply_for(&self, request_id: &str) -> String {
        format!("{}/reply/{request_id}", self.prefix)
    }

    /// 单向信息增量主题（服务订阅）。
    pub fn ingest(&self) -> String {
        format!("{}/ingest", self.prefix)
    }

    /// 事件广播主题。
    pub fn events(&self) -> String {
        format!("{}/events", self.prefix)
    }

    /// 某设备 liveliness token key。
    pub fn liveliness_token(&self, device_id: &str) -> String {
        format!("{}/liveliness/{device_id}", self.prefix)
    }

    /// 订阅所有设备的 liveliness 上线/下线。
    pub fn liveliness_all(&self) -> String {
        format!("{}/liveliness/*", self.prefix)
    }
}
