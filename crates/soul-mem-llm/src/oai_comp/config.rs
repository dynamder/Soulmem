//! OpenAI-compatible 后端的配置——**wire 细节都住在这里**。
//!
//! 字段只保留当前真正要用的：输出上限字段名、provider 扩展体、抑制 thinking 的字段、
//! 鉴权头、超时、代理与重试预算。退避参数是常量（见 `transport.rs`）——没有调用方需要调它。
//!
//! # 每组字段被谁消费
//!
//! | 字段 | 消费点 |
//! |---|---|
//! | `base_url` / `api_key` / `auth_header` / `auth_scheme` / `extra_headers` | `provider.rs`（拼 URL、装鉴权头） |
//! | `token_field` / `extra_body` / `reasoning_suppression_body` | `backend.rs::encode`（决定 wire 字段名与扩展体） |
//! | `connect_timeout` / `no_proxy` | `transport.rs::build_http_client` |
//! | `total_timeout` / `first_byte_timeout` | `backend.rs`（`tokio::time::timeout` 包住调用） |
//! | `max_retries` | `transport.rs`（传给重试策略） |
//! | `supports_streaming` | `backend.rs::stream` 的前置检查 |
//! | `name` / `model` | `BackendInfo` → trace |
//!
//! "本地 llama-server 与远程 API 的差异"应当在调用方对着本结构体表达一次
//! （参考 `soul-tune/src/engine/llm/llama_server.rs::local_config`）。

use serde_json::{Map, Value};
use std::time::Duration;

/// 输出上限映射到哪个 wire 字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenFieldPolicy {
    /// `max_tokens`。llama.cpp / vLLM / 多数 OpenAI-compatible 服务接受的字段。
    #[default]
    MaxTokens,
    /// `max_completion_tokens`。OpenAI 官方的新字段名。
    MaxCompletionTokens,
}

impl TokenFieldPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// 一个 OpenAI-compatible provider 的完整配置。
#[derive(Clone)]
pub struct OaiCompatConfig {
    /// 后端名，进 trace。
    pub name: String,
    /// API base（如 `https://api.deepseek.com/v1`、`http://127.0.0.1:8081/v1`）。
    /// 末尾斜杠会被归一化掉。
    pub base_url: String,
    pub model: String,
    /// 密钥。**只从调用方提供的配置来，不落盘、不进 trace。**
    pub api_key: Option<String>,
    /// 鉴权头名字。`None` 表示不发鉴权头（本地服务常见）。
    pub auth_header: Option<String>,
    /// 鉴权前缀。`Some("Bearer")` → `Bearer <key>`；`None` → 裸 key。
    pub auth_scheme: Option<String>,
    pub extra_headers: Vec<(String, String)>,

    pub token_field: TokenFieldPolicy,

    /// 恒定注入的 provider 扩展字段。这是"provider 差异"的通用出口：
    /// `chat_template_kwargs`、`response_format` 之类都从这里发，不必给它们各加一个字段。
    pub extra_body: Map<String, Value>,
    /// [`crate::Hints::suppress_reasoning`] 命中时**额外**注入的字段
    /// （本地 llama-server 用 `chat_template_kwargs: {"enable_thinking": false}`）。
    pub reasoning_suppression_body: Map<String, Value>,

    pub supports_streaming: bool,

    pub connect_timeout: Duration,
    /// 整次调用总时限（含响应体读取）。`None` 表示不限。
    pub total_timeout: Option<Duration>,
    /// 流式的**首字节**时限：拿到响应头即算通过，不限制流的总时长。
    pub first_byte_timeout: Option<Duration>,

    /// `None` = 按 host 自动：环回地址绕过代理，其余交给系统代理设置。
    ///
    /// 环回必须绕过：否则本地 llama-server 会被 `HTTP_PROXY` 劫持成一次外网请求。
    pub no_proxy: Option<bool>,

    /// 传输层重试次数（不含首次）。覆盖 429 / 5xx / 连接失败。
    pub max_retries: u32,
}

impl std::fmt::Debug for OaiCompatConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OaiCompatConfig")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("has_api_key", &self.api_key.is_some())
            .field("token_field", &self.token_field)
            .field("supports_streaming", &self.supports_streaming)
            .field("max_retries", &self.max_retries)
            .field("no_proxy", &self.resolved_no_proxy())
            .finish_non_exhaustive()
    }
}

impl Default for OaiCompatConfig {
    fn default() -> Self {
        Self {
            name: "oai_compat".into(),
            base_url: String::new(),
            model: String::new(),
            api_key: None,
            auth_header: Some("authorization".into()),
            auth_scheme: Some("Bearer".into()),
            extra_headers: Vec::new(),
            token_field: TokenFieldPolicy::MaxTokens,
            extra_body: Map::new(),
            reasoning_suppression_body: Map::new(),
            supports_streaming: true,
            connect_timeout: Duration::from_secs(10),
            total_timeout: Some(Duration::from_secs(120)),
            first_byte_timeout: Some(Duration::from_secs(60)),
            no_proxy: None,
            max_retries: 3,
        }
    }
}

impl OaiCompatConfig {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: normalize_base(&base_url.into()),
            model: model.into(),
            ..Default::default()
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_auth_header(mut self, header: Option<&str>, scheme: Option<&str>) -> Self {
        self.auth_header = header.map(str::to_string);
        self.auth_scheme = scheme.map(str::to_string);
        self
    }

    pub fn with_extra_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    pub fn with_token_field(mut self, policy: TokenFieldPolicy) -> Self {
        self.token_field = policy;
        self
    }

    /// 注入一个 provider 扩展字段（如 `chat_template_kwargs`、`response_format`）。
    pub fn with_extra_body(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra_body.insert(key.into(), value);
        self
    }

    /// 配置"抑制 thinking"的具体字段。
    pub fn with_reasoning_suppression(mut self, key: impl Into<String>, value: Value) -> Self {
        self.reasoning_suppression_body.insert(key.into(), value);
        self
    }

    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// 整次调用总时限（含响应体读取）。`None` 表示不限。
    pub fn with_total_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.total_timeout = timeout;
        self
    }

    /// 流式首字节时限（只覆盖"拿到响应头"）。
    pub fn with_first_byte_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.first_byte_timeout = timeout;
        self
    }

    pub fn with_no_proxy(mut self, no_proxy: bool) -> Self {
        self.no_proxy = Some(no_proxy);
        self
    }

    /// 实际生效的"绕过代理"决策。
    pub fn resolved_no_proxy(&self) -> bool {
        self.no_proxy.unwrap_or_else(|| {
            host_of(&self.base_url)
                .map(is_loopback_host)
                .unwrap_or(false)
        })
    }

    /// 配置自检。缺 base_url / model 属于"后端不可用"，调用方据此降级而不是反复重试。
    pub fn validate(&self) -> Result<(), crate::error::LlmError> {
        if self.base_url.trim().is_empty() {
            return Err(crate::error::LlmError::unavailable(format!(
                "provider `{}` 未配置 base_url",
                self.name
            )));
        }
        if self.model.trim().is_empty() {
            return Err(crate::error::LlmError::unavailable(format!(
                "provider `{}` 未配置 model",
                self.name
            )));
        }
        Ok(())
    }
}

/// 归一化 base：去掉末尾斜杠。
///
/// async-openai 用 `base + "/chat/completions"` 拼 URL，留着末尾斜杠会得到 `//chat/...`，
/// 少数反代对双斜杠直接 404。
pub fn normalize_base(base: &str) -> String {
    base.trim().trim_end_matches('/').to_string()
}

/// 从 URL 里取出 host（剥离 scheme / userinfo / 端口；支持 IPv6 字面量）。
pub fn host_of(url: &str) -> Option<&str> {
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .filter(|a| !a.is_empty())?;
    let authority = authority.rsplit('@').next()?;
    if let Some(rest) = authority.strip_prefix('[') {
        return rest.split(']').next();
    }
    authority.split(':').next()
}

/// 是否为环回地址（本地服务）。
pub fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn base_url_is_normalized() {
        assert_eq!(
            normalize_base("http://127.0.0.1:8081/v1/"),
            "http://127.0.0.1:8081/v1"
        );
        assert_eq!(
            normalize_base("  https://api.deepseek.com/v1//  "),
            "https://api.deepseek.com/v1"
        );
        assert_eq!(normalize_base(""), "");
    }

    #[test]
    fn host_extraction_handles_ports_userinfo_and_ipv6() {
        assert_eq!(host_of("http://127.0.0.1:8081/v1"), Some("127.0.0.1"));
        assert_eq!(
            host_of("https://api.deepseek.com/v1"),
            Some("api.deepseek.com")
        );
        assert_eq!(
            host_of("https://user:pw@example.com:443/v1"),
            Some("example.com")
        );
        assert_eq!(host_of("http://[::1]:8081/v1"), Some("::1"));
        assert_eq!(host_of("127.0.0.1:8081/v1"), Some("127.0.0.1"));
        assert_eq!(host_of(""), None);
    }

    #[test]
    fn loopback_detection() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("LOCALHOST"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.1.2.3"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("api.deepseek.com"));
        assert!(!is_loopback_host("192.168.1.10"));
    }

    /// 本地服务必须绕过代理：否则 `HTTP_PROXY` 会把 127.0.0.1 的请求送到代理想去解析。
    #[test]
    fn no_proxy_defaults_by_host_and_can_be_forced() {
        let local = OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1", "m");
        assert!(local.resolved_no_proxy(), "环回应自动绕过代理");

        let remote = OaiCompatConfig::new("remote", "https://api.deepseek.com/v1", "m");
        assert!(!remote.resolved_no_proxy(), "远程应尊重系统代理设置");

        assert!(remote.clone().with_no_proxy(true).resolved_no_proxy());
        assert!(!local.clone().with_no_proxy(false).resolved_no_proxy());
    }

    #[test]
    fn validate_rejects_missing_base_or_model() {
        let ok = OaiCompatConfig::new("p", "http://127.0.0.1:1/v1", "m");
        assert!(ok.validate().is_ok());

        let no_base = OaiCompatConfig::new("p", "", "m");
        let err = no_base.validate().expect_err("缺 base_url 应报错");
        assert!(err.is_unavailable(), "配置缺失属于后端不可用");

        let no_model = OaiCompatConfig::new("p", "http://127.0.0.1:1/v1", "  ");
        assert!(no_model.validate().is_err());
    }

    #[test]
    fn providers_express_their_differences_through_config() {
        let local = OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1", "qwen3-4b")
            .with_no_proxy(true)
            .with_retries(2)
            .with_reasoning_suppression("chat_template_kwargs", json!({"enable_thinking": false}));

        assert_eq!(local.token_field, TokenFieldPolicy::MaxTokens);
        assert_eq!(
            local.reasoning_suppression_body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );

        let official = OaiCompatConfig::new("openai", "https://api.openai.com/v1", "gpt-4o-mini")
            .with_api_key("sk-x")
            .with_token_field(TokenFieldPolicy::MaxCompletionTokens)
            .with_extra_body("response_format", json!({"type": "json_object"}));

        assert_eq!(official.token_field.as_str(), "max_completion_tokens");
        assert_eq!(
            official.extra_body["response_format"],
            json!({"type": "json_object"})
        );
    }

    #[test]
    fn debug_never_leaks_the_api_key() {
        let cfg = OaiCompatConfig::new("p", "http://127.0.0.1:1/v1", "m").with_api_key("sk-secret");
        let text = format!("{cfg:?}");
        assert!(!text.contains("sk-secret"), "调试输出不得泄露密钥：{text}");
        assert!(text.contains("has_api_key: true"));
    }
}
