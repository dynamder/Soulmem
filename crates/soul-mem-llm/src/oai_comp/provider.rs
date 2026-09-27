//! provider 抽象：实现 `async_openai::config::Config`，把"鉴权头 + base + 额外头"的差异
//! 收在一个地方。
//!
//! 为什么额外 header 走 `Config` 而不是每次调用传：`Chat` 的 `request_options` 在
//! async-openai 里是 `pub(crate)`，外部拿不到；`Config::headers()` 才是它公开的注入口。
//!
//! # 谁在用这里的三个方法
//!
//! async-openai 在拼请求时调用（`Client::build_request_parts`）：
//!
//! | 方法 | 结果去向 |
//! |---|---|
//! | [`Config::url`](async_openai::config::Config::url) | `/chat/completions` 拼到我们归一化后的 base 上 |
//! | [`Config::headers`](async_openai::config::Config::headers) | 鉴权头 + 额外头 |
//! | [`Config::query`](async_openai::config::Config::query) | 空（OpenAI-compatible 不需要 query） |
//!
//! 构造入口：[`OaiCompatProvider::from_config`]，由 `OaiCompatBackend::new` 调用。

use super::config::OaiCompatConfig;
use async_openai::config::Config;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use secrecy::SecretString;

/// 由 [`OaiCompatConfig`] 派生出的 provider 配置。
#[derive(Clone)]
pub(super) struct OaiCompatProvider {
    base: String,
    api_key: SecretString,
    headers: HeaderMap,
}

impl std::fmt::Debug for OaiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OaiCompatProvider")
            .field("base", &self.base)
            .field(
                "header_names",
                &self.headers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl OaiCompatProvider {
    pub(super) fn from_config(cfg: &OaiCompatConfig) -> Self {
        let mut headers = HeaderMap::new();

        if let (Some(name), Some(key)) = (cfg.auth_header.as_deref(), cfg.api_key.as_deref()) {
            let value = match cfg.auth_scheme.as_deref() {
                Some(scheme) if !scheme.is_empty() => format!("{scheme} {key}"),
                _ => key.to_string(),
            };
            insert_header(&mut headers, name, &value);
        }

        for (name, value) in &cfg.extra_headers {
            insert_header(&mut headers, name, value);
        }

        Self {
            base: cfg.base_url.clone(),
            api_key: SecretString::from(cfg.api_key.clone().unwrap_or_default()),
            headers,
        }
    }
}

/// 名字/值非法时跳过而不是 panic：这是运行期输入（可能来自配置文件）。
fn insert_header(headers: &mut HeaderMap, name: &str, value: &str) {
    match (HeaderName::try_from(name), HeaderValue::try_from(value)) {
        (Ok(name), Ok(value)) => {
            headers.insert(name, value);
        }
        _ => {
            tracing::warn!("忽略非法请求头 `{name}`");
        }
    }
}

impl Config for OaiCompatProvider {
    fn headers(&self) -> HeaderMap {
        self.headers.clone()
    }

    /// async-openai 传进来的 path 形如 `/chat/completions`，与 base 直接拼接。
    fn url(&self, path: &str) -> String {
        if path.starts_with('/') {
            format!("{}{path}", self.base)
        } else {
            format!("{}/{path}", self.base)
        }
    }

    fn query(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn api_base(&self) -> &str {
        &self.base
    }

    fn api_key(&self) -> &SecretString {
        &self.api_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    fn header(provider: &OaiCompatProvider, name: &str) -> Option<String> {
        provider
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    }

    #[test]
    fn url_joining_keeps_base_path_and_port() {
        let cfg = OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1/", "m");
        let provider = OaiCompatProvider::from_config(&cfg);
        assert_eq!(
            provider.url("/chat/completions"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
        assert_eq!(
            provider.url("chat/completions"),
            "http://127.0.0.1:8081/v1/chat/completions"
        );
        assert_eq!(provider.api_base(), "http://127.0.0.1:8081/v1");
        assert!(provider.query().is_empty());
    }

    #[test]
    fn bearer_auth_is_applied_by_default() {
        let cfg = OaiCompatConfig::new("p", "https://api.example.com/v1", "m").with_api_key("sk-x");
        let provider = OaiCompatProvider::from_config(&cfg);
        assert_eq!(
            header(&provider, "authorization").as_deref(),
            Some("Bearer sk-x")
        );
        assert_eq!(provider.api_key().expose_secret(), "sk-x");
    }

    #[test]
    fn auth_can_use_a_custom_header_and_bare_key() {
        let cfg = OaiCompatConfig::new("azure", "https://x.openai.azure.com/v1", "m")
            .with_api_key("k")
            .with_auth_header(Some("api-key"), None);
        let provider = OaiCompatProvider::from_config(&cfg);
        assert_eq!(header(&provider, "api-key").as_deref(), Some("k"));
        assert!(
            header(&provider, "authorization").is_none(),
            "自定义头名时不得再发 Authorization"
        );
    }

    #[test]
    fn local_provider_sends_no_auth_header() {
        let cfg = OaiCompatConfig::new("local", "http://127.0.0.1:8081/v1", "m")
            .with_auth_header(None, None);
        let provider = OaiCompatProvider::from_config(&cfg);
        assert!(header(&provider, "authorization").is_none());
        assert!(provider.headers().is_empty());
    }

    #[test]
    fn extra_headers_are_merged_without_dropping_auth() {
        let cfg = OaiCompatConfig::new("p", "https://api.example.com/v1", "m")
            .with_api_key("sk-x")
            .with_extra_header("x-trace-id", "abc")
            .with_extra_header("HTTP-Referer", "https://soulmem.local");
        let provider = OaiCompatProvider::from_config(&cfg);
        assert_eq!(
            header(&provider, "authorization").as_deref(),
            Some("Bearer sk-x")
        );
        assert_eq!(header(&provider, "x-trace-id").as_deref(), Some("abc"));
        assert_eq!(
            header(&provider, "http-referer").as_deref(),
            Some("https://soulmem.local")
        );
    }

    #[test]
    fn illegal_header_name_is_skipped_not_panicked() {
        let cfg = OaiCompatConfig::new("p", "http://127.0.0.1:1/v1", "m")
            .with_extra_header("bad header name", "v")
            .with_extra_header("x-ok", "v");
        let provider = OaiCompatProvider::from_config(&cfg);
        assert!(header(&provider, "x-ok").is_some());
        assert_eq!(provider.headers().len(), 1, "非法头名应被跳过");
    }

    #[test]
    fn debug_does_not_leak_the_key() {
        let cfg =
            OaiCompatConfig::new("p", "https://api.example.com/v1", "m").with_api_key("sk-secret");
        let provider = OaiCompatProvider::from_config(&cfg);
        let text = format!("{provider:?}");
        assert!(!text.contains("sk-secret"), "{text}");
    }
}
