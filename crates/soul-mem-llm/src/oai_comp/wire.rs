//! OpenAI-compatible 的 wire 类型——**全部私有**。
//!
//! 这里是整个 crate 里唯一出现 provider 字段名的地方：`max_tokens` 与
//! `max_completion_tokens` 的取舍、`chat_template_kwargs`、`reasoning_content`、
//! `finish_reason` 的原始字符串。顶层契约（[`crate::backend`]）不认识它们中的任何一个。
//!
//! 为什么自带请求类型而不是用 `CreateChatCompletionRequest`：后者没有 `extra_body`
//! 逃生口，发不出 `chat_template_kwargs` 这类 provider 扩展。`byot` 让我们传入自己的
//! 类型给 async-openai 的传输层，从而既复用它处理 URL/鉴权/错误映射/SSE，又保留
//! 完整的字段控制权。

use crate::backend::{StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// 发出去的一条消息。
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct OaiMessage {
    pub role: String,
    pub content: String,
}

/// 输出上限字段。二者择一，由后端配置决定——这是 provider 差异，不是语义差异。
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub(super) struct TokenField {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
}

impl TokenField {
    pub fn max_tokens(value: u32) -> Self {
        Self {
            max_tokens: Some(value),
            max_completion_tokens: None,
        }
    }

    pub fn max_completion_tokens(value: u32) -> Self {
        Self {
            max_tokens: None,
            max_completion_tokens: Some(value),
        }
    }
}

/// 发出去的请求。所有可选项在缺省时**不出现**在 JSON 里：
/// 有些 OpenAI-compatible 服务对未知/空参数直接 400。
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(super) struct OaiChatRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// 展开为 `{"max_tokens": n}` 或 `{"max_completion_tokens": n}`。
    #[serde(flatten)]
    pub token_limit: TokenField,
    /// provider 扩展（`chat_template_kwargs`、`response_format` …）。
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// 响应里的 token 用量。字段可缺可 null。
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
}

impl OaiUsage {
    pub fn to_usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
    }
}

/// 响应里的 assistant 消息。`content` 可能为 null（内容过滤 / 仅工具调用）。
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    /// 推理模型的思考内容（Qwen3 系在未禁用 thinking 时内容只落在这里）。
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiChoice {
    #[serde(default)]
    pub message: Option<OaiResponseMessage>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// 非流式响应。未知字段一律忽略（不同 provider 差异很大）。
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiResponse {
    #[serde(default)]
    pub choices: Vec<OaiChoice>,
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

/// 流式 chunk 的增量。
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiChunkChoice {
    #[serde(default)]
    pub delta: Option<OaiDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// 流式 chunk。有的 chunk 只带 `usage`，有的只带 `finish_reason`。
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct OaiChunk {
    #[serde(default)]
    pub choices: Vec<OaiChunkChoice>,
    #[serde(default)]
    pub usage: Option<OaiUsage>,
}

/// `finish_reason` 原文 → 语义停止原因。
///
/// 这个映射必须留在后端：它是 provider 词汇，顶层不该认识 `"length"` 这种字符串。
pub(super) fn stop_reason_from_wire(raw: Option<&str>) -> StopReason {
    match raw {
        Some("stop") => StopReason::Completed,
        Some("length") => StopReason::LengthCapped,
        Some("content_filter") => StopReason::Filtered,
        Some("tool_calls") | Some("function_call") => StopReason::ToolCall,
        _ => StopReason::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(token_limit: TokenField, extra: Map<String, Value>) -> OaiChatRequest {
        OaiChatRequest {
            model: "qwen3-4b".into(),
            messages: vec![OaiMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            temperature: None,
            stream: None,
            token_limit,
            extra,
        }
    }

    /// wire 正确性是这一层唯一的防线：字段名写错不会编译报错，只会静默改变行为。
    #[test]
    fn request_omits_absent_optional_fields() {
        let value = serde_json::to_value(request(TokenField::default(), Map::new())).unwrap();
        assert_eq!(
            value,
            json!({
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "hi"}],
            }),
            "缺省字段必须完全不出现"
        );
    }

    #[test]
    fn token_limit_maps_to_exactly_one_wire_field() {
        let max_tokens =
            serde_json::to_value(request(TokenField::max_tokens(512), Map::new())).unwrap();
        assert_eq!(max_tokens["max_tokens"], 512);
        assert!(
            max_tokens.get("max_completion_tokens").is_none(),
            "两个字段绝不能同时出现"
        );

        let max_completion =
            serde_json::to_value(request(TokenField::max_completion_tokens(512), Map::new()))
                .unwrap();
        assert_eq!(max_completion["max_completion_tokens"], 512);
        assert!(max_completion.get("max_tokens").is_none());
    }

    #[test]
    fn extra_body_is_flattened_into_the_top_level() {
        let mut extra = Map::new();
        extra.insert(
            "chat_template_kwargs".into(),
            json!({"enable_thinking": false}),
        );
        extra.insert("response_format".into(), json!({"type": "json_object"}));
        let value = serde_json::to_value(request(TokenField::max_tokens(64), extra)).unwrap();

        assert_eq!(
            value["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(value["response_format"], json!({"type": "json_object"}));
        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["model"], "qwen3-4b");
    }

    #[test]
    fn temperature_and_stream_appear_only_when_set() {
        let mut req = request(TokenField::default(), Map::new());
        req.temperature = Some(0.25);
        req.stream = Some(true);

        let value = serde_json::to_value(&req).unwrap();
        // 0.25 在 f32/f64 下都精确可表；换用 0.2 会因为 f32→JSON 的加宽变成
        // 0.20000000298023224（合法但无法精确断言），测试刻意避开。
        assert_eq!(value["temperature"], 0.25);
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn response_tolerates_missing_and_null_content() {
        let parsed: OaiResponse = serde_json::from_value(json!({
            "id": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": null}, "finish_reason": "content_filter"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": null, "total_tokens": 3}
        }))
        .expect("null content 必须能解析");

        assert_eq!(parsed.choices.len(), 1);
        let message = parsed.choices[0].message.as_ref().unwrap();
        assert_eq!(message.content, None);
        assert_eq!(message.reasoning_content, None);
        assert_eq!(
            parsed.choices[0].finish_reason.as_deref(),
            Some("content_filter")
        );
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(3));
        assert_eq!(usage.completion_tokens, None);
        assert!(!usage.is_empty());
    }

    #[test]
    fn response_tolerates_unknown_fields_and_empty_choices() {
        let parsed: OaiResponse = serde_json::from_value(json!({
            "id": "x",
            "object": "chat.completion",
            "system_fingerprint": "fp_x",
            "choices": []
        }))
        .expect("未知字段必须被忽略");
        assert!(parsed.choices.is_empty());
        assert!(parsed.usage.is_none());
        assert!(OaiUsage::default().is_empty());
    }

    #[test]
    fn chunk_parses_delta_variants() {
        let content: OaiChunk = serde_json::from_value(json!({
            "choices": [{"delta": {"content": "你"}, "finish_reason": null}]
        }))
        .unwrap();
        assert_eq!(
            content.choices[0]
                .delta
                .as_ref()
                .unwrap()
                .content
                .as_deref(),
            Some("你")
        );

        let reasoning: OaiChunk = serde_json::from_value(json!({
            "choices": [{"delta": {"reasoning_content": "想"}, "finish_reason": null}]
        }))
        .unwrap();
        assert_eq!(
            reasoning.choices[0]
                .delta
                .as_ref()
                .unwrap()
                .reasoning_content
                .as_deref(),
            Some("想")
        );

        // 只带 finish_reason / 只带 usage 的 chunk 也必须能解析
        let tail: OaiChunk = serde_json::from_value(json!({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }))
        .unwrap();
        assert_eq!(tail.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(tail.usage.unwrap().total_tokens, Some(3));

        let usage_only: OaiChunk =
            serde_json::from_value(json!({"choices": [], "usage": {"total_tokens": 2}})).unwrap();
        assert!(usage_only.choices.is_empty());
        assert_eq!(usage_only.usage.unwrap().total_tokens, Some(2));
    }

    #[test]
    fn finish_reason_mapping_covers_known_and_unknown() {
        assert_eq!(stop_reason_from_wire(Some("stop")), StopReason::Completed);
        assert_eq!(
            stop_reason_from_wire(Some("length")),
            StopReason::LengthCapped
        );
        assert_eq!(
            stop_reason_from_wire(Some("content_filter")),
            StopReason::Filtered
        );
        assert_eq!(
            stop_reason_from_wire(Some("tool_calls")),
            StopReason::ToolCall
        );
        assert_eq!(
            stop_reason_from_wire(Some("function_call")),
            StopReason::ToolCall
        );
        assert_eq!(
            stop_reason_from_wire(Some("something_new")),
            StopReason::Unknown
        );
        assert_eq!(stop_reason_from_wire(None), StopReason::Unknown);
    }
}
