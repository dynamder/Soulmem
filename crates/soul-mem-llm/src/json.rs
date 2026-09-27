//! 宽容地从 LLM 输出里取出 JSON —— 与 provider、与具体任务都无关。
//!
//! # 谁在用
//!
//! | 调用方 | 用到的入口 |
//! |---|---|
//! | `soul-mem-algo` 巩固解析器 | [`parse_json_array`] |
//! | `soul-tune` playtest 修复器 | [`extract_balanced_array`] / [`extract_balanced_object`] / [`extract_top_level_objects`] / [`strip_markdown_fences`] |
//! | `soul-tune` trace 展示 | [`split_think_blocks`] / [`strip_think_block`] / [`extract_think_content`] |
//! | 本 crate 的 provider 错误信息 | [`preview_text`] |
//!
//! # 为什么集中在一处
//!
//! 这段逻辑此前在仓库里有两份实现，且细节不一致：`soul-tune` 的 playtest 修复器用
//! **平衡括号扫描**（会跳过字符串内的括号与转义），`soul-mem-algo` 的巩固解析器用
//! `rfind('}')` 截取（尾部解释里出现 `}` 就解析失败）。这里以前者为准，作为唯一来源。
//!
//! # 顺序
//!
//! 1. 剥 `<think>` 块（推理模型的思考过程不该被当成答案）；
//! 2. 剥 markdown 代码围栏；
//! 3. 用平衡括号扫描取出完整的数组/对象；
//! 4. 反序列化。解析失败**一律返回显式错误并带上原文片段**，不静默降级为空结果。

use crate::error::{LlmError, LlmErrorKind};
use serde::de::DeserializeOwned;

/// 报错时携带的原文长度上限。
const PREVIEW_CHARS: usize = 512;

/// 按**字符**边界截断文本（不能按字节切，否则多字节字符会被切成半个）。
///
/// 共用给本模块与 provider 的错误信息：两者都需要"把原文片段带进错误里"。
pub fn preview_text(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => format!("{}…", &text[..idx]),
        None => text.to_string(),
    }
}

/// 找到下一个 `<think>` 块，返回 `(块起点, 内容起点, 内容终点, 块终点)`。
///
/// 容忍三种收尾：`</think>`、`<think/>`（模型偶发写成开标签）与**未闭合**（吃到结尾）。
fn find_next_think_block(s: &str) -> Option<(usize, usize, usize, usize)> {
    let block_start = s.find("<think>")?;
    let content_start = block_start + 7;
    let rest = &s[block_start..];

    let (closing_tag_pos, closing_tag_len) = if let Some(pos) = rest.find("</think>") {
        (pos, 8)
    } else if let Some(pos) = rest.find("<think/>") {
        (pos, 8)
    } else {
        (rest.len(), 0)
    };

    Some((
        block_start,
        content_start,
        block_start + closing_tag_pos,
        block_start + closing_tag_pos + closing_tag_len,
    ))
}

/// 剥离全部 `<think>` 块并修剪首尾空白。
pub fn strip_think_block(text: &str) -> String {
    let mut result = text.to_string();
    while let Some((block_start, _, _, block_end)) = find_next_think_block(&result) {
        result.replace_range(block_start..block_end, "");
    }
    result.trim().to_string()
}

/// 取出第一个 `<think>` 块的内容（供观测/排障，不参与解析）。
pub fn extract_think_content(text: &str) -> Option<String> {
    let (_, content_start, content_end, _) = find_next_think_block(text)?;
    Some(text[content_start..content_end].trim().to_string())
}

/// 取第一个 `[` 到最后一个 `]` 之间的片段。
///
/// **不感知嵌套与字符串**，只在内容确定是单个数组时可用；不确定时请用
/// [`extract_balanced_array`]。
pub fn extract_json_array(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// 取第一个**平衡**的 `[...]`（跳过字符串内的括号与转义）。
pub fn extract_balanced_array(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in text[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '[' if !in_string => depth += 1,
            ']' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=start + i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// 取第一个**平衡**的 `{...}`（跳过字符串内的括号与转义）。
pub fn extract_balanced_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let (end, ok) = find_matching_brace(text, start);
    if ok {
        Some(text[start..=end].to_string())
    } else {
        None
    }
}

/// 拆出**全部** `<think>` 块的内容与剩余正文。
///
/// 返回 `(各思考块, 去思考后的正文)`；没有思考块时第一项为空。
/// 用于"把模型的思考过程与答案分开呈现"的场景（playtest trace、报告）。
pub fn split_think_blocks(text: &str) -> (Vec<String>, String) {
    let mut thoughts: Vec<String> = Vec::new();
    let mut body = text.to_string();
    while let Some((block_start, content_start, content_end, block_end)) =
        find_next_think_block(&body)
    {
        thoughts.push(body[content_start..content_end].trim().to_string());
        body.replace_range(block_start..block_end, "");
    }
    (thoughts, body.trim().to_string())
}

/// 收集文本里**所有**顶层 `{...}`，拼成一个 JSON 数组。
///
/// 用于模型把多个对象平铺输出（`{...}\n{...}`）而没有套数组的情况。
pub fn extract_top_level_objects(text: &str) -> Option<String> {
    let mut objects: Vec<String> = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let (obj_end, ok) = find_matching_brace(text, i);
            if ok {
                let obj = text[i..=obj_end].trim().to_string();
                if !obj.is_empty() {
                    objects.push(obj);
                }
                i = obj_end + 1;
                continue;
            }
        }
        i += 1;
    }
    if objects.is_empty() {
        None
    } else {
        Some(format!("[{}]", objects.join(",")))
    }
}

fn find_matching_brace(text: &str, start: usize) -> (usize, bool) {
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in text[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return (start + i, true);
                }
            }
            _ => {}
        }
    }
    (text.len(), false)
}

/// 剥掉 markdown 代码围栏（` ```json ... ``` `）。
///
/// 没有围栏时原样返回（只做 trim），不做任何猜测性改写。
pub fn strip_markdown_fences(text: &str) -> String {
    let lines: Vec<&str> = text.trim().lines().collect();
    let mut result: Vec<&str> = Vec::new();
    let mut in_fence = false;
    let mut stripped = false;
    for line in &lines {
        if line.trim().starts_with("```") {
            in_fence = !in_fence;
            stripped = true;
            continue;
        }
        result.push(line);
    }
    if stripped || in_fence {
        result.join("\n").trim().to_string()
    } else {
        text.to_string()
    }
}

/// 宽容地解析一个 JSON 数组：剥 think → 提取平衡数组 → 提取围栏内数组 → 反序列化。
///
/// 失败时返回 [`LlmErrorKind::Decode`]，message 里带上原子片段——**不返回空数组**，
/// 否则"模型没给出合法 JSON"会伪装成"模型认为这里没有内容"。
pub fn parse_json_array<T: DeserializeOwned>(raw: &str) -> Result<Vec<T>, LlmError> {
    let without_think = strip_think_block(raw);
    let candidate = extract_balanced_array(&without_think)
        .or_else(|| extract_balanced_array(&strip_markdown_fences(&without_think)));

    let Some(json) = candidate else {
        return Err(decode_error("输出里找不到完整的 JSON 数组", &without_think));
    };

    serde_json::from_str(&json)
        .map_err(|error| decode_error(&format!("JSON 数组解析失败: {error}"), &json))
}

fn decode_error(what: &str, body: &str) -> LlmError {
    LlmError::new(
        LlmErrorKind::Decode,
        format!(
            "{what}（原文前 {PREVIEW_CHARS} 字符）: {}",
            preview_text(body, PREVIEW_CHARS)
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    // ---------- think 块 ----------

    #[test]
    fn strip_think_block_removes_closed_blocks() {
        assert_eq!(strip_think_block("<think>reasoning</think>ab"), "ab");
        assert_eq!(strip_think_block("a<think>x</think>b"), "ab");
    }

    #[test]
    fn strip_think_block_tolerates_the_self_closing_variant() {
        assert_eq!(strip_think_block("<think>a<think/>b"), "b");
    }

    #[test]
    fn strip_think_block_tolerates_unclosed_block() {
        assert_eq!(strip_think_block("ab<think>never closed"), "ab");
    }

    #[test]
    fn strip_think_block_removes_every_block_and_trims() {
        assert_eq!(
            strip_think_block("  <think>1</think>mid<think>2</think>  "),
            "mid"
        );
    }

    #[test]
    fn extract_think_content_reports_presence_honestly() {
        assert_eq!(
            extract_think_content("<think>reasoning</think>"),
            Some("reasoning".to_string())
        );
        assert_eq!(extract_think_content("no block here"), None);
        assert_eq!(
            extract_think_content("<think></think>"),
            Some(String::new())
        );
    }

    // ---------- 数组/对象提取 ----------

    #[test]
    fn extract_json_array_takes_first_to_last_bracket() {
        assert_eq!(
            extract_json_array(r#"prefix [{"k":"v"}] suffix"#),
            Some(r#"[{"k":"v"}]"#)
        );
        assert_eq!(extract_json_array("[1,[2]]"), Some("[1,[2]]"));
        assert_eq!(extract_json_array("no array"), None);
        assert_eq!(extract_json_array("[only open"), None);
    }

    #[test]
    fn extract_balanced_array_ignores_brackets_inside_strings() {
        let input = r#"noise [{"a":"]"}, {"b":[1,2]}] tail"#;
        assert_eq!(
            extract_balanced_array(input).as_deref(),
            Some(r#"[{"a":"]"}, {"b":[1,2]}]"#)
        );
    }

    #[test]
    fn extract_balanced_array_returns_none_when_unbalanced() {
        assert_eq!(extract_balanced_array("[1, [2]"), None);
        assert_eq!(extract_balanced_array("no array"), None);
    }

    #[test]
    fn extract_balanced_object_handles_nesting_and_escapes() {
        let input = r#"pre {"a": {"b": "}"}} post"#;
        assert_eq!(
            extract_balanced_object(input).as_deref(),
            Some(r#"{"a": {"b": "}"}}"#)
        );
        assert_eq!(extract_balanced_object(r#"{"a": 1"#), None);
        assert_eq!(extract_balanced_object("no object"), None);
    }

    #[test]
    fn extract_top_level_objects_wraps_flattened_objects_into_an_array() {
        let flattened = r#"{"a":1} noise {"b":2}"#;
        assert_eq!(
            extract_top_level_objects(flattened).as_deref(),
            Some(r#"[{"a":1},{"b":2}]"#)
        );
        assert_eq!(extract_top_level_objects("no objects"), None);
    }

    // ---------- 围栏 ----------

    #[test]
    fn strip_markdown_fences_unwraps_fenced_content() {
        assert_eq!(strip_markdown_fences("```json\n[a]\n```"), "[a]");
        assert_eq!(strip_markdown_fences("```\n[a]\n```"), "[a]");
    }

    #[test]
    fn strip_markdown_fences_leaves_plain_text_untouched() {
        assert_eq!(strip_markdown_fences("[a]"), "[a]");
        assert_eq!(strip_markdown_fences("  [a]  "), "  [a]  ");
    }

    // ---------- 宽容解析 ----------

    #[test]
    fn parse_json_array_handles_think_fences_and_noise() {
        let fenced = "```json\n[{\"a\":1}]\n```";
        assert_eq!(
            parse_json_array::<Value>(fenced).expect("应成功"),
            vec![json!({"a": 1})]
        );

        let think_wrapped = "<think>先想一下</think>[{\"a\":1},{\"b\":2}]";
        assert_eq!(
            parse_json_array::<Value>(think_wrapped)
                .expect("应成功")
                .len(),
            2
        );

        let noisy = "好的，结果如下：\n[{\"a\":1}]\n以上。";
        assert_eq!(parse_json_array::<Value>(noisy).expect("应成功").len(), 1);

        let empty: Vec<Value> = parse_json_array::<Value>("[]").expect("空数组合法");
        assert!(empty.is_empty());
    }

    /// 这一条是本次集中化的直接收益：尾部解释里带 `}` 不再解析失败。
    #[test]
    fn parse_json_array_survives_braces_in_trailing_prose() {
        let raw = r#"[{"a":1}] 说明：每个对象形如 {"a": <int>}。"#;
        let parsed = parse_json_array::<Value>(raw).expect("尾部解释不应破坏解析");
        assert_eq!(parsed, vec![json!({"a": 1})]);
    }

    #[test]
    fn parse_json_array_reports_decode_errors_with_source_text() {
        let no_array = parse_json_array::<Value>("模型今天不想输出 JSON").expect_err("应失败");
        assert_eq!(no_array.kind(), LlmErrorKind::Decode);
        assert!(
            no_array.message().contains("找不到完整的 JSON 数组"),
            "{}",
            no_array.message()
        );
        assert!(
            no_array.message().contains("模型今天不想输出"),
            "原文片段必须带出：{}",
            no_array.message()
        );

        let malformed = parse_json_array::<Value>("[{\"a\": }]").expect_err("应失败");
        assert_eq!(malformed.kind(), LlmErrorKind::Decode);
        assert!(
            malformed.message().contains("解析失败"),
            "{}",
            malformed.message()
        );
    }

    #[test]
    fn parse_json_array_accepts_strongly_typed_items() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Item {
            a: i32,
        }
        let parsed = parse_json_array::<Item>(r#"[{"a":1},{"a":2}]"#).expect("应成功");
        assert_eq!(parsed, vec![Item { a: 1 }, Item { a: 2 }]);
    }

    #[test]
    fn preview_text_truncates_on_character_boundaries() {
        assert_eq!(preview_text("abc", 5), "abc");
        assert_eq!(preview_text("abcdef", 3), "abc…");
        assert_eq!(preview_text("日本語テキスト", 2), "日本…");
        assert_eq!(preview_text("", 3), "");
    }
}
