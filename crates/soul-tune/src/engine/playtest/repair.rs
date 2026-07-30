use serde::Deserialize;

/// Returns (block_start, content_start, content_end, block_end) or None.
/// Supports `<think>..</think>`, `<think>..<think/>`, and unclosed `<think>..` variants.
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

    let content_end = block_start + closing_tag_pos;
    let block_end = content_end + closing_tag_len;

    Some((block_start, content_start, content_end, block_end))
}

pub fn strip_think_block(s: &str) -> String {
    let mut result = s.to_string();
    loop {
        match find_next_think_block(&result) {
            Some((block_start, _, _, block_end)) => {
                result.replace_range(block_start..block_end, "");
            }
            None => break,
        }
    }
    result.trim().to_string()
}

pub fn extract_think_content(s: &str) -> Option<String> {
    let (_, content_start, content_end, _) = find_next_think_block(s)?;
    Some(s[content_start..content_end].trim().to_string())
}

pub fn extract_json_array(s: &str) -> Option<&str> {
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    if end > start {
        Some(&s[start..=end])
    } else {
        None
    }
}

fn extract_balanced_array(s: &str) -> Option<String> {
    let start = s.find('[')?;
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s[start..].char_indices() {
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
                    return Some(s[start..=start + i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_markdown_fences(s: &str) -> String {
    let lines: Vec<&str> = s.trim().lines().collect();
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
        s.to_string()
    }
}

fn find_matching_brace(s: &str, start: usize) -> (usize, bool) {
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s[start..].char_indices() {
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
    (s.len(), false)
}

fn extract_top_level_objects(s: &str) -> Option<String> {
    let mut objects: Vec<String> = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let (obj_end, ok) = find_matching_brace(s, i);
            if ok {
                let obj = s[i..=obj_end].trim().to_string();
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

fn is_valid_query_json(s: &str) -> bool {
    serde_json::from_str::<Vec<RawQuery>>(s).is_ok()
}

pub fn robust_json_extract(clean: &str) -> Option<String> {
    if let Some(j) = extract_balanced_array(clean) {
        if is_valid_query_json(&j) {
            return Some(j);
        }
    }

    let stripped = strip_markdown_fences(clean);

    if let Some(j) = extract_balanced_array(&stripped) {
        if is_valid_query_json(&j) {
            return Some(j);
        }
    }

    if let Some(j) = extract_top_level_objects(clean) {
        if is_valid_query_json(&j) {
            return Some(j);
        }
    }

    if stripped != clean {
        if let Some(j) = extract_top_level_objects(&stripped) {
            if is_valid_query_json(&j) {
                return Some(j);
            }
        }
    }

    if let Some(j) = repair_json(&stripped) {
        return Some(j);
    }

    if let Some(j) = repair_json(clean) {
        return Some(j);
    }

    None
}

pub fn split_response(s: &str) -> (Option<String>, String) {
    let mut think_parts: Vec<String> = Vec::new();
    let mut body = s.to_string();
    loop {
        match find_next_think_block(&body) {
            Some((block_start, content_start, content_end, block_end)) => {
                think_parts.push(body[content_start..content_end].trim().to_string());
                body.replace_range(block_start..block_end, "");
            }
            None => break,
        }
    }
    let think = if think_parts.is_empty() {
        None
    } else {
        Some(think_parts.join("\n\n"))
    };
    (think, body.trim().to_string())
}

#[derive(Debug, Deserialize)]
pub struct RawQuery {
    pub tag: Vec<String>,
    pub variant: RawVariant,
    pub priority: u32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RawVariant {
    Situation(Vec<RawSitUnit>),
    Semantic(Vec<RawSemUnit>),
    SemanticSingle(RawSemUnit),
    BareArray(Vec<RawSemUnit>),
    BareSingle(RawSemUnit),
}

#[derive(Debug, Deserialize)]
pub struct RawSemUnit {
    #[serde(default)]
    pub concept_identifier: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawSitUnit {
    #[serde(default)]
    pub narrative: Option<String>,
    #[serde(default)]
    pub participants: Option<Vec<RawParticipant>>,
    #[serde(default)]
    pub environment: Option<RawEnvironment>,
    #[serde(default)]
    pub event: Option<Vec<RawEvent>>,
}

#[derive(Debug, Deserialize)]
pub struct RawParticipant {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawEnvironment {
    #[serde(default)]
    pub atmosphere: Option<String>,
    #[serde(default)]
    pub tone: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawEvent {
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub initiator: Option<String>,
    #[serde(default)]
    pub target: Option<String>,
}

const JSON_REPAIR_SLUG: &str = "soul-tune-json-repair-v1";
const JSON_REPAIR_SPEC: &str = r#"You are a JSON repair tool. Fix the malformed JSON array to produce valid JSON.
Fix these issues: trailing commas, unquoted keys, single quotes,
unclosed brackets/braces, extra text or markdown fences.

Correct output format:
[
  {"tag": ["personality"], "variant": [{"concept_identifier": "traits"}], "priority": 0},
  {"tag": ["event", "recent"], "variant": {"concept_identifier": "meeting", "description": "discussed timeline"}, "priority": 1}
]

Each object MUST have: <tag> (string array), <variant> (object or array of objects with concept_identifier and optional description), <priority> (integer).
Output ONLY the repaired JSON array. No markdown, no explanations."#;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub fn repair_json(bad_json: &str) -> Option<String> {
    let state = init_paw_state();
    let mut lock = state.inner.lock().ok()?;
    let f = lock.as_mut()?;
    let prompt = format!("Fix this JSON:\n{}\n\n---\nRepaired JSON:", bad_json);
    let raw = f.run(&prompt).ok()?;
    extract_balanced_array(&raw)
        .or_else(|| extract_json_array(&raw).map(|s| s.to_string()))
}

struct PawState {
    rt: tokio::runtime::Runtime,
    inner: Mutex<Option<Box<dyn paw_rs::paw_core::PawFnTrait>>>,
    mapping_path: PathBuf,
    config: paw_rs::paw_core::PawConfig,
}

static PAW: OnceLock<PawState> = OnceLock::new();

fn init_paw_state() -> &'static PawState {
    PAW.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().expect("PAW tokio runtime");
        let config = paw_rs::paw_core::PawConfig::from_env();
        let mapping_path = config.cache_dir().join("paw_id_mapping.json");
        let inner = Mutex::new(init_paw_fn_blocking(&rt, &config, &mapping_path));
        PawState { rt, inner, mapping_path, config }
    })
}

fn init_paw_fn_blocking(
    rt: &tokio::runtime::Runtime,
    config: &paw_rs::paw_core::PawConfig,
    mapping_path: &Path,
) -> Option<Box<dyn paw_rs::paw_core::PawFnTrait>> {
    rt.block_on(async {
        if let Ok(data) = std::fs::read_to_string(mapping_path) {
            if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
                if let Some(id) = map.get(JSON_REPAIR_SLUG) {
                    if let Ok(f) = paw_rs::PawFnBuilder::builder()
                        .config(config.clone())
                        .id(id)
                        .load().await
                    {
                        return Some(f);
                    }
                }
            }
        }

        use paw_rs::paw_core::{CompileRequest, PawClient};
        let client = PawClient::new(config);
        let req = CompileRequest::builder()
            .spec(JSON_REPAIR_SPEC)
            .slug(JSON_REPAIR_SLUG)
            .ephemeral(false)
            .build().ok()?;
        let program = client.compile(req).await.ok()?;
        let _ = client.download_paw(&program.id).await.ok()?;

        let mut map: HashMap<String, String> = std::fs::read_to_string(mapping_path)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or_default();
        map.insert(JSON_REPAIR_SLUG.to_string(), program.id.clone());
        let _ = std::fs::write(mapping_path, serde_json::to_string(&map).unwrap_or_default());

        paw_rs::PawFnBuilder::builder()
            .config(config.clone())
            .id(&program.id)
            .load().await.ok()
    })
}

use std::collections::HashMap;

use paw_rs::PawFnBuilder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_think_content_present() {
        let input = "before<think>reasoning here</think>after";
        let result = extract_think_content(input);
        assert_eq!(result, Some("reasoning here".into()));
    }

    #[test]
    fn test_extract_think_content_absent() {
        let input = "no think tags here";
        let result = extract_think_content(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_think_content_empty() {
        let input = "<think></think>";
        let result = extract_think_content(input);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_strip_think_single() {
        let input = "before<think>remove</think>after";
        let result = strip_think_block(input);
        assert_eq!(result, "beforeafter");
    }

    #[test]
    fn test_strip_think_multiple() {
        let input = "a<think>1</think>b<think>2</think>c";
        let result = strip_think_block(input);
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_strip_think_none() {
        let input = "plain text no tags";
        let result = strip_think_block(input);
        assert_eq!(result, "plain text no tags");
    }

    #[test]
    fn test_extract_json_array_simple() {
        let input = "some text [1, 2, 3] more text";
        let result = extract_json_array(input);
        assert_eq!(result, Some("[1, 2, 3]"));
    }

    #[test]
    fn test_extract_json_array_nested() {
        let input = "[[1,2],[3,4]]";
        let result = extract_json_array(input);
        assert_eq!(result, Some("[[1,2],[3,4]]"));
    }

    #[test]
    fn test_extract_json_array_none() {
        let input = "no brackets";
        let result = extract_json_array(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_array_only_open() {
        let input = "only [ open";
        let result = extract_json_array(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_split_response_with_think() {
        let input = "<think>reason</think>final content";
        let (think, body) = split_response(input);
        assert_eq!(think, Some("reason".into()));
        assert_eq!(body, "final content");
    }

    #[test]
    fn test_split_response_multiple_think() {
        let input = "<think>a</think>text<think>b</think>more";
        let (think, body) = split_response(input);
        assert_eq!(think, Some("a\n\nb".into()));
        assert_eq!(body, "textmore");
    }

    #[test]
    fn test_split_response_no_think() {
        let input = "just plain text";
        let (think, body) = split_response(input);
        assert!(think.is_none());
        assert_eq!(body, "just plain text");
    }

    #[test]
    fn test_split_response_self_closing_think() {
        let input = "<think>reason<think/>final content";
        let (think, body) = split_response(input);
        assert_eq!(think, Some("reason".into()));
        assert_eq!(body, "final content");
    }

    #[test]
    fn test_split_response_unclosed_think() {
        let input = "<think>unclosed thought";
        let (think, body) = split_response(input);
        assert_eq!(think, Some("unclosed thought".into()));
        assert_eq!(body, "");
    }

    #[test]
    fn test_split_response_mixed_closing() {
        let input = "<think>a</think>text<think>b<think/>more";
        let (think, body) = split_response(input);
        assert_eq!(think, Some("a\n\nb".into()));
        assert_eq!(body, "textmore");
    }

    #[test]
    fn test_split_response_standalone_selfclose_not_think() {
        let input = "no think here<think/>still no think";
        let (think, body) = split_response(input);
        assert!(think.is_none());
        assert_eq!(body, "no think here<think/>still no think");
    }

    #[test]
    fn test_strip_think_self_closing() {
        let input = "before<think>remove<think/>after";
        let result = strip_think_block(input);
        assert_eq!(result, "beforeafter");
    }

    #[test]
    fn test_strip_think_unclosed() {
        let input = "a<think>incomplete";
        let result = strip_think_block(input);
        assert_eq!(result, "a");
    }

    #[test]
    fn test_extract_think_self_closing() {
        let input = "<think>reasoning<think/>body";
        let result = extract_think_content(input);
        assert_eq!(result, Some("reasoning".into()));
    }

    #[test]
    fn test_extract_think_unclosed() {
        let input = "<think>just thinking";
        let result = extract_think_content(input);
        assert_eq!(result, Some("just thinking".into()));
    }

    #[test]
    fn test_extract_think_standalone_selfclose() {
        let input = "text<think/>more";
        let result = extract_think_content(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_robust_direct_clean_json() {
        let input = r#"[{"tag":["test"],"variant":{"Semantic":[{"concept_identifier":"x"}]},"priority":1}]"#;
        let result = robust_json_extract(input);
        assert_eq!(result.as_deref(), Some(input));
    }

    #[test]
    fn test_robust_with_surrounding_text() {
        let input = "here is the answer:\n[{\"tag\":[\"a\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"b\"}]},\"priority\":1}]\ndone";
        let result = robust_json_extract(input);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_robust_markdown_fence() {
        let input = "```json\n[{\"tag\":[\"test\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"x\"}]},\"priority\":1}]\n```";
        let result = robust_json_extract(input);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_robust_objects_no_array_wrapper() {
        let input = "{\"tag\":[\"t1\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"a\"}]},\"priority\":1}\n{\"tag\":[\"t2\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"b\"}]},\"priority\":2}";
        let result = robust_json_extract(input);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_robust_single_object_wrapped() {
        let input = "{\"tag\":[\"t1\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"a\"}]},\"priority\":1}";
        let result = robust_json_extract(input);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    #[ignore = "requires PAW service"]
    fn test_robust_failed_returns_none() {
        let input = "this is just plain text no json at all";
        let result = robust_json_extract(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_balanced_array_nested() {
        let input = r#"text [{"a": [1,2]}, {"b": [3]}] more text"#;
        let result = extract_balanced_array(input);
        assert_eq!(result.unwrap(), r#"[{"a": [1,2]}, {"b": [3]}]"#);
    }

    #[test]
    fn test_extract_balanced_array_unbalanced() {
        let input = r#"[{"a": "missing close"#;
        let result = extract_balanced_array(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_top_level_objects_multiple() {
        let input = r#"{"a":1}{"b":2}"#;
        let result = extract_top_level_objects(input);
        assert_eq!(result.unwrap(), r#"[{"a":1},{"b":2}]"#);
    }

    #[test]
    fn test_strip_markdown_fences_basic() {
        let input = "```json\n[1,2,3]\n```";
        let result = strip_markdown_fences(input);
        assert_eq!(result, "[1,2,3]");
    }

    #[test]
    fn test_strip_markdown_fences_no_fence() {
        let input = "[1,2,3]";
        let result = strip_markdown_fences(input);
        assert_eq!(result, "[1,2,3]");
    }
}
