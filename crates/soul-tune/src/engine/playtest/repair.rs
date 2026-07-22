use serde::Deserialize;

pub fn strip_think_block(s: &str) -> String {
    let mut result = s.to_string();
    loop {
        let start = result.find("<think>");
        let end = start.and_then(|s| result[s..].find("</think>").map(|e| s + e));
        match (start, end) {
            (Some(s), Some(e)) => {
                result.replace_range(s..e + 8, "");
            }
            _ => break,
        }
    }
    result.trim().to_string()
}

pub fn extract_think_content(s: &str) -> Option<String> {
    let start = s.find("<think>")?;
    let remaining = &s[start + 7..];
    let end = remaining.find("</think>")?;
    Some(remaining[..end].trim().to_string())
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

pub fn split_response(s: &str) -> (Option<String>, String) {
    let mut think_parts: Vec<String> = Vec::new();
    let mut body = s.to_string();
    loop {
        let start = body.find("<think>");
        let end = start.and_then(|s| body[s..].find("</think>").map(|e| s + e));
        match (start, end) {
            (Some(s), Some(e)) => {
                think_parts.push(body[s + 7..e].trim().to_string());
                body.replace_range(s..e + 8, "");
            }
            _ => break,
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
    extract_json_array(&raw).map(|s| s.to_string())
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
}
