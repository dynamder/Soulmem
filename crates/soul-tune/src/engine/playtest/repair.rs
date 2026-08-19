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

pub(crate) fn extract_balanced_array(s: &str) -> Option<String> {
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

pub fn robust_json_extract(clean: &str, llm: &mut dyn LlmBackend) -> Option<String> {
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

    if let Some(j) = repair_json(&stripped, llm) {
        return Some(j);
    }

    if let Some(j) = repair_json(clean, llm) {
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

fn default_priority() -> u32 {
    5
}

#[derive(Debug, Deserialize)]
pub struct RawQuery {
    pub tag: Vec<String>,
    pub variant: RawVariant,
    #[serde(default = "default_priority")]
    pub priority: u32,
}

/// 解析 LLM 输出的 variant 字段。
/// 兼容三种形态：
///   - 显式包裹：{"Semantic": [...]} / {"Situation": [...]}（推荐，LLM 提示词使用）
///   - 裸数组：  [{...}, {...}]
///   - 裸单对象：{"concept_identifier": ...} / {"narrative": ...}
/// 注意：untagged 按顺序尝试，Semantic 包裹必须先于单对象变体，
/// 否则 {"Semantic": [...]} 会被 RawSemUnit 贪婪吞掉变成空单元（历史 bug）。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RawVariant {
    Semantic {
        Semantic: Vec<RawSemUnit>,
    },
    Situation {
        Situation: Vec<RawSitUnit>,
    },
    SemanticSingle(RawSemUnit),
    SituationSingle(RawSitUnit),
    BareArray(Vec<RawSemUnit>),
    BareSingle(RawSemUnit),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawSemUnit {
    #[serde(default)]
    pub concept_identifier: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawSitUnit {
    #[serde(default, alias = "narration")]
    pub narrative: Option<String>,
    #[serde(default)]
    pub location: Option<Vec<RawLocationUnit>>,
    #[serde(default)]
    pub participants: Option<Vec<RawParticipantUnit>>,
    #[serde(default)]
    pub environment: Option<RawEnvironmentUnit>,
    #[serde(default)]
    pub event: Option<Vec<RawEventUnit>>,
}

#[derive(Debug, Deserialize)]
pub struct RawLocationUnit {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub coordinates: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawParticipantUnit {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawEnvironmentUnit {
    #[serde(default)]
    pub atmosphere: Option<String>,
    #[serde(default)]
    pub tone: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawEventUnit {
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
  {"tag": ["personality"], "variant": {"Semantic": [{"concept_identifier": "traits", "description": "..."}]}, "priority": 0},
  {"tag": ["event", "recent"], "variant": {"Situation": [{"narrative": "...", "location": [{"name": "...", "coordinates": "..."}], "participants": [{"name": "...", "role": "..."}], "environment": {"atmosphere": "...", "tone": "..."}, "event": [{"action": "...", "initiator": "...", "target": "..."}]}]}, "priority": 1}
]

Each object MUST have: <tag> (string array), <variant> (either {"Semantic": [...]} or {"Situation": [...]}, each an array of objects), <priority> (integer).
Output ONLY the repaired JSON array. No markdown, no explanations."#;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// 运行一个 PAW 函数（按 slug 惰性加载并缓存），返回原始输出。
/// `max_tokens` 限制本次生成的 token 数；PAW candle 后端默认无限制，
/// 会让小模型一路生成到填满上下文（数分钟），必须显式限制。
/// PAW 服务不可用或加载/运行失败时返回 None。
pub(crate) fn run_paw(slug: &str, spec: &str, prompt: &str, max_tokens: Option<usize>) -> Option<String> {
    let state = init_paw_state();

    // 快路径：已缓存则直接运行
    {
        let mut lock = state.fns.lock().ok()?;
        if let Some(f) = lock.get_mut(slug) {
            return run_with_limit(f, prompt, max_tokens);
        }
    }

    // 慢路径：编译/下载并缓存后运行
    let loaded = state
        .rt
        .block_on(async { load_paw_fn(&state.config, &state.mapping_path, slug, spec).await });

    let mut lock = state.fns.lock().ok()?;
    let f = lock.entry(slug.to_string()).or_insert(loaded?);
    run_with_limit(f, prompt, max_tokens)
}

fn run_with_limit(
    f: &mut Box<dyn paw_rs::paw_core::PawFnTrait>,
    prompt: &str,
    max_tokens: Option<usize>,
) -> Option<String> {
    let opts = paw_rs::paw_core::PawRuntimeOptions {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
    };
    f.run_with(prompt, &opts).ok()
}

/// 修复畸形 JSON：PAW 优先（原始管线），不可用或产出无效时用主对话 LLM 顶上。
pub fn repair_json(bad_json: &str, llm: &mut dyn LlmBackend) -> Option<String> {
    let prompt = format!("Fix this JSON:\n{}\n\n---\nRepaired JSON:", bad_json);
    let try_parse = |raw: &str| {
        extract_balanced_array(raw).or_else(|| extract_json_array(raw).map(|s| s.to_string()))
    };
    if let Some(raw) = run_paw(JSON_REPAIR_SLUG, JSON_REPAIR_SPEC, &prompt, Some(1024)) {
        if let Some(j) = try_parse(&raw) {
            return Some(j);
        }
    }
    // PAW 不可用或产出无效：暂时用主对话 LLM 顶上
    let raw = llm.chat(JSON_REPAIR_SPEC, &prompt, 1024).ok()?;
    try_parse(&raw)
}

struct PawState {
    rt: tokio::runtime::Runtime,
    fns: Mutex<HashMap<String, Box<dyn paw_rs::paw_core::PawFnTrait>>>,
    mapping_path: PathBuf,
    config: paw_rs::paw_core::PawConfig,
}

static PAW: OnceLock<PawState> = OnceLock::new();

fn init_paw_state() -> &'static PawState {
    PAW.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().expect("PAW tokio runtime");
        let config = paw_rs::paw_core::PawConfig::from_env();
        let mapping_path = config.cache_dir().join("paw_id_mapping.json");
        PawState {
            rt,
            fns: Mutex::new(HashMap::new()),
            mapping_path,
            config,
        }
    })
}

/// 尝试从映射文件恢复已编译的 PAW，或编译并下载新函数。
async fn load_paw_fn(
    config: &paw_rs::paw_core::PawConfig,
    mapping_path: &Path,
    slug: &str,
    spec: &str,
) -> Option<Box<dyn paw_rs::paw_core::PawFnTrait>> {
    // 1. 优先从映射文件恢复已编译的 PAW
    if let Ok(data) = std::fs::read_to_string(mapping_path) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
            if let Some(id) = map.get(slug) {
                if let Ok(f) = paw_rs::PawFnBuilder::builder()
                    .config(config.clone())
                    .id(id)
                    .load()
                    .await
                {
                    return Some(f);
                }
            }
        }
    }

    // 2. 编译并下载
    use paw_rs::paw_core::{CompileRequest, PawClient};
    let client = PawClient::new(config);
    let req = CompileRequest::builder()
        .spec(spec.to_string())
        .slug(slug.to_string())
        .ephemeral(false)
        .build()
        .ok()?;
    let program = client.compile(req).await.ok()?;
    let _ = client.download_paw(&program.id).await.ok()?;

    // 3. 记录 slug -> id 映射
    let mut map: HashMap<String, String> = std::fs::read_to_string(mapping_path)
        .ok()
        .and_then(|d| serde_json::from_str(&d).ok())
        .unwrap_or_default();
    map.insert(slug.to_string(), program.id.clone());
    let _ = std::fs::write(mapping_path, serde_json::to_string(&map).unwrap_or_default());

    paw_rs::PawFnBuilder::builder()
        .config(config.clone())
        .id(&program.id)
        .load()
        .await
        .ok()
}

use crate::engine::llm::LlmBackend;

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用 LLM 存根：默认返回空串（修复路径取不到 JSON 时自然失败）。
    struct MockLlm;
    impl LlmBackend for MockLlm {
        fn chat(
            &mut self,
            _system: &str,
            _user_msg: &str,
            _max_tokens: u32,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

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
        let result = robust_json_extract(input, &mut MockLlm);
        assert_eq!(result.as_deref(), Some(input));
    }

    #[test]
    fn test_robust_with_surrounding_text() {
        let input = "here is the answer:\n[{\"tag\":[\"a\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"b\"}]},\"priority\":1}]\ndone";
        let result = robust_json_extract(input, &mut MockLlm);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_robust_markdown_fence() {
        let input = "```json\n[{\"tag\":[\"test\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"x\"}]},\"priority\":1}]\n```";
        let result = robust_json_extract(input, &mut MockLlm);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_robust_objects_no_array_wrapper() {
        let input = "{\"tag\":[\"t1\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"a\"}]},\"priority\":1}\n{\"tag\":[\"t2\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"b\"}]},\"priority\":2}";
        let result = robust_json_extract(input, &mut MockLlm);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_robust_single_object_wrapped() {
        let input = "{\"tag\":[\"t1\"],\"variant\":{\"Semantic\":[{\"concept_identifier\":\"a\"}]},\"priority\":1}";
        let result = robust_json_extract(input, &mut MockLlm);
        assert!(result.is_some());
        let parsed: Vec<RawQuery> = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    #[ignore = "requires PAW service"]
    fn test_robust_failed_returns_none() {
        let input = "this is just plain text no json at all";
        let result = robust_json_extract(input, &mut MockLlm);
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

    #[test]
    fn test_wrapped_semantic_preserves_concept() {
        let input =
            r#"[{"tag":["test"],"variant":{"Semantic":[{"concept_identifier":"弹幕规则","description":"desc"}]},"priority":1}]"#;
        let parsed: Vec<RawQuery> = serde_json::from_str(input).unwrap();
        match &parsed[0].variant {
            RawVariant::Semantic { Semantic: units } => {
                assert_eq!(units.len(), 1);
                assert_eq!(units[0].concept_identifier.as_deref(), Some("弹幕规则"));
                assert_eq!(units[0].description.as_deref(), Some("desc"));
            }
            other => panic!("expected wrapped Semantic, got {:?}", other),
        }
    }

    #[test]
    fn test_wrapped_situation_preserves_fields() {
        let input = r#"[{"tag":["sit"],"variant":{"Situation":[{"narrative":"在漫展","location":[{"name":"漫展"}],"event":[{"action":"逛展","initiator":"某人"}]}]},"priority":3}]"#;
        let parsed: Vec<RawQuery> = serde_json::from_str(input).unwrap();
        match &parsed[0].variant {
            RawVariant::Situation { Situation: units } => {
                assert_eq!(units.len(), 1);
                assert_eq!(units[0].narrative.as_deref(), Some("在漫展"));
                assert_eq!(units[0].location.as_ref().unwrap().len(), 1);
                assert_eq!(
                    units[0].event.as_ref().unwrap()[0].action.as_deref(),
                    Some("逛展")
                );
            }
            other => panic!("expected wrapped Situation, got {:?}", other),
        }
    }

    #[test]
    fn test_bare_array_and_single_still_parse() {
        let bare_array = r#"[{"tag":["t"],"variant":[{"concept_identifier":"x"}],"priority":1}]"#;
        let parsed: Vec<RawQuery> = serde_json::from_str(bare_array).unwrap();
        assert!(matches!(&parsed[0].variant, RawVariant::BareArray(_)));

        let bare_single = r#"[{"tag":["t"],"variant":{"concept_identifier":"y"},"priority":1}]"#;
        let parsed: Vec<RawQuery> = serde_json::from_str(bare_single).unwrap();
        match &parsed[0].variant {
            RawVariant::SemanticSingle(u) => {
                assert_eq!(u.concept_identifier.as_deref(), Some("y"));
            }
            other => panic!("expected SemanticSingle, got {:?}", other),
        }
    }
}
