use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::Deserialize;

use crate::engine::playtest::repair::{robust_json_extract, RawQuery, RawVariant};
use crate::engine::playtest::runner::QUERY_PROMPT;

use soul_mem_query::query::retrieve::{
    MemoryRetrieveQuery, MemoryRetrieveQueryVariant, PrioritizedMemoryRetrieveQuery,
    SemanticQueryUnit,
};

struct PawQueryState {
    rt: tokio::runtime::Runtime,
    inner: Mutex<Option<Box<dyn paw_rs::paw_core::PawFnTrait>>>,
    mapping_path: PathBuf,
    config: paw_rs::paw_core::PawConfig,
    spec_hash: u64,
}

static PAW_QUERY: OnceLock<PawQueryState> = OnceLock::new();

fn init_paw_query_state(
    question_path: &Path,
) -> PawQueryState {
    let rt = tokio::runtime::Runtime::new().expect("PAW query tokio runtime");
    let config = paw_rs::paw_core::PawConfig::from_env();
    let mapping_path = config.cache_dir().join("paw_query_id_mapping.json");

    let examples = load_examples(question_path);
    let spec = build_spec(&examples);
    let spec_hash = compute_hash(&spec);

    let paw_fn = rt.block_on(async {
        compile_or_load(&config, &mapping_path, &spec, spec_hash).await
    });

    PawQueryState {
        rt,
        inner: Mutex::new(paw_fn),
        mapping_path,
        config,
        spec_hash,
    }
}

pub fn paw_generate_queries(
    question_json_dir: &Path,
    _system_prompt: &str,
    user_message: &str,
) -> Option<(Vec<PrioritizedMemoryRetrieveQuery>, String)> {
    let question_path = question_json_dir.join("question.json");
    if !question_path.exists() {
        return None;
    }

    let state = PAW_QUERY.get_or_init(|| init_paw_query_state(&question_path));

    let examples = load_examples(&question_path);
    let spec = build_spec(&examples);
    let hash = compute_hash(&spec);
    if hash != state.spec_hash {
        let new_fn = state.rt.block_on(async {
            compile_or_load(&state.config, &state.mapping_path, &spec, hash).await
        });
        if let Ok(mut lock) = state.inner.lock() {
            *lock = new_fn;
        }
    }

    let mut lock = state.inner.lock().ok()?;
    let f = lock.as_mut()?;

    let prompt = format!("用户说: \"{}\"", user_message);
    let raw = f.run(&prompt).ok()?;

    let json = robust_json_extract(&raw)?;
    let raw_queries: Vec<RawQuery> = serde_json::from_str(&json).ok()?;

    Some((convert_raw_queries(raw_queries), json))
}

fn compute_hash(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

#[derive(Debug, Deserialize)]
struct QuestionFile {
    test_cases: Vec<TestCase>,
}

#[derive(Debug, Deserialize)]
struct TestCase {
    name: String,
    description: String,
    sub_queries: Vec<serde_json::Value>,
}

struct Example {
    message: String,
    json: serde_json::Value,
}

fn load_examples(question_path: &Path) -> Vec<Example> {
    let Ok(text) = std::fs::read_to_string(question_path) else {
        return vec![];
    };
    let Ok(file) = serde_json::from_str::<QuestionFile>(&text) else {
        return vec![];
    };

    let mut seen: HashSet<String> = HashSet::new();
    let mut examples: Vec<Example> = Vec::new();

    for case in &file.test_cases {
        let desc = case.description.trim().to_string();
        if desc.is_empty() || seen.contains(&desc) {
            continue;
        }
        seen.insert(desc.clone());

        let trimmed: Vec<&serde_json::Value> = case
            .sub_queries
            .iter()
            .filter(|q| {
                q.get("priority")
                    .and_then(|p| p.as_u64())
                    .unwrap_or(0)
                    >= 5
            })
            .collect();

        if trimmed.is_empty() {
            continue;
        }

        let json = serde_json::Value::Array(
            trimmed.into_iter().take(3).cloned().collect(),
        );

        examples.push(Example {
            message: desc,
            json: json.get(0).and_then(|v| v.as_object()).map(|_| json.clone()).unwrap_or_else(|| {
                let first = json.as_array().and_then(|a| a.first()).cloned();
                first.unwrap_or(json)
            }),
        });
    }

    examples
}

fn build_spec(examples: &[Example]) -> String {
    let mut spec = String::new();
    spec.push_str(QUERY_PROMPT);

    if !examples.is_empty() {
        spec.push_str("\n\n## 角色特有示例（来自实际记忆）\n");
        spec.push_str("以下示例展示该角色记忆检索查询的准确格式。\n\n");

        for ex in examples {
            let compact = serde_json::to_string(&ex.json).unwrap_or_default();
            spec.push_str(&format!(
                "用户说: \"{}\"\n输出: {}\n\n",
                ex.message, compact
            ));
        }
    }

    spec
}

async fn compile_or_load(
    config: &paw_rs::paw_core::PawConfig,
    mapping_path: &Path,
    spec: &str,
    hash: u64,
) -> Option<Box<dyn paw_rs::paw_core::PawFnTrait>> {
    let slug = format!("soul-tune-query-{:016x}", hash);

    // Try cache load
    if let Ok(data) = std::fs::read_to_string(mapping_path) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
            if let Some(id) = map.get(&slug) {
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

    // Compile new
    if config.api_key().is_none() {
        return None;
    }

    use paw_rs::paw_core::{CompileRequest, PawClient};
    let client = PawClient::new(config);
    let req = CompileRequest::builder()
        .spec(spec)
        .compiler("paw-4b-qwen3-0.6b")
        .slug(&slug)
        .ephemeral(false)
        .build()
        .ok()?;
    let program = client.compile(req).await.ok()?;
    let _ = client.download_paw(&program.id).await.ok()?;

    // Persist mapping
    let mut map: HashMap<String, String> =
        std::fs::read_to_string(mapping_path)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or_default();
    map.insert(slug, program.id.clone());
    let _ = std::fs::write(mapping_path, serde_json::to_string(&map).unwrap_or_default());

    paw_rs::PawFnBuilder::builder()
        .config(config.clone())
        .id(&program.id)
        .load()
        .await
        .ok()
}

fn convert_raw_queries(raw: Vec<RawQuery>) -> Vec<PrioritizedMemoryRetrieveQuery> {
    raw.into_iter()
        .map(|r| {
            let units: Vec<SemanticQueryUnit> = match r.variant {
                RawVariant::Semantic(units) | RawVariant::BareArray(units) => units
                    .into_iter()
                    .map(|u| {
                        SemanticQueryUnit::new()
                            .with_concept_identifier(u.concept_identifier.unwrap_or_default())
                    })
                    .collect(),
                RawVariant::SemanticSingle(unit) | RawVariant::BareSingle(unit) => {
                    vec![SemanticQueryUnit::new()
                        .with_concept_identifier(unit.concept_identifier.unwrap_or_default())]
                }
                RawVariant::Situation(units) => units
                    .into_iter()
                    .map(|u| {
                        SemanticQueryUnit::new()
                            .with_concept_identifier(u.narrative.unwrap_or_default())
                    })
                    .collect(),
            };
            let variant = MemoryRetrieveQueryVariant::Semantic(units);
            MemoryRetrieveQuery::new(r.tag, variant).with_priority(r.priority)
        })
        .collect()
}
