use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::Embeddable;
use soul_mem_runtime::working_memory::WorkingMemory;

/// Download model files from HF mirror and place them in the correct cache
/// structure so hf-hub finds them locally and skips re-download.
fn prefill_mirror_cache() -> bool {
    use std::io::Write;
    use std::process::Command;

    let model_id = "BAAI/bge-small-zh-v1.5";
    let files = ["config.json", "tokenizer.json", "model.safetensors"];

    let cache_root = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            Path::new(&home).join(".cache").join("huggingface")
        });
    let hub_dir = cache_root.join("hub");
    let model_tag = model_id.replace('/', "--");
    let model_dir = hub_dir.join(format!("models--{}", model_tag));
    let snapshots_dir = model_dir.join("snapshots");

    // First, discover the commit hash via a HEAD request (extract from redirect Location)
    let test_url = format!(
        "https://hf-mirror.com/{}/resolve/main/config.json",
        model_id
    );
    let commit_hash = match Command::new("curl").args(["-sI", &test_url]).output() {
        Ok(out) => {
            let raw = String::from_utf8_lossy(&out.stdout);
            // Extract Location header: Location: /api/resolve-cache/.../{hash}/...
            let loc = raw
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("location:"))
                .unwrap_or("")
                .trim();
            // Location: /api/resolve-cache/models/{org}/{model}/{commit_hash}/config.json?...
            // split: [0]"" [1]"api" [2]"resolve-cache" [3]"models" [4]"{org}" [5]"{model}" [6]"{commit_hash} [7]"config.json?..."
            let hash = loc.split('/').nth(6).unwrap_or("").to_string();
            if hash.len() >= 10 {
                hash
            } else {
                String::new()
            }
        }
        Err(_) => return false,
    };
    if commit_hash.is_empty() || commit_hash.len() < 10 {
        return false;
    }

    eprintln!("  镜像站 commit: {}", commit_hash);

    // Write refs/main
    let refs_dir = model_dir.join("refs");
    let _ = std::fs::create_dir_all(&refs_dir);
    if let Ok(mut f) = std::fs::File::create(refs_dir.join("main")) {
        // hf-hub reads refs/main as raw string, no trailing whitespace
        let _ = write!(f, "{}", commit_hash);
    }

    // Download each file into snapshots/{commit_hash}/
    let snap_dir = snapshots_dir.join(&commit_hash);
    let _ = std::fs::create_dir_all(&snap_dir);

    let mut ok = true;
    for file in &files {
        let url = format!("https://hf-mirror.com/{}/resolve/main/{}", model_id, file);
        let target = snap_dir.join(file);
        if target.exists() {
            eprintln!("  {} 已缓存，跳过", file);
            continue;
        }
        let status = Command::new("curl")
            .args(["-fsSL", "-o", target.to_str().unwrap(), &url])
            .status();
        match status {
            Ok(s) if s.success() => eprintln!("  下载 {} 成功", file),
            _ => {
                eprintln!("  下载 {} 失败", file);
                let _ = std::fs::remove_file(&target);
                ok = false;
            }
        }
    }
    ok
}

pub fn get_bge_model() -> &'static BgeSmallZh {
    static MODEL: OnceLock<BgeSmallZh> = OnceLock::new();
    MODEL.get_or_init(|| match BgeSmallZh::default_cpu() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("直连 huggingface.co 失败: {e}");
            eprintln!("尝试从 HF 镜像站 hf-mirror.com 预下载模型文件...");
            if prefill_mirror_cache() {
                eprintln!("预下载完成，重试初始化...");
                BgeSmallZh::default_cpu().expect("预下载后模型初始化仍然失败")
            } else {
                panic!("所有下载方式均失败")
            }
        }
    })
}

/// Direct-deserialization types (matches core types exactly).
/// These work for hand-written fixtures but may fail on batch-generated data
/// that has JSON quirks (null time_span, missing enum wrappers).
#[derive(Debug, Deserialize)]
pub struct GraphLinkRaw {
    pub from: String,
    pub to: String,
    pub intensity: f64,
    pub link_type: MemoryLinkType,
}

#[derive(Debug, Deserialize)]
pub struct GraphNodeRaw {
    pub id: String,
    pub tags: Vec<String>,
    pub mem_type: MemoryType,
    #[serde(default)]
    pub mem_links: Vec<GraphLinkRaw>,
}

/// Lenient fixture node — uses serde_json::Value for problematic enum fields
/// so we can fix fixture-specific JSON quirks before deserializing into core types.
#[derive(Debug, Deserialize)]
struct FixtureNode {
    pub id: String,
    pub tags: Vec<String>,
    pub mem_type: Value,
    #[serde(default)]
    pub mem_links: Vec<FixtureLink>,
}

#[derive(Debug, Deserialize)]
struct FixtureLink {
    pub from: String,
    pub to: String,
    pub intensity: f64,
    pub link_type: Value,
}

/// Fix `"time_span": null` → a default ISO-8601 string.
/// Batch-generated fixtures often have null time_span in SpecificSituation,
/// but core type `DateTime<Utc>` cannot deserialize null.
fn fix_mem_type(value: &mut Value) {
    if let Value::Object(obj) = value {
        if let Some(sit) = obj.get_mut("Situation") {
            if let Some(spec) = sit.get_mut("SpecificSituation") {
                if let Value::Object(fields) = spec {
                    if let Some(Value::Null) = fields.get("time_span") {
                        fields.insert(
                            "time_span".into(),
                            Value::String("1970-01-01T00:00:00Z".into()),
                        );
                    }
                    if let Some(ctx) = fields.get_mut("context") {
                        if let Value::Object(ctx_obj) = ctx {
                            if let Some(Value::Null) = ctx_obj.get("environment") {
                                ctx_obj.insert(
                                    "environment".into(),
                                    serde_json::json!({
                                        "atmosphere": "",
                                        "tone": ""
                                    }),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Fix fixture link_type format to match serde external-tagging expectations.
/// Fixtures:   `{"Proc": {"prob": 0.8}}`
/// Serde:      `{"Proc": {"TrigToAction": {"prob": 0.8}}}`
/// Same for Situation links: `{"Situation": {}}` → `{"Situation": {"AbstractToSpecific": {}}}`
fn fix_link_type(value: &mut Value) {
    if let Value::Object(obj) = value {
        if let Some(proc_val) = obj.get_mut("Proc") {
            let already_wrapped = proc_val
                .as_object()
                .map_or(false, |m| m.contains_key("TrigToAction"));
            if !already_wrapped {
                *proc_val = serde_json::json!({"TrigToAction": proc_val.take()});
            }
        }
        if let Some(sit_val) = obj.get_mut("Situation") {
            let already_wrapped = sit_val
                .as_object()
                .map_or(false, |m| m.contains_key("AbstractToSpecific"));
            if !already_wrapped {
                *sit_val = serde_json::json!({"AbstractToSpecific": sit_val.take()});
            }
        }
    }
}

/// 从 graph JSON 加载并构建 WorkingMemory（自动执行 BGE embedding）
/// 自动修复 fixture 数据中的常见 JSON 问题：
/// - null time_span → 默认时间
/// - 缺失的 enum 包裹层 → 补全
pub fn load_graph(
    path: &Path,
) -> Result<(WorkingMemory, HashMap<String, MemoryId>), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut raw_nodes: Vec<FixtureNode> = serde_json::from_reader(reader)?;

    // Fix fixture data quirks before deserializing into core types
    for node in &mut raw_nodes {
        fix_mem_type(&mut node.mem_type);
        for link in &mut node.mem_links {
            fix_link_type(&mut link.link_type);
        }
    }

    let mut id_map: HashMap<String, MemoryId> = HashMap::new();
    for raw in &raw_nodes {
        id_map.insert(raw.id.clone(), MemoryId::new());
    }

    let mut notes: Vec<(String, MemoryNoteBuilder)> = Vec::new();
    for raw in &raw_nodes {
        let mem_id = id_map[&raw.id];

        let mem_type: MemoryType = serde_json::from_value(raw.mem_type.clone())?;
        let mut links = Vec::new();
        for l in &raw.mem_links {
            let link_type: MemoryLinkType = serde_json::from_value(l.link_type.clone())?;
            let from = id_map.get(&l.from).copied().unwrap_or(mem_id);
            let to = id_map.get(&l.to).copied().unwrap_or(mem_id);
            links.push(MemoryLink::from_tuple(from, to, link_type, l.intensity));
        }

        let builder = MemoryNoteBuilder::new(mem_type)
            .id(mem_id)
            .tags(raw.tags.clone())
            .mem_links(links);
        notes.push((raw.id.clone(), builder));
    }

    let model = get_bge_model();
    let wm = WorkingMemory::new(10);
    let cluster = wm.memory_cluster();
    cluster.write(|c| {
        for (_raw_id, builder) in notes {
            let note = builder.build().expect("MemoryNoteBuilder failed");
            let embedding = note.embed(model).expect("Embedding failed");
            c.add_single_node(EmbeddedMemoryNote { note, embedding });
        }
    });

    Ok((wm, id_map))
}

#[derive(Serialize, Deserialize)]
struct EmbeddingCache {
    id_map: HashMap<String, MemoryId>,
    notes: Vec<EmbeddedMemoryNote>,
}

fn cache_path(graph_path: &Path) -> PathBuf {
    let mut p = graph_path.to_path_buf();
    let ext = p
        .extension()
        .map(|e| format!("{}.embcache", e.to_string_lossy()))
        .unwrap_or_else(|| "embcache".into());
    p.set_extension(ext);
    p
}

pub fn cached_load_graph(
    path: &Path,
) -> Result<(WorkingMemory, HashMap<String, MemoryId>), Box<dyn std::error::Error>> {
    let cp = cache_path(path);
    if cp.exists() {
        let file = std::fs::File::open(&cp)?;
        let reader = std::io::BufReader::new(file);
        if let Ok(cache) = serde_json::from_reader::<_, EmbeddingCache>(reader) {
            let wm = WorkingMemory::new(10);
            let cluster = wm.memory_cluster();
            cluster.write(|c| {
                for note in cache.notes {
                    c.add_single_node(note);
                }
            });
            return Ok((wm, cache.id_map));
        }
    }

    let (wm, id_map) = load_graph(path)?;

    // Write cache
    let notes: Vec<EmbeddedMemoryNote> = wm
        .memory_cluster()
        .read_or_compute(|c| c.graph().node_weights().map(|n| n.clone()).collect());

    if let Ok(file) = std::fs::File::create(&cp) {
        let writer = std::io::BufWriter::new(file);
        let _ = serde_json::to_writer(
            writer,
            &EmbeddingCache {
                id_map: id_map.clone(),
                notes,
            },
        );
    }

    Ok((wm, id_map))
}

pub fn clear_embedding_cache(dir: &Path) -> usize {
    let mut count = 0;
    if !dir.is_dir() {
        return count;
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().map(|e| e == "embcache").unwrap_or(false) {
                    let _ = std::fs::remove_file(&p);
                    count += 1;
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node_raw_deserialize() {
        let json = r#"
        [
          {
            "id": "mem_a",
            "tags": ["rust", "编程"],
            "mem_type": {
              "Semantic": {
                "content": "Rust",
                "aliases": [],
                "concept_type": "Entity",
                "description": "系统编程语言"
              }
            },
            "mem_links": []
          },
          {
            "id": "mem_b",
            "tags": ["python"],
            "mem_type": {
              "Semantic": {
                "content": "Python",
                "aliases": [],
                "concept_type": "Entity",
                "description": "脚本语言"
              }
            },
            "mem_links": [
              {
                "from": "mem_a",
                "to": "mem_b",
                "intensity": 0.8,
                "link_type": {
                  "Sem": { "verb": "related", "confidence": 0.7 }
                }
              }
            ]
          }
        ]
        "#;
        let nodes: Vec<GraphNodeRaw> = serde_json::from_str(json).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, "mem_a");
        assert_eq!(nodes[1].mem_links.len(), 1);
    }

    #[test]
    fn test_fix_mem_type_null_time_span() {
        let mut val = serde_json::json!({
            "Situation": {
                "SpecificSituation": {
                    "narrative": "test",
                    "time_span": null,
                    "context": {
                        "location": null,
                        "participants": [],
                        "emotions": [],
                        "sensory_data": [],
                        "environment": { "atmosphere": "", "tone": "" },
                        "event": []
                    }
                }
            }
        });
        fix_mem_type(&mut val);
        let obj = val["Situation"]["SpecificSituation"].as_object().unwrap();
        assert_ne!(obj.get("time_span").and_then(|v| v.as_str()), Some("null"));
        assert!(obj.get("time_span").and_then(|v| v.as_str()).is_some());
    }

    #[test]
    fn test_fix_link_type_proc() {
        let mut val = serde_json::json!({"Proc": {"prob": 0.8}});
        fix_link_type(&mut val);
        assert_eq!(
            val,
            serde_json::json!({"Proc": {"TrigToAction": {"prob": 0.8}}})
        );
    }

    #[test]
    fn test_fix_link_type_sem_unaffected() {
        let mut val = serde_json::json!({"Sem": {"verb": "朋友", "confidence": 0.9}});
        fix_link_type(&mut val);
        assert_eq!(
            val,
            serde_json::json!({"Sem": {"verb": "朋友", "confidence": 0.9}})
        );
    }

    #[test]
    fn test_fix_link_type_already_wrapped() {
        let mut val = serde_json::json!({"Proc": {"TrigToAction": {"prob": 0.8}}});
        fix_link_type(&mut val);
        assert_eq!(
            val,
            serde_json::json!({"Proc": {"TrigToAction": {"prob": 0.8}}})
        );
    }

    #[test]
    fn test_fix_mem_type_null_environment() {
        let mut val = serde_json::json!({
            "Situation": {
                "SpecificSituation": {
                    "narrative": "test",
                    "time_span": null,
                    "context": {
                        "location": null,
                        "participants": [],
                        "emotions": [],
                        "sensory_data": [],
                        "environment": null,
                        "event": []
                    }
                }
            }
        });
        fix_mem_type(&mut val);
        let ctx = &val["Situation"]["SpecificSituation"]["context"];
        assert_eq!(
            ctx["environment"],
            serde_json::json!({"atmosphere": "", "tone": ""})
        );
    }

    #[test]
    fn test_load_character_graph_with_null_env() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures/example_data/格蕾修_https_zh_moegirl_org_cn_E6_A0_BC_E8_95_BE_E4_BF_AE/graph.json");
        let (_wm, id_map) =
            load_graph(&path).expect("格蕾修 graph with null environment should load");
        assert!(!id_map.is_empty());
    }

    #[test]
    fn test_fix_mem_type_semantic_unaffected() {
        let mut val = serde_json::json!({
            "Semantic": {
                "content": "Rust",
                "aliases": [],
                "concept_type": "Entity",
                "description": "desc"
            }
        });
        fix_mem_type(&mut val);
        assert_eq!(val["Semantic"]["content"], serde_json::json!("Rust"));
    }
}
