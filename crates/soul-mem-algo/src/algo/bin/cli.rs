use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use soul_mem_algo::algo::consolidate::generate::{MemorySpec, generate_memories_from_summary};
use soul_mem_algo::algo::consolidate::link::generate_links;
use soul_mem_algo::algo::consolidate::select::{ActiveNode, SelectConfig, select_active_nodes};
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::{MemoryId, MemoryNote, MemoryNoteBuilder};
use soul_mem_core::render::Render;
use soul_mem_llm::{LlmEngine, OaiCompatBackend, OaiCompatConfig};
use soul_mem_query::embedding::{Embeddable, EmbeddingGenResult, EmbeddingModel, EmbeddingVec};
use soul_mem_runtime::working_memory::WorkingMemory;
use soul_mem_runtime::working_memory::sliding_window::Summary;

const EMBED_DIM: usize = 512;
const TOP_K: usize = 8;
const SUMMARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/algo/bin/summary.in");
const OLD_NODES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/algo/bin/old_nodes.json");

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 巩固测试开始 ==");

    let summary_text =
        fs::read_to_string(SUMMARY).with_context(|| format!("读取摘要文件失败：{SUMMARY}"))?;
    if summary_text.trim().is_empty() {
        bail!("摘要为空：请写入 {SUMMARY}");
    }

    println!("本次巩固所依据的摘要：");
    println!("{summary_text}");

    let engine = build_engine()?;

    let mut summary = Summary::new();
    summary.update(summary_text.trim());
    let new_notes = generate_memories_from_summary(&summary, None, &engine)
        .await
        .context("从摘要生成记忆节点失败")?;

    println!("== 新节点 ({}) ==", new_notes.len());
    print_notes(&new_notes);
    if new_notes.is_empty() {
        println!("摘要没有拆出任何节点，结束。");
        return Ok(());
    }

    let wm = seed_working_memory()?;
    let active = select_active(&wm);
    print_active(&active, &wm);

    let links = generate_links(&wm, &active, &new_notes, &engine)
        .await
        .context("生成连接失败")?;
    print_links(&links, &active, &new_notes);
    println!("== 巩固运行完成 ==");
    Ok(())
}

fn build_engine() -> Result<LlmEngine> {
    dotenvy::dotenv().ok();

    let base_url = required_env("SOULMEM_API_BASE")?;
    let model = required_env("SOULMEM_MODEL")?;
    let config = OaiCompatConfig::new("consolidate-cli", base_url, model);
    let config = match optional_env("SOULMEM_API_KEY") {
        Some(key) => config.with_api_key(key),
        None => config.with_auth_header(None, None),
    };

    let backend = OaiCompatBackend::new(config).context("构建 LLM 后端失败")?;
    Ok(LlmEngine::new(Arc::new(backend)))
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("缺少环境变量 {name}"))
}

fn optional_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[derive(Deserialize)]
struct OldNodeSpec {
    id: Option<MemoryId>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    hits: u32,
    retrievals: Option<usize>,
    #[serde(flatten)]
    memory: MemorySpec,
}

impl OldNodeSpec {
    fn into_note(self) -> Result<MemoryNote> {
        let note = MemoryNote::from(self.memory);
        MemoryNoteBuilder::new(note.mem_type().clone())
            .id(self.id.unwrap_or_else(|| note.id()))
            .tags(self.tags)
            .create_time(note.creation_time())
            .last_accessed_time(note.last_accessed_time())
            .build()
            .context("构建旧节点失败")
    }
}

fn seed_working_memory() -> Result<WorkingMemory> {
    let raw = fs::read_to_string(OLD_NODES)
        .with_context(|| format!("读取旧节点文件失败：{OLD_NODES}"))?;
    let specs: Vec<OldNodeSpec> =
        serde_json::from_str(&raw).with_context(|| format!("解析旧节点文件失败：{OLD_NODES}"))?;

    let wm = WorkingMemory::new(20);
    for spec in specs {
        let hits = spec.hits;
        let retrievals = spec.retrievals.unwrap_or(hits as usize).max(hits as usize);
        let note = spec.into_note()?;
        let id = note.id();

        let embedded = note
            .embed_and_fuse(&ZeroEmbedding)
            .context("嵌入旧节点失败")?;
        wm.add_node(embedded);
        for _ in 0..hits {
            wm.record_retrieval(id);
        }
        let extra = retrievals - hits as usize;
        if extra > 0 {
            wm.write_records(|records| {
                if let Some(record) = records.get_mut(&id) {
                    for _ in 0..extra {
                        record.record_retrieval();
                    }
                }
            });
        }
    }
    Ok(wm)
}

fn select_active(wm: &WorkingMemory) -> Vec<ActiveNode> {
    let config = SelectConfig {
        top_k: TOP_K,
        min_cycle_hits: 1,
    };
    wm.read_records(|records| {
        wm.read_cycle_hits(|cycle_hits| select_active_nodes(records, cycle_hits, &config))
    })
}

fn print_notes(notes: &[MemoryNote]) {
    for (index, note) in notes.iter().enumerate() {
        let rendered = note.render(0.0);
        println!("  [N{}] {}  id={}", index + 1, rendered.label(), note.id());
        println!("      content: {}", rendered.content);
        if !note.tags().is_empty() {
            println!("      tags: {}", note.tags().join(", "));
        }
        if let Some(context) = &rendered.context
            && let Ok(text) = serde_json::to_string_pretty(context)
        {
            for line in text.lines() {
                println!("      {line}");
            }
        }
    }
}

fn print_active(active: &[ActiveNode], wm: &WorkingMemory) {
    println!("\n== 活跃旧节点 ({}) ==", active.len());
    if active.is_empty() {
        println!("  没有活跃节点：检查旧节点文件的 hits，否则新节点无处可连。");
        return;
    }
    for (index, node) in active.iter().enumerate() {
        let preview = wm.memory_cluster().read_or_compute(|cluster| {
            cluster
                .get_node(node.id)
                .map(|embedded| embedded.note().render(0.0).content)
                .unwrap_or_default()
        });
        println!(
            "  [A{}] hits={} total={}  {}",
            index + 1,
            node.cycle_hits,
            node.total_count,
            preview
        );
    }
}

fn print_links(links: &[MemoryLink], active: &[ActiveNode], new_notes: &[MemoryNote]) {
    println!("\n== 连接 ({}) ==", links.len());
    if links.is_empty() {
        println!("  LLM 没有返回有效边。");
        return;
    }

    let labels = labels(active, new_notes);
    for link in links {
        println!(
            "  {} -> {}  {}  intensity={}",
            label_of(&labels, link.from()),
            label_of(&labels, link.to()),
            format_link_type(link),
            link.intensity
        );
    }
}

fn labels(active: &[ActiveNode], new_notes: &[MemoryNote]) -> HashMap<MemoryId, String> {
    let mut labels = HashMap::new();
    for (index, node) in active.iter().enumerate() {
        labels.insert(node.id, format!("A{}", index + 1));
    }
    for (index, note) in new_notes.iter().enumerate() {
        labels.insert(note.id(), format!("N{}", index + 1));
    }
    labels
}

fn label_of(labels: &HashMap<MemoryId, String>, id: MemoryId) -> &str {
    labels.get(&id).map(String::as_str).unwrap_or("?")
}

fn format_link_type(link: &MemoryLink) -> String {
    match link.link_type() {
        MemoryLinkType::Sem(edge) => {
            format!("Sem(verb={:?}, confidence={})", edge.verb, edge.confidence)
        }
        other => format!("{other:?}"),
    }
}

struct ZeroEmbedding;

impl EmbeddingModel for ZeroEmbedding {
    fn infer_batch(&self, input: &[&str]) -> EmbeddingGenResult<Vec<EmbeddingVec>> {
        Ok(input
            .iter()
            .map(|_| EmbeddingVec::zero(EMBED_DIM))
            .collect())
    }

    fn infer_with_chunk(&self, _input: &str) -> EmbeddingGenResult<EmbeddingVec> {
        Ok(EmbeddingVec::zero(EMBED_DIM))
    }

    fn infer_and_fuse(&self, _input: &[&str]) -> EmbeddingGenResult<EmbeddingVec> {
        Ok(EmbeddingVec::zero(EMBED_DIM))
    }

    fn max_input_token(&self) -> usize {
        EMBED_DIM
    }

    fn dim(&self) -> usize {
        EMBED_DIM
    }
}
