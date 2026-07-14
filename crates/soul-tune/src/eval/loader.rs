use std::collections::HashMap;

use std::path::Path;

use serde::Deserialize;

use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::embedding_model::bge::BgeSmallZh;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;
use soul_mem_query::embedding::Embeddable;
use soul_mem_runtime::working_memory::WorkingMemory;

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

/// 从 graph JSON 加载并构建 WorkingMemory（自动执行 BGE embedding）
pub fn load_graph(
    path: &Path,
) -> Result<(WorkingMemory, HashMap<String, MemoryId>), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let raw_nodes: Vec<GraphNodeRaw> = serde_json::from_reader(reader)?;

    let mut id_map: HashMap<String, MemoryId> = HashMap::new();
    for raw in &raw_nodes {
        id_map.insert(raw.id.clone(), MemoryId::new());
    }

    let mut notes: Vec<(String, MemoryNoteBuilder)> = Vec::new();
    for raw in &raw_nodes {
        let mem_id = id_map[&raw.id];
        let links: Vec<MemoryLink> = raw
            .mem_links
            .iter()
            .map(|l| {
                let from = id_map.get(&l.from).copied().unwrap_or(mem_id);
                let to = id_map.get(&l.to).copied().unwrap_or(mem_id);
                MemoryLink::from_tuple(from, to, l.link_type.clone(), l.intensity)
            })
            .collect();
        let builder = MemoryNoteBuilder::new(raw.mem_type.clone())
            .id(mem_id)
            .tags(raw.tags.clone())
            .mem_links(links);
        notes.push((raw.id.clone(), builder));
    }

    let model = BgeSmallZh::default_cpu()?;
    let wm = WorkingMemory::new(10);
    let cluster = wm.memory_cluster();
    cluster.write(|c| {
        for (_raw_id, builder) in notes {
            let note = builder.build().expect("MemoryNoteBuilder failed");
            let embedding = note.embed(&model).expect("Embedding failed");
            c.add_single_node(EmbeddedMemoryNote { note, embedding });
        }
    });

    Ok((wm, id_map))
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
}
