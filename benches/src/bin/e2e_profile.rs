//! hotpath 检索链路剖析驱动（多拓扑合成图）。
//!
//! 不关注检索正确性——本工具只负责把**不同拓扑结构 + 规模**的合成记忆图灌入
//! 工作记忆，逐查询跑 `RetrDefaultPipeline`，用 hotpath 观测链路各阶段的
//! 函数级耗时分布（插桩在 soul-mem-algo 内：similarity/association/ppr/
//! bayes_action/merge 等，需 `--features hotpath` 联动开启）。
//!
//! 用法：
//!   cargo run -p soul-mem-benches --features hotpath --bin e2e_profile -- [nodes] [queries] [topo] [param]
//!
//!   topo: ring(默认) | star | dense | cluster | scale
//!   param: dense→每节点出度(默认 8)；cluster→社区数(默认 8)；scale→每节点附边数(默认 3)；其余忽略
//!
//! 语义记忆按 32 个正交"概念块"分布（确定性基向量，查询扫块），每 64 节点
//! 挂一个触发动作节点（TrigToAction），动作推理腿在所有拓扑下都保持活跃。

use std::hint::black_box;
use std::sync::Arc;

use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_algo::algo::retrieve::complex::{
    AssociateWithActionConfig, DefaultPipelineConfig, RetrDefaultPipeline,
};
use soul_mem_algo::algo::retrieve::short_only::ShortOnlyConfig;
use soul_mem_algo::algo::retrieve::similarity::SimilarityConfig;
use soul_mem_core::memory_links::proc_mem::{ProcMemLink, TrigToAction};
use soul_mem_core::memory_links::sem_mem::SemMemLink;
use soul_mem_core::memory_links::{MemoryLink, MemoryLinkType};
use soul_mem_core::memory_note::proc_mem::{Action, ActionType, ProcMemory};
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::{MemoryId, MemoryNoteBuilder, MemoryType};
use soul_mem_query::embedding::EmbeddingVec;
use soul_mem_query::embedding::note::{
    EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
};
use soul_mem_query::embedding::query::note::{
    EmbeddedMemoryRetrieveQuery, MemoryRetrieveQueryEmbedding,
};
use soul_mem_query::embedding::sem::SemanticEmbedding;
use soul_mem_query::query::retrieve::{MemoryRetrieveQuery, MemoryRetrieveQueryVariant};
use soul_mem_runtime::working_memory::WorkingMemory;

const EMB_DIM: usize = 512;
const BLOCKS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Topo {
    Ring,
    Star,
    Dense { degree: usize },
    Cluster { communities: usize },
    Scale { m: usize },
}

impl Topo {
    fn name(&self) -> &'static str {
        match self {
            Topo::Ring => "ring",
            Topo::Star => "star",
            Topo::Dense { .. } => "dense",
            Topo::Cluster { .. } => "cluster",
            Topo::Scale { .. } => "scale",
        }
    }

    fn parse(name: &str, param: usize) -> Self {
        match name {
            "star" => Topo::Star,
            "dense" => Topo::Dense {
                degree: param.max(2),
            },
            "cluster" => Topo::Cluster {
                communities: param.max(2),
            },
            "scale" => Topo::Scale { m: param.max(1) },
            _ => Topo::Ring,
        }
    }
}

/// 确定性 LCG（无 rand 依赖）。
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// 节点 i 的出边目标索引列表（依拓扑生成；权重统一 0.7，桥接 0.3）。
fn out_neighbors(i: usize, n: usize, topo: Topo, state: &mut u64) -> Vec<(usize, f32)> {
    if n <= 1 {
        return Vec::new();
    }
    match topo {
        Topo::Ring => vec![((i + 1) % n, 0.7)],
        Topo::Star => {
            if i == 0 {
                (1..n).map(|j| (j, 0.7)).collect()
            } else {
                vec![(0, 0.7)]
            }
        }
        Topo::Dense { degree } => (1..=degree).map(|d| (((i + d) % n), 0.7)).collect(),
        Topo::Cluster { communities } => {
            let gs = (n / communities).max(1);
            let group = i / gs;
            // 组内环：仅在 i+1 仍属于本组且未越界时连到 i+1（尾组可能不满）
            let in_group_next = if i + 1 >= n || (i + 1).is_multiple_of(gs) {
                None
            } else {
                Some((i + 1, 0.7))
            };
            // 组桥：每组首节点连到下一组首节点（环形）
            let bridge = if i.is_multiple_of(gs) {
                Some(((((group + 1) % communities) * gs) % n, 0.3))
            } else {
                None
            };
            in_group_next.into_iter().chain(bridge).collect()
        }
        Topo::Scale { m } => {
            if i == 0 {
                return Vec::new();
            }
            // 偏向早期节点的"伪 BA"：r^2 偏斜制造少数高入度 hub
            let mut picks = Vec::with_capacity(m);
            let mut tries = 0;
            while picks.len() < m && tries < m * 8 {
                tries += 1;
                let r = (lcg(state) as usize) % i;
                let t = (r * r) % i;
                if !picks.iter().any(|(x, _)| *x == t) {
                    picks.push((t, 0.7));
                }
            }
            picks
        }
    }
}

fn zero_vec() -> Vec<f32> {
    vec![0.0f32; EMB_DIM]
}

/// 块 `block` 的确定性 3 维基向量（查询与记忆共用，保证相似度有区分度）。
fn block_basis(block: usize) -> Vec<f32> {
    let mut v = zero_vec();
    let base = (block % BLOCKS) * 4;
    for d in 0..3 {
        v[(base + d) % EMB_DIM] = 1.0;
    }
    v
}

fn sem_link(from: MemoryId, to: MemoryId, label: String, intensity: f32) -> MemoryLink {
    MemoryLink::new(
        from,
        to,
        MemoryLinkType::Sem(SemMemLink::new(label, intensity)),
    )
}

fn sem_note(
    id: MemoryId,
    content: String,
    links: Vec<MemoryLink>,
    block: usize,
) -> EmbeddedMemoryNote {
    let note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory {
        content,
        aliases: vec![],
        concept_type: ConceptType::Entity,
        description: String::new(),
    }))
    .id(id)
    .mem_links(links)
    .build()
    .expect("note build");
    let embedding = MemoryEmbedding::new(
        EmbeddingVec::new(block_basis(block)),
        MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
            EmbeddingVec::new(zero_vec()),
            EmbeddingVec::new(zero_vec()),
            EmbeddingVec::new(zero_vec()),
        )),
    );
    EmbeddedMemoryNote { note, embedding }
}

/// 合成记忆图：`nodes` 个语义记忆 + 每 64 个挂一个触发动作节点。
fn build_working_memory(nodes: usize, topo: Topo) -> (Arc<WorkingMemory>, usize) {
    let wm = WorkingMemory::new(64);
    let ids: Vec<MemoryId> = (0..nodes).map(|_| MemoryId::new()).collect();
    let mut edge_count = 0usize;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let cluster = wm.memory_cluster();
    cluster.write(|c| {
        for i in 0..nodes {
            let action_id = if i.is_multiple_of(64) {
                Some(MemoryId::new())
            } else {
                None
            };

            let mut links: Vec<MemoryLink> = out_neighbors(i, nodes, topo, &mut state)
                .into_iter()
                .map(|(t, w)| {
                    edge_count += 1;
                    sem_link(ids[i], ids[t], format!("rel_{i}->{t}"), w)
                })
                .collect();
            if let Some(aid) = action_id {
                edge_count += 1;
                links.push(MemoryLink::new(
                    ids[i],
                    aid,
                    MemoryLinkType::Proc(ProcMemLink::TrigToAction(TrigToAction::new(0.5))),
                ));
            }

            c.add_single_node(sem_note(ids[i], format!("记忆内容 {}", i), links, i));

            if let Some(aid) = action_id {
                let action_note = MemoryNoteBuilder::new(MemoryType::Procedure(ProcMemory::new(
                    Action::new(format!("Action_{i}"), ActionType::new_speak()),
                )))
                .id(aid)
                .build()
                .expect("action note build");
                let action_embedding = MemoryEmbedding::new(
                    EmbeddingVec::new(zero_vec()),
                    MemoryEmbeddingVariant::Procedure(),
                );
                c.add_single_node(EmbeddedMemoryNote {
                    note: action_note,
                    embedding: action_embedding,
                });
            }
        }
    });
    (Arc::new(wm), edge_count)
}

fn pipeline_config() -> DefaultPipelineConfig {
    DefaultPipelineConfig {
        short_mem_with_history: ShortOnlyConfig {
            clipping_length: None,
            include_summary: true,
        },
        similarity: SimilarityConfig {
            similarity_threshold: 0.0,
            max_results: 64,
        },
        assoc_with_action: AssociateWithActionConfig {
            association: Default::default(),
            action_top_k: 3,
            ..Default::default()
        },
    }
}

#[hotpath::main]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let queries: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(150);
    let topo = Topo::parse(
        args.get(3).map(String::as_str).unwrap_or("ring"),
        args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0),
    );

    println!(
        "== e2e_profile: topo={} nodes={nodes} queries={queries} ==",
        topo.name()
    );

    let (wm, edges) = hotpath::measure_block!("1_build_graph", build_working_memory(nodes, topo));
    println!(
        "   graph: {nodes} sem nodes, {edges} edges, {} action nodes",
        nodes / 64 + 1
    );
    let config = pipeline_config();

    let mut total_hits: usize = 0;
    for j in 0..queries {
        let query_embedding =
            MemoryRetrieveQueryEmbedding::new(EmbeddingVec::new(block_basis(j * 7)));
        let embedded_query = EmbeddedMemoryRetrieveQuery {
            embedding: query_embedding,
            query: MemoryRetrieveQuery::new(
                vec![format!("profile_query_{j}")],
                MemoryRetrieveQueryVariant::Semantic(vec![]),
            ),
        };
        let request = config
            .clone()
            .into_request(Arc::clone(&wm), embedded_query, 1);

        hotpath::measure_block!("2_pipeline_retrieve", {
            let result = RetrDefaultPipeline {}.retrieve(request);
            total_hits += black_box(result.association.len());
        });
    }
    black_box(total_hits);
    println!("== done: total_association_hits={total_hits} ==");
}
