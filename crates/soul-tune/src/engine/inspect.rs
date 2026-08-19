//! 数据集检视：从 graph.json / question.json 解析结构化条目（headless 与 GUI 共用）。
//!
//! 由原 TUI `states::inspect.rs` 的**纯数据解析部分**抽取而来（渲染/交互留在 GUI 侧），
//! 供 `soul-tune inspect` 命令行与 FRB 桥接复用。

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, PartialEq)]
pub enum InspectFileType {
    Graph,
    Query,
}

pub struct LinkDisplay {
    pub from_id: String,
    pub to_id: String,
    pub target_idx: usize,
    pub link_type_desc: String,
    pub intensity: f64,
    pub is_outgoing: bool,
}

pub struct InspectEntry {
    pub id: String,
    pub summary: String,
    pub preview_lines: Vec<String>,
    pub detail_lines: Vec<String>,
    pub links: Vec<LinkDisplay>,
}

/// headless 检视结果（GUI 用 JSON 树呈现原始数据，此结构供 CLI 打印）。
pub struct InspectData {
    pub file_path: PathBuf,
    pub file_type: InspectFileType,
    pub entries: Vec<InspectEntry>,
    pub stats: Option<Vec<String>>,
}

pub fn inspect_data(file_path: PathBuf) -> InspectData {
    let content = std::fs::read_to_string(&file_path).unwrap_or_default();
    let val: serde_json::Value = serde_json::from_str(&content).unwrap_or(serde_json::Value::Null);

    let (file_type, entries) = if val.is_array() {
        (InspectFileType::Graph, parse_graph_nodes(val.as_array().unwrap()))
    } else if let Some(_cases) = val.get("test_cases").and_then(|v| v.as_array()) {
        (InspectFileType::Query, parse_query_cases(&val))
    } else if let Some(nodes) = val.get("nodes").and_then(|v| v.as_array()) {
        (InspectFileType::Graph, parse_graph_nodes(nodes))
    } else {
        (InspectFileType::Query, parse_query_cases(&val))
    };

    let stats = if file_type == InspectFileType::Graph {
        let parent = file_path.parent().unwrap_or(std::path::Path::new(""));
        let stats_path = parent.join("graph_stats.json");
        if stats_path.exists() {
            load_graph_stats(&stats_path)
        } else {
            None
        }
    } else {
        None
    };

    InspectData {
        file_path,
        file_type,
        entries,
        stats,
    }
}

fn parse_graph_nodes(arr: &[serde_json::Value]) -> Vec<InspectEntry> {
    let mut entries: Vec<InspectEntry> = Vec::new();
    // First pass: collect all link references
    for node in arr {
        let id = node
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let tags: Vec<String> = node
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let mem_type = node.get("mem_type");
        let (type_label, preview_lines, detail_lines) = format_mem_type(mem_type, &tags);

        let entry = InspectEntry {
            id: id.clone(),
            summary: format!(
                "{} [{}]  {}",
                type_label,
                tags.join(","),
                preview_lines.first().unwrap_or(&String::new())
            ),
            preview_lines: preview_lines.clone(),
            detail_lines,
            links: Vec::new(),
        };
        entries.push(entry);
    }

    // Second pass: build link index from raw JSON to avoid borrowing entries
    let id_to_idx: HashMap<&str, usize> = arr
        .iter()
        .enumerate()
        .filter_map(|(i, node)| node.get("id").and_then(|v| v.as_str()).map(|id| (id, i)))
        .collect();

    for (i, node) in arr.iter().enumerate() {
        if let Some(links) = node.get("mem_links").and_then(|v| v.as_array()) {
            for link in links {
                let from = link.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let to = link.get("to").and_then(|v| v.as_str()).unwrap_or("");
                let intensity = link
                    .get("intensity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let link_type_desc = link
                    .get("link_type")
                    .map(format_link_type)
                    .unwrap_or_default();

                let from_idx = id_to_idx.get(from).copied().unwrap_or(i);
                let to_idx = id_to_idx.get(to).copied().unwrap_or(i);

                // Outgoing from `from`
                if from_idx < entries.len() {
                    entries[from_idx].links.push(LinkDisplay {
                        from_id: from.to_string(),
                        to_id: to.to_string(),
                        target_idx: to_idx,
                        link_type_desc: link_type_desc.clone(),
                        intensity,
                        is_outgoing: true,
                    });
                }
                // Incoming to `to`
                if to_idx < entries.len() && to_idx != from_idx {
                    entries[to_idx].links.push(LinkDisplay {
                        from_id: from.to_string(),
                        to_id: to.to_string(),
                        target_idx: from_idx,
                        link_type_desc,
                        intensity,
                        is_outgoing: false,
                    });
                }
            }
        }
    }

    // Sort links so outgoing come before incoming
    for entry in &mut entries {
        entry.links.sort_by_key(|l| !l.is_outgoing);
    }

    entries
}

fn parse_query_cases(val: &serde_json::Value) -> Vec<InspectEntry> {
    let cases = match val.get("test_cases").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };

    let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("?");
    let desc = val
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut entries: Vec<InspectEntry> = Vec::new();
    let config = val.get("config");

    for tc in cases {
        let case_name = tc
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let case_desc = tc
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let sub_queries = tc
            .get("sub_queries")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        let expected: Vec<String> = tc
            .get("expected_combined_ranking")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let expected_str = if expected.is_empty() {
            "(空)".to_string()
        } else {
            format!("[{}]", expected.join(", "))
        };

        let bonus: Vec<String> = tc
            .get("bonus_combined_ranking")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let bonus_str = if bonus.is_empty() {
            "(空)".to_string()
        } else {
            format!("[{}]", bonus.join(", "))
        };

        let preview_lines = vec![
            format!("描述: {}", case_desc),
            format!("子查询数: {}", sub_queries),
            format!("期望结果: {}", expected_str),
            format!("奖励结果: {}", bonus_str),
        ];

        let mut detail_lines = Vec::new();
        detail_lines.push(format!("数据集: {}", name));
        detail_lines.push(format!("描述: {}", desc));
        detail_lines.push(String::new());
        detail_lines.push(format!("用例名: {}", case_name));
        detail_lines.push(format!("用例描述: {}", case_desc));

        if let Some(cfg) = config {
            if let Some(thresh) = cfg.get("similarity_threshold").and_then(|v| v.as_f64()) {
                detail_lines.push(format!("相似度阈值: {:.2}", thresh));
            }
            if let Some(max_r) = cfg.get("max_results").and_then(|v| v.as_u64()) {
                detail_lines.push(format!("最大结果数: {}", max_r));
            }
            if let Some(k_vals) = cfg.get("test_k_values").and_then(|v| v.as_array()) {
                let ks: Vec<String> = k_vals
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n.to_string()))
                    .collect();
                detail_lines.push(format!("测试K值: [{}]", ks.join(", ")));
            }
        }

        detail_lines.push(String::new());
        detail_lines.push(format!("子查询 ({}个):", sub_queries));
        if let Some(subs) = tc.get("sub_queries").and_then(|v| v.as_array()) {
            for (si, sq) in subs.iter().enumerate() {
                let prio = sq.get("priority").and_then(|v| v.as_u64()).unwrap_or(0);
                let tags: Vec<String> = sq
                    .get("tag")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                detail_lines.push(format!("  Q{} pri={} tags=[{}]", si, prio, tags.join(",")));
                if let Some(variant) = sq.get("variant") {
                    detail_lines.extend(format_variant_preview(variant, 4));
                }
            }
        }

        detail_lines.push(String::new());
        detail_lines.push(format!("期望排序 (combined): {}", expected_str));
        detail_lines.push(format!("奖励排序 (bonus): {}", bonus_str));
        if let Some(per_q) = tc.get("expected_per_query").and_then(|v| v.as_array()) {
            for eq in per_q {
                let qidx = eq.get("q").and_then(|v| v.as_u64()).unwrap_or(0);
                let ranking: Vec<String> = eq
                    .get("ranking")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                detail_lines.push(format!("  Q{} 期望: [{}]", qidx, ranking.join(", ")));
                let bonus_ranking: Vec<String> = eq
                    .get("bonus_ranking")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if !bonus_ranking.is_empty() {
                    detail_lines.push(format!("     奖励: [{}]", bonus_ranking.join(", ")));
                }
            }
        }

        let summary = format!("{}  [{}子查询] {}", case_name, sub_queries, case_desc);

        entries.push(InspectEntry {
            id: case_name,
            summary,
            preview_lines,
            detail_lines,
            links: Vec::new(),
        });
    }

    entries
}

fn format_mem_type(
    mem_type: Option<&serde_json::Value>,
    tags: &[String],
) -> (String, Vec<String>, Vec<String>) {
    let type_label: String;
    let mut preview_lines = Vec::new();
    let mut detail_lines = Vec::new();

    match mem_type {
        None => {
            type_label = "?".to_string();
            preview_lines.push("(无类型)".to_string());
            detail_lines.push("(无类型)".to_string());
        }
        Some(val) if val.get("Semantic").is_some() => {
            type_label = "Semantic".to_string();
            let sem = &val["Semantic"];
            let content = sem.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let desc = sem
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let aliases: Vec<String> = sem
                .get("aliases")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let concept_type = sem
                .get("concept_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            preview_lines.push(format!("内容: {}", content));
            preview_lines.push(format!("描述: {}", desc));

            detail_lines.push("类型: Semantic".to_string());
            detail_lines.push(format!("标签: [{}]", tags.join(", ")));
            detail_lines.push(format!("内容: {}", content));
            detail_lines.push(format!("别名: [{}]", aliases.join(", ")));
            detail_lines.push(format!("概念类型: {}", concept_type));
            if !desc.is_empty() {
                detail_lines.push(format!("描述: {}", desc));
            }
        }
        Some(val) if val.get("Situation").is_some() => {
            let sit = &val["Situation"];

            if let Some(spec) = sit.get("SpecificSituation") {
                type_label = "Situation".to_string();
                let narrative = spec.get("narrative").and_then(|v| v.as_str()).unwrap_or("");
                let time_span = spec
                    .get("time_span")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");

                preview_lines.push(format!("叙事: {}", narrative));
                preview_lines.push(format!("时间: {}", time_span));

                detail_lines.push("类型: Situation::SpecificSituation".to_string());
                detail_lines.push(format!("标签: [{}]", tags.join(", ")));
                detail_lines.push(format!("叙事: {}", narrative));
                detail_lines.push(format!("时间: {}", time_span));

                if let Some(ctx) = spec.get("context") {
                    if let Some(loc) = ctx.get("location").and_then(|v| v.as_object()) {
                        let name = loc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let coords = loc
                            .get("coordinates")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        detail_lines.push(format!("地点: {} ({})", name, coords));
                    }
                    if let Some(parts) = ctx.get("participants").and_then(|v| v.as_array()) {
                        for p in parts {
                            let pname = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let role = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
                            detail_lines.push(format!("参与者: {} ({})", pname, role));
                        }
                    }
                    if let Some(env) = ctx.get("environment").and_then(|v| v.as_object()) {
                        let atm = env.get("atmosphere").and_then(|v| v.as_str()).unwrap_or("");
                        let tone = env.get("tone").and_then(|v| v.as_str()).unwrap_or("");
                        detail_lines.push(format!("环境: atm={} tone={}", atm, tone));
                    }
                    if let Some(events) = ctx.get("event").and_then(|v| v.as_array()) {
                        for ev in events {
                            let action = ev.get("action").and_then(|v| v.as_str()).unwrap_or("");
                            let init = ev.get("initiator").and_then(|v| v.as_str()).unwrap_or("");
                            let tgt = ev.get("target").and_then(|v| v.as_str()).unwrap_or("");
                            detail_lines.push(format!("事件: {} → {} ({})", init, action, tgt));
                        }
                    }
                }
            } else if let Some(abs) = sit.get("AbstractSituation") {
                type_label = "AbstractSit".to_string();
                detail_lines.push("类型: Situation::AbstractSituation".to_string());
                detail_lines.push(format!("标签: [{}]", tags.join(", ")));

                if let Some(loc) = abs.get("Location") {
                    let name = loc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let coords = loc
                        .get("coordinates")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    preview_lines.push(format!("地点: {}", name));
                    detail_lines.push("子类: Location".to_string());
                    detail_lines.push(format!("名称: {}", name));
                    if !coords.is_empty() {
                        detail_lines.push(format!("坐标: {}", coords));
                    }
                } else if let Some(part) = abs.get("Participant") {
                    let name = part.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let role = part.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    preview_lines.push(format!("参与者: {}", name));
                    detail_lines.push("子类: Participant".to_string());
                    detail_lines.push(format!("名称: {}", name));
                    if !role.is_empty() {
                        detail_lines.push(format!("角色: {}", role));
                    }
                } else if let Some(env) = abs.get("Environment") {
                    let atm = env.get("atmosphere").and_then(|v| v.as_str()).unwrap_or("");
                    let tone = env.get("tone").and_then(|v| v.as_str()).unwrap_or("");
                    preview_lines.push(format!("环境: {} / {}", atm, tone));
                    detail_lines.push("子类: Environment".to_string());
                    detail_lines.push(format!("氛围: {}", atm));
                    detail_lines.push(format!("色调: {}", tone));
                } else if let Some(evt) = abs.get("Event") {
                    let action = evt.get("action").and_then(|v| v.as_str()).unwrap_or("");
                    let init = evt.get("initiator").and_then(|v| v.as_str()).unwrap_or("");
                    let tgt = evt.get("target").and_then(|v| v.as_str()).unwrap_or("");
                    preview_lines.push(format!("事件: {} → {}", init, action));
                    detail_lines.push("子类: Event".to_string());
                    detail_lines.push(format!("动作: {}", action));
                    if !init.is_empty() {
                        detail_lines.push(format!("发起者: {}", init));
                    }
                    if !tgt.is_empty() {
                        detail_lines.push(format!("目标: {}", tgt));
                    }
                } else {
                    preview_lines.push(format!("{:?}", abs));
                    detail_lines.push(format!("{:?}", abs));
                }
            } else {
                type_label = "Situation".to_string();
                preview_lines.push(format!("{:?}", sit));
                detail_lines.push(format!("{:?}", sit));
            }
        }
        Some(val) if val.get("Procedure").is_some() => {
            type_label = "Procedure".to_string();
            let action = &val["Procedure"]["action"];
            let content = action.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let action_type = action
                .get("action_type")
                .and_then(|v| v.as_str())
                .unwrap_or("?");

            preview_lines.push("类型: Procedure".to_string());
            preview_lines.push(format!("动作: {}", content));

            detail_lines.push("类型: Procedure".to_string());
            detail_lines.push(format!("标签: [{}]", tags.join(", ")));
            detail_lines.push(format!("动作: {}", content));
            detail_lines.push(format!("动作类型: {}", action_type));
        }
        Some(val) => {
            let first_key = val
                .as_object()
                .and_then(|o| o.keys().next())
                .map(|k| k.to_string())
                .unwrap_or_else(|| "?".to_string());
            type_label = first_key;
            preview_lines.push(format!("{:?}", val));
            detail_lines.push(format!("{:?}", val));
        }
    }

    (type_label, preview_lines, detail_lines)
}

fn format_link_type(val: &serde_json::Value) -> String {
    if let Some(obj) = val.as_object() {
        for (k, v) in obj {
            return match k.as_str() {
                "Sem" => {
                    let verb = v.get("verb").and_then(|x| x.as_str()).unwrap_or("?");
                    let conf = v.get("confidence").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    format!("Sem[{} conf={:.1}]", verb, conf)
                }
                "Proc" => {
                    if let Some(inner) = v.get("TrigToAction") {
                        let prob = inner.get("prob").and_then(|x| x.as_f64()).unwrap_or(0.0);
                        format!("Proc::TrigToAction[prob={:.1}]", prob)
                    } else {
                        format!("Proc[{:?}]", v)
                    }
                }
                "Situation" => {
                    if v.get("AbstractToSpecific").is_some() {
                        "Sit::AbstractToSpecific".to_string()
                    } else {
                        "Situation[...]".to_string()
                    }
                }
                "Coref" => "Coref".to_string(),
                _ => format!("{}[...]", k),
            };
        }
    }
    val.to_string()
}

fn label_key(k: &str) -> &'static str {
    match k {
        "node_count" => "节点数",
        "edge_count" => "边数",
        "node_types" => "节点类型",
        "link_types" => "边类型",
        "connected_components" => "连通分量",
        "largest_component" => "最大分量",
        "isolated_nodes" => "孤立节点",
        "global_redundancy" => "全局冗余度",
        "avg_clustering" => "平均聚类系数",
        "community_modularity" => "社区模块度",
        "intra_community_ratio" => "社区内边比",
        "gini_coefficient" => "基尼系数",
        "has_self_node" => "有自身节点",
        "self_description_ok" => "自身描述有效",
        "is_clean" => "图结构清洁",
        "is_structurally_valid" => "结构有效",
        "proc_without_incoming_proc" => "孤立Procedure数",
        "abstract_sit_type_count" => "抽象情境类型数",
        "has_proc_none" => "有空Procedure",
        _ => "?",
    }
}

fn load_graph_stats(path: &std::path::Path) -> Option<Vec<String>> {
    let content = std::fs::read_to_string(path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    let obj = val.as_object()?;

    let mut lines = Vec::new();

    for (k, v) in obj {
        let lbl = match k.as_str() {
            "node_count"
            | "edge_count"
            | "connected_components"
            | "largest_component"
            | "isolated_nodes"
            | "abstract_sit_type_count" => v.as_u64().map(|n| format!("{}: {}", label_key(k), n)),
            "global_redundancy"
            | "avg_clustering"
            | "community_modularity"
            | "intra_community_ratio"
            | "gini_coefficient" => v.as_f64().map(|n| format!("{}: {:.3}", label_key(k), n)),
            "has_self_node"
            | "self_description_ok"
            | "is_clean"
            | "is_structurally_valid"
            | "has_proc_none" => v
                .as_bool()
                .map(|b| format!("{}: {}", label_key(k), if b { "✓" } else { "✗" })),
            "proc_without_incoming_proc" => v
                .as_array()
                .map(|a| format!("{}: {}个", label_key(k), a.len())),
            "node_types" | "link_types" => v.as_object().map(|o| {
                let inner: Vec<String> = o
                    .iter()
                    .map(|(sk, sv)| format!("{}={}", sk, sv.as_u64().unwrap_or(0)))
                    .collect();
                format!("{}: {}", label_key(k), inner.join(" "))
            }),
            _ => None,
        };
        if let Some(l) = lbl {
            lines.push(l);
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines)
    }
}

fn format_variant_preview(val: &serde_json::Value, indent: usize) -> Vec<String> {
    let pad = " ".repeat(indent);
    let mut out = Vec::new();
    if let Some(obj) = val.as_object() {
        for (k, inner) in obj {
            if let Some(arr) = inner.as_array() {
                if arr.is_empty() {
                    out.push(format!("{} {}: (空)", pad, k));
                } else {
                    out.push(format!("{} {}: {}条", pad, k, arr.len()));
                    for item in arr.iter().take(3) {
                        if let Some(item_obj) = item.as_object() {
                            for (fk, fv) in item_obj {
                                if let Some(s) = fv.as_str() {
                                    if s.chars().count() > 60 {
                                        let truncated: String = s.chars().take(60).collect();
                                        out.push(format!("{}   {}: {}...", pad, fk, truncated));
                                    } else {
                                        out.push(format!("{}   {}: {}", pad, fk, s));
                                    }
                                }
                            }
                        }
                    }
                    if arr.len() > 3 {
                        out.push(format!("{}   ... 还有{}条", pad, arr.len() - 3));
                    }
                }
            } else {
                out.push(format!("{} {}: {}", pad, k, inner));
            }
        }
    }
    out
}
