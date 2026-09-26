use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;
use soul_mem_algo::algo::retrieve::{DbPrefetchConfig, PrefetchOutcome};
use soul_mem_core::memory_note::MemoryId;

use crate::engine::retrieve::dataset::SubQuery;

#[derive(Debug, Clone, Serialize)]
pub struct RankingMetrics {
    pub recall_at: Vec<(usize, f64)>,
    pub precision_at: Vec<(usize, f64)>,
    pub mrr: f64,
    pub ndcg_at: Vec<(usize, f64)>,
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActionMetrics {
    pub action_hit_rate: f64,
    pub action_recall_at: Vec<(usize, f64)>,
    /// 该用例是否带 expected_actions 真值（False 表示占位指标，不应计入统计）
    pub has_expected_actions: bool,
}

#[derive(Clone, Serialize)]
pub struct PerQueryMetrics {
    pub query_index: usize,
    pub ranking_metrics: RankingMetrics,
}

/// prefetch_db 召回观测（DB 模式每个用例记录一次）。
///
/// 回答两个问题：
/// 1. DB 召回子图有多大（候选 HNSW 命中 / 邻居扩展 / 实际写入工作记忆的节点数）；
/// 2. 期望命中有多少进入了召回子图——没进的就是 DB 路径的结构性漏召
///    （`expected_missed`），与"进了子图但精确重排没排进 top-k"的原因区分开。
///
/// 观测数据直接取自 [`prefetch_db`] 的返回值（[`PrefetchOutcome`]），
/// 与真实预取同源；子图规模仍以工作记忆实际内容为准。
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct DbRecallDetail {
    /// similarity_fetch 的 union 候选数（每槽位 KNN，去重）。
    pub candidate_count: usize,
    /// 邻居扩展新增的节点数（`visited - 候选`）。
    pub neighbor_count: usize,
    /// 实际写入用例工作记忆的子图节点数（候选 + 邻居去重后）。
    pub subgraph_count: usize,
    /// 参与召回的查询（子查询）数。
    pub query_count: usize,
    /// 每槽位 KNN 候选召回预算。
    pub candidate_k: usize,
    /// 邻居扩展跳数（0 = 不扩展）。
    pub neighbor_depth: usize,
    /// 期望节点总数（must + bonus 并集、去重）。
    pub expected_count: usize,
    /// 期望 ∩ 候选（仅靠嵌入相似召回即覆盖的部分）。
    pub expected_in_candidates: usize,
    /// 期望 ∩ 子图（候选 + 邻居后覆盖的部分）。
    pub expected_in_subgraph: usize,
    /// must 期望（判定通过的依据）中进入子图的数量。
    pub must_in_subgraph: usize,
    /// 期望中未被 DB 召回的子集（按期望顺序，为 DB 回退用例的根因候选）。
    pub expected_missed: Vec<MemoryId>,
}

impl DbRecallDetail {
    /// 由一次 `prefetch_db` 的返回值与用例期望真值构建详情。
    ///
    /// - `outcome`：预取返回值（候选/邻居 id），观测与实现同源；
    /// - `subgraph_ids`：**实际写入**工作记忆的节点集合（以工作记忆为准，
    ///   而非由返回值推断，便于发现写入侧与召回侧的偏差）；
    /// - `expected_must` / `expected_bonus` 保持调用侧顺序；并集去重后与子图比对。
    pub fn from_prefetch(
        outcome: &PrefetchOutcome,
        subgraph_ids: &[MemoryId],
        expected_must: &[MemoryId],
        expected_bonus: &[MemoryId],
        query_count: usize,
        config: DbPrefetchConfig,
    ) -> Self {
        let candidate_set: HashSet<&MemoryId> = outcome.candidates.iter().collect();
        let subgraph_set: HashSet<&MemoryId> = subgraph_ids.iter().collect();

        // must + bonus 并集，保持顺序去重
        let mut seen: HashSet<MemoryId> = HashSet::new();
        let expected_all: Vec<MemoryId> = expected_must
            .iter()
            .chain(expected_bonus.iter())
            .copied()
            .filter(|id| seen.insert(*id))
            .collect();

        let expected_in_candidates = expected_all
            .iter()
            .filter(|id| candidate_set.contains(id))
            .count();
        let expected_in_subgraph = expected_all
            .iter()
            .filter(|id| subgraph_set.contains(id))
            .count();
        let must_in_subgraph = expected_must
            .iter()
            .filter(|id| subgraph_set.contains(id))
            .count();
        let expected_missed: Vec<MemoryId> = expected_all
            .iter()
            .filter(|id| !subgraph_set.contains(id))
            .copied()
            .collect();

        DbRecallDetail {
            candidate_count: outcome.candidates.len(),
            neighbor_count: outcome.neighbors.len(),
            subgraph_count: subgraph_ids.len(),
            query_count,
            candidate_k: config.candidate_k,
            neighbor_depth: config.neighbor_depth,
            expected_count: expected_all.len(),
            expected_in_candidates,
            expected_in_subgraph,
            must_in_subgraph,
            expected_missed,
        }
    }
}

#[derive(Clone, Serialize)]
pub struct RetrieveCaseData {
    pub case_name: String,
    pub description: String,
    pub combined_retrieved_ids: Vec<MemoryId>,
    pub combined_ranking_metrics: RankingMetrics,
    /// DB 模式：prefetch_db 召回观测（候选/邻居/子图规模与期望覆盖）；直接模式为 None。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_recall: Option<DbRecallDetail>,
    pub per_query_metrics: Vec<PerQueryMetrics>,
    pub action_metrics: ActionMetrics,
    /// 该用例的期望结果中是否包含抽象情境节点（有真值才计入抽象指标）。
    pub has_expected_abstract: bool,
    /// 期望抽象节点是否出现在合并结果（相似度+PPR）中。
    pub abstract_detected: Option<bool>,
    /// 期望抽象节点是否仍被相似度直接命中（数据侧泛化是否达标的观测门）。
    pub abstract_direct_hit: Option<bool>,
    pub tag_weight: f32,
    pub variant_weight: f32,
    pub id_names: Option<Arc<HashMap<MemoryId, NodeSummary>>>,
    pub expected_combined_ranking: Vec<MemoryId>,
    pub bonus_combined_ranking: Vec<MemoryId>,
    pub graph_names: Option<Arc<HashMap<MemoryId, String>>>,
    pub sub_queries: Vec<SubQuery>,
}

#[derive(Clone, Serialize)]
pub struct NodeSummary {
    pub tags: Vec<String>,
    pub type_label: String,
    pub primary: String,
    pub secondary: String,
}

pub struct DrilldownSections {
    pub header_lines: Vec<String>,
    pub metrics_rows: Vec<String>,
    pub subquery_items: Vec<SubQueryItem>,
    pub comparison_rows: Vec<ComparisonRow>,
    /// DB 模式：prefetch_db 召回观测的可读行（候选/邻居/子图/期望覆盖/漏召列表）。
    pub db_recall_lines: Vec<String>,
}

pub struct SubQueryItem {
    pub index: usize,
    pub mrr: f64,
    pub hit_rate: f64,
}

pub struct ComparisonRow {
    pub position: usize,
    pub retrieved: Option<RetrievedEntry>,
    pub expected: Option<String>,
    pub is_hit: bool,
}

pub struct RetrievedEntry {
    pub name: String,
    pub id: MemoryId,
}

pub fn build_drilldown_sections(data: &RetrieveCaseData) -> DrilldownSections {
    let mut sections = DrilldownSections {
        header_lines: Vec::new(),
        metrics_rows: Vec::new(),
        subquery_items: Vec::new(),
        comparison_rows: Vec::new(),
        db_recall_lines: Vec::new(),
    };

    sections
        .header_lines
        .push(format!(" 用例: {}", data.case_name));
    let passed =
        data.combined_ranking_metrics.hit_rate > 0.0 || data.combined_ranking_metrics.mrr > 0.0;
    sections
        .header_lines
        .push(format!(" 状态: {}", if passed { "通过" } else { "失败" }));

    sections
        .metrics_rows
        .push("  K     Recall    Precision  NDCG".to_string());
    for (k, r) in &data.combined_ranking_metrics.recall_at {
        let p = data
            .combined_ranking_metrics
            .precision_at
            .iter()
            .find(|(pk, _)| pk == k)
            .map(|(_, v)| v)
            .unwrap_or(&0.0);
        let n = data
            .combined_ranking_metrics
            .ndcg_at
            .iter()
            .find(|(nk, _)| nk == k)
            .map(|(_, v)| v)
            .unwrap_or(&0.0);
        sections
            .metrics_rows
            .push(format!("  @{:<2}   {:.4}    {:.4}    {:.4}", k, r, p, n));
    }
    sections.metrics_rows.push(format!(
        "  MRR: {:.4}     Hit: {:.2}",
        data.combined_ranking_metrics.mrr, data.combined_ranking_metrics.hit_rate
    ));

    for m in &data.per_query_metrics {
        sections.subquery_items.push(SubQueryItem {
            index: m.query_index,
            mrr: m.ranking_metrics.mrr,
            hit_rate: m.ranking_metrics.hit_rate,
        });
    }

    // DB 模式：prefetch_db 召回观测详情（候选/邻居/子图/期望覆盖/漏召列表）
    if let Some(r) = &data.db_recall {
        sections.db_recall_lines.push(format!(
            "  候选召回: {} 节点（{} 查询 × 每槽位预算 {}）",
            r.candidate_count, r.query_count, r.candidate_k
        ));
        if r.neighbor_depth == 0 {
            sections
                .db_recall_lines
                .push("  邻居扩展: 关闭（深度 0）".to_string());
        } else {
            sections.db_recall_lines.push(format!(
                "  邻居扩展: +{} 节点（深度 {}）",
                r.neighbor_count, r.neighbor_depth
            ));
        }
        sections
            .db_recall_lines
            .push(format!("  工作记忆子图: {} 节点", r.subgraph_count));
        sections.db_recall_lines.push(format!(
            "  期望覆盖: {}/{} 进入子图（候选内 {}，must {}）",
            r.expected_in_subgraph, r.expected_count, r.expected_in_candidates, r.must_in_subgraph
        ));
        if r.expected_missed.is_empty() {
            sections
                .db_recall_lines
                .push("  期望全部进入召回子图 ✓".to_string());
        } else {
            sections
                .db_recall_lines
                .push("  期望未召回（DB 结构性漏召候选）:".to_string());
            for id in &r.expected_missed {
                let name = data
                    .graph_names
                    .as_ref()
                    .and_then(|m| m.get(id))
                    .cloned()
                    .unwrap_or_else(|| format!("{id:?}"));
                sections.db_recall_lines.push(format!("    - {name}"));
            }
        }
    }

    let n_max = data
        .combined_retrieved_ids
        .len()
        .min(10)
        .max(data.expected_combined_ranking.len().min(5));
    for pos in 0..n_max {
        let retrieved = data.combined_retrieved_ids.get(pos).map(|id| {
            let name = data
                .graph_names
                .as_ref()
                .and_then(|m| m.get(id))
                .cloned()
                .unwrap_or_default();
            RetrievedEntry { name, id: *id }
        });
        let expected = data.expected_combined_ranking.get(pos).map(|eid| {
            data.graph_names
                .as_ref()
                .and_then(|m| m.get(eid))
                .cloned()
                .unwrap_or_default()
        });
        let is_hit = retrieved
            .as_ref()
            .map(|r| data.expected_combined_ranking.contains(&r.id))
            .unwrap_or(false);
        sections.comparison_rows.push(ComparisonRow {
            position: pos + 1,
            retrieved,
            expected,
            is_hit,
        });
    }

    sections
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_query::query::retrieve::MemoryRetrieveQueryVariant;

    fn make_id() -> MemoryId {
        MemoryId::new()
    }

    fn mock_case_data() -> RetrieveCaseData {
        let id1 = make_id();
        let id2 = make_id();
        let mut names = HashMap::new();
        names.insert(id1, "node_1".to_string());
        names.insert(id2, "node_2".to_string());
        RetrieveCaseData {
            case_name: "test_case".into(),
            description: "test".into(),
            combined_retrieved_ids: vec![id1, id2],
            combined_ranking_metrics: RankingMetrics {
                recall_at: vec![(1, 0.5), (3, 1.0)],
                precision_at: vec![(1, 1.0), (3, 0.667)],
                mrr: 1.0,
                ndcg_at: vec![(1, 1.0), (3, 0.8)],
                hit_rate: 1.0,
            },
            db_recall: None,
            per_query_metrics: vec![PerQueryMetrics {
                query_index: 0,
                ranking_metrics: RankingMetrics {
                    recall_at: vec![(1, 1.0)],
                    precision_at: vec![(1, 1.0)],
                    mrr: 1.0,
                    ndcg_at: vec![(1, 1.0)],
                    hit_rate: 1.0,
                },
            }],
            action_metrics: ActionMetrics {
                action_hit_rate: 1.0,
                action_recall_at: vec![(1, 1.0)],
                has_expected_actions: false,
            },
            has_expected_abstract: false,
            abstract_detected: None,
            abstract_direct_hit: None,
            tag_weight: 0.4,
            variant_weight: 0.6,
            id_names: None,
            expected_combined_ranking: vec![id1],
            bonus_combined_ranking: vec![],
            graph_names: Some(Arc::new(names)),
            sub_queries: vec![SubQuery {
                priority: 1,
                tags: vec!["test".into()],
                variant: MemoryRetrieveQueryVariant::Semantic(vec![]),
            }],
        }
    }

    #[test]
    fn test_build_drilldown_sections_normal() {
        let data = mock_case_data();
        let sections = build_drilldown_sections(&data);
        assert!(!sections.header_lines.is_empty());
        assert!(!sections.metrics_rows.is_empty());
        assert_eq!(sections.subquery_items.len(), 1);
        assert!(!sections.comparison_rows.is_empty());
    }

    #[test]
    fn test_build_drilldown_header_contains_name() {
        let data = mock_case_data();
        let sections = build_drilldown_sections(&data);
        let h = sections.header_lines.join(" ");
        assert!(h.contains("test_case"));
    }

    #[test]
    fn test_build_drilldown_edge_no_ids() {
        let mut data = mock_case_data();
        data.combined_retrieved_ids.clear();
        data.expected_combined_ranking.clear();
        let sections = build_drilldown_sections(&data);
        assert!(sections.comparison_rows.is_empty());
    }

    #[test]
    fn test_ranking_metrics_clone() {
        let rm = RankingMetrics {
            recall_at: vec![(1, 0.5)],
            precision_at: vec![(1, 0.5)],
            mrr: 0.5,
            ndcg_at: vec![(1, 0.5)],
            hit_rate: 0.5,
        };
        let c = rm.clone();
        assert!((c.mrr - 0.5).abs() < 1e-6);
    }

    // ── DbRecallDetail::from_prefetch ──

    fn ids(n: usize) -> Vec<MemoryId> {
        (0..n).map(|_| MemoryId::new()).collect()
    }

    fn outcome(candidates: &[MemoryId], neighbors: &[MemoryId]) -> PrefetchOutcome {
        PrefetchOutcome {
            candidates: candidates.to_vec(),
            neighbors: neighbors.to_vec(),
        }
    }

    #[test]
    fn test_db_recall_detail_full_coverage() {
        let candidate = ids(3); // a0..a2 进入候选
        let neighbors = ids(1); // b0 邻居
        let subgraph: Vec<MemoryId> = candidate.iter().chain(neighbors.iter()).copied().collect();
        let must: Vec<MemoryId> = vec![candidate[0], neighbors[0]];
        let bonus: Vec<MemoryId> = vec![candidate[1]];
        let detail = DbRecallDetail::from_prefetch(
            &outcome(&candidate, &neighbors),
            &subgraph,
            &must,
            &bonus,
            2,
            DbPrefetchConfig::new(20, 1),
        );
        assert_eq!(detail.candidate_count, 3);
        assert_eq!(detail.neighbor_count, 1);
        assert_eq!(detail.subgraph_count, 4);
        assert_eq!(detail.query_count, 2);
        assert_eq!(detail.candidate_k, 20);
        assert_eq!(detail.neighbor_depth, 1);
        assert_eq!(detail.expected_count, 3);
        assert_eq!(detail.expected_in_candidates, 2); // a0, a1（邻居 b0 不在候选）
        assert_eq!(detail.expected_in_subgraph, 3);
        assert_eq!(detail.must_in_subgraph, 2);
        assert!(detail.expected_missed.is_empty());
    }

    #[test]
    fn test_db_recall_detail_missed_keeps_expected_order() {
        let candidate = ids(1);
        let subgraph = candidate.clone();
        let b = MemoryId::new();
        let a = MemoryId::new();
        let must = vec![b, a]; // 期望顺序：b 在前
        let detail = DbRecallDetail::from_prefetch(
            &outcome(&candidate, &[]),
            &subgraph,
            &must,
            &[],
            1,
            DbPrefetchConfig::new(4, 1),
        );
        assert_eq!(detail.expected_count, 2);
        assert_eq!(detail.expected_in_subgraph, 0);
        // 漏召按期望原序（b 在前），便于 UI 对照期望排名
        assert_eq!(detail.expected_missed, vec![b, a]);
        // a/b 与候选 id 均不同 → 候选内覆盖为 0
        assert_eq!(detail.expected_in_candidates, 0);
    }

    #[test]
    fn test_db_recall_detail_bonus_dedup() {
        let (x, y) = (MemoryId::new(), MemoryId::new());
        // must 与 bonus 重复的 y 只计一次
        let detail = DbRecallDetail::from_prefetch(
            &outcome(&[x], &[]),
            &[x, y],
            &[x, y],
            &[y],
            1,
            DbPrefetchConfig::new(4, 1),
        );
        assert_eq!(detail.expected_count, 2);
        assert_eq!(detail.expected_in_subgraph, 2);
        assert_eq!(detail.must_in_subgraph, 2);
        assert!(detail.expected_missed.is_empty());
    }

    #[test]
    fn test_db_recall_detail_depth_zero_is_recorded() {
        // 邻居扩展关闭时，观测里必须能看出来（深度 0 与"扩展了但一个没召到"不同）
        let x = MemoryId::new();
        let detail = DbRecallDetail::from_prefetch(
            &outcome(&[x], &[]),
            &[x],
            &[x],
            &[],
            1,
            DbPrefetchConfig::new(20, 0),
        );
        assert_eq!(detail.neighbor_depth, 0);
        assert_eq!(detail.neighbor_count, 0);
    }

    #[test]
    fn test_build_drilldown_db_recall_lines() {
        let mut data = mock_case_data();
        let (x, y) = (MemoryId::new(), MemoryId::new());
        let detail = DbRecallDetail::from_prefetch(
            &outcome(&[x], &[y]),
            &[x, y],
            &[x],
            &[MemoryId::new()],
            1,
            DbPrefetchConfig::new(20, 2),
        );
        data.db_recall = Some(detail);
        // 期望含一个不在子图的节点 → 出现"期望未召回"段
        let sections = build_drilldown_sections(&data);
        assert!(!sections.db_recall_lines.is_empty());
        let joined = sections.db_recall_lines.join("\n");
        assert!(joined.contains("候选召回: 1"));
        assert!(joined.contains("邻居扩展: +1 节点（深度 2）"));
        assert!(joined.contains("期望未召回"));
        // 全被召回时给出 ✓ 行
        data.db_recall = Some(DbRecallDetail::from_prefetch(
            &outcome(&[x], &[y]),
            &[x, y],
            &[x, y],
            &[],
            1,
            DbPrefetchConfig::new(20, 1),
        ));
        let sections2 = build_drilldown_sections(&data);
        assert!(
            sections2
                .db_recall_lines
                .join("\n")
                .contains("期望全部进入召回子图 ✓")
        );
        // 深度 0 时给出"扩展关闭"而非 +0
        data.db_recall = Some(DbRecallDetail::from_prefetch(
            &outcome(&[x], &[]),
            &[x],
            &[x],
            &[],
            1,
            DbPrefetchConfig::new(20, 0),
        ));
        let sections3 = build_drilldown_sections(&data);
        assert!(
            sections3
                .db_recall_lines
                .join("\n")
                .contains("邻居扩展: 关闭（深度 0）")
        );
    }
}
