//! retrieve：多 query 检索编排（检索管线 → 收集全文 → 组装响应）。
//!
//! 对应 orchestration 的查询主流程与 Svc 规划区：
//! 每个带优先级 query 走 `DefaultPipeline`，命中的 MemoryNote 合并去重（保留更高分），
//! 再连同滑动窗口短期上下文一起返回。输入/输出均为 protobuf 消息。

use super::SoulMemService;
use crate::error::{Error, Result};
use crate::service::information_role;
use crate::wire::convert::{note_to_proto, query_from_proto};
use crate::wire::pb;
use soul_mem_algo::algo::retrieve::RetrStrategy;
use soul_mem_algo::algo::retrieve::complex::default_pipeline::RetrDefaultPipeline;
use soul_mem_core::memory_note::{MemoryId, MemoryNote};
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::embedding::query::note::EmbeddedMemoryRetrieveQuery;
use soul_mem_runtime::working_memory::WorkingMemory;
use std::collections::BTreeMap;
use std::sync::Arc;

/// 一次检索请求最多容纳的 query 数量（防滥用）。
pub const MAX_QUERIES_PER_REQUEST: usize = 16;

impl SoulMemService {
    /// 执行检索：返回 MemoryNote 命中集合 + 短期上下文。
    pub async fn retrieve(&self, req: pb::RetrieveRequest) -> Result<pb::RetrieveResponse> {
        if req.queries.is_empty() {
            return Err(Error::InvalidArgument(
                "retrieve requires at least one query".into(),
            ));
        }
        if req.queries.len() > MAX_QUERIES_PER_REQUEST {
            return Err(Error::InvalidArgument(format!(
                "too many queries: {} exceeds {MAX_QUERIES_PER_REQUEST}",
                req.queries.len()
            )));
        }

        let wm = self.wm_arc().await;

        // 多 query：逐个走默认管线。
        let mut by_id: BTreeMap<MemoryId, (f64, MemoryNote)> = BTreeMap::new();
        for pq in &req.queries {
            let query = pq
                .query
                .as_ref()
                .ok_or_else(|| Error::InvalidArgument("PrioritizedQuery.query is required".into()))
                .and_then(query_from_proto)?;
            let embedded = query
                .embed(&*self.core.model)
                .map_err(|e| Error::internal(format!("embed query: {e}")))?;
            let embedded_query = EmbeddedMemoryRetrieveQuery {
                embedding: embedded,
                query,
            };

            let pipeline_request = self.core.pipeline.clone().into_request(
                Arc::clone(&wm),
                embedded_query,
                pq.priority,
            );
            let result = RetrDefaultPipeline {}.retrieve(pipeline_request);

            for (id, score) in result.association {
                if let Some(note) = fetch_note(&wm, id) {
                    by_id
                        .entry(id)
                        .and_modify(|(cur_score, _)| {
                            if score > *cur_score {
                                *cur_score = score;
                            }
                        })
                        .or_insert((score, note));
                }
            }
        }
        drop(wm);

        // 命中的检索记账。
        let hit_ids: Vec<MemoryId> = by_id.keys().copied().collect();
        if !hit_ids.is_empty() {
            self.with_wm(move |wm| {
                for id in hit_ids {
                    wm.record_retrieval(id);
                }
            })
            .await?;
        }

        let mut hits = Vec::with_capacity(by_id.len());
        for (score, note) in by_id.into_values() {
            hits.push(pb::NoteHit {
                score,
                note: Some(note_to_proto(&note)?),
            });
        }

        // 短期上下文（滑动窗口 + 摘要）。
        let wm = self.wm_arc().await;
        let short_history = wm
            .sliding_window()
            .get_windows()
            .iter()
            .map(|info| pb::ShortItem {
                role: information_role(info).to_string(),
                content: info.get_str().to_string(),
            })
            .collect::<Vec<_>>();
        let summary = wm.sliding_window().get_summary().to_string();
        drop(wm);

        Ok(pb::RetrieveResponse {
            hits,
            summary,
            short_history,
        })
    }
}

/// 从 cluster 取某 id 对应 MemoryNote 的副本。
fn fetch_note(wm: &WorkingMemory, id: MemoryId) -> Option<MemoryNote> {
    let handle = wm.memory_cluster();
    handle.read_or_compute(|cluster| cluster.get_node(id).map(|n| n.note.clone()))
}
