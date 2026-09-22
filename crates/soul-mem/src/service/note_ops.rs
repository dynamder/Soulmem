//! note_ops：按 id 读/写 MemoryNote，以及用户反馈。
//!
//! 对应 orchestration「外部接口：对指定 id 的 MemoryNote 读写」。
//! 写入采用 upsert 语义：存在则覆盖（刷新 embedding），不存在则新增。

use super::SoulMemService;
use crate::error::{Error, Result};
use crate::wire::convert::{feedback_from_proto, note_from_proto, note_to_proto, parse_memory_id};
use crate::wire::pb;
use soul_mem_query::embedding::Embeddable;
use soul_mem_query::embedding::note::EmbeddedMemoryNote;

impl SoulMemService {
    /// 按 id 读取一条 MemoryNote。
    pub async fn read_note(&self, req: pb::ReadNoteRequest) -> Result<pb::ReadNoteResponse> {
        let id = parse_memory_id(&req.id)?;
        let wm = self.wm_arc().await;
        let handle = wm.memory_cluster();
        let note = handle.read_or_compute(|c| c.get_node(id).map(|n| n.note.clone()));
        drop(wm);
        let note = note.ok_or_else(|| Error::NotFound(format!("MemoryNote {id} not found")))?;
        Ok(pb::ReadNoteResponse {
            note: Some(note_to_proto(&note)?),
        })
    }

    /// 写入（新增或覆盖）一条 MemoryNote。
    ///
    /// upsert 语义：存在则**原地替换 note 与 embedding 并刷新边**，不存在则新增。
    /// 关键点：不使用 `remove_node + add_node`，因为 `remove_node` 会连同 `Record`
    /// （检索计数/反馈历史）一起删除，从而静默清空行为信号。
    pub async fn write_note(&self, req: pb::WriteNoteRequest) -> Result<pb::WriteNoteResponse> {
        let note_pb = req
            .note
            .ok_or_else(|| Error::InvalidArgument("WriteNoteRequest.note is required".into()))?;
        let note = note_from_proto(note_pb)?;
        let id = note.id();

        // 用当前模型为该 Note 生成 embedding（新增/覆盖都会刷新向量）。
        let embedding = note
            .embed(&*self.core.model)
            .map_err(|e| Error::internal(format!("embed MemoryNote: {e}")))?;
        let note_clone = note.clone();

        let created = self
            .with_wm(move |wm| {
                let handle = wm.memory_cluster();
                let existed = handle.read_or_compute(|c| c.contains_node(id));
                if existed {
                    // 原地替换内容与向量，保留该 id 对应的 Record（行为历史）。
                    handle.write(|c| {
                        if let Some(node) = c.get_node_mut(id) {
                            node.note = note_clone;
                            node.embedding = embedding;
                        }
                    });
                    handle.write(|c| c.refresh_node(&id));
                    false
                } else {
                    wm.add_node(EmbeddedMemoryNote {
                        note: note_clone,
                        embedding,
                    });
                    true
                }
            })
            .await?;

        Ok(pb::WriteNoteResponse {
            id: id.to_string(),
            created,
        })
    }

    /// 对某条 MemoryNote 添加用户反馈。
    pub async fn feedback(&self, req: pb::Feedback) -> Result<pb::Ack> {
        let id = parse_memory_id(&req.note_id)?;
        let uf = feedback_from_proto(req.kind)?;
        let label = match uf {
            soul_mem_runtime::working_memory::record::UserFeedback::Positive => "positive",
            soul_mem_runtime::working_memory::record::UserFeedback::Negative => "negative",
            soul_mem_runtime::working_memory::record::UserFeedback::Neutral => "neutral",
            soul_mem_runtime::working_memory::record::UserFeedback::None => "none",
        };
        let existed = self
            .with_wm(|wm| {
                if !wm.memory_cluster().read_or_compute(|c| c.contains_node(id)) {
                    return false;
                }
                wm.add_feedback(id, uf);
                true
            })
            .await?;
        if !existed {
            return Err(Error::NotFound(format!("MemoryNote {id} not found")));
        }
        Ok(pb::Ack {
            message: format!("feedback {label} recorded for {id}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, EmbeddingMode};
    use crate::store::Store;
    use crate::wire::convert::note_to_proto;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};

    fn test_config() -> Config {
        Config {
            device_id: "note-ops-test".to_string(),
            zenoh_key_prefix: "soulmem_note_ops".to_string(),
            window_capacity: 8,
            similarity_threshold: 0.05,
            similarity_max_results: 8,
            llm_base_url: "http://127.0.0.1:9/v1".to_string(),
            llm_api_key: "demo".to_string(),
            llm_model: "demo".to_string(),
            persist_interval_secs: 0,
            consolidate_interval_secs: 0,
            forget_interval_secs: 0,
            embedding_mode: EmbeddingMode::Hash,
            store_path: None,
        }
    }

    fn sem_note() -> soul_mem_core::memory_note::MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "周会".to_string(),
            ConceptType::Entity,
            "每周一次的团队例会".to_string(),
        )))
        .build()
        .expect("build note")
    }

    /// 回归：upsert 覆盖同一 id 时必须保留 Record（检索计数/反馈历史）。
    #[tokio::test]
    async fn write_note_upsert_preserves_record() -> Result<()> {
        let service = SoulMemService::from_config(&test_config(), Store::default(), None).await?;
        let note = sem_note();
        let id = note.id();
        let note_pb = note_to_proto(&note)?;

        let first = service
            .write_note(pb::WriteNoteRequest {
                note: Some(note_pb.clone()),
            })
            .await?;
        assert!(first.created, "first write should create");

        // 制造行为历史：一次检索计数 + 一次正反馈。
        service.with_wm(|wm| wm.record_retrieval(id)).await?;
        service
            .feedback(pb::Feedback {
                note_id: id.to_string(),
                kind: pb::FeedbackKind::FeedbackPositive as i32,
            })
            .await?;

        let before = service
            .with_wm(|wm| {
                let r = wm.records().get(&id).expect("record exists");
                (r.retrieval_count(), r.feedback_score())
            })
            .await?;
        assert_eq!(before, (1, 1));

        // 覆盖写：内容/向量刷新，但 Record 必须保留。
        let second = service
            .write_note(pb::WriteNoteRequest {
                note: Some(note_pb),
            })
            .await?;
        assert!(!second.created, "second write should be an update");

        let after = service
            .with_wm(|wm| {
                let r = wm.records().get(&id).expect("record still exists");
                (r.retrieval_count(), r.feedback_score())
            })
            .await?;
        assert_eq!(after, (1, 1), "upsert must not clear Record");
        Ok(())
    }
}
