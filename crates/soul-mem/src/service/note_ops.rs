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
            .with_wm(|wm| {
                let existed = wm.memory_cluster().read_or_compute(|c| c.contains_node(id));
                if existed {
                    let _ = wm.remove_node(id);
                }
                wm.add_node(EmbeddedMemoryNote {
                    note: note_clone,
                    embedding,
                });
                !existed
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
