//! 演示场景：全部经 zenoh 订阅/发布（protobuf 载荷）完成。
//!
//! 步骤与 plan.md §7（单机验收场景）对应：ping → ingest → write → retrieve → read → feedback → control，
//! 并观察 liveliness 心跳。

use crate::Opts;
use anyhow::{Result, anyhow};
use soul_mem::wire::convert::note_to_proto;
use soul_mem::wire::pb;
use soul_mem::zenoh::ZenohClient;

pub async fn run(client: &ZenohClient, opts: &Opts) -> Result<()> {
    let _ = opts;

    // 1) Ping。
    let ping = client.ping().await.map_err(anyhow::Error::from)?;
    log::info!("zenoh ping ok, server device_id={}", ping.device_id);

    // 2) Ingest。
    let ingest_req = pb::IngestRequest {
        deltas: vec![
            pb::InfoDelta {
                role: pb::MessageRole::RoleUser as i32,
                content: "今天我们要讨论每周例会安排".to_string(),
            },
            pb::InfoDelta {
                role: pb::MessageRole::RoleAssistant as i32,
                content: "好的，我已经记下例会的话题".to_string(),
            },
        ],
    };
    log::info!(
        "ingest result: {}",
        client
            .ingest(&ingest_req)
            .await
            .map_err(anyhow::Error::from)?
            .message
    );

    // 3) WriteNote：写入一条语义记忆“周会”。
    let note = build_sem_note();
    let written = client
        .write_note(&pb::WriteNoteRequest {
            note: Some(note_to_proto(&note)?),
        })
        .await
        .map_err(anyhow::Error::from)?;
    log::info!(
        "write_note ok: id={} created={}",
        written.id,
        written.created
    );

    // 4) Retrieve。
    let req = retrieve_req("周会");
    let resp = client.retrieve(&req).await.map_err(anyhow::Error::from)?;
    dump_retrieve("zenoh", &resp);
    let hit_id = resp
        .hits
        .iter()
        .filter_map(|h| h.note.as_ref())
        .find(|n| n.id == written.id)
        .map(|n| n.id.clone())
        .ok_or_else(|| {
            anyhow!(
                "expected retrieved hit for written note, got {}",
                resp.hits.len()
            )
        })?;

    // 5) ReadNote。
    let read = client
        .read_note(&hit_id)
        .await
        .map_err(anyhow::Error::from)?;
    log::info!(
        "read_note ok, id={}",
        read.note.map(|n| n.id).unwrap_or_default()
    );

    // 6) Feedback。
    let fb = pb::Feedback {
        note_id: hit_id,
        kind: pb::FeedbackKind::FeedbackPositive as i32,
    };
    log::info!(
        "feedback result: {}",
        client
            .feedback(&fb)
            .await
            .map_err(anyhow::Error::from)?
            .message
    );

    // 7) Control：强制 persist。
    let ctrl = client
        .control(pb::ControlKind::ControlPersist)
        .await
        .map_err(anyhow::Error::from)?;
    log::info!(
        "control persist result: ok={} message={}",
        ctrl.ok,
        ctrl.message
    );

    // 8) 观察 liveliness 心跳（best-effort）。
    log::info!("observing liveliness for a few seconds...");
    let seen = client
        .observe_liveliness("", 3)
        .await
        .map_err(anyhow::Error::from)?;
    if seen == 0 {
        log::info!(
            "no liveliness sample within 3s (best-effort; ping 已证明服务可达，不影响其余步骤)"
        );
    } else {
        log::info!("liveliness observed {seen} sample(s)");
    }

    log::info!("scenario completed successfully (device side).");
    Ok(())
}

fn dump_retrieve(channel: &str, resp: &pb::RetrieveResponse) {
    log::info!(
        "[{channel}] retrieve hits={} short_history={} summary_len={}",
        resp.hits.len(),
        resp.short_history.len(),
        resp.summary.len()
    );
    for h in &resp.hits {
        if let Some(note) = &h.note {
            log::info!("  hit id={} score={:.3}", note.id, h.score);
        }
    }
}

fn retrieve_req(concept: &str) -> pb::RetrieveRequest {
    pb::RetrieveRequest {
        queries: vec![pb::PrioritizedQuery {
            priority: 10,
            query: Some(pb::MemoryRetrieveQuery {
                tag: Vec::new(),
                variant: Some(pb::memory_retrieve_query::Variant::Semantic(
                    pb::SemanticQueryList {
                        units: vec![pb::SemanticQueryUnit {
                            concept_identifier: Some(concept.to_string()),
                            description: None,
                        }],
                    },
                )),
            }),
        }],
    }
}

/// 构造一条语义记忆用于写入。
fn build_sem_note() -> soul_mem_core::memory_note::MemoryNote {
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};
    MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
        "周会".to_string(),
        ConceptType::Entity,
        "每周一次的团队例会".to_string(),
    )))
    .tags(vec!["会议".to_string()])
    .build()
    .expect("build memory note")
}
