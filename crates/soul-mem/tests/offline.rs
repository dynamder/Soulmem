//! 离线集成测试：不依赖任何网络（不打开 zenoh session）。
//!
//! 覆盖 service 编排层的本地全链路：ingest → write → retrieve → read → feedback → control(persist)，
//! 以及经 FileStore 的「落盘 → 重启恢复」。网络相关能力由 `mock-device` 手动演示，不纳入自动化测试。
//! 消息类型为 `proto/soul_mem.proto` 生成的 protobuf 类型。

use soul_mem::config::{Config, EmbeddingMode};
use soul_mem::error::Result;
use soul_mem::service::SoulMemService;
use soul_mem::store::Store;
use soul_mem::wire::convert::note_to_proto;
use soul_mem::wire::pb;
use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
use soul_mem_core::memory_note::{MemoryNoteBuilder, MemoryType};

fn demo_config(store_file: std::path::PathBuf) -> Config {
    Config {
        device_id: "offline-test".to_string(),
        zenoh_key_prefix: "soulmem_offline".to_string(),
        window_capacity: 20,
        similarity_threshold: 0.05,
        similarity_max_results: 16,
        llm_base_url: "http://127.0.0.1:9/v1".to_string(),
        llm_api_key: "demo".to_string(),
        llm_model: "demo".to_string(),
        persist_interval_secs: 0,
        consolidate_interval_secs: 0,
        forget_interval_secs: 0,
        embedding_mode: EmbeddingMode::Hash,
        store_path: Some(store_file),
    }
}

fn build_sem_note() -> soul_mem_core::memory_note::MemoryNote {
    MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
        "周会".to_string(),
        ConceptType::Entity,
        "每周一次的团队例会".to_string(),
    )))
    .tags(vec!["会议".to_string()])
    .build()
    .expect("build memory note")
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

async fn build_service(config: &Config) -> Result<SoulMemService> {
    let store = Store::from_config(config.store_path.clone());
    let loaded = store.load().await?;
    SoulMemService::from_config(config, store, loaded).await
}

#[tokio::test(flavor = "multi_thread")]
async fn offline_flow_and_restart_restore() -> Result<()> {
    let dir = tempfile::tempdir().map_err(soul_mem::error::Error::internal)?;
    let store_file = dir.path().join("snapshot.json");
    let cfg = demo_config(store_file.clone());

    let service = build_service(&cfg).await?;

    // ingest（窗口容量 20 > 2 条，不会触发需要网络的 LLM 摘要）。
    let ack = service
        .ingest(pb::IngestRequest {
            deltas: vec![
                pb::InfoDelta {
                    role: pb::MessageRole::RoleUser as i32,
                    content: "今天我们来谈谈周会安排".to_string(),
                },
                pb::InfoDelta {
                    role: pb::MessageRole::RoleAssistant as i32,
                    content: "好的，我会记录周会话题".to_string(),
                },
            ],
        })
        .await?;
    assert!(
        ack.message.contains("ingested 2"),
        "unexpected ack: {ack:?}"
    );

    // write → retrieve → read → feedback。
    let note = build_sem_note();
    let note_id = note.id().to_string();
    let written = service
        .write_note(pb::WriteNoteRequest {
            note: Some(note_to_proto(&note)?),
        })
        .await?;
    assert_eq!(written.id, note_id);

    let resp = service.retrieve(retrieve_req("周会")).await?;
    assert!(
        resp.hits
            .iter()
            .filter_map(|h| h.note.as_ref())
            .any(|n| n.id == note_id),
        "expected retrieved hit for note {note_id}, got {} hits",
        resp.hits.len()
    );
    assert!(
        !resp.short_history.is_empty(),
        "short history should contain ingested deltas"
    );

    let read = service
        .read_note(pb::ReadNoteRequest {
            id: note_id.clone(),
        })
        .await?;
    assert_eq!(read.note.map(|n| n.id).unwrap_or_default(), note_id);

    let ack = service
        .feedback(pb::Feedback {
            note_id: note_id.clone(),
            kind: pb::FeedbackKind::FeedbackPositive as i32,
        })
        .await?;
    assert!(
        ack.message.contains("positive"),
        "unexpected feedback ack: {ack:?}"
    );

    // control persist（本地落盘，无网络）。
    let ctrl = service
        .control(pb::Control {
            kind: pb::ControlKind::ControlPersist as i32,
        })
        .await?;
    assert!(ctrl.ok, "persist control should succeed: {ctrl:?}");

    // 重启：从文件快照恢复。
    let service2 = build_service(&cfg).await?;
    let resp2 = service2.retrieve(retrieve_req("周会")).await?;
    assert!(
        resp2
            .hits
            .iter()
            .filter_map(|h| h.note.as_ref())
            .any(|n| n.id == note_id),
        "after restart, note should be restored, got {} hits",
        resp2.hits.len()
    );

    Ok(())
}
