//! 巩固：摘要 → 新记忆节点 → 与活跃节点连边 → 写回长期存储。
//!
//! [`consolidate`] 是最上层封装：只读取工作记忆，返回一个由调用方负责应用的
//! [`ConsolidationPatch`]，本身不修改任何状态。

use soul_mem_core::memory_links::MemoryLink;
use soul_mem_core::memory_note::MemoryNote;
use soul_mem_llm::{LlmEngine, LlmError};
use soul_mem_runtime::working_memory::WorkingMemory;
use soul_mem_runtime::working_memory::sliding_window::Summary;
use thiserror::Error;

use self::generate::generate_memories_from_summary;
use self::link::generate_links;
use self::select::{ActiveNode, SelectConfig, select_active_nodes};

pub mod generate;
pub mod link;
pub mod persist;
pub mod select;

/// 一次巩固产出的、待应用到工作记忆的变更。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConsolidationPatch {
    /// 从摘要中新拆出的记忆节点，尚未嵌入。
    pub new_notes: Vec<MemoryNote>,
    /// 活跃节点与新节点之间的新边；两端都在本次巩固的已知节点集合里。
    pub links: Vec<MemoryLink>,
}

#[derive(Debug, Error)]
pub enum ConsolidationError {
    /// 第一次 LLM 调用（摘要 → 节点）失败；此时没有生成任何节点。
    #[error("generating memory notes from summary failed: {0}")]
    Generate(#[source] LlmError),

    /// 第二次 LLM 调用（节点 → 边）失败。已生成的节点一并返回，调用方可以
    /// 直接使用它们，或只重试连边。
    #[error("generating links for consolidated notes failed: {source}")]
    Link {
        #[source]
        source: LlmError,
        notes: Vec<MemoryNote>,
    },
}

impl ConsolidationError {
    /// 连边阶段失败时，本次已经成功生成的新节点；生成阶段失败时为 `None`。
    pub fn generated_notes(&self) -> Option<&[MemoryNote]> {
        match self {
            Self::Generate(_) => None,
            Self::Link { notes, .. } => Some(notes),
        }
    }
}

/// 巩固最上层封装：摘要 → 新节点 → 与活跃节点连边。
///
/// `working_mem` 只读：它的提取记录（`Record` 与本工作循环命中数）用于择出活跃
/// 节点，其记忆簇用于给连边提示词提供节点内容。返回的 patch 由调用方负责嵌入、
/// 写入工作记忆并持久化，本函数不产生任何可变状态。
///
/// 摘要为空或没有拆出节点时返回空 patch，不发起后续 LLM 调用。
pub async fn consolidate(
    summary: &Summary,
    working_mem: &WorkingMemory,
    engine: &LlmEngine,
) -> Result<ConsolidationPatch, ConsolidationError> {
    if summary.get().trim().is_empty() {
        return Ok(ConsolidationPatch::default());
    }

    let new_notes = generate_memories_from_summary(summary, None, engine)
        .await
        .map_err(ConsolidationError::Generate)?;
    if new_notes.is_empty() {
        return Ok(ConsolidationPatch {
            new_notes,
            links: Vec::new(),
        });
    }

    let active = active_nodes(working_mem);
    let links = match generate_links(working_mem, &active, &new_notes, engine).await {
        Ok(links) => links,
        Err(source) => {
            return Err(ConsolidationError::Link {
                source,
                notes: new_notes,
            });
        }
    };

    Ok(ConsolidationPatch { new_notes, links })
}

fn active_nodes(working_mem: &WorkingMemory) -> Vec<ActiveNode> {
    working_mem.read_records(|records| {
        working_mem.read_cycle_hits(|cycle_hits| {
            select_active_nodes(records, cycle_hits, &SelectConfig::default())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use parking_lot::Mutex;
    use soul_mem_core::memory_note::MemoryId;
    use soul_mem_core::memory_note::MemoryNoteBuilder;
    use soul_mem_core::memory_note::MemoryType;
    use soul_mem_core::memory_note::situation_mem::{
        Context, Environment, Location, Participant, SpecificSituation, SituationType,
    };
    use soul_mem_llm::{BackendInfo, ChatBackend, Completion, EventStream, LlmErrorKind, Task};
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::{
        EmbeddedMemoryNote, MemoryEmbedding, MemoryEmbeddingVariant,
    };
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const GENERATED_JSON: &str = r#"[{"type":"semantic","content":"new fact","description":"desc","concept_type":"Entity"}]"#;

    #[derive(Clone, Copy)]
    enum LinkStep {
        Edges,
        Fail,
    }

    struct Backend {
        calls: AtomicUsize,
        link_step: LinkStep,
        link_prompt: Mutex<Option<String>>,
    }

    impl Backend {
        fn new(link_step: LinkStep) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                link_step,
                link_prompt: Mutex::new(None),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        fn link_prompt(&self) -> String {
            self.link_prompt.lock().clone().unwrap_or_default()
        }
    }

    #[async_trait]
    impl ChatBackend for Backend {
        fn info(&self) -> BackendInfo {
            BackendInfo::new("consolidate-test")
        }

        async fn complete(&self, task: Task) -> Result<Completion, LlmError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                return Ok(Completion::new(GENERATED_JSON));
            }

            let prompt = task
                .messages
                .last()
                .map(|message| message.text.as_str())
                .unwrap_or_default();
            *self.link_prompt.lock() = Some(prompt.to_string());

            match self.link_step {
                LinkStep::Fail => Err(LlmError::new(LlmErrorKind::Timeout, "link failed")),
                LinkStep::Edges => {
                    let new_id = find_prompt_id(prompt, true).ok_or_else(|| {
                        LlmError::new(LlmErrorKind::Internal, "new node id missing from prompt")
                    })?;
                    let active_id = find_prompt_id(prompt, false).ok_or_else(|| {
                        LlmError::new(LlmErrorKind::Internal, "active node id missing from prompt")
                    })?;
                    let unknown = MemoryId::new();
                    Ok(Completion::new(format!(
                        r#"[{{"from":"{active_id}","to":"{new_id}","link_type":{{"Sem":{{"verb":"relates","confidence":1.0}}}}}},{{"from":"{new_id}","to":"{new_id}","link_type":{{"Sem":{{"verb":"self","confidence":1.0}}}}}},{{"from":"{unknown}","to":"{new_id}","link_type":{{"Sem":{{"verb":"ghost","confidence":1.0}}}}}}]"#
                    )))
                }
            }
        }

        async fn stream(&self, _task: Task) -> Result<EventStream, LlmError> {
            Err(LlmError::unsupported("test backend has no stream"))
        }
    }

    fn find_prompt_id(prompt: &str, new: bool) -> Option<String> {
        prompt
            .lines()
            .filter(|line| line.starts_with("- "))
            .find(|line| line.contains("[NEW]") == new)
            .and_then(|line| line.strip_prefix("- "))
            .and_then(|line| line.split_whitespace().next())
            .map(str::to_string)
    }

    fn summary(text: &str) -> Summary {
        let mut summary = Summary::new();
        summary.update(text);
        summary
    }

    fn specific_note() -> MemoryNote {
        MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(
            SpecificSituation::new(
                String::new(),
                Utc::now(),
                Context::new(
                    Some(Location {
                        name: "火锅店".to_string(),
                        coordinates: String::new(),
                    }),
                    vec![Participant {
                        name: "小白".to_string(),
                        role: "朋友".to_string(),
                    }],
                    vec![],
                    vec![],
                    Environment {
                        atmosphere: "热闹".to_string(),
                        tone: String::new(),
                    },
                    vec![],
                ),
            ),
        )))
        .build()
        .unwrap()
    }

    fn embedded(note: MemoryNote) -> EmbeddedMemoryNote {
        let embedding = MemoryEmbedding::new(
            EmbeddingVec::zero(512),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                EmbeddingVec::zero(512),
                EmbeddingVec::zero(512),
                EmbeddingVec::zero(512),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    #[tokio::test]
    async fn consolidate_returns_valid_links_and_builds_prompt_without_empty_sections() {
        let working_mem = WorkingMemory::new(10);
        let active = specific_note();
        let active_id = active.id();
        working_mem.add_node(embedded(active));
        working_mem.record_retrieval(active_id);

        let backend = Arc::new(Backend::new(LinkStep::Edges));
        let engine = LlmEngine::new(backend.clone()).with_whole_call_retries(0);
        let patch = consolidate(&summary("a new fact"), &working_mem, &engine)
            .await
            .unwrap();

        assert_eq!(backend.call_count(), 2);
        assert_eq!(patch.new_notes.len(), 1);
        assert_eq!(patch.links.len(), 1, "只应保留两端已知且非自环的边");
        assert_eq!(patch.links[0].from(), active_id);
        assert_eq!(patch.links[0].to(), patch.new_notes[0].id());

        let prompt = backend.link_prompt();
        assert!(
            prompt.contains("[specific_situation] 地点=火锅店"),
            "{prompt}"
        );
        assert!(prompt.contains("参与者=小白"), "{prompt}");
        assert!(prompt.contains("氛围=热闹"), "{prompt}");
        assert!(!prompt.contains("情绪="), "{prompt}");
        assert!(!prompt.contains("事件="), "{prompt}");
        assert!(prompt.contains("[NEW]"), "{prompt}");

        assert_eq!(
            working_mem
                .memory_cluster()
                .read_or_compute(|cluster| cluster.graph().node_count()),
            1,
            "consolidate 不应修改工作记忆"
        );
    }

    #[tokio::test]
    async fn link_failure_returns_generated_notes_and_keeps_working_memory_unchanged() {
        let working_mem = WorkingMemory::new(10);
        let active = specific_note();
        let active_id = active.id();
        working_mem.add_node(embedded(active));
        working_mem.record_retrieval(active_id);

        let backend = Arc::new(Backend::new(LinkStep::Fail));
        let engine = LlmEngine::new(backend.clone()).with_whole_call_retries(0);
        let error = consolidate(&summary("a new fact"), &working_mem, &engine)
            .await
            .unwrap_err();

        assert!(matches!(error, ConsolidationError::Link { .. }));
        let notes = error.generated_notes().expect("连边失败必须保留已生成节点");
        assert_eq!(notes.len(), 1);
        assert!(matches!(
            notes[0].mem_type(),
            MemoryType::Semantic(semantic) if semantic.content == "new fact"
        ));
        assert_eq!(backend.call_count(), 2);
        assert_eq!(
            working_mem
                .memory_cluster()
                .read_or_compute(|cluster| cluster.graph().node_count()),
            1
        );
    }
}
