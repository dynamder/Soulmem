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
