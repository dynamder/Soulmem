use std::collections::HashMap;

use soul_mem_core::memory_note::MemoryId;
use soul_mem_runtime::working_memory::record::Record;

/// 一个在本次工作循环中被提取过的记忆节点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveNode {
    pub id: MemoryId,
    pub cycle_hits: u32,
    pub total_count: usize,
}

#[derive(Debug, Clone)]
pub struct SelectConfig {
    pub top_k: usize,
    pub min_cycle_hits: u32,
}

impl Default for SelectConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            min_cycle_hits: 1,
        }
    }
}

/// 从本工作循环的提取命中中选出 top-k 活跃节点。
pub fn select_active_nodes(
    records: &HashMap<MemoryId, Record>,
    cycle_hits: &HashMap<MemoryId, u32>,
    config: &SelectConfig,
) -> Vec<ActiveNode> {
    let mut nodes: Vec<ActiveNode> = cycle_hits
        .iter()
        .filter(|(_, hits)| **hits >= config.min_cycle_hits)
        .map(|(id, hits)| ActiveNode {
            id: *id,
            cycle_hits: *hits,
            total_count: records.get(id).map(Record::retrieval_count).unwrap_or(0),
        })
        .collect();

    nodes.sort_by(|a, b| {
        b.cycle_hits
            .cmp(&a.cycle_hits)
            .then_with(|| b.total_count.cmp(&a.total_count))
            .then_with(|| a.id.cmp(&b.id))
    });
    nodes.truncate(config.top_k);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_with_retrievals(id: MemoryId, count: usize) -> Record {
        let mut record = Record::new(id);
        for _ in 0..count {
            record.record_retrieval();
        }
        record
    }

    #[test]
    fn test_top_k_truncates_and_ranks_by_cycle_hits() {
        let ids: Vec<MemoryId> = (0..5).map(|_| MemoryId::new()).collect();
        let mut records = HashMap::new();
        let mut hits = HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            records.insert(*id, record_with_retrievals(*id, 1));
            hits.insert(*id, i as u32 + 1); // 1..=5
        }

        let cfg = SelectConfig {
            top_k: 3,
            min_cycle_hits: 1,
        };
        let got = select_active_nodes(&records, &hits, &cfg);

        assert_eq!(got.len(), 3);
        assert_eq!(got.iter().map(|n| n.cycle_hits).collect::<Vec<_>>(), vec![5, 4, 3]);
        assert_eq!(got[0].id, ids[4]);
        assert_eq!(got[1].id, ids[3]);
        assert_eq!(got[2].id, ids[2]);
    }

    #[test]
    fn test_ties_break_by_total_count_then_id() {
        let ids: Vec<MemoryId> = (0..4).map(|_| MemoryId::new()).collect();
        let mut records = HashMap::new();
        let mut hits = HashMap::new();
        // 前两个累计次数高，后两个累计次数低；四个的本循环命中数相同
        let totals = [7usize, 5, 2, 2];
        for (i, id) in ids.iter().enumerate() {
            records.insert(*id, record_with_retrievals(*id, totals[i]));
            hits.insert(*id, 2u32);
        }

        let cfg = SelectConfig::default();
        let got = select_active_nodes(&records, &hits, &cfg);

        // 累计次数降序，最后两个并列时 id 升序
        let mut low_pair = [ids[2], ids[3]];
        low_pair.sort();
        let expected = vec![ids[0], ids[1], low_pair[0], low_pair[1]];
        assert_eq!(got.iter().map(|n| n.id).collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_min_cycle_hits_filters_low_signal_nodes() {
        let frequent = MemoryId::new();
        let glancing = MemoryId::new();
        let mut records = HashMap::new();
        let mut hits = HashMap::new();
        records.insert(frequent, record_with_retrievals(frequent, 4));
        records.insert(glancing, record_with_retrievals(glancing, 1));
        hits.insert(frequent, 4u32);
        hits.insert(glancing, 1u32);

        let cfg = SelectConfig {
            top_k: 8,
            min_cycle_hits: 2,
        };
        let got = select_active_nodes(&records, &hits, &cfg);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, frequent);
        assert_eq!(got[0].total_count, 4);
    }

    #[test]
    fn test_empty_cycle_hits_returns_empty() {
        let records = HashMap::new();
        let hits = HashMap::new();

        let got = select_active_nodes(&records, &hits, &SelectConfig::default());

        assert!(got.is_empty());
    }

    #[test]
    fn test_hit_without_record_degrades_to_zero_count() {
        let orphan = MemoryId::new();
        let records = HashMap::new();
        let mut hits = HashMap::new();
        hits.insert(orphan, 3u32);

        let got = select_active_nodes(&records, &hits, &SelectConfig::default());

        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, orphan);
        assert_eq!(got[0].total_count, 0);
    }

    #[test]
    fn test_top_k_zero_returns_empty() {
        let id = MemoryId::new();
        let mut records = HashMap::new();
        let mut hits = HashMap::new();
        records.insert(id, record_with_retrievals(id, 1));
        hits.insert(id, 9u32);

        let cfg = SelectConfig {
            top_k: 0,
            min_cycle_hits: 1,
        };
        assert!(select_active_nodes(&records, &hits, &cfg).is_empty());
    }
}
