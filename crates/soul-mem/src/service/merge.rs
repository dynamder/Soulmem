//! 多 query 结果合并。
//!
//! 参照 `docs/architecture/orchestration.md`：*"以每一个 MemoryNote 为单位，将分数
//! 加权平均，取 top-k"*。这里以 query 的 `priority` 为权重，对某节点在它出现过的
//! 各 query 中的分数做加权平均（未出现的 query 不计入分母）。

use std::collections::HashMap;

use soul_mem_core::memory_note::MemoryId;

/// 对若干条 `(priority, 打分序列)` 做优先级加权平均，返回降序 top-k。
///
/// 权重取 `priority`；某节点只在它出现的序列里累积分子与分母，因此不会因为
/// "没被某条 query 命中"而被摊薄。若某节点的权重和恰为 0（所有命中的 query
/// `priority` 都是 0），退化为未加权平均，而不是一律给 0——否则全部候选同分、
/// 排序失去意义。
pub fn merge_scored<'a>(
    series: impl Iterator<Item = (u32, &'a [(MemoryId, f64)])>,
    top_k: usize,
) -> Vec<(MemoryId, f64)> {
    // (加权分和, 权重和, 原始分和, 命中次数)
    let mut acc: HashMap<MemoryId, (f64, f64, f64, u32)> = HashMap::new();
    for (priority, scored) in series {
        let weight = priority as f64;
        for (id, score) in scored {
            let slot = acc.entry(*id).or_insert((0.0, 0.0, 0.0, 0));
            slot.0 += weight * *score;
            slot.1 += weight;
            slot.2 += *score;
            slot.3 += 1;
        }
    }
    let mut merged: Vec<(MemoryId, f64)> = acc
        .into_iter()
        .map(|(id, (weighted, weight, raw, count))| {
            let value = if weight > 0.0 {
                weighted / weight
            } else if count > 0 {
                raw / count as f64
            } else {
                0.0
            };
            (id, value)
        })
        .collect();
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    merged.truncate(top_k);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> MemoryId {
        MemoryId::new()
    }

    #[test]
    fn weighted_average_ignores_absent_queries() {
        let a = id();
        let b = id();
        // query1 priority=1 命中 a,b；query2 priority=3 仅命中 a
        let s1 = vec![(a, 0.2), (b, 0.8)];
        let s2 = vec![(a, 0.6)];
        let merged = merge_scored([(1, s1.as_slice()), (3, s2.as_slice())].into_iter(), 10);
        let get = |id| merged.iter().find(|(i, _)| *i == id).map(|(_, v)| *v);
        // a: (1*0.2 + 3*0.6)/(1+3) = 2.0/4 = 0.5
        assert!((get(a).expect("a") - 0.5).abs() < 1e-9);
        // b: 仅出现于 query1，分母只有 1
        assert!((get(b).expect("b") - 0.8).abs() < 1e-9);
    }

    #[test]
    fn all_zero_priority_falls_back_to_plain_average() {
        let a = id();
        let s1 = vec![(a, 0.2)];
        let s2 = vec![(a, 0.8)];
        let merged = merge_scored([(0, s1.as_slice()), (0, s2.as_slice())].into_iter(), 10);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn truncates_to_top_k_in_descending_order() {
        let scored: Vec<(MemoryId, f64)> = (0..5).map(|n| (id(), n as f64)).collect();
        let merged = merge_scored([(1, scored.as_slice())].into_iter(), 3);
        assert_eq!(merged.len(), 3);
        assert!(merged.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn empty_input_yields_empty() {
        let merged = merge_scored(std::iter::empty(), 5);
        assert!(merged.is_empty());
    }
}
