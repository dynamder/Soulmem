use std::collections::HashSet;

use soul_mem_core::memory_note::MemoryId;

use crate::engine::retrieve::data::{ActionMetrics, RankingMetrics};

pub fn dcg(relevance: &[f64]) -> f64 {
    relevance
        .iter()
        .enumerate()
        .map(|(i, &r)| (2_f64.powf(r) - 1.0) / (i as f64 + 2.0).log2())
        .sum()
}

pub fn compute_ranking_metrics(
    retrieved: &[MemoryId],
    ground_truth: &[MemoryId],
    test_k_values: &[usize],
) -> RankingMetrics {
    let gt_set: HashSet<&MemoryId> = ground_truth.iter().collect();
    let num_relevant = ground_truth.len();
    if num_relevant == 0 {
        return RankingMetrics {
            recall_at: test_k_values.iter().map(|&k| (k, 0.0)).collect(),
            precision_at: test_k_values.iter().map(|&k| (k, 0.0)).collect(),
            mrr: 0.0,
            ndcg_at: test_k_values.iter().map(|&k| (k, 0.0)).collect(),
            hit_rate: 0.0,
        };
    }

    let max_k = *test_k_values
        .iter()
        .max()
        .unwrap_or(&1)
        .min(&retrieved.len());

    let mut recall = Vec::new();
    let mut precision = Vec::new();
    let mut ndcg_scores = Vec::new();
    let mut rr = 0.0;
    let mut hit = false;

    let mut ideal_relevance: Vec<f64> = ground_truth.iter().map(|_| 1.0).collect();
    ideal_relevance.truncate(max_k);
    if ideal_relevance.len() < max_k {
        ideal_relevance.extend(vec![0.0; max_k - ideal_relevance.len()]);
    }
    let ideal_dcg = dcg(&ideal_relevance);

    for &k in test_k_values {
        let k = k.min(retrieved.len());
        let retrieved_k = &retrieved[..k];
        let relevant_count = retrieved_k.iter().filter(|id| gt_set.contains(*id)).count();

        let r = relevant_count as f64 / num_relevant as f64;
        let p = relevant_count as f64 / k as f64;
        recall.push((k, r));
        precision.push((k, p));

        let actual_relevance: Vec<f64> = retrieved_k
            .iter()
            .map(|id| if gt_set.contains(id) { 1.0 } else { 0.0 })
            .collect();
        let actual_dcg = dcg(&actual_relevance);
        let ndcg = if ideal_dcg > 0.0 {
            actual_dcg / ideal_dcg
        } else {
            0.0
        };
        ndcg_scores.push((k, ndcg));

        if !hit && relevant_count > 0 {
            let first_rel = retrieved_k
                .iter()
                .position(|id| gt_set.contains(id))
                .unwrap_or(usize::MAX);
            if first_rel < k {
                rr = 1.0 / (first_rel as f64 + 1.0);
                hit = true;
            }
        }
    }

    RankingMetrics {
        recall_at: recall,
        precision_at: precision,
        mrr: rr,
        ndcg_at: ndcg_scores,
        hit_rate: if hit { 1.0 } else { 0.0 },
    }
}

pub fn compute_action_metrics(
    retrieved_actions: &[MemoryId],
    expected_actions: &[MemoryId],
    test_k_values: &[usize],
) -> ActionMetrics {
    let gt_set: HashSet<&MemoryId> = expected_actions.iter().collect();
    let total_expected = expected_actions.len();

    let hit = retrieved_actions.iter().any(|id| gt_set.contains(id));

    let recall_at = test_k_values
        .iter()
        .map(|&k| {
            let k = k.min(retrieved_actions.len());
            let count = retrieved_actions[..k]
                .iter()
                .filter(|id| gt_set.contains(*id))
                .count();
            let r = if total_expected > 0 {
                count as f64 / total_expected as f64
            } else {
                0.0
            };
            (k, r)
        })
        .collect();

    ActionMetrics {
        action_hit_rate: if hit { 1.0 } else { 0.0 },
        action_recall_at: recall_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_note::MemoryId;

    fn make_ids(count: usize) -> Vec<MemoryId> {
        (0..count).map(|_| MemoryId::new()).collect()
    }

    #[test]
    fn test_dcg_identical_relevance() {
        let rel = vec![1.0, 1.0, 1.0];
        let score = dcg(&rel);
        assert!(score > 0.0);
    }

    #[test]
    fn test_dcg_zero() {
        let rel = vec![0.0, 0.0, 0.0];
        let score = dcg(&rel);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_ranking_metrics_perfect_match() {
        let ids = make_ids(1);
        let retrieved: Vec<MemoryId> = ids.iter().copied().collect();
        let ground_truth: Vec<MemoryId> = ids.iter().copied().collect();
        let metrics = compute_ranking_metrics(&retrieved, &ground_truth, &[1]);
        for (_, r) in &metrics.recall_at {
            assert!((*r - 1.0).abs() < 1e-6);
        }
        assert!((metrics.mrr - 1.0).abs() < 1e-6);
        assert!((metrics.hit_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ranking_metrics_no_match() {
        let relevant = make_ids(3);
        let retrieved = make_ids(5);
        let metrics = compute_ranking_metrics(&retrieved, &relevant, &[1, 3, 5]);
        for (_, r) in &metrics.recall_at {
            assert!((*r - 0.0).abs() < 1e-6);
        }
        assert!((metrics.mrr - 0.0).abs() < 1e-6);
        assert!((metrics.hit_rate - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ranking_metrics_partial_match() {
        let relevant = make_ids(2);
        let (a, b) = (relevant[0], relevant[1]);
        let mut retrieved = make_ids(3);
        retrieved[0] = a;
        retrieved[2] = b;
        let metrics = compute_ranking_metrics(&retrieved, &relevant, &[1, 2, 3]);
        assert!((metrics.recall_at[0].1 - 0.5).abs() < 1e-6);
        assert!((metrics.recall_at[2].1 - 1.0).abs() < 1e-6);
        assert!((metrics.mrr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ranking_metrics_mrr_position() {
        let relevant = make_ids(1);
        let target = relevant[0];
        let mut retrieved = make_ids(5);
        retrieved[3] = target;
        let metrics = compute_ranking_metrics(&retrieved, &relevant, &[5]);
        assert!((metrics.mrr - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_action_metrics_hit() {
        let actions = make_ids(3);
        let retrieved = vec![actions[0]];
        let metrics = compute_action_metrics(&retrieved, &actions, &[1, 3]);
        assert!((metrics.action_hit_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_action_metrics_no_hit() {
        let actions = make_ids(3);
        let retrieved = make_ids(3);
        let metrics = compute_action_metrics(&retrieved, &actions, &[1, 3]);
        assert!((metrics.action_hit_rate - 0.0).abs() < 1e-6);
    }
}
