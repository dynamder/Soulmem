use chrono::{DateTime, Utc};

/// 激活次数计入遗忘的最大有效值（超过此值的激活不再额外减缓遗忘）
pub const DEFAULT_MAX_ACTIVATION_CAP: usize = 50;

/// 艾宾浩斯遗忘曲线衰减度计算
///
/// 公式: R(t) = e^(-t / τ)
///   t = 从创建到当前经过的小时数
///   τ = adjusted_half_life / ln(2)
///   adjusted_half_life = base_half_life_hours * (1.0 + active_factor * capped_retrievals)
///   capped_retrievals = min(retrieval_count, max_activation_cap)
///
/// 半衰期 (R = 0.5) = adjusted_half_life
/// 当 retrieval_count 增加时（最多计入 max_activation_cap 次），半衰期延长，衰减变慢
///
/// 返回值: 0.0 ~ 1.0 的强度值
///   - 1.0 = 完全新鲜（刚刚创建）
///   - 0.0 = 完全遗忘
pub fn ebbinghaus_decay(
    create_time: DateTime<Utc>,
    retrieval_count: usize,
    current_time: DateTime<Utc>,
    base_half_life_hours: f32,
    active_factor: f32,
    max_activation_cap: usize,
) -> f32 {
    let elapsed_hours = (current_time - create_time).num_hours() as f32;
    if elapsed_hours <= 0.0 {
        return 1.0;
    }
    let capped = retrieval_count.min(max_activation_cap);
    let adjusted_half_life = base_half_life_hours * (1.0 + active_factor * capped as f32);
    let tau = adjusted_half_life / std::f32::consts::LN_2;
    (-elapsed_hours / tau).exp()
}

/// 计算缺失度 (1.0 - decay)
/// 返回值: 0.0 ~ 1.0，越大表示遗忘越多
pub fn compute_missing_degree(
    create_time: DateTime<Utc>,
    retrieval_count: usize,
    current_time: DateTime<Utc>,
    base_half_life_hours: f32,
    active_factor: f32,
    max_activation_cap: usize,
) -> f32 {
    1.0 - ebbinghaus_decay(
        create_time,
        retrieval_count,
        current_time,
        base_half_life_hours,
        active_factor,
        max_activation_cap,
    )
}

/// 计算边衰减后的强度
pub fn edge_decay_intensity(
    original_intensity: f64,
    create_time: DateTime<Utc>,
    retrieval_count: usize,
    current_time: DateTime<Utc>,
    base_half_life_hours: f32,
    active_factor: f32,
    max_activation_cap: usize,
) -> f64 {
    original_intensity
        * ebbinghaus_decay(
            create_time,
            retrieval_count,
            current_time,
            base_half_life_hours,
            active_factor,
            max_activation_cap,
        ) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_fresh_node() {
        let now = Utc::now();
        let d = ebbinghaus_decay(now, 0, now, 24.0, 0.1, 50);
        assert!((d - 1.0).abs() < 1e-6, "fresh node should have intensity 1.0");
    }

    #[test]
    fn test_half_life_baseline() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let after = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
        let d = ebbinghaus_decay(created, 0, after, 24.0, 0.0, 50);
        assert!((d - 0.5).abs() < 0.01, "after one half-life, expected ~0.5, got {}", d);
    }

    #[test]
    fn test_retrieval_slows_decay() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let after = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
        let d0 = ebbinghaus_decay(created, 0, after, 24.0, 0.1, 50);
        let d1 = ebbinghaus_decay(created, 10, after, 24.0, 0.1, 50);
        assert!(d1 > d0, "more retrievals should slow decay: {} vs {}", d0, d1);
    }

    #[test]
    fn test_activation_cap() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let after = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
        let d50 = ebbinghaus_decay(created, 50, after, 24.0, 0.1, 50);
        let d200 = ebbinghaus_decay(created, 200, after, 24.0, 0.1, 50);
        assert!(
            (d50 - d200).abs() < 0.001,
            "activations beyond cap should not affect decay: {} vs {}",
            d50, d200
        );
    }

    #[test]
    fn test_missing_degree_range() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let far_future = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let md = compute_missing_degree(created, 0, far_future, 24.0, 0.0, 50);
        assert!(md > 0.9, "long time should give high missing degree");
    }

    #[test]
    fn test_edge_decay() {
        let created = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let after = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();
        let decayed = edge_decay_intensity(1.0, created, 0, after, 24.0, 0.0, 50);
        assert!((decayed - 0.5).abs() < 0.01, "edge should decay with node");
    }
}
