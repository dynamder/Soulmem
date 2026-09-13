use chrono::{DateTime, Utc};

/// 激活次数计入遗忘的最大有效值（超过此值的激活不再额外减缓遗忘）
pub const DEFAULT_MAX_ACTIVATION_CAP: usize = 50;

/// 艾宾浩斯遗忘曲线衰减度计算
///
/// # 公式
///   `R(t) = e^(-t / τ)`
///   - `t` = 从创建到当前经过的小时数
///   - `τ` = adjusted_half_life / ln(2)
///   - `adjusted_half_life = base_half_life_hours × (1.0 + active_factor × capped_retrievals)`
///   - `capped_retrievals = min(retrieval_count, max_activation_cap)`
///
/// 半衰期 (R = 0.5) = adjusted_half_life。
/// retrieval_count 最多计入 max_activation_cap 次，超出不再减缓遗忘。
///
/// # 返回值
/// `0.0 ~ 1.0`，`1.0` = 完全新鲜（刚刚创建），`0.0` = 完全遗忘
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

/// 计算缺失度（`1.0 - decay`）。
/// 返回值 `0.0 ~ 1.0`，越大表示遗忘越多。
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

/// 计算边衰减后的强度。
/// 返回 `original_intensity × decay`，强度随节点遗忘等比衰减。
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

/// 根据旧缺失度与时间差增量计算新的缺失度。
///
/// 由 `old_missing_degree`（在 `old_time` 时刻的状态）推算 `current_time` 时刻的缺失度，
/// 避免每次从创建时间重新计算，支持惰性更新。
///
/// 公式：`1 - (1 - old_missing_degree) × e^(-Δt / τ)`
///   - `Δt` = current_time - old_time 经过的小时数（**分数小时**，不截断）
///   - `τ` = adjusted_half_life / ln(2)，半衰期随激活次数延长（受 cap 限制）
///
/// # 为什么必须用分数小时，不能用 `num_hours()`
///
/// 调用方（`compute_and_update_missing_degree`、`compute_all_missing_degrees`、
/// `decay_edge`、`decay_graph_edge`）拿到结果后会**无条件**执行
/// `set_last_forget_time(current_time)`——已流逝的时间只会被**消费掉**，不会被延后。
/// 因此本函数必须把亚小时的时间也计入衰减：一旦截断到整小时，那部分时间就被**丢弃**，
/// 每十几分钟交互一次的负载下缺失度永远不涨，遗忘会完全冻结，衰减的复合律也不再成立。
/// 用秒换算成分数小时是对刷新粒度不敏感的前提。
pub fn update_missing_degree_incremental(
    old_missing_degree: f32,
    old_time: DateTime<Utc>,
    current_time: DateTime<Utc>,
    retrieval_count: usize,
    base_half_life_hours: f32,
    active_factor: f32,
    max_activation_cap: usize,
) -> f32 {
    let elapsed_hours = (current_time - old_time).num_seconds() as f32 / 3600.0;
    if elapsed_hours <= 0.0 {
        return old_missing_degree;
    }
    let capped = retrieval_count.min(max_activation_cap);
    let adjusted_half_life = base_half_life_hours * (1.0 + active_factor * capped as f32);
    let tau = adjusted_half_life / std::f32::consts::LN_2;
    let retention = (-elapsed_hours / tau).exp();
    1.0 - (1.0 - old_missing_degree) * retention
}

/// 计算节点经历指定时长后的强度。
///
/// # 参数
/// - `duration_hours` — 时长（小时）
/// - `initial_intensity` — 初始强度（0.0 ~ 1.0）
/// - `activation_count` — 已激活次数
/// - `active_factor` — 激活次数影响系数
/// - `half_life_hours` — 半衰期（小时）
///
/// # 公式
/// `强度 = initial_intensity × e^(-duration / τ)`
///   - `τ = adjusted_half_life / ln(2)`
///   - `adjusted_half_life = half_life_hours × (1 + active_factor × min(activation_count, CAP))`
///
/// # 案例
/// 半衰期 24h、激活 5 次（影响系数 0.1）→ 调整半衰期 `24×(1+0.1×5)=36h`，
/// τ ≈ 51.94h，初始强度 1.0 经 48h 后强度 ≈ 0.397。
pub fn node_intensity_after(
    duration_hours: f32,
    initial_intensity: f32,
    activation_count: usize,
    active_factor: f32,
    half_life_hours: f32,
) -> f32 {
    if duration_hours <= 0.0 {
        return initial_intensity;
    }
    let capped = activation_count.min(DEFAULT_MAX_ACTIVATION_CAP);
    let adjusted_half_life = half_life_hours * (1.0 + active_factor * capped as f32);
    let tau = adjusted_half_life / std::f32::consts::LN_2;
    initial_intensity * (-duration_hours / tau).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    const BASE_HALF_LIFE: f32 = 24.0;
    const ACTIVE_FACTOR: f32 = 0.1;
    const CAP: usize = 50;

    fn base_time() -> DateTime<Utc> {
        match Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0) {
            chrono::LocalResult::Single(t) => t,
            other => panic!("固定时钟解析失败: {other:?}"),
        }
    }

    fn step(old_md: f32, from: DateTime<Utc>, to: DateTime<Utc>) -> f32 {
        update_missing_degree_incremental(
            old_md,
            from,
            to,
            /*retrieval_count=*/ 0,
            BASE_HALF_LIFE,
            ACTIVE_FACTOR,
            CAP,
        )
    }

    /// 钉住衰减增量的**复合律**：把总时长切成若干段逐段施加，必须等于一次性施加总时长。
    ///
    /// 这既是数学上正确的契约，也是"与刷新粒度无关"这一可观察行为的直接表述。
    /// 它与 `test_sub_hour_elapsed_produces_nonzero_decay` 一起锁死分数小时语义——
    /// 两者都必须保持通过，否则说明 `update_missing_degree_incremental` 又退回了
    /// 会被刷新粒度影响的实现。
    #[test]
    fn test_incremental_decay_composes_over_sub_hour_refresh() {
        let base = base_time();

        // 参照：一次性跨 4 小时
        let once = step(0.0, base, base + Duration::hours(4));
        assert!(once > 0.0, "前置条件：跨 4 小时应产生衰减，实际 {once}");

        // 复现：每 30 分钟刷新一次，累计同样 4 小时
        let mut md = 0.0f32;
        let mut last = base;
        for step_no in 1..=8 {
            let now = base + Duration::minutes(30 * step_no);
            md = step(md, last, now);
            last = now;
        }

        assert!(
            (md - once).abs() < 1e-3,
            "衰减不满足复合律（亚小时刷新丢弃了流逝时间）：逐步累积 md={md}，一次性 md={once}"
        );
    }

    /// 最小可观察证据：30 分钟流逝必须产生非零衰减（曾被 `num_hours()` 截断为 0）。
    #[test]
    fn test_sub_hour_elapsed_produces_nonzero_decay() {
        let base = base_time();
        let after_30min = step(0.0, base, base + Duration::minutes(30));
        assert!(
            after_30min > 0.0,
            "30 分钟流逝被 num_hours() 截断为 0，衰减完全丢失：{after_30min}"
        );
    }
}
