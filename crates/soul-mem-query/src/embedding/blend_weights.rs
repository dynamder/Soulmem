/// 嵌入计算链中所有可调权重参数。
/// 默认值与当前代码中的硬编码值完全一致。
#[derive(Debug, Clone, PartialEq)]
pub struct BlendWeights {
    // — tag/variant 顶层融合 —
    // tag 通道权重降为 0.3：标签在语义不匹配时会拖累 narrative/概念命中，
    // 且标签共现（如"人物"）易抬升无关实体，因此降低其主导作用。
    pub tag: f32,
    pub variant: f32,

    /// 检索评分中 embedding 余弦相似度所占权重，`1 - string_blend_alpha` 为 Jaro-Winkler 字符串得分权重。
    /// 用于 `EmbeddedMemoryNote::compute_fused` 的顶层混合。
    pub string_blend_alpha: f32,

    // — Semantic 子权重 —
    pub sem_concept: f32,     // concept vs description (0.5)
    pub sem_description: f32, // (0.5)

    // — Situation: Location —
    pub sit_location_name: f32,
    pub sit_location_coord: f32,

    // — Situation: Participant —
    pub sit_participant_name: f32,
    pub sit_participant_role: f32,

    // — Situation: Environment —
    pub sit_env_atmosphere: f32,
    pub sit_env_tone: f32,

    // — Situation: Event —
    pub sit_event_initiator: f32,
    pub sit_event_target: f32,
    pub sit_event_action: f32,
    /// 当只有 initiator 时 action 的权重（initiator = 1 - this）
    pub sit_event_initiator_only_action: f32,
    /// 当只有 target 时 action 的权重（target = 1 - this）
    pub sit_event_target_only_action: f32,
}

impl Default for BlendWeights {
    fn default() -> Self {
        Self {
            tag: 0.3,
            variant: 0.7,
            string_blend_alpha: 0.6,
            sem_concept: 0.5,
            sem_description: 0.5,
            sit_location_name: 0.6,
            sit_location_coord: 0.4,
            sit_participant_name: 0.6,
            sit_participant_role: 0.4,
            sit_env_atmosphere: 0.5,
            sit_env_tone: 0.5,
            sit_event_initiator: 0.3,
            sit_event_target: 0.3,
            sit_event_action: 0.4,
            sit_event_initiator_only_action: 0.6,
            sit_event_target_only_action: 0.6,
        }
    }
}

impl BlendWeights {
    /// 归一化到「每一组权重之和为 1」的等价权重。
    ///
    /// 评分契约要求各通道的加权和落在 `[0, 1]`，这只在每组权重之和为 1、且单个权重
    /// 都属于 `[0, 1]` 时成立。例如 `{ tag: 0.9, variant: 0.9 }` 在两条通道都完美匹配时给出 1.8。
    /// `MemoryRetrieveQueryEmbedding::with_weights` 会先调用本方法。
    pub fn normalized(&self) -> Self {
        let d = Self::default();
        let [tag, variant] = normalize_group([self.tag, self.variant], [d.tag, d.variant]);
        let [sem_concept, sem_description] = normalize_group(
            [self.sem_concept, self.sem_description],
            [d.sem_concept, d.sem_description],
        );
        let [sit_location_name, sit_location_coord] = normalize_group(
            [self.sit_location_name, self.sit_location_coord],
            [d.sit_location_name, d.sit_location_coord],
        );
        let [sit_participant_name, sit_participant_role] = normalize_group(
            [self.sit_participant_name, self.sit_participant_role],
            [d.sit_participant_name, d.sit_participant_role],
        );
        let [sit_env_atmosphere, sit_env_tone] = normalize_group(
            [self.sit_env_atmosphere, self.sit_env_tone],
            [d.sit_env_atmosphere, d.sit_env_tone],
        );
        let [sit_event_initiator, sit_event_target, sit_event_action] = normalize_group(
            [
                self.sit_event_initiator,
                self.sit_event_target,
                self.sit_event_action,
            ],
            [
                d.sit_event_initiator,
                d.sit_event_target,
                d.sit_event_action,
            ],
        );

        Self {
            tag,
            variant,
            string_blend_alpha: clamp_unit(self.string_blend_alpha, d.string_blend_alpha),
            sem_concept,
            sem_description,
            sit_location_name,
            sit_location_coord,
            sit_participant_name,
            sit_participant_role,
            sit_env_atmosphere,
            sit_env_tone,
            sit_event_initiator,
            sit_event_target,
            sit_event_action,
            sit_event_initiator_only_action: clamp_unit(
                self.sit_event_initiator_only_action,
                d.sit_event_initiator_only_action,
            ),
            sit_event_target_only_action: clamp_unit(
                self.sit_event_target_only_action,
                d.sit_event_target_only_action,
            ),
        }
    }
}

/// 把一组相关权重按和归一化。组内出现非有限值/负值、或和不为正时整组回退到 `defaults`，
/// 避免把「未配置」变成 NaN 或反向权重。
fn normalize_group<const N: usize>(values: [f32; N], defaults: [f32; N]) -> [f32; N] {
    let sum: f32 = values.iter().sum();
    if values.iter().all(|v| v.is_finite() && *v >= 0.0) && sum > 0.0 {
        values.map(|v| v / sum)
    } else {
        defaults
    }
}

/// 单权重钳到 `[0, 1]`；非有限值退回默认。
fn clamp_unit(value: f32, default: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn pairs_are_normalized_to_unit_sum() {
        let bw = BlendWeights {
            tag: 0.9,
            variant: 0.9,
            sem_concept: 2.0,
            sem_description: 6.0,
            ..BlendWeights::default()
        };
        let n = bw.normalized();
        assert_close(n.tag, 0.5);
        assert_close(n.variant, 0.5);
        assert_close(n.sem_concept, 0.25);
        assert_close(n.sem_description, 0.75);
    }

    #[test]
    fn event_triple_is_normalized_to_unit_sum() {
        let bw = BlendWeights {
            sit_event_initiator: 1.0,
            sit_event_target: 1.0,
            sit_event_action: 2.0,
            ..BlendWeights::default()
        };
        let n = bw.normalized();
        assert_close(n.sit_event_initiator, 0.25);
        assert_close(n.sit_event_target, 0.25);
        assert_close(n.sit_event_action, 0.5);
    }

    #[test]
    fn zero_within_a_positive_group_is_kept() {
        let bw = BlendWeights {
            sit_location_name: 1.0,
            sit_location_coord: 0.0,
            ..BlendWeights::default()
        };
        let n = bw.normalized();
        assert_close(n.sit_location_name, 1.0);
        assert_close(n.sit_location_coord, 0.0);
    }

    #[test]
    fn zero_sum_group_falls_back_to_default() {
        let d = BlendWeights::default();
        let n = BlendWeights {
            tag: 0.0,
            variant: 0.0,
            ..BlendWeights::default()
        }
        .normalized();
        assert_eq!((n.tag, n.variant), (d.tag, d.variant));
    }

    #[test]
    fn negative_or_non_finite_group_falls_back_to_default() {
        let d = BlendWeights::default();
        let n = BlendWeights {
            sem_concept: -1.0,
            sem_description: 0.5,
            ..BlendWeights::default()
        }
        .normalized();
        assert_eq!(
            (n.sem_concept, n.sem_description),
            (d.sem_concept, d.sem_description)
        );

        let n = BlendWeights {
            sit_env_atmosphere: f32::NAN,
            sit_env_tone: 1.0,
            ..BlendWeights::default()
        }
        .normalized();
        assert_eq!(
            (n.sit_env_atmosphere, n.sit_env_tone),
            (d.sit_env_atmosphere, d.sit_env_tone)
        );
    }

    #[test]
    fn single_weights_are_clamped() {
        let d = BlendWeights::default();
        let n = BlendWeights {
            string_blend_alpha: 2.0,
            sit_event_initiator_only_action: -0.5,
            sit_event_target_only_action: f32::INFINITY,
            ..BlendWeights::default()
        }
        .normalized();
        assert_eq!(n.string_blend_alpha, 1.0);
        assert_eq!(n.sit_event_initiator_only_action, 0.0);
        assert_eq!(
            n.sit_event_target_only_action,
            d.sit_event_target_only_action
        );
    }
}
