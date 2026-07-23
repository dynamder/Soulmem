/// 嵌入计算链中所有可调权重参数。
/// 默认值与当前代码中的硬编码值完全一致。
#[derive(Debug, Clone, PartialEq)]
pub struct BlendWeights {
    // — tag/variant 顶层融合 —
    pub tag: f32,
    pub variant: f32,

    // — Semantic 子权重 —
    pub sem_concept_main: f32, // concept vs aliases within concept_id score (0.5)
    pub sem_concept_aliases: f32, // (0.5)
    pub sem_concept: f32,      // concept vs description (0.5)
    pub sem_description: f32,  // (0.5)

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
            tag: 0.4,
            variant: 0.6,
            sem_concept_main: 0.5,
            sem_concept_aliases: 0.5,
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
