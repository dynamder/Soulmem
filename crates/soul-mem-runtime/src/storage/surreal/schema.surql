-- SoulMem SurrealDB schema（memory_note / memory_link）
-- 由 crates/soul-mem-runtime/src/storage/surreal/repository.rs 经 include_str! 引入，
-- 幂等执行（全部 IF NOT EXISTS）。
-- 槽位列名与 mapper::EmbeddingSlot::column() 保持一致（有测试防漂移）。

-- ============ memory_note ============
DEFINE TABLE IF NOT EXISTS memory_note SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS memory_id ON TABLE memory_note TYPE string;
DEFINE FIELD IF NOT EXISTS tags ON TABLE memory_note TYPE array<string>;
DEFINE FIELD IF NOT EXISTS retrieval_count ON TABLE memory_note TYPE int;
DEFINE FIELD IF NOT EXISTS create_time ON TABLE memory_note TYPE datetime;
DEFINE FIELD IF NOT EXISTS last_accessed_time ON TABLE memory_note TYPE datetime;
DEFINE FIELD IF NOT EXISTS missing_degree ON TABLE memory_note TYPE float;
DEFINE FIELD IF NOT EXISTS last_forget_time ON TABLE memory_note TYPE datetime;
DEFINE FIELD IF NOT EXISTS mem_type ON TABLE memory_note TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS variant_emb ON TABLE memory_note TYPE any;

-- 槽位列（多列 HNSW，每列一个索引；缺失列 = NONE，不建值）
DEFINE FIELD IF NOT EXISTS tag_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sem_content_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sem_aliases_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sem_description_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_narrative_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_loc_name_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_loc_coord_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_part_name_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_part_role_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_env_atmosphere_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_env_tone_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_event_action_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_event_initiator_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_ctx_event_target_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_loc_name_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_loc_coord_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_part_name_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_part_role_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_env_atmosphere_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_env_tone_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_event_action_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_event_initiator_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS sit_event_target_emb ON TABLE memory_note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS fused_self_emb ON TABLE memory_note TYPE option<array<float>>;

-- 唯一键 + 向量索引
DEFINE INDEX IF NOT EXISTS idx_memory_note_memory_id ON TABLE memory_note COLUMNS memory_id UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_tag ON TABLE memory_note FIELDS tag_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sem_content ON TABLE memory_note FIELDS sem_content_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sem_aliases ON TABLE memory_note FIELDS sem_aliases_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sem_description ON TABLE memory_note FIELDS sem_description_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_narrative ON TABLE memory_note FIELDS sit_narrative_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_loc_name ON TABLE memory_note FIELDS sit_ctx_loc_name_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_loc_coord ON TABLE memory_note FIELDS sit_ctx_loc_coord_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_part_name ON TABLE memory_note FIELDS sit_ctx_part_name_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_part_role ON TABLE memory_note FIELDS sit_ctx_part_role_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_env_atmosphere ON TABLE memory_note FIELDS sit_ctx_env_atmosphere_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_env_tone ON TABLE memory_note FIELDS sit_ctx_env_tone_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_event_action ON TABLE memory_note FIELDS sit_ctx_event_action_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_event_initiator ON TABLE memory_note FIELDS sit_ctx_event_initiator_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_ctx_event_target ON TABLE memory_note FIELDS sit_ctx_event_target_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_loc_name ON TABLE memory_note FIELDS sit_loc_name_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_loc_coord ON TABLE memory_note FIELDS sit_loc_coord_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_part_name ON TABLE memory_note FIELDS sit_part_name_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_part_role ON TABLE memory_note FIELDS sit_part_role_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_env_atmosphere ON TABLE memory_note FIELDS sit_env_atmosphere_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_env_tone ON TABLE memory_note FIELDS sit_env_tone_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_event_action ON TABLE memory_note FIELDS sit_event_action_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_event_initiator ON TABLE memory_note FIELDS sit_event_initiator_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_sit_event_target ON TABLE memory_note FIELDS sit_event_target_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;
DEFINE INDEX IF NOT EXISTS idx_memory_note_fused_self ON TABLE memory_note FIELDS fused_self_emb HNSW DIMENSION 512 TYPE F32 DISTANCE COSINE;

-- ============ memory_link（图边记录）============
DEFINE TABLE IF NOT EXISTS memory_link SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS link_id ON TABLE memory_link TYPE string;
DEFINE FIELD IF NOT EXISTS `in` ON TABLE memory_link TYPE record<memory_note>;
DEFINE FIELD IF NOT EXISTS out ON TABLE memory_link TYPE record<memory_note>;
DEFINE FIELD IF NOT EXISTS intensity ON TABLE memory_link TYPE number;
DEFINE FIELD IF NOT EXISTS missing_degree ON TABLE memory_link TYPE float;
DEFINE FIELD IF NOT EXISTS last_forget_time ON TABLE memory_link TYPE datetime;
DEFINE FIELD IF NOT EXISTS link_type ON TABLE memory_link TYPE object FLEXIBLE;

DEFINE INDEX IF NOT EXISTS idx_memory_link_link_id ON TABLE memory_link COLUMNS link_id UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_memory_link_in ON TABLE memory_link COLUMNS `in`;
DEFINE INDEX IF NOT EXISTS idx_memory_link_out ON TABLE memory_link COLUMNS out;
