pub const TABLE_MEMORY_NOTE: &str = "memory_note";
pub const TABLE_MEMORY_LINK: &str = "memory_link";
pub const TABLE_RETRIEVAL_EVENT: &str = "retrieval_event";
pub const TABLE_FEEDBACK_EVENT: &str = "feedback_event";

pub const BOOTSTRAP_STATEMENTS: &[&str] = &[
    "DEFINE TABLE memory_note SCHEMAFULL;",
    "DEFINE FIELD id ON TABLE memory_note TYPE string;",
    "DEFINE FIELD tags ON TABLE memory_note TYPE array<string>;",
    "DEFINE FIELD retrieval_count ON TABLE memory_note TYPE int;",
    "DEFINE FIELD create_time ON TABLE memory_note TYPE datetime;",
    "DEFINE FIELD last_accessed_time ON TABLE memory_note TYPE datetime;",
    "DEFINE FIELD kind ON TABLE memory_note TYPE string;",
    "DEFINE FIELD situation_subtype ON TABLE memory_note TYPE option<string>;",
    "DEFINE FIELD payload ON TABLE memory_note TYPE object;",
    "DEFINE INDEX idx_memory_note_id ON TABLE memory_note COLUMNS id UNIQUE;",
    "DEFINE INDEX idx_memory_note_kind ON TABLE memory_note COLUMNS kind;",
    "DEFINE TABLE memory_link SCHEMAFULL;",
    "DEFINE FIELD id ON TABLE memory_link TYPE string;",
    "DEFINE FIELD from ON TABLE memory_link TYPE string;",
    "DEFINE FIELD to ON TABLE memory_link TYPE string;",
    "DEFINE FIELD intensity ON TABLE memory_link TYPE number;",
    "DEFINE FIELD kind ON TABLE memory_link TYPE string;",
    "DEFINE FIELD payload ON TABLE memory_link TYPE object;",
    "DEFINE INDEX idx_memory_link_id ON TABLE memory_link COLUMNS id UNIQUE;",
    "DEFINE INDEX idx_memory_link_from ON TABLE memory_link COLUMNS from;",
    "DEFINE INDEX idx_memory_link_to ON TABLE memory_link COLUMNS to;",
    "DEFINE TABLE retrieval_event SCHEMAFULL;",
    "DEFINE FIELD memory_id ON TABLE retrieval_event TYPE string;",
    "DEFINE FIELD occurred_at ON TABLE retrieval_event TYPE datetime;",
    "DEFINE FIELD query_fingerprint ON TABLE retrieval_event TYPE option<string>;",
    "DEFINE FIELD score ON TABLE retrieval_event TYPE option<number>;",
    "DEFINE INDEX idx_retrieval_event_memory_id ON TABLE retrieval_event COLUMNS memory_id;",
    "DEFINE INDEX idx_retrieval_event_occurred_at ON TABLE retrieval_event COLUMNS occurred_at;",
    "DEFINE TABLE feedback_event SCHEMAFULL;",
    "DEFINE FIELD memory_id ON TABLE feedback_event TYPE string;",
    "DEFINE FIELD occurred_at ON TABLE feedback_event TYPE datetime;",
    "DEFINE FIELD feedback ON TABLE feedback_event TYPE string;",
    "DEFINE INDEX idx_feedback_event_memory_id ON TABLE feedback_event COLUMNS memory_id;",
    "DEFINE INDEX idx_feedback_event_occurred_at ON TABLE feedback_event COLUMNS occurred_at;",
];

pub const UPSERT_MEMORY_NOTE: &str = r#"
UPSERT type::thing($table, $record_id) CONTENT $content;
"#;

pub const GET_MEMORY_NOTE_BY_ID: &str = r#"
SELECT * FROM type::thing($table, $record_id);
"#;

pub const DELETE_MEMORY_NOTE_BY_ID: &str = r#"
DELETE type::thing($table, $record_id);
"#;

pub const UPSERT_MEMORY_LINK: &str = r#"
UPSERT type::thing($table, $record_id) CONTENT $content;
"#;

pub const DELETE_MEMORY_LINK_BY_ID: &str = r#"
DELETE type::thing($table, $record_id);
"#;

pub const LIST_OUTBOUND_LINKS: &str = r#"
SELECT * FROM memory_link WHERE from = $from_id;
"#;

pub const LIST_INBOUND_LINKS: &str = r#"
SELECT * FROM memory_link WHERE to = $to_id;
"#;

pub const INSERT_RETRIEVAL_EVENT: &str = r#"
CREATE retrieval_event CONTENT $content;
"#;

pub const INSERT_FEEDBACK_EVENT: &str = r#"
CREATE feedback_event CONTENT $content;
"#;

pub const VECTOR_SIMILARITY_SEARCH: &str = r#"
-- placeholder:
-- SELECT id, vector::similarity::cosine(embedding, $query_embedding) AS score
-- FROM memory_note
-- WHERE score >= $min_score
-- ORDER BY score DESC
-- LIMIT $limit;
"#;
