// memory_note
pub const MEMORY_NOTE_SCHEMA: &[&str] = &[
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
];

// memory_link
pub const MEMORY_LINK_SCHEMA: &[&str] = &[
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
];

// retrieval_event
pub const RETRIEVAL_EVENT_SCHEMA: &[&str] = &[
    "DEFINE TABLE retrieval_event SCHEMAFULL;",
    "DEFINE FIELD memory_id ON TABLE retrieval_event TYPE string;",
    "DEFINE FIELD occurred_at ON TABLE retrieval_event TYPE datetime;",
    "DEFINE FIELD score ON TABLE retrieval_event TYPE option<number>;",
    "DEFINE INDEX idx_retrieval_event_memory_id ON TABLE retrieval_event COLUMNS memory_id;",
    "DEFINE INDEX idx_retrieval_event_occurred_at ON TABLE retrieval_event COLUMNS occurred_at;",
];

// feedback_event
pub const FEEDBACK_EVENT_SCHEMA: &[&str] = &[
    "DEFINE TABLE feedback_event SCHEMAFULL;",
    "DEFINE FIELD memory_id ON TABLE feedback_event TYPE string;",
    "DEFINE FIELD occurred_at ON TABLE feedback_event TYPE datetime;",
    "DEFINE FIELD feedback ON TABLE feedback_event TYPE string;",
    "DEFINE INDEX idx_feedback_event_memory_id ON TABLE feedback_event COLUMNS memory_id;",
    "DEFINE INDEX idx_feedback_event_occurred_at ON TABLE feedback_event COLUMNS occurred_at;",
];


// 初始化
pub const SCHEMA_GROUPS: &[&[&str]] = &[
    MEMORY_NOTE_SCHEMA,
    MEMORY_LINK_SCHEMA,
    RETRIEVAL_EVENT_SCHEMA,
    FEEDBACK_EVENT_SCHEMA,
];

pub fn bootstrap_statements() -> Vec<&'static str> {
    SCHEMA_GROUPS
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect()
}

// 查询
pub const UPSERT_NOTE: &str = r#"
UPSERT type::record($table, $record_id) CONTENT $content;
"#;

pub const GET_NOTE_BY_ID: &str = r#"
SELECT * FROM type::record($table, $record_id);
"#;

pub const DELETE_NOTE_BY_ID: &str = r#"
DELETE type::record($table, $record_id);
"#;


pub const UPSERT_LINK: &str = r#"
UPSERT type::record($table, $record_id) CONTENT $content;
"#;

pub const DELETE_LINK_BY_ID: &str = r#"
DELETE type::record($table, $record_id);
"#;

pub const LIST_OUT_LINKS: &str = r#"
SELECT * FROM memory_link WHERE from = $from_id;
"#;

pub const LIST_IN_LINKS: &str = r#"
SELECT * FROM memory_link WHERE to = $to_id;
"#;

// 写入检索和反馈
pub const INSERT_RETRIEVAL_EVENT: &str = r#"
CREATE retrieval_event CONTENT $content;
"#;

pub const INSERT_FEEDBACK_EVENT: &str = r#"
CREATE feedback_event CONTENT $content;
"#;


// todo：向量相似检索语句，后续接入 embedding 字段和索引后再补全。

