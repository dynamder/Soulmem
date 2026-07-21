// memory_note
pub const MEMORY_NOTE_SCHEMA: &[&str] = &[
    "DEFINE TABLE IF NOT EXISTS memory_note SCHEMAFULL;",
    "DEFINE FIELD IF NOT EXISTS id ON TABLE memory_note TYPE string;",
    "DEFINE FIELD IF NOT EXISTS tags ON TABLE memory_note TYPE array<string>;",
    "DEFINE FIELD IF NOT EXISTS retrieval_count ON TABLE memory_note TYPE int;",
    "DEFINE FIELD IF NOT EXISTS create_time ON TABLE memory_note TYPE datetime;",
    "DEFINE FIELD IF NOT EXISTS last_accessed_time ON TABLE memory_note TYPE datetime;",
    "DEFINE FIELD IF NOT EXISTS kind ON TABLE memory_note TYPE string;",
    "DEFINE FIELD IF NOT EXISTS situation_subtype ON TABLE memory_note TYPE option<string>;",
    "DEFINE FIELD IF NOT EXISTS embedding ON TABLE memory_note TYPE option<array<float>>;",
    "DEFINE FIELD IF NOT EXISTS payload ON TABLE memory_note TYPE object FLEXIBLE;",
    "DEFINE INDEX IF NOT EXISTS idx_memory_note_id ON TABLE memory_note COLUMNS id UNIQUE;",
    "DEFINE INDEX IF NOT EXISTS idx_memory_note_embedding ON TABLE memory_note FIELDS embedding HNSW DIMENSION 512 DIST COSINE TYPE F32;",
];

// memory_link
pub const MEMORY_LINK_SCHEMA: &[&str] = &[
    "DEFINE TABLE IF NOT EXISTS memory_link SCHEMAFULL;",
    "DEFINE FIELD IF NOT EXISTS id ON TABLE memory_link TYPE string;",
    "DEFINE FIELD IF NOT EXISTS from ON TABLE memory_link TYPE string;",
    "DEFINE FIELD IF NOT EXISTS to ON TABLE memory_link TYPE string;",
    "DEFINE FIELD IF NOT EXISTS intensity ON TABLE memory_link TYPE number;",
    "DEFINE FIELD IF NOT EXISTS kind ON TABLE memory_link TYPE string;",
    "DEFINE FIELD IF NOT EXISTS payload ON TABLE memory_link TYPE object FLEXIBLE;",
    "DEFINE INDEX IF NOT EXISTS idx_memory_link_id ON TABLE memory_link COLUMNS id UNIQUE;",
    "DEFINE INDEX IF NOT EXISTS idx_memory_link_from ON TABLE memory_link COLUMNS from;",
    "DEFINE INDEX IF NOT EXISTS idx_memory_link_to ON TABLE memory_link COLUMNS to;",
];

// retrieval_event
pub const RETRIEVAL_EVENT_SCHEMA: &[&str] = &[
    "DEFINE TABLE IF NOT EXISTS retrieval_event SCHEMAFULL;",
    "DEFINE FIELD IF NOT EXISTS memory_id ON TABLE retrieval_event TYPE string;",
    "DEFINE FIELD IF NOT EXISTS occurred_at ON TABLE retrieval_event TYPE datetime;",
    "DEFINE FIELD IF NOT EXISTS score ON TABLE retrieval_event TYPE option<number>;",
    "DEFINE INDEX IF NOT EXISTS idx_retrieval_event_memory_id ON TABLE retrieval_event COLUMNS memory_id;",
    "DEFINE INDEX IF NOT EXISTS idx_retrieval_event_occurred_at ON TABLE retrieval_event COLUMNS occurred_at;",
];

// feedback_event
pub const FEEDBACK_EVENT_SCHEMA: &[&str] = &[
    "DEFINE TABLE IF NOT EXISTS feedback_event SCHEMAFULL;",
    "DEFINE FIELD IF NOT EXISTS memory_id ON TABLE feedback_event TYPE string;",
    "DEFINE FIELD IF NOT EXISTS occurred_at ON TABLE feedback_event TYPE datetime;",
    "DEFINE FIELD IF NOT EXISTS feedback ON TABLE feedback_event TYPE string;",
    "DEFINE INDEX IF NOT EXISTS idx_feedback_event_memory_id ON TABLE feedback_event COLUMNS memory_id;",
    "DEFINE INDEX IF NOT EXISTS idx_feedback_event_occurred_at ON TABLE feedback_event COLUMNS occurred_at;",
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
SELECT * FROM ONLY type::record($table, $record_id);
"#;

pub const DELETE_NOTE_BY_ID: &str = r#"
DELETE type::record($table, $record_id);
"#;

pub const SET_NOTE_EMBEDDING: &str = r#"
UPDATE type::record($table, $record_id) MERGE { embedding: $embedding };
"#;

pub const UPSERT_LINK: &str = r#"
UPSERT type::record($table, $record_id) CONTENT $content;
"#;

pub const GET_LINK_BY_ID: &str = r#"
SELECT * FROM ONLY type::record($table, $record_id);
"#;

pub const DELETE_LINK_BY_ID: &str = r#"
DELETE type::record($table, $record_id);
"#;

pub const DELETE_LINKS_BY_MEMORY_ID: &str = r#"
DELETE memory_link WHERE from = $memory_id OR to = $memory_id;
"#;

pub const LIST_OUT_LINKS: &str = r#"
SELECT * FROM memory_link WHERE from = $from_id;
"#;

pub const LIST_IN_LINKS: &str = r#"
SELECT * FROM memory_link WHERE to = $to_id;
"#;

// 写入检索和反馈
pub const INSERT_RETRIEVAL_EVENT: &str = r#"
CREATE type::record($table, $record_id) CONTENT $content;
"#;

pub const INSERT_FEEDBACK_EVENT: &str = r#"
CREATE type::record($table, $record_id) CONTENT $content;
"#;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_definitions_are_repeatable() {
        assert!(
            bootstrap_statements()
                .iter()
                .all(|statement| statement.contains("IF NOT EXISTS"))
        );
    }

    #[test]
    fn test_payload_fields_allow_nested_data() {
        assert!(
            MEMORY_NOTE_SCHEMA
                .iter()
                .any(|statement| statement.contains("payload") && statement.contains("FLEXIBLE"))
        );
        assert!(
            MEMORY_LINK_SCHEMA
                .iter()
                .any(|statement| statement.contains("payload") && statement.contains("FLEXIBLE"))
        );
    }
}
