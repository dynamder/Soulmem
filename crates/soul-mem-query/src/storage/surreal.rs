// SurrealDB 的具体实现

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Serialize, de::DeserializeOwned};
use soul_mem_core::{
    memory_links::MemoryLink,
    memory_note::{MemoryId, MemoryNote},
};
use surrealdb::{
    Surreal,
    engine::any::{Any, connect},
    opt::auth::Root,
    types::{RecordId, RecordIdKey, SerdeWrapper, SurrealValue, Value as SurrealValueData},
};
use uuid::Uuid;

use super::{
    error::{StorageError, StorageResult},
    model::{
        EventWindow, FeedbackEventRecord, MemoryLinkRecord, MemoryNoteRecord, RetrievalEventRecord,
        SimilarityHit, SimilarityQuery,
    },
    repository::MemoryRepository,
    surql,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurrealConnectionConfig {
    pub endpoint: String,
    pub namespace: String,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl SurrealConnectionConfig {
    // 创建数据库连接配置。
    pub fn new(
        endpoint: impl Into<String>,
        namespace: impl Into<String>,
        database: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            namespace: namespace.into(),
            database: database.into(),
            username: None,
            password: None,
        }
    }

    // 添加数据库用户名和密码。
    pub fn with_auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }
}

impl Default for SurrealConnectionConfig {
    fn default() -> Self {
        Self {
            endpoint: "ws://127.0.0.1:8000".to_string(),
            namespace: "soulmem".to_string(),
            database: "memory".to_string(),
            username: None,
            password: None,
        }
    }
}

const TABLE_MEMORY_NOTE: &str = "memory_note";
const TABLE_MEMORY_LINK: &str = "memory_link";
const TABLE_RETRIEVAL_EVENT: &str = "retrieval_event";
const TABLE_FEEDBACK_EVENT: &str = "feedback_event";
const NOTE_EMBEDDING_DIMENSION: usize = 512;
const NOTE_VECTOR_SEARCH_EF: usize = 40;

#[derive(Debug, Default)]
struct RepositoryState {
    connected: bool,
    bootstrapped: bool,
}

#[derive(Clone)]
pub struct SurrealMemoryRepository {
    config: SurrealConnectionConfig,
    db: Arc<RwLock<Option<Surreal<Any>>>>,
    state: Arc<RwLock<RepositoryState>>,
}

impl SurrealMemoryRepository {
    // 创建数据库仓储对象。
    pub fn new(config: SurrealConnectionConfig) -> Self {
        Self {
            config,
            db: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(RepositoryState::default())),
        }
    }

    // 返回数据库连接配置。
    pub fn config(&self) -> &SurrealConnectionConfig {
        &self.config
    }

    // 连接数据库并选择 namespace 和 database。
    pub async fn connect(&self) -> StorageResult<()> {
        if self.is_connected() {
            return Ok(());
        }

        if self.config.username.is_some() ^ self.config.password.is_some() {
            return Err(StorageError::invalid_data(
                "SurrealDB auth requires both username and password",
            ));
        }

        let db = connect(self.config.endpoint.as_str())
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to connect to SurrealDB at {}: {err}",
                    self.config.endpoint
                ))
            })?;

        if let (Some(username), Some(password)) = (
            self.config.username.as_deref(),
            self.config.password.as_deref(),
        ) {
            db.signin(Root {
                username: username.to_owned(),
                password: password.to_owned(),
            })
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to sign in to SurrealDB at {}: {err}",
                    self.config.endpoint
                ))
            })?;
        }

        db.use_ns(&self.config.namespace)
            .use_db(&self.config.database)
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to select namespace/database {}/{}: {err}",
                    self.config.namespace, self.config.database
                ))
            })?;

        *self.db.write() = Some(db);
        let mut guard = self.state.write();
        guard.connected = true;
        Ok(())
    }

    // 检查数据库是否已连接。
    pub fn is_connected(&self) -> bool {
        self.state.read().connected
    }

    // 初始化数据库表、字段和索引。
    pub async fn bootstrap_schema(&self) -> StorageResult<()> {
        self.ensure_connected()?;
        let db = self.db()?;
        for statement in self.bootstrap_statements() {
            let response = db.query(statement).await.map_err(|err| {
                StorageError::backend(format!(
                    "failed to execute bootstrap statement `{statement}`: {err}"
                ))
            })?;
            response.check().map_err(|err| {
                StorageError::backend(format!(
                    "bootstrap statement `{statement}` returned an error: {err}"
                ))
            })?;
        }

        let mut guard = self.state.write();
        guard.bootstrapped = true;
        Ok(())
    }

    // 返回所有数据库初始化语句。
    pub fn bootstrap_statements(&self) -> Vec<&'static str> {
        surql::bootstrap_statements()
    }

    // 获取数据库客户端。
    fn db(&self) -> StorageResult<Surreal<Any>> {
        self.db.read().clone().ok_or_else(|| {
            StorageError::backend("SurrealDB client is not available; call connect() first")
        })
    }

    // 检查数据库连接状态。
    fn ensure_connected(&self) -> StorageResult<()> {
        if self.state.read().connected && self.db.read().is_some() {
            Ok(())
        } else {
            Err(StorageError::backend(
                "SurrealDB repository is not connected; call connect() first",
            ))
        }
    }

    // 检查数据库和 schema 是否已准备好。
    fn ensure_bootstrapped(&self) -> StorageResult<()> {
        let guard = self.state.read();
        if !guard.connected {
            return Err(StorageError::backend(
                "SurrealDB repository is not connected; call connect() first",
            ));
        }
        if !guard.bootstrapped {
            return Err(StorageError::backend(
                "schema is not bootstrapped; call bootstrap() first",
            ));
        }
        Ok(())
    }

    // 准备写入数据库的记录内容。
    fn content_without_id<T>(
        value: &T,
        datetime_fields: &[(&str, chrono::DateTime<chrono::Utc>)],
    ) -> StorageResult<SurrealValueData>
    where
        T: Serialize + DeserializeOwned + Clone + 'static,
    {
        let mut surreal_value = SerdeWrapper(value.clone()).into_value();
        let object = match &mut surreal_value {
            SurrealValueData::Object(object) => object,
            _ => {
                return Err(StorageError::backend(
                    "storage record content must serialize to a SurrealDB object",
                ));
            }
        };
        object.remove("id");
        for (field, datetime) in datetime_fields {
            object.insert(
                (*field).to_string(),
                SurrealValueData::Datetime((*datetime).into()),
            );
        }
        Ok(surreal_value)
    }

    // 检查 embedding 是否有效。
    fn validate_note_embedding(embedding: &[f32]) -> StorageResult<()> {
        if embedding.len() != NOTE_EMBEDDING_DIMENSION {
            return Err(StorageError::invalid_data(format!(
                "note embedding dimension mismatch: expected {}, got {}",
                NOTE_EMBEDDING_DIMENSION,
                embedding.len()
            )));
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(StorageError::invalid_data(
                "note embedding contains non-finite values",
            ));
        }
        Ok(())
    }

    // 把记录 ID 转成普通字符串。
    fn normalize_record_key(raw_id: &str) -> StorageResult<String> {
        if !raw_id.contains(':') {
            return Ok(raw_id.to_string());
        }

        let record_id = RecordId::parse_simple(raw_id).map_err(|err| {
            StorageError::backend(format!(
                "failed to parse SurrealDB record id `{raw_id}`: {err}"
            ))
        })?;

        match record_id.key {
            RecordIdKey::String(key) => Ok(key
                .strip_prefix('`')
                .and_then(|key| key.strip_suffix('`'))
                .unwrap_or(&key)
                .to_string()),
            RecordIdKey::Uuid(uuid) => Ok(uuid.to_string()),
            RecordIdKey::Number(number) => Ok(number.to_string()),
            _ => Err(StorageError::backend(format!(
                "unsupported SurrealDB record key type in `{raw_id}`"
            ))),
        }
    }

    // 把数据库返回的 ID 转成字符串。
    fn normalize_record_value(value: &SurrealValueData) -> StorageResult<String> {
        match value {
            SurrealValueData::RecordId(record_id) => match &record_id.key {
                RecordIdKey::String(key) => Ok(key.clone()),
                RecordIdKey::Uuid(uuid) => Ok(uuid.to_string()),
                RecordIdKey::Number(number) => Ok(number.to_string()),
                _ => Err(StorageError::backend(format!(
                    "unsupported SurrealDB record key type in `{record_id:?}`"
                ))),
            },
            SurrealValueData::String(raw_id) => Self::normalize_record_key(raw_id),
            _ => Err(StorageError::backend(
                "SurrealDB record has an unsupported `id` value",
            )),
        }
    }

    // 把数据库记录转成 JSON。
    fn normalize_record_json(value: SurrealValueData) -> StorageResult<serde_json::Value> {
        let normalized_id = match &value {
            SurrealValueData::Object(object) => object
                .get("id")
                .ok_or_else(|| StorageError::backend("SurrealDB record is missing `id`"))
                .and_then(Self::normalize_record_value)?,
            _ => {
                return Err(StorageError::backend(
                    "SurrealDB query did not return an object record",
                ));
            }
        };

        let mut json = value.into_json_value();
        let object = json.as_object_mut().ok_or_else(|| {
            StorageError::backend("SurrealDB query did not return an object record")
        })?;
        object.insert("id".to_string(), serde_json::Value::String(normalized_id));
        Ok(json)
    }

    // 读取一条可选记录。
    fn decode_optional_record<T>(value: Option<SurrealValueData>) -> StorageResult<Option<T>>
    where
        T: DeserializeOwned,
    {
        value
            .map(|value| {
                let json = Self::normalize_record_json(value)?;
                Ok(serde_json::from_value(json)?)
            })
            .transpose()
    }

    // 读取多条数据库记录。
    fn decode_record_list<T>(values: Vec<SurrealValueData>) -> StorageResult<Vec<T>>
    where
        T: DeserializeOwned,
    {
        values
            .into_iter()
            .map(|value| {
                let json = Self::normalize_record_json(value)?;
                Ok(serde_json::from_value(json)?)
            })
            .collect()
    }

    // 处理向量检索结果。
    fn decode_similarity_hits(
        values: Vec<SurrealValueData>,
        min_score: f32,
    ) -> StorageResult<Vec<SimilarityHit>> {
        values
            .into_iter()
            .map(|value| {
                let memory_id = match &value {
                    SurrealValueData::Object(object) => object
                        .get("id")
                        .ok_or_else(|| {
                            StorageError::backend("SurrealDB similarity query row is missing `id`")
                        })
                        .and_then(Self::normalize_record_value)?,
                    _ => {
                        return Err(StorageError::backend(
                            "SurrealDB similarity query did not return an object row",
                        ));
                    }
                };
                let json = value.into_json_value();
                let object = json.as_object().ok_or_else(|| {
                    StorageError::backend("SurrealDB similarity query did not return an object row")
                })?;
                let score = object
                    .get("score")
                    .and_then(|value| value.as_f64())
                    .ok_or_else(|| {
                        StorageError::backend(
                            "SurrealDB similarity query row is missing a numeric `score`",
                        )
                    })? as f32;

                Ok(SimilarityHit { memory_id, score })
            })
            .filter_map(|hit| match hit {
                Ok(hit) if hit.score >= min_score => Some(Ok(hit)),
                Ok(_) => None,
                Err(err) => Some(Err(err)),
            })
            .collect()
    }

    // 生成事件查询语句。
    fn build_event_list_query(table: &str, window: EventWindow) -> StorageResult<String> {
        if let (Some(start), Some(end)) = (window.start, window.end)
            && start > end
        {
            return Err(StorageError::invalid_data(
                "event window start must not be later than end",
            ));
        }

        let mut query = format!("SELECT * FROM {table} WHERE memory_id = $memory_id");

        if window.start.is_some() {
            query.push_str(" AND occurred_at >= $start");
        }
        if window.end.is_some() {
            query.push_str(" AND occurred_at <= $end");
        }

        query.push_str(" ORDER BY occurred_at ASC;");
        Ok(query)
    }

    // 生成返回 Top-K 记忆的查询语句。
    fn build_similarity_query(query: &SimilarityQuery) -> StorageResult<String> {
        if query.limit == 0 {
            return Err(StorageError::invalid_data(
                "similarity query limit must be greater than 0",
            ));
        }
        if !query.min_score.is_finite() {
            return Err(StorageError::invalid_data(
                "similarity query min_score must be finite",
            ));
        }
        Self::validate_note_embedding(&query.embedding)?;

        let mut sql = format!(
            "SELECT id, 1 - vector::distance::knn() AS score FROM memory_note WHERE embedding != NONE AND embedding <|{},{}|> $query_embedding",
            query.limit, NOTE_VECTOR_SEARCH_EF
        );

        if !query.kinds.is_empty() {
            sql.push_str(" AND (");
            for (index, _) in query.kinds.iter().enumerate() {
                if index > 0 {
                    sql.push_str(" OR ");
                }
                sql.push_str(&format!("kind = $kind_{index}"));
            }
            sql.push(')');
        }

        if !query.tags_any.is_empty() {
            sql.push_str(" AND (");
            for (index, _) in query.tags_any.iter().enumerate() {
                if index > 0 {
                    sql.push_str(" OR ");
                }
                sql.push_str(&format!("tags CONTAINS $tag_{index}"));
            }
            sql.push(')');
        }

        sql.push_str(&format!(" ORDER BY score DESC LIMIT {};", query.limit));
        Ok(sql)
    }

    // 查询记忆关联的事件。
    async fn list_event_records<T>(
        &self,
        table: &str,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<Vec<T>>
    where
        T: DeserializeOwned,
    {
        self.ensure_bootstrapped()?;
        let memory_id_str = memory_id.to_string();
        let query = Self::build_event_list_query(table, window)?;
        let db = self.db()?;
        let mut request = db.query(query).bind(("memory_id", memory_id_str.clone()));
        if let Some(start) = window.start {
            request = request.bind(("start", start));
        }
        if let Some(end) = window.end {
            request = request.bind(("end", end));
        }

        let response = request.await.map_err(|err| {
            StorageError::backend(format!(
                "failed to list `{table}` rows for memory `{memory_id_str}` from SurrealDB: {err}"
            ))
        })?;
        let mut response = response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while listing `{table}` rows for memory `{memory_id_str}`: {err}"
            ))
        })?;

        let records = response.take(0).map_err(|err| {
            StorageError::backend(format!(
                "failed to decode `{table}` rows for memory `{memory_id_str}` from SurrealDB: {err}"
            ))
        })?;
        Self::decode_record_list(records)
    }

    // 根据 ID 查询关系。
    async fn get_link_record(&self, link_id: &str) -> StorageResult<Option<MemoryLinkRecord>> {
        self.ensure_bootstrapped()?;
        let db = self.db()?;
        let response = db
            .query(surql::GET_LINK_BY_ID)
            .bind(("table", TABLE_MEMORY_LINK))
            .bind(("record_id", link_id.to_string()))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to load memory_link `{link_id}` from SurrealDB: {err}"
                ))
            })?;
        let mut response = response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while loading memory_link `{link_id}`: {err}"
            ))
        })?;

        let record = response.take(0).map_err(|err| {
            StorageError::backend(format!(
                "failed to decode memory_link `{link_id}` from SurrealDB: {err}"
            ))
        })?;
        Self::decode_optional_record(record)
    }
}

impl Default for SurrealMemoryRepository {
    fn default() -> Self {
        Self::new(SurrealConnectionConfig::default())
    }
}

#[async_trait]
impl MemoryRepository for SurrealMemoryRepository {
    // 连接数据库并初始化 schema。
    async fn bootstrap(&self) -> StorageResult<()> {
        if !self.is_connected() {
            self.connect().await?;
        }
        self.bootstrap_schema().await
    }

    // 在事务中新增或更新记忆节点。
    async fn upsert_note(&self, note: &MemoryNote) -> StorageResult<MemoryNoteRecord> {
        self.ensure_bootstrapped()?;
        let record = MemoryNoteRecord::from_note(note)?;
        let content = Self::content_without_id(
            &record,
            &[
                ("create_time", record.create_time),
                ("last_accessed_time", record.last_accessed_time),
            ],
        )?;
        let db = self.db()?;
        let transaction = db.begin().await.map_err(|err| {
            StorageError::backend(format!("failed to begin note upsert transaction: {err}"))
        })?;

        let result = transaction
            .query(surql::UPSERT_NOTE)
            .bind(("table", TABLE_MEMORY_NOTE))
            .bind(("record_id", record.id.clone()))
            .bind(("content", content))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to upsert memory_note `{}` in transaction: {err}",
                    record.id
                ))
            })
            .and_then(|response| {
                response.check().map_err(|err| {
                    StorageError::backend(format!(
                        "SurrealDB returned an error while upserting memory_note `{}` in transaction: {err}",
                        record.id
                    ))
                }).map(|_| ())
            });

        match result {
            Ok(()) => {
                transaction.commit().await.map_err(|err| {
                    StorageError::backend(format!("failed to commit note upsert transaction: {err}"))
                })?;
                Ok(record)
            }
            Err(err) => {
                transaction.cancel().await.map_err(|cancel_err| {
                    StorageError::backend(format!(
                        "{err}; failed to rollback note upsert transaction: {cancel_err}"
                    ))
                })?;
                Err(err)
            }
        }
    }

    // 原子保存节点、节点关系和 embedding。
    async fn save_note_bundle(
        &self,
        note: &MemoryNote,
        embedding: Vec<f32>,
    ) -> StorageResult<MemoryNoteRecord> {
        self.ensure_bootstrapped()?;
        Self::validate_note_embedding(&embedding)?;

        let mut record = MemoryNoteRecord::from_note(note)?;
        record.embedding = Some(embedding);
        let content = Self::content_without_id(
            &record,
            &[
                ("create_time", record.create_time),
                ("last_accessed_time", record.last_accessed_time),
            ],
        )?;
        let db = self.db()?;
        let transaction = db.begin().await.map_err(|err| {
            StorageError::backend(format!("failed to begin note bundle transaction: {err}"))
        })?;

        let result = async {
            let response = transaction
                .query(surql::UPSERT_NOTE)
                .bind(("table", TABLE_MEMORY_NOTE))
                .bind(("record_id", record.id.clone()))
                .bind(("content", content))
                .await
                .map_err(|err| {
                    StorageError::backend(format!(
                        "failed to upsert memory_note `{}` in transaction: {err}",
                        record.id
                    ))
                })?;
            response.check().map_err(|err| {
                StorageError::backend(format!(
                    "SurrealDB returned an error while upserting memory_note `{}` in transaction: {err}",
                    record.id
                ))
            })?;

            for link in note.links() {
                let link_record = MemoryLinkRecord::from_link(link)?;
                let link_content = Self::content_without_id(&link_record, &[])?;
                let response = transaction
                    .query(surql::UPSERT_LINK)
                    .bind(("table", TABLE_MEMORY_LINK))
                    .bind(("record_id", link_record.id.clone()))
                    .bind(("content", link_content))
                    .await
                    .map_err(|err| {
                        StorageError::backend(format!(
                            "failed to upsert memory_link `{}` in transaction: {err}",
                            link_record.id
                        ))
                    })?;
                response.check().map_err(|err| {
                    StorageError::backend(format!(
                        "SurrealDB returned an error while upserting memory_link `{}` in transaction: {err}",
                        link_record.id
                    ))
                })?;
            }

            Ok::<_, StorageError>(())
        }
        .await;

        match result {
            Ok(()) => {
                transaction.commit().await.map_err(|err| {
                    StorageError::backend(format!("failed to commit note bundle transaction: {err}"))
                })?;
                Ok(record)
            }
            Err(err) => {
                transaction.cancel().await.map_err(|cancel_err| {
                    StorageError::backend(format!(
                        "{err}; failed to rollback note bundle transaction: {cancel_err}"
                    ))
                })?;
                Err(err)
            }
        }
    }

    // 原子保存多个记忆节点及其关系。
    async fn upsert_notes(&self, notes: &[MemoryNote]) -> StorageResult<Vec<MemoryNoteRecord>> {
        self.ensure_bootstrapped()?;
        let db = self.db()?;
        let transaction = db.begin().await.map_err(|err| {
            StorageError::backend(format!("failed to begin notes transaction: {err}"))
        })?;
        let mut records = Vec::with_capacity(notes.len());

        let result = async {
            for note in notes {
                let record = MemoryNoteRecord::from_note(note)?;
                let content = Self::content_without_id(
                    &record,
                    &[
                        ("create_time", record.create_time),
                        ("last_accessed_time", record.last_accessed_time),
                    ],
                )?;
                let response = transaction
                    .query(surql::UPSERT_NOTE)
                    .bind(("table", TABLE_MEMORY_NOTE))
                    .bind(("record_id", record.id.clone()))
                    .bind(("content", content))
                    .await
                    .map_err(|err| {
                        StorageError::backend(format!(
                            "failed to upsert memory_note `{}` in transaction: {err}",
                            record.id
                        ))
                    })?;
                response.check().map_err(|err| {
                    StorageError::backend(format!(
                        "SurrealDB returned an error while upserting memory_note `{}` in transaction: {err}",
                        record.id
                    ))
                })?;

                for link in note.links() {
                    let link_record = MemoryLinkRecord::from_link(link)?;
                    let link_content = Self::content_without_id(&link_record, &[])?;
                    let response = transaction
                        .query(surql::UPSERT_LINK)
                        .bind(("table", TABLE_MEMORY_LINK))
                        .bind(("record_id", link_record.id.clone()))
                        .bind(("content", link_content))
                        .await
                        .map_err(|err| {
                            StorageError::backend(format!(
                                "failed to upsert memory_link `{}` in transaction: {err}",
                                link_record.id
                            ))
                        })?;
                    response.check().map_err(|err| {
                        StorageError::backend(format!(
                            "SurrealDB returned an error while upserting memory_link `{}` in transaction: {err}",
                            link_record.id
                        ))
                    })?;
                }
                records.push(record);
            }
            Ok::<_, StorageError>(())
        }
        .await;

        match result {
            Ok(()) => {
                transaction.commit().await.map_err(|err| {
                    StorageError::backend(format!("failed to commit notes transaction: {err}"))
                })?;
                Ok(records)
            }
            Err(err) => {
                transaction.cancel().await.map_err(|cancel_err| {
                    StorageError::backend(format!(
                        "{err}; failed to rollback notes transaction: {cancel_err}"
                    ))
                })?;
                Err(err)
            }
        }
    }

    // 根据 ID 查询记忆节点。
    async fn get_note(&self, memory_id: MemoryId) -> StorageResult<Option<MemoryNoteRecord>> {
        self.ensure_bootstrapped()?;
        let key = memory_id.to_string();
        let db = self.db()?;
        let response = db
            .query(surql::GET_NOTE_BY_ID)
            .bind(("table", TABLE_MEMORY_NOTE))
            .bind(("record_id", key.clone()))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to load memory_note `{key}` from SurrealDB: {err}"
                ))
            })?;
        let mut response = response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while loading memory_note `{key}`: {err}"
            ))
        })?;

        let record = response.take(0).map_err(|err| {
            StorageError::backend(format!(
                "failed to decode memory_note `{key}` from SurrealDB: {err}"
            ))
        })?;
        Self::decode_optional_record(record)
    }

    // 查询记忆节点和它的出边关系。
    async fn load_note(&self, memory_id: MemoryId) -> StorageResult<Option<MemoryNote>> {
        let note_record = match self.get_note(memory_id).await? {
            Some(record) => record,
            None => return Ok(None),
        };

        let links = self
            .list_outbound_links(memory_id)
            .await?
            .into_iter()
            .map(|record| record.to_link())
            .collect::<StorageResult<Vec<_>>>()?;

        Ok(Some(note_record.to_note(links)?))
    }

    // 删除记忆节点和关联关系。
    async fn delete_note(&self, memory_id: MemoryId) -> StorageResult<bool> {
        self.ensure_bootstrapped()?;
        let key = memory_id.to_string();
        if self.get_note(memory_id).await?.is_none() {
            return Ok(false);
        }

        let db = self.db()?;
        let transaction = db.begin().await.map_err(|err| {
            StorageError::backend(format!("failed to begin delete note transaction: {err}"))
        })?;

        let result = async {
            let delete_note = transaction
                .query(surql::DELETE_NOTE_BY_ID)
                .bind(("table", TABLE_MEMORY_NOTE))
                .bind(("record_id", key.clone()))
                .await
                .map_err(|err| {
                    StorageError::backend(format!(
                        "failed to delete memory_note `{key}` in transaction: {err}"
                    ))
                })?;
            delete_note.check().map_err(|err| {
                StorageError::backend(format!(
                    "SurrealDB returned an error while deleting memory_note `{key}` in transaction: {err}"
                ))
            })?;

            let delete_links = transaction
                .query(surql::DELETE_LINKS_BY_MEMORY_ID)
                .bind(("memory_id", key.clone()))
                .await
                .map_err(|err| {
                    StorageError::backend(format!(
                        "failed to delete memory_link rows attached to `{key}` in transaction: {err}"
                    ))
                })?;
            delete_links.check().map_err(|err| {
                StorageError::backend(format!(
                    "SurrealDB returned an error while deleting memory_link rows attached to `{key}` in transaction: {err}"
                ))
            })?;

            Ok::<_, StorageError>(())
        }
        .await;

        match result {
            Ok(()) => {
                transaction.commit().await.map_err(|err| {
                    StorageError::backend(format!("failed to commit delete note transaction: {err}"))
                })?;
                Ok(true)
            }
            Err(err) => {
                transaction.cancel().await.map_err(|cancel_err| {
                    StorageError::backend(format!(
                        "{err}; failed to rollback delete note transaction: {cancel_err}"
                    ))
                })?;
                Err(err)
            }
        }
    }

    // 保存记忆节点的 embedding。
    async fn set_note_embedding(
        &self,
        memory_id: MemoryId,
        embedding: Vec<f32>,
    ) -> StorageResult<()> {
        self.ensure_bootstrapped()?;
        Self::validate_note_embedding(&embedding)?;

        let key = memory_id.to_string();
        if self.get_note(memory_id).await?.is_none() {
            return Err(StorageError::not_found(format!(
                "memory_note `{key}` does not exist"
            )));
        }

        let db = self.db()?;
        let response = db
            .query(surql::SET_NOTE_EMBEDDING)
            .bind(("table", TABLE_MEMORY_NOTE))
            .bind(("record_id", key.clone()))
            .bind(("embedding", embedding))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to set embedding for memory_note `{key}` in SurrealDB: {err}"
                ))
            })?;
        response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while setting embedding for memory_note `{key}`: {err}"
            ))
        })?;

        Ok(())
    }

    // 查询记忆节点的 embedding。
    async fn get_note_embedding(&self, memory_id: MemoryId) -> StorageResult<Option<Vec<f32>>> {
        Ok(self
            .get_note(memory_id)
            .await?
            .and_then(|record| record.embedding))
    }

    // 新增或更新记忆关系。
    async fn upsert_link(&self, link: &MemoryLink) -> StorageResult<MemoryLinkRecord> {
        self.ensure_bootstrapped()?;
        let record = MemoryLinkRecord::from_link(link)?;
        let content = Self::content_without_id(&record, &[])?;
        let db = self.db()?;
        let response = db
            .query(surql::UPSERT_LINK)
            .bind(("table", TABLE_MEMORY_LINK))
            .bind(("record_id", record.id.clone()))
            .bind(("content", content))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to upsert memory_link `{}` into SurrealDB: {err}",
                    record.id
                ))
            })?;
        response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while upserting memory_link `{}`: {err}",
                record.id
            ))
        })?;

        Ok(record)
    }

    // 根据 ID 删除记忆关系。
    async fn delete_link(&self, link_id: &str) -> StorageResult<bool> {
        self.ensure_bootstrapped()?;
        if self.get_link_record(link_id).await?.is_none() {
            return Ok(false);
        }

        let db = self.db()?;
        let response = db
            .query(surql::DELETE_LINK_BY_ID)
            .bind(("table", TABLE_MEMORY_LINK))
            .bind(("record_id", link_id.to_string()))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to delete memory_link `{link_id}` from SurrealDB: {err}"
                ))
            })?;
        response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while deleting memory_link `{link_id}`: {err}"
            ))
        })?;

        Ok(true)
    }

    // 查询节点发出的关系。
    async fn list_outbound_links(
        &self,
        memory_id: MemoryId,
    ) -> StorageResult<Vec<MemoryLinkRecord>> {
        self.ensure_bootstrapped()?;
        let from = memory_id.to_string();
        let db = self.db()?;
        let response = db
            .query(surql::LIST_OUT_LINKS)
            .bind(("from_id", from.clone()))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to list outbound memory_link rows for `{from}`: {err}"
                ))
            })?;
        let mut response = response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while listing outbound memory_link rows for `{from}`: {err}"
            ))
        })?;

        let links = response.take(0).map_err(|err| {
            StorageError::backend(format!(
                "failed to decode outbound memory_link rows for `{from}`: {err}"
            ))
        })?;
        Self::decode_record_list(links)
    }

    // 查询指向节点的关系。
    async fn list_inbound_links(
        &self,
        memory_id: MemoryId,
    ) -> StorageResult<Vec<MemoryLinkRecord>> {
        self.ensure_bootstrapped()?;
        let to = memory_id.to_string();
        let db = self.db()?;
        let response = db
            .query(surql::LIST_IN_LINKS)
            .bind(("to_id", to.clone()))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to list inbound memory_link rows for `{to}`: {err}"
                ))
            })?;
        let mut response = response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while listing inbound memory_link rows for `{to}`: {err}"
            ))
        })?;

        let links = response.take(0).map_err(|err| {
            StorageError::backend(format!(
                "failed to decode inbound memory_link rows for `{to}`: {err}"
            ))
        })?;
        Self::decode_record_list(links)
    }

    // 保存一次检索事件。
    async fn append_retrieval_event(&self, mut event: RetrievalEventRecord) -> StorageResult<()> {
        self.ensure_bootstrapped()?;
        let record_id = match event.id.take() {
            Some(id) if id.trim().is_empty() => {
                return Err(StorageError::invalid_data(
                    "retrieval event id cannot be empty",
                ));
            }
            Some(id) => id,
            None => Uuid::new_v4().to_string(),
        };
        event.id = Some(record_id.clone());

        let content = Self::content_without_id(&event, &[("occurred_at", event.occurred_at)])?;
        let db = self.db()?;
        let response = db
            .query(surql::INSERT_RETRIEVAL_EVENT)
            .bind(("table", TABLE_RETRIEVAL_EVENT))
            .bind(("record_id", record_id.clone()))
            .bind(("content", content))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to append retrieval_event `{record_id}` to SurrealDB: {err}"
                ))
            })?;
        response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while appending retrieval_event `{record_id}`: {err}"
            ))
        })?;

        Ok(())
    }

    // 保存一次反馈事件。
    async fn append_feedback_event(&self, mut event: FeedbackEventRecord) -> StorageResult<()> {
        self.ensure_bootstrapped()?;
        let record_id = match event.id.take() {
            Some(id) if id.trim().is_empty() => {
                return Err(StorageError::invalid_data(
                    "feedback event id cannot be empty",
                ));
            }
            Some(id) => id,
            None => Uuid::new_v4().to_string(),
        };
        event.id = Some(record_id.clone());

        let content = Self::content_without_id(&event, &[("occurred_at", event.occurred_at)])?;
        let db = self.db()?;
        let response = db
            .query(surql::INSERT_FEEDBACK_EVENT)
            .bind(("table", TABLE_FEEDBACK_EVENT))
            .bind(("record_id", record_id.clone()))
            .bind(("content", content))
            .await
            .map_err(|err| {
                StorageError::backend(format!(
                    "failed to append feedback_event `{record_id}` to SurrealDB: {err}"
                ))
            })?;
        response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while appending feedback_event `{record_id}`: {err}"
            ))
        })?;

        Ok(())
    }

    // 查询记忆的检索事件。
    async fn list_retrieval_events(
        &self,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<Vec<RetrievalEventRecord>> {
        self.list_event_records(TABLE_RETRIEVAL_EVENT, memory_id, window)
            .await
    }

    // 查询记忆的反馈事件。
    async fn list_feedback_events(
        &self,
        memory_id: MemoryId,
        window: EventWindow,
    ) -> StorageResult<Vec<FeedbackEventRecord>> {
        self.list_event_records(TABLE_FEEDBACK_EVENT, memory_id, window)
            .await
    }

    // 返回相似度最高的 Top-K 记忆。
    async fn query_similar_notes(
        &self,
        query: SimilarityQuery,
    ) -> StorageResult<Vec<SimilarityHit>> {
        self.ensure_bootstrapped()?;
        let sql = Self::build_similarity_query(&query)?;
        let db = self.db()?;
        let mut request = db
            .query(sql)
            .bind(("query_embedding", query.embedding.clone()));
        for (index, kind) in query.kinds.iter().enumerate() {
            request = request.bind((format!("kind_{index}"), kind.as_str()));
        }
        for (index, tag) in query.tags_any.iter().enumerate() {
            request = request.bind((format!("tag_{index}"), tag.clone()));
        }

        let response = request.await.map_err(|err| {
            StorageError::backend(format!(
                "failed to run similarity search against SurrealDB: {err}"
            ))
        })?;
        let mut response = response.check().map_err(|err| {
            StorageError::backend(format!(
                "SurrealDB returned an error while running similarity search: {err}"
            ))
        })?;

        let hits = response.take(0).map_err(|err| {
            StorageError::backend(format!(
                "failed to decode similarity search results from SurrealDB: {err}"
            ))
        })?;
        Self::decode_similarity_hits(hits, query.min_score)
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use soul_mem_core::{
        memory_links::{MemoryLink, MemoryLinkType, sem_mem::SemMemLink},
        memory_note::{
            MemoryNoteBuilder, MemoryType,
            sem_mem::{ConceptType, SemMemory},
        },
    };

    use super::*;
    use crate::storage::model::FeedbackValue;

    #[test]
    fn test_build_event_list_query_validates_window() {
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let sql = SurrealMemoryRepository::build_event_list_query(
            TABLE_RETRIEVAL_EVENT,
            EventWindow::new(Some(start), Some(end)),
        )
        .expect("build event query");

        assert!(sql.contains("occurred_at >= $start"));
        assert!(sql.contains("occurred_at <= $end"));
        assert!(sql.contains("ORDER BY occurred_at ASC"));

        let error = SurrealMemoryRepository::build_event_list_query(
            TABLE_RETRIEVAL_EVENT,
            EventWindow::new(Some(end), Some(start)),
        )
        .expect_err("reject inverted event window");
        assert!(matches!(error, StorageError::InvalidData(_)));
    }

    #[test]
    fn test_build_similarity_query_validates_input_and_filters() {
        let mut query = SimilarityQuery::new(vec![0.0; NOTE_EMBEDDING_DIMENSION]);
        query.tags_any.push("rust".to_string());
        query
            .kinds
            .push(super::super::model::MemoryNoteKind::Semantic);

        let sql = SurrealMemoryRepository::build_similarity_query(&query)
            .expect("build similarity query");
        assert!(sql.contains("kind = $kind_0"));
        assert!(sql.contains("tags CONTAINS $tag_0"));

        query.embedding.pop();
        assert!(matches!(
            SurrealMemoryRepository::build_similarity_query(&query),
            Err(StorageError::InvalidData(_))
        ));

        query.embedding.push(0.0);
        query.min_score = f32::NAN;
        assert!(matches!(
            SurrealMemoryRepository::build_similarity_query(&query),
            Err(StorageError::InvalidData(_))
        ));
    }

    #[test]
    fn test_content_without_id_preserves_other_fields() {
        let event = RetrievalEventRecord {
            id: Some("event-id".to_string()),
            memory_id: Uuid::new_v4().to_string(),
            occurred_at: Utc::now(),
            score: Some(0.8),
        };

        let content = SurrealMemoryRepository::content_without_id(
            &event,
            &[("occurred_at", event.occurred_at)],
        )
        .expect("serialize event content");
        let SurrealValueData::Object(object) = content else {
            panic!("event content must be an object");
        };

        assert!(!object.contains_key("id"));
        assert!(object.contains_key("memory_id"));
        assert!(matches!(
            object.get("occurred_at"),
            Some(SurrealValueData::Datetime(_))
        ));
    }

    #[test]
    fn test_normalize_quoted_record_key() {
        let uuid = Uuid::new_v4().to_string();
        let raw_id = format!("memory_note:`{uuid}`");

        assert_eq!(
            SurrealMemoryRepository::normalize_record_key(&raw_id).expect("normalize record key"),
            uuid
        );
    }

    #[tokio::test]
    #[ignore = "requires a running SurrealDB instance at ws://127.0.0.1:8000"]
    async fn test_repository_roundtrip() {
        let repo = SurrealMemoryRepository::new(SurrealConnectionConfig::new(
            "ws://127.0.0.1:8000",
            "test",
            "main",
        ).with_auth("test", "test"));
        repo.bootstrap().await.expect("bootstrap");
        repo.bootstrap().await.expect("repeat bootstrap");

        let source_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Rust".to_string(),
            ConceptType::Entity,
            "language".to_string(),
        )))
        .build()
        .expect("build source note");
        let target_note = MemoryNoteBuilder::new(MemoryType::Semantic(SemMemory::new(
            "Cargo".to_string(),
            ConceptType::Entity,
            "package manager".to_string(),
        )))
        .build()
        .expect("build target note");

        let saved = repo.upsert_note(&source_note).await.expect("upsert source");
        repo.upsert_note(&target_note).await.expect("upsert target");
        let loaded = repo.get_note(source_note.id()).await.expect("get source");

        let loaded = loaded.expect("exists");
        assert_eq!(loaded.id, saved.id);
        assert_eq!(loaded.payload, saved.payload);

        let mut embedding = vec![0.0; NOTE_EMBEDDING_DIMENSION];
        embedding[0] = 1.0;
        repo.set_note_embedding(source_note.id(), embedding.clone())
            .await
            .expect("set embedding");
        assert_eq!(
            repo.get_note_embedding(source_note.id())
                .await
                .expect("get embedding"),
            Some(embedding.clone())
        );

        let hits = repo
            .query_similar_notes(SimilarityQuery::new(embedding))
            .await
            .expect("query similar notes");
        assert!(
            hits.iter()
            .any(|hit| hit.memory_id == source_note.id().to_string())
        );

        repo.upsert_note(&source_note)
            .await
            .expect("upsert source without embedding");
        assert_eq!(
            repo.get_note_embedding(source_note.id())
                .await
                .expect("get cleared embedding"),
            None
        );

        let link = MemoryLink::new(
            source_note.id(),
            target_note.id(),
            MemoryLinkType::Sem(SemMemLink::new("uses".to_string(), 0.8, 0.9)),
        );
        let saved_link = repo.upsert_link(&link).await.expect("upsert link");
        assert_eq!(
            repo.list_outbound_links(source_note.id())
                .await
                .expect("list outbound links"),
            vec![saved_link.clone()]
        );
        assert_eq!(
            repo.list_inbound_links(target_note.id())
                .await
                .expect("list inbound links"),
            vec![saved_link]
        );
        assert_eq!(
            repo.load_note(source_note.id())
                .await
                .expect("load note")
                .expect("source exists")
                .links(),
            &[link]
        );

        let mut retrieval_event = RetrievalEventRecord::new(source_note.id());
        retrieval_event.score = Some(0.9);
        repo.append_retrieval_event(retrieval_event)
            .await
            .expect("append retrieval event");
        let retrieval_events = repo
            .list_retrieval_events(source_note.id(), EventWindow::all())
            .await
            .expect("list retrieval events");
        assert_eq!(retrieval_events.len(), 1);
        assert_eq!(retrieval_events[0].score, Some(0.9));
        assert_eq!(
            repo.get_retrieval_event_stats(source_note.id(), EventWindow::all())
                .await
                .expect("get retrieval stats")
                .total,
            1
        );

        repo.append_feedback_event(FeedbackEventRecord::new(
            source_note.id(),
            FeedbackValue::Positive,
        ))
        .await
        .expect("append feedback event");
        let feedback_events = repo
            .list_feedback_events(source_note.id(), EventWindow::all())
            .await
            .expect("list feedback events");
        assert_eq!(feedback_events.len(), 1);
        assert_eq!(feedback_events[0].feedback, FeedbackValue::Positive);
        assert_eq!(
            repo.get_feedback_event_stats(source_note.id(), EventWindow::all())
                .await
                .expect("get feedback stats")
                .total,
            1
        );

        assert!(
            repo.delete_note(source_note.id())
                .await
                .expect("delete source note")
        );
        assert!(
            repo.list_inbound_links(target_note.id())
                .await
                .expect("list links after delete")
                .is_empty()
        );
        assert!(
            repo.delete_note(target_note.id())
                .await
                .expect("delete target note")
        );
    }
}
