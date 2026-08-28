//! SurrealDB 仓储层：`MemoryRepository` 的 surrealdb 实现（多列 HNSW 方案）。
//!
//! - 连接：嵌入式 SurrealKv（`connect`，生产）或 Mem（`connect_mem`，测试）。
//! - Schema：`include_str!("schema.surql")` 幂等初始化。
//! - 写：`split_embedded` → 事务内全量 CONTENT 替换 note 行 + 重建出边（先删后写）。
//!   不用 MERGE——深合并会残留外部标签 enum 的旧变体键（详见 `MemoryRepository::upsert_notes`）。
//! - 读：`memory_id` 过滤取回 NoteRow，`variant_emb` 还原嵌入，`memory_link` 边合并链接。
//! - 召回：查询嵌入 flatten 成槽位 → 每槽位 `<|k, EF|>` KNN → union 候选 → fetch_notes
//!   （精确重排留在调用方，DB 只做候选召回）。

use std::collections::{HashMap, HashSet};
use std::path::Path;

use async_trait::async_trait;
use serde_json::Value;

use soul_mem_core::memory_links::{LinkId, MemoryLink};
use soul_mem_core::memory_note::MemoryId;
use soul_mem_query::embedding::note::{EmbeddedMemoryNote, MemoryEmbedding};
use surrealdb::Surreal;
#[cfg(test)]
use surrealdb::engine::local::Mem;
use surrealdb::engine::local::{Db, SurrealKv};
use surrealdb::types::SurrealValue;

use super::super::{EntityKind, MemoryRepository, StorageError, StorageResult};
use super::mapper::note::{NoteRow, flatten_embedding};
use super::mapper::{MemoryIdCodec, record_str_to_memory_id, split_embedded};

/// SurrealDB 仓储实现。`db` 为嵌入式连接（SurrealKv / Mem）。
pub struct SurrealRepository {
    db: Surreal<Db>,
}

impl SurrealRepository {
    /// 连接嵌入式 SurrealKv（磁盘持久化，生产路径）。
    pub async fn connect(
        path: impl AsRef<Path>,
        character: impl AsRef<str>,
    ) -> Result<Self, surrealdb::Error> {
        let db = Surreal::new::<SurrealKv>(path.as_ref()).await?;
        db.use_ns("soulmem").use_db(character.as_ref()).await?;
        Ok(Self { db })
    }

    /// 连接嵌入式内存库（测试路径，进程退出即失）。
    #[cfg(test)]
    pub async fn connect_mem() -> Result<Self, surrealdb::Error> {
        let db = Surreal::new::<Mem>(()).await?;
        db.use_ns("soulmem").use_db("soulmem").await?;
        Ok(Self { db })
    }

    /// 幂等初始化 schema（`include_str!` 引入 `schema.surql`）。
    pub async fn init_schema(&self) -> StorageResult<()> {
        self.db.query(include_str!("schema.surql")).await?;
        Ok(())
    }

    /// 批量写：整个 batch 一个事务（原子性），事务内逐条 SDK `upsert().content()` 全量替换。
    ///
    /// 注：3.2.4 无可靠的批量 upsert 路径（探针实证：`UPSERT/UPDATE MERGE` 的数组 CONTENT 不被支持；
    /// `INSERT $batch ON DUPLICATE KEY UPDATE` 的 `$value` 引用会丢字段）。故用事务内循环。
    async fn upsert_batch(&self, mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()> {
        if mem_notes.is_empty() {
            return Ok(());
        }
        let txn = self.db.clone().begin().await?;
        let result: StorageResult<()> = async {
            for embedded in mem_notes {
                Self::write_one(&txn, embedded).await?;
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                txn.commit().await?;
                Ok(())
            }
            Err(e) => {
                txn.cancel().await?;
                Err(e)
            }
        }
    }

    /// 事务内写一条：note 行（SDK content 全量替换）+ 重建出边（先删旧再写新）。
    async fn write_one(
        txn: &surrealdb::method::Transaction<Db>,
        embedded: EmbeddedMemoryNote,
    ) -> StorageResult<()> {
        let (row, link_rows) = split_embedded(embedded)?;
        let note_key = row.id.to_string();
        let note_rid = row.id.to_record_id();
        // NoteRow 派生 SurrealValue：into_value() 产出 SDK Value（datetime → Datetime）。
        // 全量 CONTENT 替换：None 槽位 → Value::None，旧列一并清空，保证与变体严格一致
        // （MERGE 深合并会残留旧变体键与旧槽位向量，见 `MemoryRepository::upsert_notes`）。
        let note_sur_value = row.into_value();
        let link_sdks: Vec<(String, surrealdb::types::Value)> = link_rows
            .iter()
            .map(|lr| (lr.id.to_string(), lr.clone().into_value()))
            .collect();

        txn.upsert::<Option<Value>>(("memory_note", note_key.as_str()))
            .content(note_sur_value)
            .await?;
        // 重建出边：先删旧（in = 本 note），再写新（按端点删边无 SDK 等价，手写 query）。
        // 语义：边表始终以 payload 的 note.links 为完整真相源做全量同步；
        // payload 的 links 即该 note 的完整边集（links 为空即表达「无出边」，同步后清空，非数据丢失）。
        txn.query("DELETE memory_link WHERE `in` = $rid")
            .bind(("rid", note_rid))
            .await?;
        for (lr_key, lr_sdk) in &link_sdks {
            txn.upsert::<Option<Value>>(("memory_link", lr_key.as_str()))
                .content(lr_sdk.clone())
                .await?;
        }
        Ok(())
    }
}

/// 事务内取回 note 行（`memory_id IN` 过滤）。
async fn query_notes_tx(
    txn: &surrealdb::method::Transaction<Db>,
    mem_ids: &[MemoryId],
) -> StorageResult<Vec<NoteRow>> {
    if mem_ids.is_empty() {
        return Ok(Vec::new());
    }
    let id_strs: Vec<String> = mem_ids.iter().map(|id| id.to_string()).collect();
    let mut res = txn
        .query("SELECT * FROM memory_note WHERE memory_id IN $ids")
        .bind(("ids", id_strs))
        .await?;
    let raw: Vec<Value> = res.take(0)?;
    let rows = raw
        .into_iter()
        .map(serde_json::from_value::<NoteRow>)
        .collect::<Result<Vec<NoteRow>, serde_json::Error>>()?;
    Ok(rows)
}

/// 事务内取回指向 `mem_ids` 的出边（`in` = 这些节点），按 from 分组还原为 MemoryLink。
///
/// 手动解析行而非 `LinkRow` 反序列化：DB 返回的 record 值序列化为带反引号的字符串
/// （如 ``memory_note:`<uuid>``），`RecordId` 的 Deserialize 不接受字符串形状。
async fn query_out_links_tx(
    txn: &surrealdb::method::Transaction<Db>,
    mem_ids: &[MemoryId],
) -> StorageResult<HashMap<MemoryId, Vec<MemoryLink>>> {
    if mem_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let id_strs: Vec<String> = mem_ids.iter().map(|id| id.to_string()).collect();
    let mut res = txn
        .query(
            "SELECT * FROM memory_link \
             WHERE `in` IN (SELECT VALUE id FROM memory_note WHERE memory_id IN $ids)",
        )
        .bind(("ids", id_strs))
        .await?;
    let raw: Vec<Value> = res.take(0)?;
    let mut by_from: HashMap<MemoryId, Vec<MemoryLink>> = HashMap::new();
    for v in raw {
        let from = record_str_to_memory_id(v.get("in").and_then(|x| x.as_str()).unwrap_or(""))?;
        let to = record_str_to_memory_id(v.get("out").and_then(|x| x.as_str()).unwrap_or(""))?;
        let link_type: soul_mem_core::memory_links::MemoryLinkType = serde_json::from_value(
            v.get("link_type").cloned().unwrap_or(Value::Null),
        )
        .map_err(|e| StorageError::Serialize {
            kind: EntityKind::Link,
            source: e,
        })?;
        let link = soul_mem_core::memory_links::MemoryLinkBuilder::new(from, to, link_type)
            .id(LinkId::from(
                uuid::Uuid::parse_str(v.get("link_id").and_then(|x| x.as_str()).unwrap_or(""))
                    .map_err(|_| StorageError::InvalidArgument("bad link_id".into()))?,
            ))
            .intensity(v.get("intensity").and_then(|x| x.as_f64()).unwrap_or(1.0))
            .missing_degree(
                v.get("missing_degree")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0) as f32,
            )
            .last_forget_time(
                serde_json::from_value(v.get("last_forget_time").cloned().unwrap_or(Value::Null))
                    .map_err(|e| StorageError::Serialize {
                    kind: EntityKind::Link,
                    source: e,
                })?,
            )
            .build();
        by_from.entry(link.from()).or_default().push(link);
    }
    Ok(by_from)
}

/// 事务内组装：note 行 + 出边 → `EmbeddedMemoryNote`（读事务的一致快照核心）。
async fn fetch_notes_tx(
    txn: &surrealdb::method::Transaction<Db>,
    mem_ids: &[MemoryId],
) -> StorageResult<Vec<EmbeddedMemoryNote>> {
    let rows = query_notes_tx(txn, mem_ids).await?;
    let links = query_out_links_tx(txn, mem_ids).await?;
    let mut by_id: HashMap<MemoryId, NoteRow> = rows.into_iter().map(|r| (r.id, r)).collect();

    let mut out = Vec::with_capacity(by_id.len());
    for id in mem_ids {
        if let Some(row) = by_id.remove(id) {
            let note_links = links.get(id).cloned().unwrap_or_default();
            out.push(row.into_embedded(note_links)?);
        }
    }
    Ok(out)
}

#[async_trait]
impl MemoryRepository for SurrealRepository {
    async fn upsert_notes(&self, mem_notes: Vec<EmbeddedMemoryNote>) -> StorageResult<()> {
        self.upsert_batch(mem_notes).await
    }

    async fn fetch_neighbors(
        &self,
        source_ids: &[MemoryId],
        depth: usize,
    ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
        if source_ids.is_empty() {
            return Ok(Vec::new());
        }
        // 读事务：BFS 逐层 + 取回全部在一个事务内（一致快照）
        let txn = self.db.clone().begin().await?;
        let result: StorageResult<Vec<EmbeddedMemoryNote>> = async {
            let mut visited: HashSet<MemoryId> = source_ids.iter().copied().collect();
            let mut frontier: Vec<MemoryId> = source_ids.to_vec();

            for _ in 0..depth {
                if frontier.is_empty() {
                    break;
                }
                let id_strs: Vec<String> = frontier.iter().map(|id| id.to_string()).collect();
                let mut res = txn
                    .query(
                        "SELECT `in`, out FROM memory_link \
                         WHERE `in` IN (SELECT VALUE id FROM memory_note WHERE memory_id IN $ids) \
                            OR out IN (SELECT VALUE id FROM memory_note WHERE memory_id IN $ids)",
                    )
                    .bind(("ids", id_strs))
                    .await?;
                let edges: Vec<Value> = res.take(0)?;
                let mut next: Vec<MemoryId> = Vec::new();
                for e in edges {
                    for key in ["in", "out"] {
                        if let Some(mid) = e
                            .get(key)
                            .and_then(|x| x.as_str())
                            .and_then(|v| record_str_to_memory_id(v).ok())
                            && visited.insert(mid)
                        {
                            next.push(mid);
                        }
                    }
                }
                frontier = next;
            }

            let source_set: HashSet<MemoryId> = source_ids.iter().copied().collect();
            let mut neighbors: Vec<MemoryId> = visited.difference(&source_set).copied().collect();
            neighbors.sort();
            fetch_notes_tx(&txn, &neighbors).await
        }
        .await;
        txn.cancel().await?; // 读事务，丢弃
        result
    }

    async fn fetch_notes(&self, mem_ids: &[MemoryId]) -> StorageResult<Vec<EmbeddedMemoryNote>> {
        if mem_ids.is_empty() {
            return Ok(Vec::new());
        }
        let txn = self.db.clone().begin().await?;
        let result = fetch_notes_tx(&txn, mem_ids).await;
        txn.cancel().await?; // 读事务，丢弃
        result
    }

    async fn similarity_fetch(
        &self,
        embeddings: Vec<MemoryEmbedding>,
        candidate_k: usize,
    ) -> StorageResult<Vec<EmbeddedMemoryNote>> {
        // 读事务：全部槽位 KNN + 取回在一个事务内（一致快照）
        let txn = self.db.clone().begin().await?;
        let result: StorageResult<Vec<EmbeddedMemoryNote>> = async {
            let k = candidate_k.max(1) as i64; // 每槽位召回数（精确重排在调用方）
            let ef = k * 2;
            let mut candidates: Vec<MemoryId> = Vec::new();

            for embedding in embeddings {
                let slots = flatten_embedding(embedding)?;
                for (slot, vec) in slots {
                    if vec.is_zero() {
                        continue; // 零向量（如空 tag）不参与 KNN
                    }
                    let col = slot.column();
                    let sql =
                        format!("SELECT memory_id FROM memory_note WHERE {col} <|{k}, {ef}|> $q");
                    // EmbeddingVec 未实现 SurrealValue，into_inner 零拷贝取出 Vec<f32> 绑定
                    let q = vec.into_inner();
                    let mut res = txn.query(sql).bind(("q", q)).await?;
                    let hits: Vec<Value> = res.take(0)?;
                    for h in hits {
                        if let Some(uuid) = h
                            .get("memory_id")
                            .and_then(|x| x.as_str())
                            .and_then(|s| uuid::Uuid::parse_str(s).ok())
                        {
                            let mid = MemoryId::from(uuid);
                            if !candidates.contains(&mid) {
                                candidates.push(mid);
                            }
                        }
                    }
                }
            }

            fetch_notes_tx(&txn, &candidates).await
        }
        .await;
        txn.cancel().await?; // 读事务，丢弃
        result
    }

    async fn remove_notes(&self, mem_ids: &[MemoryId]) -> StorageResult<()> {
        if mem_ids.is_empty() {
            return Ok(());
        }
        // 写事务：删边 + 删 note 原子
        let txn = self.db.clone().begin().await?;
        let result: StorageResult<()> = async {
            let id_strs: Vec<String> = mem_ids.iter().map(|id| id.to_string()).collect();
            // 删边（按端点删无 SDK 等价，手写 query）；note 删除走 SDK delete
            txn.query(
                "DELETE memory_link WHERE `in` IN (SELECT VALUE id FROM memory_note WHERE memory_id IN $ids) \
                  OR out IN (SELECT VALUE id FROM memory_note WHERE memory_id IN $ids)",
            )
            .bind(("ids", id_strs))
            .await?;
            for id in mem_ids {
                txn.delete::<Option<Value>>(("memory_note", id.to_string().as_str()))
                    .await?;
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                txn.commit().await?;
                Ok(())
            }
            Err(e) => {
                txn.cancel().await?;
                Err(e)
            }
        }
    }

    async fn remove_links(&self, link_ids: &[LinkId]) -> StorageResult<()> {
        if link_ids.is_empty() {
            return Ok(());
        }
        // 写事务：批量删除原子
        let txn = self.db.clone().begin().await?;
        let result: StorageResult<()> = async {
            for id in link_ids {
                txn.delete::<Option<Value>>(("memory_link", id.to_string().as_str()))
                    .await?;
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                txn.commit().await?;
                Ok(())
            }
            Err(e) => {
                txn.cancel().await?;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soul_mem_core::memory_links::{
        LinkId, MemoryLinkBuilder, MemoryLinkType, sem_mem::SemMemLink,
    };
    use soul_mem_core::memory_note::MemoryNoteBuilder;
    use soul_mem_core::memory_note::MemoryType;
    use soul_mem_core::memory_note::sem_mem::{ConceptType, SemMemory};
    use soul_mem_core::memory_note::situation_mem::{AbstractSituation, Location};
    use soul_mem_query::embedding::EmbeddingVec;
    use soul_mem_query::embedding::note::MemoryEmbeddingVariant;
    use soul_mem_query::embedding::sem::SemanticEmbedding;
    use soul_mem_query::embedding::situation::location::LocationEmbedding;
    use soul_mem_query::embedding::situation::{AbstractSituationEmbedding, SituationEmbedding};

    /// 512 维单位向量（[c, sqrt(1-c^2), 0...]，schema HNSW DIMENSION 512）
    fn unit(c: f32) -> EmbeddingVec {
        let mut v = vec![0.0f32; 512];
        v[0] = c;
        v[1] = (1.0 - c * c).sqrt();
        EmbeddingVec::new(v)
    }

    fn sem_note(tag: f32, content: f32) -> EmbeddedMemoryNote {
        let mem_type = MemoryType::Semantic(SemMemory {
            content: "concept".into(),
            aliases: vec![],
            concept_type: ConceptType::Entity,
            description: "desc".into(),
        });
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding = MemoryEmbedding::new(
            unit(tag),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                unit(content),
                unit(content),
                unit(0.5),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    fn sit_note(narrative: f32) -> EmbeddedMemoryNote {
        let mem_type = MemoryType::Situation(
            AbstractSituation::Location(Location {
                name: "place".into(),
                coordinates: "".into(),
            })
            .into(),
        );
        let note = MemoryNoteBuilder::new(mem_type).build().unwrap();
        let embedding = MemoryEmbedding::new(
            unit(0.0),
            MemoryEmbeddingVariant::Situation(SituationEmbedding::Abstract(
                AbstractSituationEmbedding::Location(LocationEmbedding {
                    name: unit(narrative),
                    coordinates: unit(0.5),
                }),
            )),
        );
        EmbeddedMemoryNote { note, embedding }
    }

    /// 给 note 追加一条指向 `to` 的出边（记录链接 id 供后续删除测试）
    fn with_link(mut embedded: EmbeddedMemoryNote, to: MemoryId) -> (EmbeddedMemoryNote, LinkId) {
        let link = MemoryLinkBuilder::new(
            embedded.note.id(),
            to,
            MemoryLinkType::Sem(SemMemLink::new("relates".into(), 1.0)),
        )
        .build();
        let link_id = link.id();
        embedded.note.links_mut().push(link);
        (embedded, link_id)
    }

    /// 以指定 id 重建 note（MemoryNote 无 set_id，builder 为准；字段值从原 note 复制）
    fn rebuild_with_id(mut embedded: EmbeddedMemoryNote, id: MemoryId) -> EmbeddedMemoryNote {
        let tags = embedded.note.tags().to_vec();
        let missing = embedded.note.missing_degree();
        let mem_type = embedded.note.mem_type().clone();
        let retr = embedded.note.retrieval_count();
        let create = embedded.note.creation_time();
        let accessed = embedded.note.last_accessed_time();
        let last_forget = embedded.note.last_forget_time();
        let new_note = MemoryNoteBuilder::new(mem_type)
            .id(id)
            .tags(tags)
            .retrieval_count(retr)
            .create_time(create)
            .last_accessed_time(accessed)
            .missing_degree(missing)
            .last_forget_time(last_forget)
            .build()
            .unwrap();
        embedded.note = new_note;
        embedded
    }

    async fn repo() -> SurrealRepository {
        let r = SurrealRepository::connect_mem().await.unwrap();
        r.init_schema().await.unwrap();
        r
    }

    #[tokio::test]
    async fn upsert_fetch_roundtrip_semantic() {
        let repo = repo().await;
        let expected = sem_note(0.3, 1.0);
        let id = expected.note().id();

        repo.upsert_notes(vec![expected.clone()])
            .await
            .unwrap();
        let fetched = repo.fetch_notes(&[id]).await.unwrap();
        assert_eq!(fetched.len(), 1);
        let f = &fetched[0];
        assert_eq!(f.note().id(), id);
        assert_eq!(f.note().mem_type(), expected.note().mem_type());
        assert_eq!(f.note().tags(), expected.note().tags());
        assert_eq!(f.note().missing_degree(), expected.note().missing_degree());
        assert_eq!(f.embedding(), expected.embedding());
    }

    #[tokio::test]
    async fn upsert_with_links_roundtrip() {
        let repo = repo().await;
        let a = sem_note(0.3, 1.0);
        let b = sem_note(0.2, 0.5);
        let a_id = a.note().id();
        let b_id = b.note().id();
        let (a, _link_id) = with_link(a, b_id);

        repo.upsert_notes(vec![a, b])
            .await
            .unwrap();
        let fetched = repo.fetch_notes(&[a_id]).await.unwrap();
        assert_eq!(fetched.len(), 1);
        let links = fetched[0].note().links();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].from(), a_id);
        assert_eq!(links[0].to(), b_id);
    }

    #[tokio::test]
    async fn fetch_neighbors_breadth() {
        let repo = repo().await;
        let a = sem_note(0.3, 1.0);
        let b = sem_note(0.2, 0.5);
        let c = sit_note(0.9);
        let c_id = c.note().id();
        let a_id = a.note().id();
        let b_id = b.note().id();
        let (b, _) = with_link(b, c_id);
        let (a, _) = with_link(a, b_id);

        repo.upsert_notes(vec![a, b, c])
            .await
            .unwrap();

        let d1 = repo.fetch_neighbors(&[a_id], 1).await.unwrap();
        let mut ids1: Vec<_> = d1.iter().map(|e| e.note().id()).collect();
        ids1.sort();
        assert_eq!(ids1, vec![b_id], "depth 1 reaches B only");

        let d2 = repo.fetch_neighbors(&[a_id], 2).await.unwrap();
        let mut ids2: Vec<_> = d2.iter().map(|e| e.note().id()).collect();
        ids2.sort();
        let mut expect2 = vec![b_id, c_id];
        expect2.sort();
        assert_eq!(ids2, expect2, "depth 2 reaches B and C");
    }

    #[tokio::test]
    async fn similarity_fetch_variant_isolation() {
        let repo = repo().await;
        let a = sem_note(0.3, 1.0);
        let b = sem_note(0.2, 0.2);
        let c = sit_note(1.0);
        let a_id = a.note().id();
        let b_id = b.note().id();
        let c_id = c.note().id();
        repo.upsert_notes(vec![a, b, c])
            .await
            .unwrap();

        // 语义查询（零 tag，只走 variant 通道）：content ~ [1,0,0...] → 只召回语义记忆
        // （情境节点的 sem_* 为 NONE；tag 通道跨变体是设计行为，这里特意用零 tag 验证变体隔离）
        let q_sem = MemoryEmbedding::new(
            EmbeddingVec::zero(512),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                unit(1.0),
                unit(1.0),
                unit(0.5),
            )),
        );
        let hits = repo.similarity_fetch(vec![q_sem], 2).await.unwrap();
        let mut ids: Vec<_> = hits.iter().map(|e| e.note().id()).collect();
        ids.sort();
        let mut expect_sem = vec![a_id, b_id];
        expect_sem.sort();
        assert_eq!(
            ids, expect_sem,
            "semantic query must not recall situation note"
        );

        // 情境查询（零 tag）：narrative 经 fused_self 通道 → 只召回情境记忆
        let q_sit = MemoryEmbedding::new(
            EmbeddingVec::zero(512),
            MemoryEmbeddingVariant::Situation(SituationEmbedding::Abstract(
                AbstractSituationEmbedding::Location(LocationEmbedding {
                    name: unit(1.0),
                    coordinates: unit(0.5),
                }),
            )),
        );
        let hits = repo.similarity_fetch(vec![q_sit], 2).await.unwrap();
        let mut ids: Vec<_> = hits.iter().map(|e| e.note().id()).collect();
        ids.sort();
        assert_eq!(
            ids,
            vec![c_id],
            "situation query must not recall semantic notes"
        );
    }

    #[tokio::test]
    async fn remove_notes_cleans_edges() {
        let repo = repo().await;
        let a = sem_note(0.3, 1.0);
        let b = sem_note(0.2, 0.5);
        let a_id = a.note().id();
        let b_id = b.note().id();
        let (a, _) = with_link(a, b_id);
        repo.upsert_notes(vec![a, b])
            .await
            .unwrap();

        repo.remove_notes(&[a_id]).await.unwrap();
        assert!(repo.fetch_notes(&[a_id]).await.unwrap().is_empty());
        // b 的入边（来自 a）应随 a 删除而清理
        let mut res = repo
            .db
            .query("SELECT VALUE id FROM memory_link WHERE out = $rid")
            .bind(("rid", b_id.to_record_id()))
            .await
            .unwrap();
        let edges: Vec<Value> = res.take(0).unwrap();
        assert!(edges.is_empty(), "edge from removed note must be gone");
    }

    #[tokio::test]
    async fn remove_links_by_id() {
        let repo = repo().await;
        let a = sem_note(0.3, 1.0);
        let b = sem_note(0.2, 0.5);
        let a_id = a.note().id();
        let b_id = b.note().id();
        let (a, link_id) = with_link(a, b_id);
        repo.upsert_notes(vec![a, b])
            .await
            .unwrap();

        repo.remove_links(&[link_id]).await.unwrap();
        let fetched = repo.fetch_notes(&[a_id]).await.unwrap();
        assert!(fetched[0].note().links().is_empty());
    }

    /// 同 id 二次全量写（唯一写模式）：payload 为完整快照，覆盖字段生效、其余字段取自快照本身。
    #[tokio::test]
    async fn upsert_same_id_overwrites_fields() {
        let repo = repo().await;
        let id_value = {
            let mut n = sem_note(0.3, 1.0);
            n.note.set_missing_degree(0.1);
            n.note.id()
        };
        repo.upsert_notes(vec![rebuild_with_id(sem_note(0.3, 1.0), id_value)])
            .await
            .unwrap();

        // 同 id 二次全量写：payload 为完整快照（仅 missing_degree 变化）
        let mut second = sem_note(0.3, 1.0);
        second.note.set_missing_degree(0.9);
        let second = rebuild_with_id(second, id_value);
        repo.upsert_notes(vec![second]).await.unwrap();

        let fetched = repo.fetch_notes(&[id_value]).await.unwrap();
        assert_eq!(fetched.len(), 1, "upsert 不丢记录");
        assert_eq!(fetched[0].note().missing_degree(), 0.9, "覆盖提供字段");
        assert_eq!(
            fetched[0].note().mem_type(),
            sem_note(0.3, 1.0).note().mem_type()
        );
        assert_eq!(fetched[0].note().retrieval_count(), 0);
    }

    /// 全量替换对「变体切换」的正确性回归：Semantic → Situation 全量覆盖后，
    /// 旧变体键与旧槽位向量被清空——读回正常（无 MERGE 深合并的双变体键残留）、
    /// KNN 不再误召回（无旧 sem_content_emb 残留）。
    #[tokio::test]
    async fn upsert_variant_switch_cleans_old_slots() {
        let repo = repo().await;
        let sem = sem_note(0.3, 1.0);
        let id = sem.note().id();
        repo.upsert_notes(vec![sem]).await.unwrap();

        // 变体切换：Situation note 全量覆盖同一 id
        repo.upsert_notes(vec![rebuild_with_id(sit_note(0.9), id)])
            .await
            .unwrap();

        // 读回正常（mem_type 为 Situation，无双变体键残留）
        let fetched = repo.fetch_notes(&[id]).await.unwrap();
        assert_eq!(fetched.len(), 1);
        assert!(
            matches!(fetched[0].note().mem_type(), MemoryType::Situation(_)),
            "变体切换后 mem_type 应为 Situation"
        );

        // 旧 sem 槽位已清空：sem 查询不得误召回该 note
        let q_sem = MemoryEmbedding::new(
            EmbeddingVec::zero(512),
            MemoryEmbeddingVariant::Semantic(SemanticEmbedding::new(
                unit(1.0),
                unit(1.0),
                unit(0.5),
            )),
        );
        let hits = repo.similarity_fetch(vec![q_sem], 2).await.unwrap();
        assert!(
            hits.iter().all(|e| e.note().id() != id),
            "变体切换后 sem 查询不得误召回"
        );

        // sit 查询仍能召回该 note（新槽位生效）
        let q_sit = MemoryEmbedding::new(
            EmbeddingVec::zero(512),
            MemoryEmbeddingVariant::Situation(SituationEmbedding::Abstract(
                AbstractSituationEmbedding::Location(LocationEmbedding {
                    name: unit(1.0),
                    coordinates: unit(0.5),
                }),
            )),
        );
        let hits = repo.similarity_fetch(vec![q_sit], 2).await.unwrap();
        assert!(
            hits.iter().any(|e| e.note().id() == id),
            "变体切换后 sit 查询应能召回"
        );
    }
}
