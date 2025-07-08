use std::rc::Rc;
use rayon::prelude::*;
use std::sync::Arc;
use surrealdb::{RecordId, Response, Surreal};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::opt::{IntoResource, Resource};
use surrealdb::sql::{Thing, Value};
use tokio::task::{id, JoinHandle};
use crate::memory::{MemoryLink, MemoryNote, MemoryNoteBuilder};

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct SurrealMemoryNoteWrapper {
    pub content : String, //内容，自然文本
    pub id : Thing, //uuid
    pub keywords : Vec<String>,//关键词
    pub links : Vec<MemoryLink>,//链接记忆
    pub retrieval_count : u32,//被检索次数
    pub timestamp : u64,//创建时间
    pub last_accessed : u64,//最后访问时间
    pub context : String,//记忆情景
    pub evolution_history : Vec<String>,//进化历史
    pub category : String,//分类,用作Surreal db 的表名
    pub tags : Vec<String>,//标签，（认知，行为）
}
impl From<SurrealMemoryNoteWrapper> for MemoryNote {
    fn from(wrapper: SurrealMemoryNoteWrapper) -> Self {
        MemoryNoteBuilder::new(wrapper.content)
            .category(wrapper.category)
            .links(wrapper.links)
            .tags(wrapper.tags)
            .context(wrapper.context)
            .evolution_history(wrapper.evolution_history)
            .timestamp(wrapper.timestamp)
            .last_accessed(wrapper.last_accessed)
            .retrieval_count(wrapper.retrieval_count)
            .keywords(wrapper.keywords)
            .id(wrapper.id.id.to_string())
            .build()
    }
}
pub struct RecordIdProxy(RecordId);
impl AsRef<RecordId> for RecordIdProxy {
    fn as_ref(&self) -> &RecordId {
        &self.0
    }
}
impl From<RecordId> for RecordIdProxy {
    fn from(record_id: RecordId) -> Self {
        RecordIdProxy(record_id)
    }
}
impl Into<RecordId> for RecordIdProxy {
    fn into(self) -> RecordId {
        self.0
    }
}

#[allow(unused)]
const RAYON_THRESHOLD: usize = 100;

#[allow(dead_code)]
pub struct RelateQueryBuilder {
    relation: String,
}
impl RelateQueryBuilder {
    pub fn new(relation: impl AsRef<str>) -> Self {
        RelateQueryBuilder {
            relation: relation.as_ref().to_string(),
        }
    }
    pub fn build(self) -> String {
        format!("RELATE $in->{}->$out SET intensity = $value;",self.relation)
    }
}
pub struct NeighborQueryBuilder {
    depth: usize,
    relation: Option<String>,
    table_name: Option<String>,
    start_node: Option<RecordId>,
}
impl NeighborQueryBuilder {
    pub fn new(depth: usize) -> Self {
        NeighborQueryBuilder {
            depth,
            relation: None,
            table_name: None,
            start_node: None,
        }
    }
    pub fn relation(mut self, relation: impl AsRef<str>) -> Self {
        self.relation = Some(relation.as_ref().to_string());
        self
    }
    pub fn table_name(mut self, table_name: impl AsRef<str>) -> Self {
        self.table_name = Some(table_name.as_ref().to_string());
        self
    }
    pub fn start_node(mut self, start_node: RecordId) -> Self {
        self.start_node = Some(start_node);
        self
    }
    pub fn build(self) -> String {
        if let Some(start_node) = self.start_node {
            format!(
                "SELECT @.{{..{}+collect}}->{}->{} AS neighbors FROM {} FETCH neighbors;",
                self.depth,
                self.relation.as_ref().unwrap_or(&"?".to_string()),
                self.table_name.as_ref().unwrap_or(&"?".to_string()),
                start_node.to_string(),
            )
        } else {
            format!(
                "SELECT @.{{..{}+collect}}->{}->{} AS neighbors FROM $start_node FETCH neighbors;",
                self.depth,
                self.relation.as_ref().unwrap_or(&"?".to_string()),
                self.table_name.as_ref().unwrap_or(&"?".to_string())
            )
        }
    }
}
pub struct IdQueryBuilder;
impl IdQueryBuilder {
    pub fn new() -> Self {
        Self
    }
    pub fn build(self, ids: impl IntoIterator<Item = impl AsRef<RecordId>>) -> String {
        let mut query = ids.into_iter()
            .fold("SELECT * FROM ".to_string(), |acc, id| {
                acc + &format!("{},", id.as_ref())
            });
        query.pop();
        query

    }
}

pub fn format_record_id(table: impl AsRef<str>, id: impl AsRef<str>) -> String {
    Thing::from((table.as_ref(), id.as_ref())).to_raw()
}
#[allow(dead_code)]
pub struct SurrealGraphRetriever {
    db: Arc<Surreal<Db>>,
    capacity: usize,
}
#[allow(dead_code)]
impl SurrealGraphRetriever {
    pub async fn new(local_address: impl AsRef<str>, capacity: Option<usize>) -> Result<Self> {
        Ok(
            Self {
                db: Arc::new(Surreal::new::<RocksDb>(local_address.as_ref()).with_capacity(capacity.unwrap_or(0)).await?),
                capacity: capacity.unwrap_or(100),
            }
        )
    }
    pub async fn use_ns_db(&self, ns: impl AsRef<str>, db: impl AsRef<str>) -> Result<()> {
        Ok(self.db.use_ns(ns.as_ref()).use_db(db.as_ref()).await?)
    }
    pub async fn raw_query(&self, query: impl AsRef<str>) -> Result<Response> {
        Ok(self.db.query(query.as_ref()).await?)
    }
    pub async fn query_neighbors(&self,depth: usize, start_record_id: RecordId, table: Option<impl AsRef<str>>, relation: Option<impl AsRef<str>>) -> Result<Vec<MemoryNote>> {
        let relation = relation.map(|r| r.as_ref().to_string()).unwrap_or("?".to_string());
        let table =table.map(|t| t.as_ref().to_string()).unwrap_or("?".to_string());
        let query = NeighborQueryBuilder::new(depth)
            .relation(relation)
            .table_name(table)
            .build();
        //println!("{}",query);
        let mut res =self.db.query(query)
            .bind(("start_node", start_record_id))
            .await?;
        let p_res = res.take::<Vec<Vec<SurrealMemoryNoteWrapper>>>((0,"neighbors"))
            .with_context(|| "Error querying neighbors")?
            .into_iter()
            .flatten()
            .map(|wrapper| wrapper.into())
            .collect();//TODO:???????????????????????????????????????????????????? WTF???????????????????
        //println!("{:?}",p_res);
        Ok(p_res)
    }
    pub async fn query_neighbors_batch(&self, queries: Vec<String>) -> Result<Vec<Vec<MemoryNote>>> {  //use neighbor query builder to build query strings, should identify the RecordId when building
        let len = queries.len();
        let queries = queries.into_iter()
            .fold("".to_string(),|acc, query| acc + &query + "\n");
        let mut res = self.db.query(queries).await?;
        let mut p_res = Vec::with_capacity(len);
        for i in 0..len {
            p_res.push(
                res.take::<Vec<Vec<SurrealMemoryNoteWrapper>>>((i,"neighbors"))
                    .with_context(|| "Error querying neighbors")?
                    .into_iter()
                    .flatten()
                    .map(|wrapper| wrapper.into())
                    .collect()//TODO:???????????????????????????????????????????????????? WTF???????????????????
            )
        }
        Ok(p_res)
    }
    pub async fn query_ids(&self, ids: impl IntoIterator<Item = impl AsRef<RecordId>>) -> Result<Vec<MemoryNote>> {
        Ok(self.db.query(IdQueryBuilder::new().build(ids))
            .await?
            .take::<Vec<SurrealMemoryNoteWrapper>>(0)
            .with_context(|| "Error querying ids")?
            .into_iter()
            .map(|wrapper| wrapper.into())
            .collect())
    }
    //TODO:test it
    pub async fn batch_query_ids(&self,ids: Vec<impl IntoIterator<Item = impl AsRef<RecordId>> + Send + 'static>) -> Result<Vec<Vec<MemoryNote>>,Vec<anyhow::Error>> {
        fn create_task( _self: &SurrealGraphRetriever,
                        record_ids: impl IntoIterator<Item = impl AsRef<RecordId>> + Send + 'static,
                        semaphore: Arc<tokio::sync::Semaphore>,
        ) -> JoinHandle<Result<Vec<MemoryNote>, anyhow::Error>> {
            let db = Arc::clone(&_self.db);
            tokio::spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| anyhow::anyhow!("Error acquiring semaphore: {}", e))?;

                Ok(db.query(IdQueryBuilder::new().build(record_ids))
                    .await?
                    .take::<Vec<SurrealMemoryNoteWrapper>>(0)
                    .with_context(|| "Error querying ids")?
                    .into_iter()
                    .map(|wrapper| wrapper.into())
                    .collect())

            })
        }
        let len = ids.len();
        let use_parallel = len >= RAYON_THRESHOLD;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.capacity));
        let mut errors = Vec::new();
        let mut tasks = Vec::with_capacity(len);
        if use_parallel {
            // 并行处理分支
            tasks = ids
                .into_par_iter()
                .map(|item| {
                    create_task(self, item, Arc::clone(&semaphore))
                })
                .collect::<Vec<_>>();
        } else {
            // 串行处理分支
            for item in ids {
                tasks.push(create_task(self, item, Arc::clone(&semaphore)));
            }
        }
        let mut res_notes = Vec::with_capacity(len);
        // 统一结果处理
        for task in tasks {
            match task.await {
                Ok(Ok(res)) => {res_notes.push(res)}
                Ok(Err(e)) => errors.push(e),
                Err(join_err) => errors.push(anyhow::anyhow!("Error joining task: {:?}", join_err)),
            }
        }
        if errors.is_empty() {
            Ok(res_notes)
        } else {
            Err(errors)
        }
    }

    /// 通用任务执行器 - 处理并行/串行任务调度
    async fn execute_tasks<T, F>(
        &self,
        items: Vec<T>,
        create_task: F,
        use_parallel: bool,
    ) -> Result<(), Vec<anyhow::Error>>
    where
        T: Send + 'static,
        F: Fn(&Self, T, Arc<tokio::sync::Semaphore>) -> JoinHandle<Result<(), anyhow::Error>>
        + Send
        + Sync
        + Copy,
    {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.capacity));
        let mut errors = Vec::new();
        let mut tasks = Vec::with_capacity(items.len());
        if use_parallel {
            // 并行处理分支
            tasks = items
                .into_par_iter()
                .map(|item| create_task(self, item, Arc::clone(&semaphore)))
                .collect::<Vec<_>>();
        } else {
            // 串行处理分支
            for item in items {
                tasks.push(create_task(self, item, Arc::clone(&semaphore)));
            }
        }
        // 统一结果处理
        for task in tasks {
            match task.await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => errors.push(e),
                Err(join_err) => errors.push(anyhow::anyhow!("Error joining task: {:?}", join_err)),
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    async fn upsert_notes_node(&self, records: Vec<MemoryNote>) -> Result<(), Vec<anyhow::Error>> {
        let len = records.len();
        //根据数据量大小，选择是否使用rayon并行操作
        let use_parallel = len >= RAYON_THRESHOLD;

        let processed_records = if use_parallel {
            records
                .into_par_iter()
                .map(|record| {
                    ((record.category.as_str().to_owned(), record.id().to_string()), record)
                })
                .collect::<Vec<_>>()
        } else {
            records
                .into_iter()
                .map(|record| {
                    ((record.category.as_str().to_owned(), record.id().to_string()), record)
                })
                .collect::<Vec<_>>()
        };

        fn create_task( _self: &SurrealGraphRetriever,
                        (key, record): ((String, String), MemoryNote),
                        semaphore: Arc<tokio::sync::Semaphore>,
        ) -> JoinHandle<Result<(), anyhow::Error>> {
            let db = Arc::clone(&_self.db);
            tokio::spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| anyhow::anyhow!("Error acquiring semaphore: {}", e))?;

                db.upsert(key.clone())
                    .content(record)
                    .await
                    .map(|_:Option<SurrealMemoryNoteWrapper>| ())
                    .context(format!("Error upserting record: {}", format_record_id(key.0, key.1.as_str())))?;
                Ok(())
            })
        }
        self.execute_tasks(processed_records, create_task, use_parallel).await
    }
    pub async fn define_unique_relation_index(db: &Arc<Surreal<Db>>, relation: impl AsRef<str>) -> Result<()>{
        db.query(
            format!("DEFINE TABLE IF NOT EXISTS {} SCHEMALESS;",relation.as_ref())
        ).await?;
        db.query(
            format!("DEFINE INDEX IF NOT EXISTS unique_relationships ON TABLE {} COLUMNS in, out UNIQUE;",relation.as_ref())
        ).await?;
        Ok(())
    }
    async fn upsert_notes_edge(&self,  edges: Vec<(RecordId, MemoryLink)>) -> Result<(), Vec<anyhow::Error>> {
        let use_parallel = edges.len() >= RAYON_THRESHOLD;
        fn create_task(
            _self: &SurrealGraphRetriever,
            (in_id, links): (RecordId, MemoryLink),
            semaphore: Arc<tokio::sync::Semaphore>,
        ) -> JoinHandle<Result<(), anyhow::Error>> {
            let db = Arc::clone(&_self.db);
            tokio::spawn(async move {
                let _permit = semaphore.acquire_owned().await
                    .map_err(|e| anyhow::anyhow!("Error acquiring semaphore: {}",e))?;

                SurrealGraphRetriever::define_unique_relation_index(&db, links.relation.clone()).await?;
                let out_id = links.as_surreal_id();
                let res = db.query(RelateQueryBuilder::new(links.relation).build())
                    .bind(("in", in_id))
                    .bind(("out", out_id))
                    .bind(("value", links.intensity))
                    .await
                    .context("Error upserting edge")?;


                Ok(())
            })
        }
        self.execute_tasks(edges, create_task, use_parallel).await
    }
    pub async fn upsert_notes(&self, records: Vec<MemoryNote>) -> Result<(), Vec<anyhow::Error>> {
        let edge_param = if records.len() >= RAYON_THRESHOLD {
            records.par_iter()
                .flat_map_iter(|record| {
                    record.links().iter().map(|link| {
                        (record.surreal_id(), link.clone())
                    })
                }).collect::<Vec<_>>()
        }else {
            records.iter()
                .flat_map(|record| {
                    record.links().iter().map(|link| {
                        (record.surreal_id(), link.clone())
                    })
                }).collect::<Vec<_>>()
        };
        let res_node = self.upsert_notes_node(records).await;
        let res_edge = self.upsert_notes_edge(edge_param).await;
        if res_node.is_ok() && res_edge.is_ok() {
            Ok(())
        } else {
            let mut err = Vec::new();
            if res_node.is_err() {
                err.extend(res_node.unwrap_err())
            }
            if res_edge.is_err() {
                err.extend(res_edge.unwrap_err())
            }
            Err(err)
        }
    }
}

mod test {
    use crate::memory::MemoryNoteBuilder;
    use super::*;
    async fn prepare_db_connect() -> SurrealGraphRetriever {
        let db = SurrealGraphRetriever::new("./test/mydatabase.db", None).await.unwrap();
        db.use_ns_db("test", "test").await.unwrap();
        db
    }
    fn prepare_test_data() -> Vec<MemoryNote> {
        vec![
            MemoryNoteBuilder::new("test1")
                .id("test1")
                .category("test")
                .links(vec![MemoryLink::new("test2", Some("default"),"test",1u32)])
                .build(),
            MemoryNoteBuilder::new("test2")
                .id("test2")
                .category("test")
                .links(vec![MemoryLink::new("test3", Some("default"),"test",1u32)])
                .build(),
            MemoryNoteBuilder::new("test3")
                .id("test3")
                .category("test")
                .links(vec![MemoryLink::new("test1", Some("default"),"test",1u32)])
                .build(),
        ]
    }
    #[tokio::test]
    async fn test_upsert_notes() {
        let db = prepare_db_connect().await;
        let data = prepare_test_data();
        db.upsert_notes(data).await.unwrap();
    }
    #[tokio::test]
    async fn test_query_notes() {
        let db = prepare_db_connect().await;
        let response = db.query_neighbors(2,RecordId::from_table_key("test","test1"), Some("test"),None::<String>).await.unwrap();
        println!("response: {:?}", response);
        assert_eq!(response.len(), 2,"the response is {:?}",response);
    }
    #[tokio::test]
    async fn test_query_neighbors_batch() { 
        let db = prepare_db_connect().await;
        let response = db.query_neighbors_batch(
            vec![
                NeighborQueryBuilder::new(2)
                    .start_node(RecordId::from_table_key("test","test1"))
                    .build(),
                NeighborQueryBuilder::new(2)
                    .start_node(RecordId::from_table_key("test","test2"))
                    .build(),
            ]
        ).await.unwrap();
        assert_eq!(response.len(), 2,"the response is {:?}",response);
        for res in response {
            println!("response: {:?}", res);
            assert_eq!(res.len(), 2,"the response is {:?}",res);
        }
    }
    #[tokio::test]
    async fn test_query_id() {
        let db = prepare_db_connect().await;
        let response = db.query_ids(
            vec![
                RecordIdProxy::from(RecordId::from_table_key("test","test1")),
                RecordIdProxy::from(RecordId::from_table_key("test","test2"))
            ]
        ).await.unwrap();
        assert_eq!(response.len(), 2,"the response is {response:?}");
        for (i,res) in response.iter().enumerate() {
            println!("response: {res:?}");
            let mut id = res.id().to_string();
            id.pop();
            assert_eq!(id, "test","the response is {:?}",res);
        }
    }
    #[tokio::test]
    async fn test_batch_query_ids() {
        let db = prepare_db_connect().await;
        let response = db.batch_query_ids(
            vec![
                vec![RecordIdProxy::from(RecordId::from_table_key("test","test1"))],
                vec![RecordIdProxy::from(RecordId::from_table_key("test","test2"))]
            ]
        ).await.unwrap();
        assert_eq!(response.len(), 2,"the response is {:?}",response);
        for res in response {
            println!("response: {:?}", res);
            assert_eq!(res.len(), 1,"the response is {:?}",res);
        }
    }
}