use rayon::prelude::*;
use std::sync::Arc;
use surrealdb::{Response, Surreal};
use anyhow::{Context, Result};
use surrealdb::engine::local::{Db, RocksDb};
use tokio::task::JoinHandle;
use crate::memory::{MemoryLink, MemoryNote};

#[allow(unused)]
const RAYON_THRESHOLD: usize = 100;
#[allow(unused)]
const EDGE_RELATE_QUERY: &str = r#"
RELATE $in->$relation->$out SET intensity = $value;
"#;
const NEIGHBOR_QUERY_UNIQUE: &str = r#"
SELECT VALUE array::distinct(array::flatten(
    (SELECT VALUE out
     FROM SELECT ->?[?]*1..$depth AS out
     FROM $starts)
))
WHERE = $depth
"#;
const NEIGHBOR_QUERY_SINGLE: &str = r#"
SELECT ->{}->{}.* FROM $in_node;
RETURN array::flatten($in_node->{}->{}.*)
"#;
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
    pub async fn query_neighbors(&self, record_id: impl AsRef<str>, table: impl AsRef<str>, relation: Option<impl AsRef<str>>) -> Result<Vec<MemoryNote>> {
        let relation = relation.map(|r| r.as_ref().to_string()).unwrap_or("?".to_string());
        let table =table.as_ref();
        let query = format!(
            "SELECT ->{}->{}.* FROM $in_node;RETURN array::flatten($in_node->{}->{}.*)",
            &relation,
            &table,
            &relation,
            &table
        );
        self.db.query(query)
            .bind(("in_node", record_id.as_ref().to_string()))
            .await?
            .take::<Vec<MemoryNote>>(1)
            .with_context(|| "Error querying neighbors") //TODO
    }
    pub fn format_record_id(table: impl AsRef<str>, id: impl AsRef<str>) -> String {
        format!("{}:{}", table.as_ref(), id.as_ref())
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
                    (Self::format_record_id(record.category.as_str(), record.id()), record)
                })
                .collect::<Vec<_>>()
        } else {
            records
                .into_iter()
                .map(|record| {
                    (Self::format_record_id(record.category.as_str(), record.id()), record)
                })
                .collect::<Vec<_>>()
        };

        fn create_task( _self: &SurrealGraphRetriever,
                        (key, record): (String, MemoryNote),
                        semaphore: Arc<tokio::sync::Semaphore>,
        ) -> JoinHandle<Result<(), anyhow::Error>> {
            let db = Arc::clone(&_self.db);

            tokio::spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| anyhow::anyhow!("Error acquiring semaphore: {}", e))?;

                db.upsert(&key)
                    .content(record)
                    .await
                    .map(|_:Vec<()>| ())
                    .context(format!("Error upserting record: {}", key))?;
                Ok(())
            })
        }
        self.execute_tasks(processed_records, create_task, use_parallel).await
    }
    async fn upsert_notes_edge(&self,  edges: Vec<(String, MemoryLink)>) -> Result<(), Vec<anyhow::Error>> {
        let use_parallel = edges.len() >= RAYON_THRESHOLD;
        fn create_task(
            _self: &SurrealGraphRetriever,
            (in_id, links): (String, MemoryLink),
            semaphore: Arc<tokio::sync::Semaphore>,
        ) -> JoinHandle<Result<(), anyhow::Error>> {
            let db = Arc::clone(&_self.db);
            tokio::spawn(async move {
                let _permit = semaphore.acquire_owned().await
                    .map_err(|e| anyhow::anyhow!("Error acquiring semaphore: {}",e))?;

                db.query(EDGE_RELATE_QUERY)
                    .bind(("in", in_id))
                    .bind(("relation", links.relation))
                    .bind(("out", links.id))
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
                        (record.id().to_string(), link.clone())
                    })
                }).collect::<Vec<_>>()
        }else {
            records.iter()
                .flat_map(|record| {
                    record.links().iter().map(|link| {
                        (record.id().to_string(), link.clone())
                    })
                }).collect::<Vec<_>>()
        };
        let res_node = self.upsert_notes_node(records).await;
        let res_edge = self.upsert_notes_edge(edge_param).await;
        if res_node.is_ok() && res_edge.is_ok() {
            Ok(())
        } else {
            Err(res_node.unwrap_err().into_iter().chain(res_edge.unwrap_err().into_iter()).collect())
        }
    }
}