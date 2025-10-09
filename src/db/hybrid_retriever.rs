// //TODO:test it
// mod config;
//
//
// use super::{qdrant_retriever, surreal_retriever};
// use anyhow::{Context, Result};
// use qdrant_client::qdrant;
// use surrealdb::RecordId;
// use crate::db::hybrid_retriever::config::{QdrantConfig, SurrealConfig};
// use crate::memory::MemoryNote;
//
// pub struct HybridRetriever {
//     qdrant_retriever: qdrant_retriever::QdrantRetriever,
//     surreal_retriever: surreal_retriever::SurrealGraphRetriever,
// }
//
// impl HybridRetriever{
//     pub async fn new(qdrant_config: QdrantConfig, surreal_config: SurrealConfig) -> Result<Self> {
//         Ok(HybridRetriever {
//             qdrant_retriever: qdrant_retriever::QdrantRetriever::new(
//                 qdrant_config.collection_name(),
//                 Some(qdrant_config.port()),
//                 qdrant_config.keep_alive(),
//                 qdrant_config.embedding_model().to_owned(), //TODO:test this
//             )?,
//             surreal_retriever: surreal_retriever::SurrealGraphRetriever::new(
//                 surreal_config.local_address(),
//                 Some(surreal_config.capacity()),
//             ).await?
//         })
//     }
//     pub async fn surreal_ns_db(&self,ns: impl AsRef<str>, db: impl AsRef<str>) -> Result<()> {
//         self.surreal_retriever.use_ns_db(ns,db).await
//     }
//     pub async fn upsert_notes(&self, notes: Vec<MemoryNote>) -> Result<()> {
//         self.qdrant_retriever.add_points(&notes).await?;
//         let surreal_res = self.surreal_retriever.upsert_notes(notes).await;
//         if surreal_res.is_err() {
//             return surreal_res
//                 .map_err(|e|e.into_iter()
//                     .fold(anyhow::anyhow!("Error upserting notes to surreal:"), |acc, e| acc.context(e)))
//
//         }
//         Ok(())
//     }
//     pub async fn retrieve_related_notes(&self, queries: &[impl AsRef<str>], k: u64, filter: Option<impl Into<qdrant::Filter> + Clone>) -> Result<Vec<Vec<MemoryNote>>> {
//         let raw_ids = self.qdrant_retriever.query(queries, k, filter).await?;
//         let processed_ids = qdrant_retriever::QdrantRetriever::parse_response_to_record_id(raw_ids.1);
//         let res = self.surreal_retriever.batch_query_ids(processed_ids).await;
//         if let Err(res) = res {
//             Err(res
//                 .into_iter()
//                 .fold(anyhow::anyhow!("Error querying surreal:"), |acc, e| acc.context(e)))
//         }else{
//             Ok(res.unwrap())
//         }
//     }
//     pub async fn retrieve_notes_by_id(&self, ids: impl IntoIterator<Item = impl AsRef<RecordId>>) -> Result<Vec<MemoryNote>>{
//         self.surreal_retriever.query_ids(ids).await
//     }
//     pub async fn retrieve_neighbor_notes(&self, depth: usize, start_record_id: RecordId, table: Option<impl AsRef<str>>, relation: Option<impl AsRef<str>>) -> Result<Vec<MemoryNote>> {
//         self.surreal_retriever.query_neighbors(depth, start_record_id, table, relation).await
//     }
//     pub async fn retrieve_neighbor_notes_batch(&self, queries: Vec<String>) -> Result<Vec<Vec<MemoryNote>>> { //use neighbor query builders to build the queries
//         self.surreal_retriever.query_neighbors_batch(queries).await
//     }
//
// }
// mod test {
//     use fastembed::EmbeddingModel;
//     use qdrant_client::qdrant::Filter;
//     use crate::memory::{MemoryLink, MemoryNoteBuilder};
//     use super::*;
//     async fn prepare_db() -> HybridRetriever {
//         let hybrid = HybridRetriever::new(
//             QdrantConfig::new("hybrid_test", EmbeddingModel::AllMiniLML6V2),
//             SurrealConfig::new("./test/hybrid_test.db", None)
//         ).await.unwrap();
//         hybrid.surreal_ns_db("test", "test").await.unwrap();
//         hybrid
//     }
//     fn prepare_test_data() -> Vec<MemoryNote> {
//         vec![
//             MemoryNoteBuilder::new("test1")
//                 .category("test")
//                 .id("d69691ad-42d7-433d-96b5-99436c4e9c21")
//                 .links(vec![MemoryLink::new("test2", Some("default"),"test",1f32)])
//                 .build(),
//             MemoryNoteBuilder::new("test2")
//                 .category("test")
//                 .id("6067153d-aceb-4a43-ac11-cf2a801ed32c")
//                 .links(vec![MemoryLink::new("test3", Some("default"),"test",1f32)])
//                 .build(),
//             MemoryNoteBuilder::new("test3")
//                 .category("test")
//                 .id("006b9abb-5506-4d8c-b2eb-f366452a4e53")
//                 .links(vec![MemoryLink::new("test1", Some("default"),"test",1f32)])
//                 .build(),
//         ]
//     }
//     #[tokio::test]
//     async fn test_hybrid_upsert() {
//         let hybrid = prepare_db().await;
//         let data = prepare_test_data();
//         hybrid.upsert_notes(data).await.unwrap()
//     }
//     #[tokio::test]
//     async fn test_retrieve_related_notes() {
//         let hybrid = prepare_db().await;
//         let res = hybrid.retrieve_related_notes(&["test"], 10, None::<Filter>).await.unwrap();
//         assert_eq!(res.len(), 1);
//         assert_eq!(res[0].len(), 3);
//         println!("{:?}", res[0]);
//     }
// }