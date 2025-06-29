use std::collections::HashMap;
use qdrant_client::qdrant::{CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId, PointStruct, PointsIdsList, PointsSelector, PointsUpdateOperation, Query, QueryBatchPointsBuilder, QueryBatchResponse, QueryPoints, QueryPointsBuilder, UpdateBatchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder};
use qdrant_client::{Payload, Qdrant};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use anyhow::{Error, Result};
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::points_update_operation::{Operation, OverwritePayload};
use crate::memory::MemoryNote;
use crate::soul_embedding::CalcEmbedding;

#[allow(dead_code)]
pub struct QdrantRetriever {
    client: Qdrant,
    collection_name: String,
    embedding_model : TextEmbedding,
}
#[allow(dead_code)]
impl QdrantRetriever {
    pub fn new(collection_name : &str, port: Option<u32>, keep_alive: bool, embedding_model: EmbeddingModel) -> Result<Self> {
        let port = port.unwrap_or(6334);
        let _client = if keep_alive {
            Qdrant::from_url(format!("http://localhost:{port}").as_str() ).keep_alive_while_idle().build()?
        }else{
            Qdrant::from_url(format!("http://localhost:{port}").as_str()).build()?
        };

        let _embedding_model = TextEmbedding::try_new(
            InitOptions::new(embedding_model).with_show_download_progress(true),
        )?;
        Ok(
            QdrantRetriever {
                client: _client,
                collection_name: collection_name.to_owned(),
                embedding_model: _embedding_model
            }
        )
    }
    async fn select_or_create_collection(&self,embedding_len: u64) -> Result<()> {
        if !self.client.collection_exists(&self.collection_name).await? {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection_name)
                        .vectors_config(VectorParamsBuilder::new(embedding_len, Distance::Cosine)),
                )
                .await?;
        }
        Ok(())
    }
    pub async fn add_points(&self,payloads: Vec<MemoryNote>) -> Result<()> {

        // dbg!(&payloads);
        //dbg!(serde_json::json!(payloads.get(0)));

        // use the document part of the payload for embedding vector
        let embeddings = payloads.calc_embedding(&self.embedding_model)?;

        //dbg!(&embeddings);

        if embeddings.is_empty() || embeddings.len() != payloads.len() {
            return Err(Error::msg("Embedding Failed, empty embedding"));
        }

        self.select_or_create_collection(embeddings[0].len() as u64).await?;
        
        let points: Vec<PointStruct> = embeddings
            .into_iter()
            .zip(payloads.into_iter())
            .filter_map(|(vector,note)| {
                if let Ok(point_payload) = note.as_payload() {
                    Some(PointStruct::new(note.id(), vector, point_payload))
                }else {
                    None
                }
            })
            .collect();

        //dbg!(&points);

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.collection_name,points)
            ).await?;

        Ok(())
    }
    pub async fn delete_points(&self,ids: &[impl AsRef<str>]) -> Result<(), Error> {
        let point_ids = ids.iter()
            .map(|id| {
                PointId::from(id.as_ref())
            })
            .collect::<Vec<_>>();
        self.client.delete_points(
            DeletePointsBuilder::new(&self.collection_name).points(
                PointsIdsList {
                    ids: point_ids,
                }
            ).wait(true)
        ).await?;
        Ok(())
    }
    
    pub async fn query(&self, queries: &[impl AsRef<str>], k: u64, filter: Option<impl Into<Filter> + Clone>) -> Result<(f64, QueryBatchResponse),Error> {
        let embeddings = queries
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .calc_embedding(&self.embedding_model)?;



        let query_batch_info: Vec<QueryPoints> = embeddings.into_iter()
            .map(|embedding| {
                QueryPointsBuilder::new(&self.collection_name)
                    .query(embedding)
                    .limit(k)
                    .with_payload(true)
            })
            .map(|builder| {
                if let Some(filter) = filter.clone() {
                    builder.filter(filter.clone())
                }else{
                    builder
                }
            })
            .map(|builder| builder.build())
            .collect();


        let response = self.client.query_batch(
            QueryBatchPointsBuilder::new(
                &self.collection_name,
                query_batch_info,
            )
        ).await?;

        //dbg!(&response.result);

        let time = response.time;

        Ok((time,response))
    }

    //TODO: tests it
    pub async fn update_points_payload(&self, payloads: impl Into<Vec<(String,Payload)>>) -> Result<(), Error> {
        self.client.update_points_batch(
            UpdateBatchPointsBuilder::new(
                &self.collection_name,
                payloads
                    .into()
                    .into_iter()
                    .map(|(id,payload)| {
                        PointsUpdateOperation {
                            operation: Some(Operation::OverwritePayload( OverwritePayload {
                                points_selector: Some(PointsSelector {
                                    points_selector_one_of: Some(PointsSelectorOneOf::Points(
                                        PointsIdsList {
                                            ids: vec![PointId::from(id)]
                                        }
                                    ))
                                }),
                                payload: HashMap::from(payload),
                                ..Default::default()
                            }))
                        }
                    })
                    .collect::<Vec<PointsUpdateOperation>>()
            ).wait(true)
        ).await?;
        Ok(())
    }
    pub async fn query_by_ids(&self, ids: &[impl AsRef<str>]) -> Result<QueryBatchResponse, Error> {

        let query_builds: Vec<QueryPoints>= ids
            .iter()
            .map(|id| {
                QueryPointsBuilder::new(&self.collection_name)
                    .query(Query::new_nearest(
                        PointId::from(id.as_ref())
                    ))
                    .build()
            })
            .collect();

       Ok(self.client
            .query_batch(
                QueryBatchPointsBuilder::new(
                    &self.collection_name,
                    query_builds,
                )
            ).await?)
    }
    pub fn parse_response_to_notes(response: QueryBatchResponse) -> Vec<Vec<MemoryNote>> {
        response.result
            .into_iter()
            .map(|points| {
                points.result
                    .into_iter()
                    .filter_map(|score_point| {
                        if score_point.id.is_some() {
                            match MemoryNote::try_from(score_point.payload) {
                                Ok(note) => Some(note),
                                Err(e) => {
                                    eprintln!("Error parsing note: {e:?}");
                                    None
                                }
                            }
                        }else{
                            None
                        }

                    }).collect()
            }).collect()
    }
}

mod test {
    #[allow(unused_imports)]
    use super::*;
    
    #[allow(dead_code)]
    fn is_valid_test_content(s: &str) -> bool {
        if let Some((prefix, suffix)) = s.split_once('_') {
            if !prefix.starts_with("test") || suffix != "content" {
                return false;
            }

            let number_part = &prefix[4..];
            !number_part.is_empty() && number_part.chars().all(|c| c.is_ascii_digit())
        } else {
            false
        }
    }
    #[tokio::test]
    async fn test_new_retriever() {
        QdrantRetriever::new(
            "test_qdrant",
            None,
            false,
            EmbeddingModel::AllMiniLML6V2
        ).unwrap();
    }
    #[tokio::test]
    async fn test_add_points() {
        let retriever = QdrantRetriever::new(
            "test_qdrant",
            None,
            false,
            EmbeddingModel::AllMiniLML6V2
        ).unwrap();
        retriever.add_points(
            vec![
                MemoryNote::new("test1_content"),
                MemoryNote::new("test2_content"),
                MemoryNote::new("test3_content"),
                MemoryNote::new("test4_content"),
                MemoryNote::new("test5_content"),
            ]
        ).await.unwrap();
    }
    #[tokio::test]
    async fn test_query_and_parse() {
        let retriever = QdrantRetriever::new(
            "test_qdrant",
            None,
            false,
            EmbeddingModel::AllMiniLML6V2
        ).unwrap();
        retriever.add_points(
            vec![
                MemoryNote::new("test1_content"),
                MemoryNote::new("test2_content"),
                MemoryNote::new("test3_content"),
                MemoryNote::new("test4_content"),
                MemoryNote::new("test5_content"),
            ]
        ).await.unwrap();
        let (_,res) = retriever.query(&vec!["test"], 5, None::<Filter>).await.unwrap();
        let res = QdrantRetriever::parse_response_to_notes(res);
        res.iter().for_each(|notes|{
            notes.iter().for_each(|note|{
                if !is_valid_test_content(note.content.as_str()) {
                    panic!("Invalid test content:{}", note.content)
                }
            })
        })
    }
}

