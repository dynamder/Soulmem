use fastembed::EmbeddingModel;

pub struct QdrantConfig {
    collection_name: String, 
    port: u32, 
    keep_alive: bool, 
    embedding_model: EmbeddingModel
}
impl QdrantConfig {
    pub fn new(collection_name: impl Into<String>, embedding_model: EmbeddingModel) -> Self {
        QdrantConfig {
            collection_name: collection_name.into(),
            port: 6334,
            keep_alive: true,
            embedding_model
        }
    }
    pub fn collection_name(&self) -> &str {
        self.collection_name.as_str()
    }
    pub fn port(&self) -> u32 {
        self.port
    }
    pub fn keep_alive(&self) -> bool {
        self.keep_alive
    }
    pub fn embedding_model(&self) -> &EmbeddingModel {
        &self.embedding_model
    }
}
pub struct QdrantConfigBuilder {
    collection_name: String,
    port: Option<u32>,
    keep_alive: Option<bool>,
    embedding_model: EmbeddingModel
}
impl QdrantConfigBuilder { 
    pub fn new(collection_name: impl Into<String>, embedding_model: EmbeddingModel) -> Self {
        QdrantConfigBuilder {
            collection_name: collection_name.into(),
            port: None,
            keep_alive: None,
            embedding_model
        }
    }
    pub fn port(mut self, port: u32) -> Self {
        self.port = Some(port);
        self
    }
    pub fn keep_alive(mut self, keep_alive: bool) -> Self {
        self.keep_alive = Some(keep_alive);
        self
    }
    pub fn build(self) -> QdrantConfig {
        QdrantConfig {
            collection_name: self.collection_name,
            port: self.port.unwrap_or(6334),
            keep_alive: self.keep_alive.unwrap_or(true),
            embedding_model: self.embedding_model
        }
    }
}
pub struct SurrealConfig {
    local_address: String, 
    capacity: usize
}
impl SurrealConfig {
    pub fn new(local_address: impl Into<String>, capacity: Option<usize>) -> Self {
        SurrealConfig {
            local_address: local_address.into(),
            capacity: capacity.unwrap_or(100),
        }
    }
    pub fn local_address(&self) -> &str {
        self.local_address.as_str()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
