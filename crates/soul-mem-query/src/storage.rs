pub mod error;
pub mod model;
pub mod repository;
pub mod surql;
pub mod surreal;

pub use error::{StorageError, StorageResult};
pub use model::{
    EventStats, EventWindow, FeedbackEventRecord, FeedbackValue, MemoryLinkKind, MemoryLinkRecord,
    MemoryNoteKind, MemoryNoteRecord, RetrievalEventRecord, SimilarityHit, SimilarityQuery,
};
pub use repository::MemoryRepository;
pub use surreal::{SurrealConnectionConfig, SurrealMemoryRepository};
