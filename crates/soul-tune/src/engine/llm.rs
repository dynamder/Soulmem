pub mod backend;

#[cfg(feature = "candle")]
pub mod qwen35;

#[cfg(feature = "llamacpp")]
pub mod llama_server;

#[cfg(feature = "candle")]
pub mod candle_llm;

pub use backend::LlmBackend;

#[cfg(feature = "llamacpp")]
pub use llama_server::LlamaServer;

#[cfg(feature = "candle")]
pub use candle_llm::{CandleLlm, CandleLlmConfig};
