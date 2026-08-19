pub mod backend;

#[cfg(feature = "llamacpp")]
pub mod llama_server;

#[cfg(feature = "llamacpp")]
pub mod resolver;

#[cfg(feature = "candle")]
pub mod qwen35;

#[cfg(feature = "candle")]
pub mod candle_llm;

pub use backend::LlmBackend;

#[cfg(feature = "llamacpp")]
pub use llama_server::LlamaServer;

#[cfg(feature = "llamacpp")]
// 对外（soul-tune-api）经全路径使用这些解析器函数；本 crate 内部未必直接引用。
#[allow(unused_imports)]
pub use resolver::{find_cached_model, probe_health, probe_status, resolve_llm};

#[cfg(feature = "candle")]
pub use candle_llm::{CandleLlm, CandleLlmConfig};
