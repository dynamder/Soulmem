//! 服务错误类型。
//!
//! 仅描述服务编排层自身的失败；底层 crate 的错误通过 `#[from]` 归类进来。
//! 错误信息**不得包含提示词正文或密钥**（见 `AGENTS.md` §4）。

/// 服务错误。
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// 配置缺失或非法。
    #[error("配置错误: {0}")]
    Config(String),

    /// 请求内容非法（如未知的控制信号）。会以 `Reply { ok: false }` 返回给调用方。
    #[error("请求非法: {0}")]
    BadRequest(String),

    /// 存储层错误。
    #[error("存储错误: {0}")]
    Storage(#[from] soul_mem_runtime::storage::StorageError),

    /// LLM 调用错误。
    #[error("LLM 错误: {0}")]
    Llm(#[from] soul_mem_llm::LlmError),

    /// 嵌入生成错误（构造查询或记忆向量时）。
    #[error("嵌入错误: {0}")]
    Embedding(String),

    /// 其他内部错误。
    #[error("内部错误: {0}")]
    Internal(String),
}

/// 服务结果别名。
pub type ServiceResult<T> = Result<T, ServiceError>;
