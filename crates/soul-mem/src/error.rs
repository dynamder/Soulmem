//! 统一错误类型与对外错误码。
//!
//! 下层 crate 以 `anyhow`/`thiserror` 报错，网络层（zenoh）与存储也各有错误，
//! 此处收敛为一种 `Error`，并给出稳定的对外错误码，便于各通道序列化回调用方。

use std::fmt;

/// 对外可见的稳定错误码（字符串，便于跨通道/跨语言传递）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    InvalidArgument,
    NotFound,
    AlreadyExists,
    Unavailable,
    FailedPrecondition,
    Internal,
    Unimplemented,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidArgument => "invalid_argument",
            ErrorCode::NotFound => "not_found",
            ErrorCode::AlreadyExists => "already_exists",
            ErrorCode::Unavailable => "unavailable",
            ErrorCode::FailedPrecondition => "failed_precondition",
            ErrorCode::Internal => "internal",
            ErrorCode::Unimplemented => "unimplemented",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// soul-mem 服务统一错误。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("already exists: {0}")]
    AlreadyExists(String),

    #[error("unavailable: {0}")]
    Unavailable(String),

    #[error("failed precondition: {0}")]
    FailedPrecondition(String),

    #[error("unimplemented: {0}")]
    Unimplemented(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    /// 返回稳定的对外错误码。
    pub fn code(&self) -> ErrorCode {
        match self {
            Error::InvalidArgument(_) => ErrorCode::InvalidArgument,
            Error::NotFound(_) => ErrorCode::NotFound,
            Error::AlreadyExists(_) => ErrorCode::AlreadyExists,
            Error::Unavailable(_) => ErrorCode::Unavailable,
            Error::FailedPrecondition(_) => ErrorCode::FailedPrecondition,
            Error::Unimplemented(_) => ErrorCode::Unimplemented,
            Error::Internal(_) => ErrorCode::Internal,
        }
    }

    /// 包装任意错误为内部错误，避免底层错误细节直接外泄。
    pub fn internal<E: fmt::Display>(err: E) -> Self {
        Error::Internal(err.to_string())
    }
}

/// 便捷：从下层 crate 常见的 `anyhow` 错误构造内部错误。
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Internal(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
