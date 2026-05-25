use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("invalid identifier `{value}`: {reason}")]
    InvalidIdentifier { value: String, reason: &'static str },

    #[error("data contract validation failed: {0}")]
    Validation(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DataError>;
