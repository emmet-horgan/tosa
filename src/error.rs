use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to deserialize flatbuffer: {0}")]
    Deserialize(String),

    #[error("unsupported op: {0}")]
    UnsupportedOp(String),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("dtype mismatch: {0}")]
    DTypeMismatch(String),

    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("broadcast error: {0}")]
    Broadcast(String),

    #[error("graph error: {0}")]
    GraphError(String),

    #[error("control flow error: {0}")]
    ControlFlow(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
