use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConvertError {
    /// Operators of the source model that no lowering is registered for,
    /// with how many times each occurs.
    #[error("unsupported operator(s): {}", describe_unsupported(.0))]
    Unsupported(Vec<(String, usize)>),

    /// A lowering rejected a node (unsupported attribute combination,
    /// unexpected operand types, ...).
    #[error("cannot lower {op} node '{node}': {reason}")]
    Lowering { op: String, node: String, reason: String },

    /// The source model could not be parsed.
    #[error("invalid {format} model: {reason}")]
    Frontend { format: String, reason: String },

    #[error("no frontend for {0}")]
    NoFrontend(String),

    #[error(transparent)]
    Ir(#[from] tosa_ir::Error),
}

fn describe_unsupported(ops: &[(String, usize)]) -> String {
    ops.iter().map(|(op, n)| format!("{op} (x{n})")).collect::<Vec<_>>().join(", ")
}

pub type Result<T> = std::result::Result<T, ConvertError>;
