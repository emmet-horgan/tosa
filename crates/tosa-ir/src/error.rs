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

    /// The caller did not supply a tensor the graph's main block declares as an input.
    #[error("missing input tensor '{0}'")]
    MissingInput(String),

    /// The caller supplied a tensor the graph's main block does not declare as an input.
    #[error("unknown input tensor '{0}'")]
    UnknownInput(String),

    /// An input tensor does not have the dtype or shape the graph declares for it.
    #[error("input '{name}' {problem}")]
    InputMismatch { name: String, problem: String },

    /// The graph finished without producing one of its declared outputs.
    #[error("output tensor '{0}' was not produced")]
    MissingOutput(String),

    /// A tensor's element count overflows or exceeds the interpreter's size limit.
    #[error("tensor too large: {0}")]
    TensorTooLarge(String),

    /// An error raised while executing one operator, with its position in the graph.
    ///
    /// Operators inside COND_IF and WHILE_LOOP bodies nest: the outer error
    /// names the control-flow operator and `source` the operator inside it.
    #[error("{op} (operator {index} in block '{block}'): {source}")]
    Operator {
        block: String,
        index: usize,
        op: &'static str,
        #[source]
        source: Box<Error>,
    },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    /// The error with every [`Error::Operator`] context layer removed.
    pub fn root_cause(&self) -> &Error {
        let mut e = self;
        while let Error::Operator { source, .. } = e {
            e = source;
        }
        e
    }

    /// The `(block, index, op)` of the innermost operator that failed, if any.
    pub fn failing_operator(&self) -> Option<(&str, usize, &'static str)> {
        let mut found = None;
        let mut e = self;
        while let Error::Operator { block, index, op, source } = e {
            found = Some((block.as_str(), *index, *op));
            e = source;
        }
        found
    }

    /// Wrap this error in an [`Error::Operator`] context layer.
    pub fn in_operator(self, block: &str, index: usize, op: &'static str) -> Error {
        Error::Operator { block: block.to_string(), index, op, source: Box::new(self) }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
