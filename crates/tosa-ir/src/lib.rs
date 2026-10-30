//! TOSA graph IR: owned graph types parsed from and serialized to the TOSA
//! flatbuffer format, a builder for constructing graphs, and the tensor and
//! element types shared by the interpreter and the converters.

pub mod builder;
pub mod error;
pub mod float8;
pub(crate) mod generated;
pub mod ir;
pub mod npy;
pub mod serialize;
pub mod tensor;
pub mod text;

pub use builder::{BlockBuilder, GraphBuilder, TensorRef};
pub use error::{Error, Result};
pub use ir::{
    BasicBlock, DType, Graph, NanMode, Op, OpAttribute, Operator, Region, ResizeMode, RoundingMode, TensorDef, Version,
};
pub use tensor::{Tensor, TensorData};
