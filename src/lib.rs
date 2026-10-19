pub mod broadcast;
pub mod error;
pub mod float8;
pub mod generated;
pub mod interpreter;
pub mod ir;
pub mod npy;
pub mod ops;
pub mod stats;
pub mod tensor;

pub use error::{Error, Result};
pub use interpreter::Interpreter;
pub use ir::{DType, Graph, Op};
pub use stats::{FlopCounter, HookSet, InterpreterHook, NoopHook, OpTrace, StatsCollector};
pub use tensor::{Tensor, TensorData};

use std::collections::HashMap;

pub fn run_graph(data: &[u8], inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let interp = Interpreter::from_flatbuffer(data)?;
    let mut hook = NoopHook;
    interp.run(inputs, &mut hook)
}

pub fn run_graph_with_stats(
    data: &[u8],
    inputs: HashMap<String, Tensor>,
) -> Result<(HashMap<String, Tensor>, StatsCollector)> {
    let interp = Interpreter::from_flatbuffer(data)?;
    let mut hook = StatsCollector::new();
    let outputs = interp.run(inputs, &mut hook)?;
    Ok((outputs, hook))
}

pub fn run_graph_with_hooks(
    data: &[u8],
    inputs: HashMap<String, Tensor>,
    hooks: &mut dyn InterpreterHook,
) -> Result<HashMap<String, Tensor>> {
    let interp = Interpreter::from_flatbuffer(data)?;
    interp.run(inputs, hooks)
}
