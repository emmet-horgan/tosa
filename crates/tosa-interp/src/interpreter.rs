//! The interpreter: validates a graph when it is created, then executes its
//! main block, and the sub-graphs of control-flow operators, following the
//! [`ExecutionPlan`] of each block.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::ir::{BasicBlock, DType, Graph, TensorDef};
use crate::ops;
use crate::plan::ExecutionPlan;
use crate::stats::{InterpreterHook, NoopHook, OpInfo};
use crate::tensor::{Tensor, checked_element_count};

/// Highest tensor rank supported, as in the reference model (TOSA level
/// 8K's MAX_RANK).
pub const MAX_RANK: usize = 6;

/// Deepest nesting of COND_IF / WHILE_LOOP bodies before a run is aborted,
/// protecting against runaway recursion on malformed graphs.
pub const MAX_NESTING: usize = 64;

/// Default limit on the size of any one tensor (16 GiB); see
/// [`Interpreter::set_max_tensor_bytes`].
pub const DEFAULT_MAX_TENSOR_BYTES: usize = 1 << 34;

/// Index of a block in its graph: (region index, block index).
pub type BlockIndex = (usize, usize);

/// Execution context of one block: its tensors, plus shared access to the
/// graph, the persistent variables and the hook.
pub struct InterpreterState<'g, 'r> {
    graph: &'g Graph,
    plans: &'g HashMap<BlockIndex, ExecutionPlan>,
    block: &'g BasicBlock,
    plan: &'g ExecutionPlan,
    tensors: HashMap<String, Tensor>,
    tensor_defs: HashMap<&'g str, &'g TensorDef>,
    variables: &'r mut HashMap<String, Tensor>,
    hook: &'r mut dyn InterpreterHook,
    depth: usize,
    max_tensor_bytes: usize,
}

impl<'g> InterpreterState<'g, '_> {
    pub fn get_tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| Error::TensorNotFound(name.to_string()))
    }

    pub fn set_tensor(&mut self, tensor: Tensor) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    pub fn remove_tensor(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    pub fn rename_tensor(&mut self, from: &str, to: &str) {
        if let Some(mut t) = self.tensors.remove(from) {
            t.name = to.to_string();
            self.tensors.insert(to.to_string(), t);
        }
    }

    pub fn find_tensor_def(&self, name: &str) -> Option<&'g TensorDef> {
        self.tensor_defs.get(name).copied()
    }

    /// A block (sub-graph) of the graph being executed, by block or region name.
    pub fn find_block(&self, name: &str) -> Option<&'g BasicBlock> {
        self.graph.find_block(name)
    }

    /// The block being executed.
    pub fn block(&self) -> &'g BasicBlock {
        self.block
    }

    /// The persistent variable `name`, if it has been initialized.
    pub fn variable(&self, name: &str) -> Option<&Tensor> {
        self.variables.get(name)
    }

    /// Set the persistent variable `name` to `value`.
    pub fn set_variable(&mut self, name: &str, mut value: Tensor) {
        value.name = name.to_string();
        self.variables.insert(name.to_string(), value);
    }

    /// Element count of an output of `shape` and `dtype`, or an error if it
    /// overflows or exceeds the tensor size limit. Operators call this before
    /// allocating an output whose size does not follow from their inputs.
    pub fn checked_count(&self, op: &str, shape: &[usize], dtype: Option<DType>) -> Result<usize> {
        let count = checked_element_count(shape).map_err(|e| Error::TensorTooLarge(format!("{op}: {e}")))?;
        let bytes = count.checked_mul(dtype.map_or(1, DType::element_size_bytes));
        match bytes {
            Some(b) if b <= self.max_tensor_bytes => Ok(count),
            _ => Err(Error::TensorTooLarge(format!(
                "{op}: output of shape {shape:?} exceeds the limit of {} bytes",
                self.max_tensor_bytes
            ))),
        }
    }

    /// Run the block `block_name` with `input_tensor_names` bound to its
    /// inputs, then store its outputs under `output_tensor_names`.
    pub fn eval_sub_block(
        &mut self,
        block_name: &str,
        input_tensor_names: &[String],
        output_tensor_names: &[String],
    ) -> Result<()> {
        let index = self
            .graph
            .find_block_index(block_name)
            .ok_or_else(|| Error::GraphError(format!("block '{}' not found", block_name)))?;
        let block = self.graph.block_at(index).expect("index from find_block_index");
        let plan = &self.plans[&index];
        if self.depth >= MAX_NESTING {
            return Err(Error::ControlFlow(format!("sub-graph nesting deeper than {MAX_NESTING}")));
        }

        let mut child_tensors: HashMap<String, Tensor> = HashMap::new();
        for (block_input_name, parent_name) in block.inputs.iter().zip(input_tensor_names) {
            let tensor = self.get_tensor(parent_name)?;
            let mut t = tensor.clone();
            t.name = block_input_name.clone();
            self.hook.on_tensor_alloc(&t.name, &t.shape, t.dtype(), t.size_bytes());
            child_tensors.insert(block_input_name.clone(), t);
        }

        let mut child = InterpreterState {
            graph: self.graph,
            plans: self.plans,
            block,
            plan,
            tensors: child_tensors,
            tensor_defs: tensor_def_map(block),
            variables: &mut *self.variables,
            hook: &mut *self.hook,
            depth: self.depth + 1,
            max_tensor_bytes: self.max_tensor_bytes,
        };
        execute_block(&mut child)?;

        for (block_output_name, parent_output_name) in block.outputs.iter().zip(output_tensor_names) {
            let mut t = child
                .tensors
                .remove(block_output_name)
                .ok_or_else(|| Error::MissingOutput(format!("{block_output_name} of block '{}'", block.name)))?;
            t.name = parent_output_name.clone();
            self.tensors.insert(parent_output_name.clone(), t);
        }
        Ok(())
    }
}

fn tensor_def_map(block: &BasicBlock) -> HashMap<&str, &TensorDef> {
    block.tensors.iter().map(|td| (td.name.as_str(), td)).collect()
}

/// A validated graph ready to run.
///
/// Creating an interpreter checks every block (tensor ranks and sizes,
/// undefined tensors, duplicate producers, dependency cycles) and computes
/// its execution plan. Persistent variables (VARIABLE / VARIABLE_READ /
/// VARIABLE_WRITE) keep their values across calls to [`Interpreter::run`].
#[derive(Debug)]
pub struct Interpreter {
    graph: Graph,
    plans: HashMap<BlockIndex, ExecutionPlan>,
    variables: HashMap<String, Tensor>,
    max_tensor_bytes: usize,
}

impl Interpreter {
    pub fn new(graph: Graph) -> Result<Self> {
        graph.main_block().ok_or_else(|| Error::GraphError("no main block found".to_string()))?;
        let mut plans = HashMap::new();
        for (index, block) in graph.blocks() {
            validate_block(block)?;
            plans.insert(index, ExecutionPlan::new(block)?);
        }
        Ok(Interpreter { graph, plans, variables: HashMap::new(), max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES })
    }

    pub fn from_flatbuffer(data: &[u8]) -> Result<Self> {
        Self::new(Graph::from_flatbuffer(data)?)
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The execution plan of the block named `block` (block or region name).
    pub fn plan(&self, block: &str) -> Option<&ExecutionPlan> {
        self.plans.get(&self.graph.find_block_index(block)?)
    }

    /// Limit the size of any single tensor; larger outputs are rejected with
    /// [`Error::TensorTooLarge`] instead of being allocated.
    pub fn set_max_tensor_bytes(&mut self, bytes: usize) {
        self.max_tensor_bytes = bytes;
    }

    /// Current values of the graph's persistent variables.
    pub fn variables(&self) -> &HashMap<String, Tensor> {
        &self.variables
    }

    /// Set a persistent variable before a run (for example to restore state).
    pub fn set_variable(&mut self, name: &str, mut value: Tensor) {
        value.name = name.to_string();
        self.variables.insert(name.to_string(), value);
    }

    /// Forget every persistent variable; the next run re-initializes them.
    pub fn reset_variables(&mut self) {
        self.variables.clear();
    }

    /// Execute the main block on `inputs`, which must hold exactly the
    /// block's declared inputs with their declared types and shapes.
    pub fn run(
        &mut self,
        inputs: HashMap<String, Tensor>,
        hook: &mut dyn InterpreterHook,
    ) -> Result<HashMap<String, Tensor>> {
        let Interpreter { graph, plans, variables, max_tensor_bytes } = self;
        let block = graph.main_block().expect("checked in Interpreter::new");
        let tensor_defs = tensor_def_map(block);
        let inputs = check_inputs(block, &tensor_defs, inputs)?;
        for t in inputs.values() {
            hook.on_tensor_alloc(&t.name, &t.shape, t.dtype(), t.size_bytes());
        }

        let mut state = InterpreterState {
            graph,
            plans,
            block,
            plan: &plans[&(0, 0)],
            tensors: inputs,
            tensor_defs,
            variables,
            hook,
            depth: 0,
            max_tensor_bytes: *max_tensor_bytes,
        };
        execute_block(&mut state)?;

        let mut outputs = HashMap::new();
        for output_name in &block.outputs {
            let tensor = state.tensors.remove(output_name).ok_or_else(|| Error::MissingOutput(output_name.clone()))?;
            outputs.insert(output_name.clone(), tensor);
        }
        Ok(outputs)
    }
}

/// Static checks on a block's tensor declarations.
fn validate_block(block: &BasicBlock) -> Result<()> {
    for t in &block.tensors {
        if t.dtype != DType::Shape && t.shape.len() > MAX_RANK {
            return Err(Error::ShapeMismatch(format!(
                "tensor '{}' has rank {}, above the maximum of {MAX_RANK}",
                t.name,
                t.shape.len()
            )));
        }
        if t.shape.iter().all(|&d| d >= 0) {
            let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
            checked_element_count(&shape).map_err(|e| Error::TensorTooLarge(format!("tensor '{}': {e}", t.name)))?;
        }
    }
    Ok(())
}

/// Check the caller's inputs against the block's declared inputs.
fn check_inputs(
    block: &BasicBlock,
    defs: &HashMap<&str, &TensorDef>,
    mut inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    if let Some(extra) = inputs.keys().find(|k| !block.inputs.contains(k)) {
        return Err(Error::UnknownInput(extra.clone()));
    }
    for name in &block.inputs {
        let tensor = inputs.get_mut(name).ok_or_else(|| Error::MissingInput(name.clone()))?;
        tensor.name = name.clone();
        let Some(def) = defs.get(name.as_str()) else { continue };
        let dtype_ok = tensor.dtype() == def.dtype || (def.dtype == DType::Shape && tensor.dtype() == DType::I64);
        if !dtype_ok {
            return Err(Error::InputMismatch {
                name: name.clone(),
                problem: format!("has dtype {:?} but the graph declares {:?}", tensor.dtype(), def.dtype),
            });
        }
        let declared_static = def.shape.iter().all(|&d| d >= 0);
        let declared: Vec<usize> = def.shape.iter().map(|&d| d.max(0) as usize).collect();
        if declared_static && tensor.shape != declared {
            return Err(Error::InputMismatch {
                name: name.clone(),
                problem: format!("has shape {:?} but the graph declares {declared:?}", tensor.shape),
            });
        }
        if tensor.data.len() != tensor.element_count() {
            return Err(Error::InputMismatch {
                name: name.clone(),
                problem: format!("has {} elements but shape {:?}", tensor.data.len(), tensor.shape),
            });
        }
    }
    Ok(inputs)
}

/// Run every operator of `state`'s block in plan order, dropping tensors
/// after their last use.
fn execute_block(state: &mut InterpreterState) -> Result<()> {
    let block = state.block;
    let plan = state.plan;

    for name in &plan.unused_inputs {
        if let Some(t) = state.tensors.remove(name) {
            state.hook.on_tensor_free(name, t.size_bytes());
        }
    }

    for (pos, &op_idx) in plan.order.iter().enumerate() {
        let op = &block.operators[op_idx];
        let op_info = OpInfo { op: op.op, op_index: op_idx, block: &block.name, depth: state.depth };

        let input_refs: Vec<&Tensor> = op.inputs.iter().filter_map(|name| state.tensors.get(name.as_str())).collect();
        state.hook.pre_op(&op_info, &input_refs);

        ops::dispatch(state, op).map_err(|e| e.in_operator(&block.name, op_idx, op.op.name()))?;

        let output_refs: Vec<&Tensor> = op.outputs.iter().filter_map(|name| state.tensors.get(name.as_str())).collect();
        for t in &output_refs {
            state.hook.on_tensor_alloc(&t.name, &t.shape, t.dtype(), t.size_bytes());
        }
        let input_refs: Vec<&Tensor> = op.inputs.iter().filter_map(|name| state.tensors.get(name.as_str())).collect();
        state.hook.post_op(&op_info, &input_refs, &output_refs);

        for name in &plan.free_after[pos] {
            if let Some(t) = state.tensors.remove(name) {
                state.hook.on_tensor_free(name, t.size_bytes());
            }
        }
    }
    Ok(())
}

/// Run `graph`'s main block once without observing it.
pub fn run_once(graph: Graph, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    Interpreter::new(graph)?.run(inputs, &mut NoopHook)
}
