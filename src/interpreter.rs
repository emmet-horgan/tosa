use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{Error, Result};
use crate::ir::{BasicBlock, Graph, Op, Operator, TensorDef};
use crate::ops;
use crate::stats::{InterpreterHook, NoopHook, OpInfo};
use crate::tensor::Tensor;

pub struct InterpreterState<'g> {
    graph: &'g Graph,
    block: &'g BasicBlock,
    tensors: HashMap<String, Tensor>,
    tensor_defs: HashMap<String, &'g TensorDef>,
}

impl<'g> InterpreterState<'g> {
    pub fn get_tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| Error::TensorNotFound(name.to_string()))
    }

    pub fn set_tensor(&mut self, tensor: Tensor) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    pub fn remove_tensor(&mut self, name: &str) {
        self.tensors.remove(name);
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

    pub fn eval_sub_block(
        &mut self,
        block_name: &str,
        input_tensor_names: &[String],
        output_tensor_names: &[String],
    ) -> Result<()> {
        self.eval_sub_block_with_rename(block_name, input_tensor_names, output_tensor_names)
    }

    pub fn eval_sub_block_with_rename(
        &mut self,
        block_name: &str,
        input_tensor_names: &[String],
        output_tensor_names: &[String],
    ) -> Result<()> {
        let block = self
            .graph
            .find_block(block_name)
            .ok_or_else(|| Error::GraphError(format!("block '{}' not found", block_name)))?;

        // Build child state with input tensors copied in
        let mut child_tensors: HashMap<String, Tensor> = HashMap::new();

        for (i, block_input_name) in block.inputs.iter().enumerate() {
            if let Some(parent_name) = input_tensor_names.get(i)
                && let Some(tensor) = self.tensors.get(parent_name)
            {
                let mut t = tensor.clone();
                t.name = block_input_name.clone();
                child_tensors.insert(block_input_name.clone(), t);
            }
        }

        let mut child_tensor_defs: HashMap<String, &'g TensorDef> = HashMap::new();
        for td in &block.tensors {
            child_tensor_defs.insert(td.name.clone(), td);
        }

        let mut child_state =
            InterpreterState { graph: self.graph, block, tensors: child_tensors, tensor_defs: child_tensor_defs };

        // Execute the child block
        execute_block_inner(&mut child_state, &mut NoopHook)?;

        // Copy outputs back
        for (i, block_output_name) in block.outputs.iter().enumerate() {
            if let Some(parent_output_name) = output_tensor_names.get(i)
                && let Some(tensor) = child_state.tensors.remove(block_output_name)
            {
                let mut t = tensor;
                t.name = parent_output_name.clone();
                self.tensors.insert(parent_output_name.clone(), t);
            }
        }

        Ok(())
    }
}

pub struct Interpreter {
    graph: Graph,
}

impl Interpreter {
    pub fn new(graph: Graph) -> Self {
        Interpreter { graph }
    }

    pub fn from_flatbuffer(data: &[u8]) -> Result<Self> {
        let graph = Graph::from_flatbuffer(data)?;
        Ok(Self::new(graph))
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
        hook: &mut dyn InterpreterHook,
    ) -> Result<HashMap<String, Tensor>> {
        let block = self.graph.main_block().ok_or_else(|| Error::GraphError("no main block found".to_string()))?;

        let mut tensor_defs: HashMap<String, &TensorDef> = HashMap::new();
        for td in &block.tensors {
            tensor_defs.insert(td.name.clone(), td);
        }

        let mut state = InterpreterState { graph: &self.graph, block, tensors: inputs, tensor_defs };

        execute_block_inner(&mut state, hook)?;

        // Collect outputs
        let mut outputs = HashMap::new();
        for output_name in &block.outputs {
            if let Some(tensor) = state.tensors.remove(output_name) {
                outputs.insert(output_name.clone(), tensor);
            }
        }

        Ok(outputs)
    }
}

/// Highest tensor rank supported, as in the reference model (TOSA level
/// 8K's MAX_RANK).
pub const MAX_RANK: usize = 6;

fn execute_block_inner(state: &mut InterpreterState, hook: &mut dyn InterpreterHook) -> Result<()> {
    let block = state.block;

    if let Some(t) = block.tensors.iter().find(|t| t.dtype != crate::ir::DType::Shape && t.shape.len() > MAX_RANK) {
        return Err(Error::ShapeMismatch(format!(
            "tensor '{}' has rank {}, above the maximum of {MAX_RANK}",
            t.name,
            t.shape.len()
        )));
    }

    // Build producer map: output_tensor_name → op_index
    let mut producer_map: HashMap<&str, usize> = HashMap::new();
    for (i, op) in block.operators.iter().enumerate() {
        for out_name in &op.outputs {
            producer_map.insert(out_name.as_str(), i);
        }
    }

    // Track which tensors are available
    let mut available: HashSet<String> = state.tensors.keys().cloned().collect();

    // Track which ops have been executed
    let num_ops = block.operators.len();
    let mut executed = vec![false; num_ops];

    // Build the ready queue: ops whose inputs are all available
    let mut ready_queue: VecDeque<usize> = VecDeque::new();
    for (i, op) in block.operators.iter().enumerate() {
        if op_is_ready(op, &available) {
            ready_queue.push_back(i);
        }
    }

    let mut ops_executed = 0;
    while let Some(op_idx) = ready_queue.pop_front() {
        if executed[op_idx] {
            continue;
        }

        let op = &block.operators[op_idx];

        // Gather input tensor refs for the hook
        let input_refs: Vec<&Tensor> = op.inputs.iter().filter_map(|name| state.tensors.get(name.as_str())).collect();

        let op_info = OpInfo { op: op.op, op_index: op_idx };

        hook.pre_op(&op_info, &input_refs);

        // Execute the operator
        ops::dispatch(state, op)?;

        // Gather output tensor refs for the hook
        let output_refs: Vec<&Tensor> = op.outputs.iter().filter_map(|name| state.tensors.get(name.as_str())).collect();

        // Notify hook about allocations
        for t in &output_refs {
            hook.on_tensor_alloc(&t.name, &t.shape, t.dtype(), t.size_bytes());
        }

        let input_refs: Vec<&Tensor> = op.inputs.iter().filter_map(|name| state.tensors.get(name.as_str())).collect();
        hook.post_op(&op_info, &input_refs, &output_refs);

        executed[op_idx] = true;
        ops_executed += 1;

        // Mark outputs as available and check downstream ops
        for out_name in &op.outputs {
            available.insert(out_name.clone());
        }

        // Check if any new ops are ready
        for (i, next_op) in block.operators.iter().enumerate() {
            if !executed[i] && op_is_ready(next_op, &available) && !ready_queue.contains(&i) {
                ready_queue.push_back(i);
            }
        }
    }

    if ops_executed < num_ops {
        // Check if unexecuted ops are all ones we don't support yet
        let unexecuted: Vec<&Operator> =
            block.operators.iter().enumerate().filter(|(i, _)| !executed[*i]).map(|(_, op)| op).collect();

        if !unexecuted.is_empty() {
            let names: Vec<&str> = unexecuted.iter().map(|o| o.op.name()).collect();
            return Err(Error::GraphError(format!("could not execute all ops, stuck on: {:?}", names)));
        }
    }

    Ok(())
}

fn op_is_ready(op: &Operator, available: &HashSet<String>) -> bool {
    // CONST ops are always ready (they produce their own data)
    if op.op == Op::Const {
        return true;
    }
    op.inputs.iter().all(|name| available.contains(name))
}
