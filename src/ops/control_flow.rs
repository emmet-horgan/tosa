//! COND_IF and WHILE_LOOP.
//!
//! Sub-graphs are named blocks (or regions) of the graph. Before running one,
//! its signature is checked against the operator as the reference model
//! does: the numbers of inputs and outputs must match and every tensor must
//! agree in element type and shape.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{BasicBlock, DType, OpAttribute, Operator, TensorDef};
use crate::ops::common::invalid;
use crate::tensor::TensorData;

/// Upper bound on WHILE_LOOP iterations, to fail rather than hang on a graph
/// that never terminates.
const MAX_WHILE_LOOP_ITERATIONS: usize = 1_000_000;

fn block<'g>(state: &InterpreterState<'g>, op: &str, name: &str) -> Result<&'g BasicBlock> {
    state.find_block(name).ok_or_else(|| invalid(op, format!("sub-graph '{name}' not found")))
}

fn block_def<'g>(block: &'g BasicBlock, name: &str) -> Result<&'g TensorDef> {
    block
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| Error::GraphError(format!("block '{}' has no tensor '{name}'", block.name)))
}

/// Check that a tensor of `dtype` and `shape` matches sub-graph tensor `def`.
fn check_matches(op: &str, what: &str, dtype: DType, shape: &[i32], def: &TensorDef) -> Result<()> {
    if dtype != def.dtype || shape != def.shape.as_slice() {
        return Err(invalid(
            op,
            format!(
                "{what} is {dtype:?} {shape:?} but sub-graph tensor '{}' is {:?} {:?}",
                def.name, def.dtype, def.shape
            ),
        ));
    }
    Ok(())
}

/// Element type and shape of an operator input (from its value).
fn input_signature(state: &InterpreterState, name: &str) -> Result<(DType, Vec<i32>)> {
    let t = state.get_tensor(name)?;
    Ok((t.dtype(), t.shape.iter().map(|&d| d as i32).collect()))
}

/// Element type and shape of an operator output (from its declaration).
fn output_signature(state: &InterpreterState, op: &str, name: &str) -> Result<(DType, Vec<i32>)> {
    let def = state
        .find_tensor_def(name)
        .ok_or_else(|| invalid(op, format!("output '{name}' is not declared")))?;
    Ok((def.dtype, def.shape.clone()))
}

/// Check that `block` takes `inputs` and produces `outputs` (operator tensor names).
fn check_block(
    state: &InterpreterState,
    op: &str,
    block: &BasicBlock,
    inputs: &[String],
    outputs: Option<&[String]>,
) -> Result<()> {
    if block.inputs.len() != inputs.len() {
        return Err(invalid(op, format!("sub-graph '{}' takes {} inputs, got {}", block.name, block.inputs.len(), inputs.len())));
    }
    for (name, block_input) in inputs.iter().zip(&block.inputs) {
        let (dtype, shape) = input_signature(state, name)?;
        check_matches(op, &format!("input '{name}'"), dtype, &shape, block_def(block, block_input)?)?;
    }
    if let Some(outputs) = outputs {
        if block.outputs.len() != outputs.len() {
            return Err(invalid(
                op,
                format!("sub-graph '{}' produces {} outputs, expected {}", block.name, block.outputs.len(), outputs.len()),
            ));
        }
        for (name, block_output) in outputs.iter().zip(&block.outputs) {
            let (dtype, shape) = output_signature(state, op, name)?;
            check_matches(op, &format!("output '{name}'"), dtype, &shape, block_def(block, block_output)?)?;
        }
    }
    Ok(())
}

/// The value of a single-element BOOL tensor.
fn scalar_bool(state: &InterpreterState, op: &str, name: &str) -> Result<bool> {
    match &state.get_tensor(name)?.data {
        TensorData::Bool(v) if v.len() == 1 => Ok(v[0]),
        _ => Err(invalid(op, "condition must be a single BOOL element")),
    }
}

pub fn eval_cond_if(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "COND_IF";
    let OpAttribute::CondIf { then_graph, else_graph } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    let (cond_name, inputs) = op.inputs.split_first().ok_or_else(|| invalid(OP, "missing condition"))?;
    let condition = scalar_bool(state, OP, cond_name)?;
    let then_block = block(state, OP, then_graph)?;
    let else_block = block(state, OP, else_graph)?;
    check_block(state, OP, then_block, inputs, Some(&op.outputs))?;
    check_block(state, OP, else_block, inputs, Some(&op.outputs))?;

    let chosen = if condition { then_graph } else { else_graph };
    state.eval_sub_block(chosen, inputs, &op.outputs)
}

pub fn eval_while_loop(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "WHILE_LOOP";
    let OpAttribute::WhileLoop { cond_graph, body_graph } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    if op.inputs.is_empty() || op.inputs.len() != op.outputs.len() {
        return Err(invalid(OP, "the loop needs at least one input and as many outputs as inputs"));
    }
    for (input, output) in op.inputs.iter().zip(&op.outputs) {
        if input_signature(state, input)? != output_signature(state, OP, output)? {
            return Err(invalid(OP, format!("input '{input}' and output '{output}' differ in type or shape")));
        }
    }
    let cond_block = block(state, OP, cond_graph)?;
    let body_block = block(state, OP, body_graph)?;
    check_block(state, OP, cond_block, &op.inputs, None)?;
    check_block(state, OP, body_block, &op.inputs, Some(&op.outputs))?;
    match cond_block.outputs.as_slice() {
        [out] => {
            let def = block_def(cond_block, out)?;
            if def.dtype != DType::Bool || def.shape.iter().product::<i32>() != 1 {
                return Err(invalid(OP, "the condition graph must produce a single BOOL element"));
            }
        }
        _ => return Err(invalid(OP, "the condition graph must have exactly one output")),
    }

    // Iterate on private copies so the operator's input tensors are untouched.
    let carried: Vec<String> = (0..op.inputs.len()).map(|i| format!("{}#while_state{i}", op.outputs[0])).collect();
    for (input, name) in op.inputs.iter().zip(&carried) {
        let mut t = state.get_tensor(input)?.clone();
        t.name = name.clone();
        state.set_tensor(t);
    }
    let cond_out = format!("{}#while_cond", op.outputs[0]);
    let next: Vec<String> = carried.iter().map(|n| format!("{n}#next")).collect();
    let mut finished = false;
    for _ in 0..MAX_WHILE_LOOP_ITERATIONS {
        state.eval_sub_block(cond_graph, &carried, std::slice::from_ref(&cond_out))?;
        let keep_going = scalar_bool(state, OP, &cond_out)?;
        state.remove_tensor(&cond_out);
        if !keep_going {
            finished = true;
            break;
        }
        state.eval_sub_block(body_graph, &carried, &next)?;
        for (n, c) in next.iter().zip(&carried) {
            state.rename_tensor(n, c);
        }
    }
    if !finished {
        return Err(Error::ControlFlow(format!(
            "{OP}: no termination after {MAX_WHILE_LOOP_ITERATIONS} iterations"
        )));
    }
    for (c, output) in carried.iter().zip(&op.outputs) {
        state.rename_tensor(c, output);
    }
    Ok(())
}
