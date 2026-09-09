use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::TensorData;

const MAX_WHILE_LOOP_ITERATIONS: usize = 10000;

pub fn eval_cond_if(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (then_block_name, else_block_name) = match &op.attribute {
        OpAttribute::CondIf {
            then_graph,
            else_graph,
        } => (then_graph.clone(), else_graph.clone()),
        _ => {
            return Err(Error::ControlFlow(
                "COND_IF: missing CondIf attribute".to_string(),
            ))
        }
    };

    // First input is the boolean condition
    let cond_tensor = state.get_tensor(&op.inputs[0])?;
    let condition = match &cond_tensor.data {
        TensorData::Bool(v) => v.first().copied().unwrap_or(false),
        _ => {
            return Err(Error::DTypeMismatch(
                "COND_IF: condition must be Bool".to_string(),
            ))
        }
    };

    let block_name = if condition {
        &then_block_name
    } else {
        &else_block_name
    };

    // Remaining inputs (after the condition) are passed to the sub-block
    let block_input_names: Vec<String> = op.inputs[1..].to_vec();
    let block_output_names: Vec<String> = op.outputs.clone();

    state.eval_sub_block(block_name, &block_input_names, &block_output_names)?;
    Ok(())
}

pub fn eval_while_loop(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (cond_block_name, body_block_name) = match &op.attribute {
        OpAttribute::WhileLoop {
            cond_graph,
            body_graph,
        } => (cond_graph.clone(), body_graph.clone()),
        _ => {
            return Err(Error::ControlFlow(
                "WHILE_LOOP: missing WhileLoop attribute".to_string(),
            ))
        }
    };

    // Loop inputs are the op's inputs; they get updated each iteration
    let loop_tensor_names: Vec<String> = op.inputs.clone();

    for iteration in 0..MAX_WHILE_LOOP_ITERATIONS {
        // Evaluate condition block
        // The cond block takes the loop tensors as input and produces a single bool output
        let cond_output_name = format!("__while_cond_out_{}", iteration);
        state.eval_sub_block_with_rename(
            &cond_block_name,
            &loop_tensor_names,
            &[cond_output_name.clone()],
        )?;

        let cond_result = state.get_tensor(&cond_output_name)?;
        let should_continue = match &cond_result.data {
            TensorData::Bool(v) => v.first().copied().unwrap_or(false),
            _ => {
                return Err(Error::ControlFlow(
                    "WHILE_LOOP: cond block must output Bool".to_string(),
                ))
            }
        };
        // Clean up temporary condition tensor
        state.remove_tensor(&cond_output_name);

        if !should_continue {
            break;
        }

        // Evaluate body block: takes loop tensors, produces updated loop tensors
        let body_output_names: Vec<String> = loop_tensor_names
            .iter()
            .map(|n| format!("{}_iter{}", n, iteration))
            .collect();

        state.eval_sub_block_with_rename(
            &body_block_name,
            &loop_tensor_names,
            &body_output_names,
        )?;

        // Update loop tensors to point to body outputs
        for (old_name, new_name) in loop_tensor_names.iter().zip(body_output_names.iter()) {
            state.rename_tensor(new_name, old_name);
        }
    }

    // Copy final loop tensors to op outputs
    for (loop_name, output_name) in loop_tensor_names.iter().zip(op.outputs.iter()) {
        if loop_name != output_name {
            let tensor = state.get_tensor(loop_name)?.clone();
            let mut output = tensor;
            output.name = output_name.clone();
            state.set_tensor(output);
        }
    }

    Ok(())
}
