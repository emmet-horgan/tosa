use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::Operator;
use crate::tensor::{Tensor, TensorData, shape_from_ir};

pub fn eval_const(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let output_name = op.outputs.first().ok_or_else(|| Error::GraphError("CONST op has no output".to_string()))?;

    let tensor_def =
        state.find_tensor_def(output_name).ok_or_else(|| Error::TensorNotFound(output_name.clone()))?.clone();

    let shape = shape_from_ir(&tensor_def.shape);
    let count: usize = shape.iter().product();
    let data = match &tensor_def.data {
        Some(bytes) => TensorData::from_raw_bytes(tensor_def.dtype, bytes, count)?,
        None => TensorData::zeros(tensor_def.dtype, count)?,
    };

    let tensor = Tensor::new(output_name.clone(), shape, data);
    state.set_tensor(tensor);
    Ok(())
}

pub fn eval_const_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // CONST_SHAPE is identical to CONST — it produces a shape tensor
    eval_const(state, op)
}

pub fn eval_identity(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input_name = op.inputs.first().ok_or_else(|| Error::GraphError("IDENTITY op has no input".to_string()))?;
    let output_name = op.outputs.first().ok_or_else(|| Error::GraphError("IDENTITY op has no output".to_string()))?;

    let input = state.get_tensor(input_name)?.clone();
    let output = Tensor::new(output_name.clone(), input.shape.clone(), input.data);
    state.set_tensor(output);
    Ok(())
}

pub fn eval_variable(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // VARIABLE declares a variable tensor — initialize it from the tensor def data or zeros
    eval_const(state, op)
}

pub fn eval_variable_read(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // VARIABLE_READ: copy the variable tensor to the output
    let input_name = op.inputs.first().ok_or_else(|| Error::GraphError("VARIABLE_READ: no input".to_string()))?;
    let output_name = op.outputs.first().ok_or_else(|| Error::GraphError("VARIABLE_READ: no output".to_string()))?;

    let input = state.get_tensor(input_name)?.clone();
    let output = Tensor::new(output_name.clone(), input.shape.clone(), input.data);
    state.set_tensor(output);
    Ok(())
}

pub fn eval_variable_write(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // VARIABLE_WRITE: write the input value into the variable tensor
    let input_name = op.inputs.get(1).ok_or_else(|| Error::GraphError("VARIABLE_WRITE: no value input".to_string()))?;
    let var_name =
        op.inputs.first().ok_or_else(|| Error::GraphError("VARIABLE_WRITE: no variable input".to_string()))?;

    let value = state.get_tensor(input_name)?.clone();
    let output = Tensor::new(var_name.clone(), value.shape.clone(), value.data);
    state.set_tensor(output);
    Ok(())
}
