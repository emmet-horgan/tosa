//! CONST, CONST_SHAPE, IDENTITY and the variable (persistent state)
//! operators VARIABLE, VARIABLE_READ and VARIABLE_WRITE.
//!
//! Variables live in the [`Interpreter`](crate::Interpreter), not in the
//! block's tensors, so they keep their values across runs. VARIABLE
//! initializes a variable the first time it executes (from the tensor's
//! constant data, or zeros); VARIABLE_READ copies the variable into its
//! output tensor; VARIABLE_WRITE's inputs are the variable and the value to
//! store. The execution plan keeps these operators in program order per
//! variable.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{Operator, TensorDef};
use crate::ops::common::{check_declared_output, expect_operands, invalid};
use crate::tensor::{Tensor, TensorData, shape_from_ir};

/// The tensor a declaration describes: its constant data, or zeros.
fn materialize(state: &InterpreterState, op: &str, def: &TensorDef) -> Result<Tensor> {
    let shape = shape_from_ir(&def.shape);
    let count = state.checked_count(op, &shape, Some(def.dtype))?;
    let data = match &def.data {
        Some(bytes) => TensorData::from_raw_bytes(def.dtype, bytes, count)?,
        None => TensorData::zeros(def.dtype, count)?,
    };
    Ok(Tensor::new(def.name.clone(), shape, data))
}

pub fn eval_const(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CONST";
    expect_operands(op, OP, 0, 1)?;
    let output_name = &op.outputs[0];
    let def = state.find_tensor_def(output_name).ok_or_else(|| Error::TensorNotFound(output_name.clone()))?;
    let tensor = materialize(state, OP, def)?;
    state.set_tensor(tensor);
    Ok(())
}

pub fn eval_const_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // CONST_SHAPE is CONST producing a SHAPE value.
    eval_const(state, op)
}

pub fn eval_identity(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    expect_operands(op, "IDENTITY", 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let output = Tensor::new(op.outputs[0].clone(), input.shape.clone(), input.data.clone());
    state.set_tensor(output);
    Ok(())
}

/// VARIABLE: declare (and on first execution initialize) the variable named
/// by the output.
pub fn eval_variable(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "VARIABLE";
    expect_operands(op, OP, 0, 1)?;
    let name = &op.outputs[0];
    if state.variable(name).is_none() {
        let def = state.find_tensor_def(name).ok_or_else(|| Error::TensorNotFound(name.clone()))?;
        let value = materialize(state, OP, def)?;
        state.set_variable(name, value);
    }
    Ok(())
}

/// The current value of variable `name`, initializing it from its
/// declaration if no VARIABLE operator has run yet.
fn variable_value(state: &mut InterpreterState, op: &str, name: &str) -> Result<Tensor> {
    if let Some(v) = state.variable(name) {
        return Ok(v.clone());
    }
    let def = state
        .find_tensor_def(name)
        .filter(|d| d.is_variable)
        .ok_or_else(|| invalid(op, format!("'{name}' is not a variable")))?;
    let value = materialize(state, op, def)?;
    state.set_variable(name, value.clone());
    Ok(value)
}

pub fn eval_variable_read(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "VARIABLE_READ";
    expect_operands(op, OP, 1, 1)?;
    let mut value = variable_value(state, OP, &op.inputs[0])?;
    if let Some(declared) = check_declared_output(state, OP, &op.outputs[0], &value.shape)?
        && declared != value.dtype()
    {
        return Err(Error::DTypeMismatch(format!(
            "{OP}: output is declared {declared:?} but the variable is {:?}",
            value.dtype()
        )));
    }
    value.name = op.outputs[0].clone();
    state.set_tensor(value);
    Ok(())
}

pub fn eval_variable_write(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "VARIABLE_WRITE";
    expect_operands(op, OP, 2, 0)?;
    let (var_name, value_name) = (&op.inputs[0], &op.inputs[1]);
    let current = variable_value(state, OP, var_name)?;
    let value = state.get_tensor(value_name)?;
    if value.dtype() != current.dtype() || value.shape != current.shape {
        return Err(invalid(
            OP,
            format!(
                "value is {:?} {:?} but variable '{var_name}' is {:?} {:?}",
                value.dtype(),
                value.shape,
                current.dtype(),
                current.shape
            ),
        ));
    }
    let value = value.clone();
    state.set_variable(var_name, value);
    Ok(())
}
