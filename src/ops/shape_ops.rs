use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::{Tensor, TensorData};

fn get_shape_i64(state: &InterpreterState, name: &str) -> Result<Vec<i64>> {
    let t = state.get_tensor(name)?;
    match &t.data {
        TensorData::I64(v) => Ok(v.clone()),
        TensorData::I32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        _ => Err(Error::DTypeMismatch(format!(
            "shape op: expected integer tensor for '{}', got {:?}",
            name,
            t.data.dtype()
        ))),
    }
}

fn output_shape_tensor(state: &mut InterpreterState, op: &Operator, data: Vec<i64>) -> Result<()> {
    let output_name = &op.outputs[0];
    let shape = vec![data.len()];
    state.set_tensor(Tensor::new(output_name.clone(), shape, TensorData::I64(data)));
    Ok(())
}

pub fn eval_add_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_sub_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_mul_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_concat_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let mut result = Vec::new();
    for input_name in &op.inputs {
        let v = get_shape_i64(state, input_name)?;
        result.extend(v);
    }
    output_shape_tensor(state, op, result)
}

pub fn eval_slice_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = get_shape_i64(state, &op.inputs[0])?;
    let start = get_shape_i64(state, &op.inputs[1])?;
    let size = get_shape_i64(state, &op.inputs[2])?;

    let s = start.first().copied().unwrap_or(0) as usize;
    let n = size.first().copied().unwrap_or(input.len() as i64) as usize;
    let result = input[s..s + n].to_vec();
    output_shape_tensor(state, op, result)
}

pub fn eval_exp2_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let result: Vec<i64> = a.iter().map(|&x| 1i64 << x).collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_log2_ceil_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let result: Vec<i64> = a
        .iter()
        .map(|&x| {
            if x <= 1 {
                0
            } else {
                64 - (x - 1).leading_zeros() as i64
            }
        })
        .collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_log2_floor_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let result: Vec<i64> = a
        .iter()
        .map(|&x| if x <= 0 { 0 } else { 63 - x.leading_zeros() as i64 })
        .collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_max_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| (*x).max(*y)).collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_min_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| (*x).min(*y)).collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_mod_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| if *y != 0 { x % y } else { 0 })
        .collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_div_ceil_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            if *y == 0 {
                0
            } else {
                (x + y - 1) / y
            }
        })
        .collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_div_floor_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    let result: Vec<i64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| if *y == 0 { 0 } else { x.div_euclid(*y) })
        .collect();
    output_shape_tensor(state, op, result)
}

pub fn eval_dim(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let axis = match &op.attribute {
        OpAttribute::ArgMax { axis } => *axis as usize,
        _ => 0,
    };
    let dim_val = if axis < input.shape.len() {
        input.shape[axis] as i64
    } else {
        0
    };
    output_shape_tensor(state, op, vec![dim_val])
}

pub fn eval_assert_equal_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = get_shape_i64(state, &op.inputs[0])?;
    let b = get_shape_i64(state, &op.inputs[1])?;
    if a != b {
        return Err(Error::ShapeMismatch(format!(
            "ASSERT_EQUAL_SHAPE: {:?} != {:?}",
            a, b
        )));
    }
    Ok(())
}
