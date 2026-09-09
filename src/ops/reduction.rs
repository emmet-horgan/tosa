use crate::broadcast::flat_to_nd;
use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::{Tensor, TensorData};

/// Compute the output shape for a reduction: input shape with the reduced axis set to 1 (kept).
fn reduce_output_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut out_shape = input_shape.to_vec();
    if axis < out_shape.len() {
        out_shape[axis] = 1;
    }
    out_shape
}

/// Extract the axis from the op attribute, returning it as a usize.
fn get_reduce_axis(op: &Operator, input_rank: usize) -> Result<usize> {
    let axis_i32 = match &op.attribute {
        OpAttribute::Reduce { axis } => *axis,
        _ => {
            return Err(Error::ShapeMismatch(
                "reduction op missing Reduce attribute".to_string(),
            ));
        }
    };
    let axis = if axis_i32 < 0 {
        (input_rank as i32 + axis_i32) as usize
    } else {
        axis_i32 as usize
    };
    if axis >= input_rank {
        return Err(Error::ShapeMismatch(format!(
            "reduction axis {} out of range for rank {}",
            axis, input_rank
        )));
    }
    Ok(axis)
}

/// Maps a flat index in the output to the range of flat indices in the input that
/// contribute to this output element, yielding each input flat index in turn.
/// This iterates along the reduced axis.
fn reduce_iterate<'a>(
    out_flat: usize,
    out_shape: &'a [usize],
    input_shape: &'a [usize],
    axis: usize,
) -> impl Iterator<Item = usize> + 'a {
    let nd = flat_to_nd(out_flat, out_shape);
    let axis_len = input_shape[axis];
    (0..axis_len).map(move |k| {
        // Compute flat index into input
        let mut flat = 0usize;
        let mut stride = 1usize;
        for dim in (0..input_shape.len()).rev() {
            let idx = if dim == axis { k } else { nd[dim] };
            flat += idx * stride;
            stride *= input_shape[dim];
        }
        flat
    })
}

/// Macro to implement a numeric reduction across F32, I32, I8, I16, I64 dtypes.
macro_rules! reduce_numeric {
    ($state:expr, $op:expr, $op_name:expr, $init:expr, $fold:expr, $finalize:expr) => {{
        let input = $state.get_tensor(&$op.inputs[0])?;
        let axis = get_reduce_axis($op, input.shape.len())?;
        let out_shape = reduce_output_shape(&input.shape, axis);
        let out_count: usize = out_shape.iter().product::<usize>().max(1);
        let output_name = &$op.outputs[0];
        let axis_len = input.shape[axis];

        let data = match &input.data {
            TensorData::F32(v) => {
                let init_fn = $init;
                let fold_fn = $fold;
                let finalize_fn = $finalize;
                let mut result = Vec::with_capacity(out_count);
                for out_flat in 0..out_count {
                    let mut acc: f64 = init_fn(v[0] as f64);
                    let mut first = true;
                    for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
                        let val = v[in_flat] as f64;
                        if first {
                            acc = init_fn(val);
                            first = false;
                        } else {
                            acc = fold_fn(acc, val);
                        }
                    }
                    result.push(finalize_fn(acc, axis_len) as f32);
                }
                TensorData::F32(result)
            }
            TensorData::I32(v) => {
                let init_fn = $init;
                let fold_fn = $fold;
                let finalize_fn = $finalize;
                let mut result = Vec::with_capacity(out_count);
                for out_flat in 0..out_count {
                    let mut acc: f64 = init_fn(v[0] as f64);
                    let mut first = true;
                    for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
                        let val = v[in_flat] as f64;
                        if first {
                            acc = init_fn(val);
                            first = false;
                        } else {
                            acc = fold_fn(acc, val);
                        }
                    }
                    result.push(finalize_fn(acc, axis_len) as i32);
                }
                TensorData::I32(result)
            }
            TensorData::I8(v) => {
                let init_fn = $init;
                let fold_fn = $fold;
                let finalize_fn = $finalize;
                let mut result = Vec::with_capacity(out_count);
                for out_flat in 0..out_count {
                    let mut acc: f64 = init_fn(v[0] as f64);
                    let mut first = true;
                    for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
                        let val = v[in_flat] as f64;
                        if first {
                            acc = init_fn(val);
                            first = false;
                        } else {
                            acc = fold_fn(acc, val);
                        }
                    }
                    result.push(finalize_fn(acc, axis_len) as i8);
                }
                TensorData::I8(result)
            }
            TensorData::I16(v) => {
                let init_fn = $init;
                let fold_fn = $fold;
                let finalize_fn = $finalize;
                let mut result = Vec::with_capacity(out_count);
                for out_flat in 0..out_count {
                    let mut acc: f64 = init_fn(v[0] as f64);
                    let mut first = true;
                    for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
                        let val = v[in_flat] as f64;
                        if first {
                            acc = init_fn(val);
                            first = false;
                        } else {
                            acc = fold_fn(acc, val);
                        }
                    }
                    result.push(finalize_fn(acc, axis_len) as i16);
                }
                TensorData::I16(result)
            }
            TensorData::I64(v) => {
                let init_fn = $init;
                let fold_fn = $fold;
                let finalize_fn = $finalize;
                let mut result = Vec::with_capacity(out_count);
                for out_flat in 0..out_count {
                    let mut acc: f64 = init_fn(v[0] as f64);
                    let mut first = true;
                    for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
                        let val = v[in_flat] as f64;
                        if first {
                            acc = init_fn(val);
                            first = false;
                        } else {
                            acc = fold_fn(acc, val);
                        }
                    }
                    result.push(finalize_fn(acc, axis_len) as i64);
                }
                TensorData::I64(result)
            }
            _ => {
                return Err(Error::DTypeMismatch(format!(
                    "{}: unsupported dtype {:?}",
                    $op_name,
                    input.data.dtype()
                )));
            }
        };

        $state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
        Ok(())
    }};
}

pub fn eval_reduce_sum(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    reduce_numeric!(
        state,
        op,
        "REDUCE_SUM",
        |v: f64| v,
        |acc: f64, v: f64| acc + v,
        |acc: f64, _axis_len: usize| acc
    )
}

pub fn eval_reduce_max(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    reduce_numeric!(
        state,
        op,
        "REDUCE_MAX",
        |v: f64| v,
        |acc: f64, v: f64| if v > acc { v } else { acc },
        |acc: f64, _axis_len: usize| acc
    )
}

pub fn eval_reduce_min(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    reduce_numeric!(
        state,
        op,
        "REDUCE_MIN",
        |v: f64| v,
        |acc: f64, v: f64| if v < acc { v } else { acc },
        |acc: f64, _axis_len: usize| acc
    )
}

pub fn eval_reduce_product(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    reduce_numeric!(
        state,
        op,
        "REDUCE_PRODUCT",
        |v: f64| v,
        |acc: f64, v: f64| acc * v,
        |acc: f64, _axis_len: usize| acc
    )
}

/// REDUCE_ALL: logical AND along axis. Input and output are Bool.
pub fn eval_reduce_all(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let axis = get_reduce_axis(op, input.shape.len())?;
    let out_shape = reduce_output_shape(&input.shape, axis);
    let out_count: usize = out_shape.iter().product::<usize>().max(1);
    let output_name = &op.outputs[0];

    let v = input
        .data
        .as_bool_slice()
        .ok_or_else(|| Error::DTypeMismatch("REDUCE_ALL: input must be Bool".to_string()))?;

    let mut result = Vec::with_capacity(out_count);
    for out_flat in 0..out_count {
        let mut acc = true;
        for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
            if !v[in_flat] {
                acc = false;
                break;
            }
        }
        result.push(acc);
    }

    state.set_tensor(Tensor::new(
        output_name.clone(),
        out_shape,
        TensorData::Bool(result),
    ));
    Ok(())
}

/// REDUCE_ANY: logical OR along axis. Input and output are Bool.
pub fn eval_reduce_any(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let axis = get_reduce_axis(op, input.shape.len())?;
    let out_shape = reduce_output_shape(&input.shape, axis);
    let out_count: usize = out_shape.iter().product::<usize>().max(1);
    let output_name = &op.outputs[0];

    let v = input
        .data
        .as_bool_slice()
        .ok_or_else(|| Error::DTypeMismatch("REDUCE_ANY: input must be Bool".to_string()))?;

    let mut result = Vec::with_capacity(out_count);
    for out_flat in 0..out_count {
        let mut acc = false;
        for in_flat in reduce_iterate(out_flat, &out_shape, &input.shape, axis) {
            if v[in_flat] {
                acc = true;
                break;
            }
        }
        result.push(acc);
    }

    state.set_tensor(Tensor::new(
        output_name.clone(),
        out_shape,
        TensorData::Bool(result),
    ));
    Ok(())
}
