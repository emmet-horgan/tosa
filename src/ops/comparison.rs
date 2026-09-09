use crate::broadcast::{broadcast_index, broadcast_shapes, flat_to_nd};
use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::Operator;
use crate::tensor::{Tensor, TensorData};

/// Helper macro to dispatch a comparison operation across supported numeric types.
/// Produces a Bool output tensor. Supports broadcasting.
macro_rules! eval_comparison {
    ($state:expr, $op:expr, $op_name:expr, $cmp:expr) => {{
        let a = $state.get_tensor(&$op.inputs[0])?;
        let b = $state.get_tensor(&$op.inputs[1])?;
        let output_name = &$op.outputs[0];

        let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
        let out_count: usize = out_shape.iter().product::<usize>().max(1);

        let cmp_fn = $cmp;

        let result: Vec<bool> = match (&a.data, &b.data) {
            (TensorData::F32(av), TensorData::F32(bv)) => {
                let mut res = Vec::with_capacity(out_count);
                for flat in 0..out_count {
                    let nd = flat_to_nd(flat, &out_shape);
                    let ai = broadcast_index(&nd, &a.shape, &out_shape);
                    let bi = broadcast_index(&nd, &b.shape, &out_shape);
                    res.push(cmp_fn(av[ai] as f64, bv[bi] as f64));
                }
                res
            }
            (TensorData::I32(av), TensorData::I32(bv)) => {
                let mut res = Vec::with_capacity(out_count);
                for flat in 0..out_count {
                    let nd = flat_to_nd(flat, &out_shape);
                    let ai = broadcast_index(&nd, &a.shape, &out_shape);
                    let bi = broadcast_index(&nd, &b.shape, &out_shape);
                    res.push(cmp_fn(av[ai] as f64, bv[bi] as f64));
                }
                res
            }
            (TensorData::I8(av), TensorData::I8(bv)) => {
                let mut res = Vec::with_capacity(out_count);
                for flat in 0..out_count {
                    let nd = flat_to_nd(flat, &out_shape);
                    let ai = broadcast_index(&nd, &a.shape, &out_shape);
                    let bi = broadcast_index(&nd, &b.shape, &out_shape);
                    res.push(cmp_fn(av[ai] as f64, bv[bi] as f64));
                }
                res
            }
            (TensorData::I16(av), TensorData::I16(bv)) => {
                let mut res = Vec::with_capacity(out_count);
                for flat in 0..out_count {
                    let nd = flat_to_nd(flat, &out_shape);
                    let ai = broadcast_index(&nd, &a.shape, &out_shape);
                    let bi = broadcast_index(&nd, &b.shape, &out_shape);
                    res.push(cmp_fn(av[ai] as f64, bv[bi] as f64));
                }
                res
            }
            (TensorData::I64(av), TensorData::I64(bv)) => {
                let mut res = Vec::with_capacity(out_count);
                for flat in 0..out_count {
                    let nd = flat_to_nd(flat, &out_shape);
                    let ai = broadcast_index(&nd, &a.shape, &out_shape);
                    let bi = broadcast_index(&nd, &b.shape, &out_shape);
                    res.push(cmp_fn(av[ai] as f64, bv[bi] as f64));
                }
                res
            }
            _ => {
                return Err(Error::DTypeMismatch(format!(
                    "{}: unsupported or mismatched dtypes {:?} and {:?}",
                    $op_name,
                    a.data.dtype(),
                    b.data.dtype(),
                )));
            }
        };

        $state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::Bool(result)));
        Ok(())
    }};
}

pub fn eval_equal(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_comparison!(state, op, "EQUAL", |a: f64, b: f64| a == b)
}

pub fn eval_greater(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_comparison!(state, op, "GREATER", |a: f64, b: f64| a > b)
}

pub fn eval_greater_equal(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_comparison!(state, op, "GREATER_EQUAL", |a: f64, b: f64| a >= b)
}

/// SELECT: inputs are [condition(Bool), a, b]. Output[i] = if cond[i] then a[i] else b[i].
/// The condition, a, and b tensors are broadcast-compatible. Output dtype matches a/b.
pub fn eval_select(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let cond = state.get_tensor(&op.inputs[0])?;
    let a = state.get_tensor(&op.inputs[1])?;
    let b = state.get_tensor(&op.inputs[2])?;
    let output_name = &op.outputs[0];

    let cond_bools = cond
        .data
        .as_bool_slice()
        .ok_or_else(|| Error::DTypeMismatch("SELECT: condition must be Bool".to_string()))?;

    // Broadcast all three shapes together
    let ab_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_shape = broadcast_shapes(&cond.shape, &ab_shape)?;
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    macro_rules! select_typed {
        ($variant:ident, $av:expr, $bv:expr) => {{
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ci = broadcast_index(&nd, &cond.shape, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(if cond_bools[ci] { $av[ai] } else { $bv[bi] });
            }
            TensorData::$variant(result)
        }};
    }

    let data = match (&a.data, &b.data) {
        (TensorData::F32(av), TensorData::F32(bv)) => select_typed!(F32, av, bv),
        (TensorData::I32(av), TensorData::I32(bv)) => select_typed!(I32, av, bv),
        (TensorData::I8(av), TensorData::I8(bv)) => select_typed!(I8, av, bv),
        (TensorData::I16(av), TensorData::I16(bv)) => select_typed!(I16, av, bv),
        (TensorData::I64(av), TensorData::I64(bv)) => select_typed!(I64, av, bv),
        (TensorData::Bool(av), TensorData::Bool(bv)) => select_typed!(Bool, av, bv),
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "SELECT: a and b must have the same dtype, got {:?} and {:?}",
                a.data.dtype(),
                b.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}
