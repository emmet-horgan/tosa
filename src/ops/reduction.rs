//! REDUCE_ALL, REDUCE_ANY, REDUCE_MAX, REDUCE_MIN, REDUCE_PRODUCT and
//! REDUCE_SUM along one axis; the output keeps the input rank with the
//! reduced dimension set to 1.
//!
//! Accumulation follows the reference model: elements are combined in index
//! order; FP16 sums and FP16/BF16 products are rounded to the element type at
//! every step, while BF16 sums accumulate in FP32 and are rounded once.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, NanMode, OpAttribute, Operator};
use crate::ops::common::{
    check_declared_output, check_nan_mode, expect_operands, float_data, float_highest, float_lowest,
    float_values, int_data, int_range, int_values, invalid, nan_max, nan_min, round_to,
};
use crate::tensor::{Tensor, TensorData};

struct Reduction {
    dtype: DType,
    out_shape: Vec<usize>,
    outer: usize,
    len: usize,
    inner: usize,
    nan_mode: NanMode,
}

impl Reduction {
    /// Combine the `len` elements of each reduction run with `f`, starting from `init`.
    fn fold<T: Copy>(&self, v: &[T], init: T, f: impl Fn(T, T) -> T) -> Vec<T> {
        let mut out = Vec::with_capacity(self.outer * self.inner);
        for o in 0..self.outer {
            for i in 0..self.inner {
                out.push((0..self.len).fold(init, |acc, a| f(acc, v[(o * self.len + a) * self.inner + i])));
            }
        }
        out
    }
}

fn eval_reduce(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    allowed: &[DType],
    f: impl FnOnce(&Reduction, &TensorData) -> Result<TensorData>,
) -> Result<()> {
    let OpAttribute::Reduce { axis, nan_mode } = op.attribute else {
        return Err(invalid(name, "missing attribute"));
    };
    expect_operands(op, name, 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let dtype = input.dtype();
    if !allowed.contains(&dtype) {
        return Err(Error::DTypeMismatch(format!("{name}: unsupported dtype {dtype:?}")));
    }
    let rank = input.shape.len();
    if rank == 0 || axis < 0 || axis as usize >= rank {
        return Err(invalid(name, format!("axis {axis} out of range for rank {rank}")));
    }
    let axis = axis as usize;
    let mut out_shape = input.shape.clone();
    out_shape[axis] = 1;
    let r = Reduction {
        dtype,
        outer: input.shape[..axis].iter().product(),
        len: input.shape[axis],
        inner: input.shape[axis + 1..].iter().product(),
        out_shape,
        nan_mode,
    };
    if let Some(declared) = check_declared_output(state, name, &op.outputs[0], &r.out_shape)?
        && declared != dtype
    {
        return Err(Error::DTypeMismatch(format!("{name}: output dtype {declared:?} must be {dtype:?}")));
    }
    let data = f(&r, &input.data)?;
    state.set_tensor(Tensor::new(op.outputs[0].clone(), r.out_shape, data));
    Ok(())
}

const MINMAX: &[DType] = &[DType::F16, DType::Bf16, DType::F32, DType::I8, DType::I16, DType::I32, DType::I64];

pub fn eval_reduce_all(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_reduce(state, op, "REDUCE_ALL", &[DType::Bool], |r, d| match d {
        TensorData::Bool(v) => Ok(TensorData::Bool(r.fold(v, true, |a, b| a && b))),
        _ => unreachable!(),
    })
}

pub fn eval_reduce_any(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_reduce(state, op, "REDUCE_ANY", &[DType::Bool], |r, d| match d {
        TensorData::Bool(v) => Ok(TensorData::Bool(r.fold(v, false, |a, b| a || b))),
        _ => unreachable!(),
    })
}

pub fn eval_reduce_max(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "REDUCE_MAX";
    eval_reduce(state, op, OP, MINMAX, |r, d| {
        let mode = check_nan_mode(OP, r.nan_mode)?;
        if let Some(v) = int_values(d) {
            return int_data(r.dtype, r.fold(&v, int_range(r.dtype).0, i64::max));
        }
        let init = if mode == NanMode::Ignore { f32::NAN } else { float_lowest(r.dtype) };
        float_data(r.dtype, r.fold(&float_values(d).unwrap(), init, |a, b| nan_max(a, b, mode)))
    })
}

pub fn eval_reduce_min(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "REDUCE_MIN";
    eval_reduce(state, op, OP, MINMAX, |r, d| {
        let mode = check_nan_mode(OP, r.nan_mode)?;
        if let Some(v) = int_values(d) {
            return int_data(r.dtype, r.fold(&v, int_range(r.dtype).1, i64::min));
        }
        let init = if mode == NanMode::Ignore { f32::NAN } else { float_highest(r.dtype) };
        float_data(r.dtype, r.fold(&float_values(d).unwrap(), init, |a, b| nan_min(a, b, mode)))
    })
}

pub fn eval_reduce_product(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_reduce(state, op, "REDUCE_PRODUCT", &[DType::F16, DType::Bf16, DType::F32], |r, d| {
        let dt = r.dtype;
        float_data(dt, r.fold(&float_values(d).unwrap(), 1.0, |a, b| round_to(dt, a * b)))
    })
}

pub fn eval_reduce_sum(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let allowed = [DType::F16, DType::Bf16, DType::F32, DType::I32, DType::I64];
    eval_reduce(state, op, "REDUCE_SUM", &allowed, |r, d| {
        if let Some(v) = int_values(d) {
            let dt = r.dtype;
            let wrap = |x: i64| if dt == DType::I32 { x as i32 as i64 } else { x };
            return int_data(dt, r.fold(&v, 0, |a, b| wrap(a.wrapping_add(b))));
        }
        // FP16 accumulates in FP16; BF16 and FP32 in FP32.
        let acc_t = if r.dtype == DType::F16 { DType::F16 } else { DType::F32 };
        float_data(r.dtype, r.fold(&float_values(d).unwrap(), 0.0, |a, b| round_to(acc_t, a + b)))
    })
}
