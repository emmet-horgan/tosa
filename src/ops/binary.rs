use crate::broadcast::{broadcast_index, broadcast_shapes, flat_to_nd};
use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::{Tensor, TensorData};

// ---------------------------------------------------------------------------
// Generic binary element-wise dispatch
// ---------------------------------------------------------------------------

/// Evaluate a binary element-wise op that supports all numeric types.
/// `f32_op` is used for F32, F16, and BF16 (promote-compute-truncate).
/// `i8_op` / `i16_op` / `i32_op` / `i64_op` handle the integer variants.
fn eval_binary_elementwise<F32Op, I8Op, I16Op, I32Op, I64Op>(
    state: &mut InterpreterState,
    op: &Operator,
    f32_op: F32Op,
    i8_op: I8Op,
    i16_op: I16Op,
    i32_op: I32Op,
    i64_op: I64Op,
) -> Result<()>
where
    F32Op: Fn(f32, f32) -> f32,
    I8Op: Fn(i8, i8) -> i8,
    I16Op: Fn(i16, i16) -> i16,
    I32Op: Fn(i32, i32) -> i32,
    I64Op: Fn(i64, i64) -> i64,
{
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    let data = match (&a.data, &b.data) {
        (TensorData::F32(av), TensorData::F32(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(f32_op(av[ai], bv[bi]));
            }
            TensorData::F32(result)
        }
        (TensorData::F16(av), TensorData::F16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let r = f32_op(av[ai].to_f32(), bv[bi].to_f32());
                result.push(half::f16::from_f32(r));
            }
            TensorData::F16(result)
        }
        (TensorData::Bf16(av), TensorData::Bf16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let r = f32_op(av[ai].to_f32(), bv[bi].to_f32());
                result.push(half::bf16::from_f32(r));
            }
            TensorData::Bf16(result)
        }
        (TensorData::I8(av), TensorData::I8(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i8_op(av[ai], bv[bi]));
            }
            TensorData::I8(result)
        }
        (TensorData::I16(av), TensorData::I16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i16_op(av[ai], bv[bi]));
            }
            TensorData::I16(result)
        }
        (TensorData::I32(av), TensorData::I32(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i32_op(av[ai], bv[bi]));
            }
            TensorData::I32(result)
        }
        (TensorData::I48(av), TensorData::I48(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i64_op(av[ai], bv[bi]));
            }
            TensorData::I48(result)
        }
        (TensorData::I64(av), TensorData::I64(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i64_op(av[ai], bv[bi]));
            }
            TensorData::I64(result)
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "binary op {:?}: unsupported dtype combination {:?} and {:?}",
                op.op,
                a.data.dtype(),
                b.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// Evaluate a binary element-wise op for integer types only.
fn eval_binary_int_only<I8Op, I16Op, I32Op, I64Op>(
    state: &mut InterpreterState,
    op: &Operator,
    i8_op: I8Op,
    i16_op: I16Op,
    i32_op: I32Op,
    i64_op: I64Op,
) -> Result<()>
where
    I8Op: Fn(i8, i8) -> i8,
    I16Op: Fn(i16, i16) -> i16,
    I32Op: Fn(i32, i32) -> i32,
    I64Op: Fn(i64, i64) -> i64,
{
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    let data = match (&a.data, &b.data) {
        (TensorData::I8(av), TensorData::I8(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i8_op(av[ai], bv[bi]));
            }
            TensorData::I8(result)
        }
        (TensorData::I16(av), TensorData::I16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i16_op(av[ai], bv[bi]));
            }
            TensorData::I16(result)
        }
        (TensorData::I32(av), TensorData::I32(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i32_op(av[ai], bv[bi]));
            }
            TensorData::I32(result)
        }
        (TensorData::I48(av), TensorData::I48(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i64_op(av[ai], bv[bi]));
            }
            TensorData::I48(result)
        }
        (TensorData::I64(av), TensorData::I64(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(i64_op(av[ai], bv[bi]));
            }
            TensorData::I64(result)
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "integer binary op {:?}: unsupported dtype combination {:?} and {:?}",
                op.op,
                a.data.dtype(),
                b.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// Evaluate a binary element-wise op for Bool tensors only.
fn eval_binary_bool<BoolOp>(
    state: &mut InterpreterState,
    op: &Operator,
    bool_op: BoolOp,
) -> Result<()>
where
    BoolOp: Fn(bool, bool) -> bool,
{
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    let data = match (&a.data, &b.data) {
        (TensorData::Bool(av), TensorData::Bool(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(bool_op(av[ai], bv[bi]));
            }
            TensorData::Bool(result)
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "logical op {:?}: expected Bool tensors, got {:?} and {:?}",
                op.op,
                a.data.dtype(),
                b.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

// ---------------------------------------------------------------------------
// Macro to reduce boilerplate for simple binary ops
// ---------------------------------------------------------------------------

/// Defines a simple binary op that applies element-wise across all numeric types.
macro_rules! define_binary_op {
    (
        $fn_name:ident,
        f32: $f32_expr:expr,
        i8: $i8_expr:expr,
        i16: $i16_expr:expr,
        i32: $i32_expr:expr,
        i64: $i64_expr:expr
    ) => {
        pub fn $fn_name(state: &mut InterpreterState, op: &Operator) -> Result<()> {
            eval_binary_elementwise(
                state,
                op,
                $f32_expr,
                $i8_expr,
                $i16_expr,
                $i32_expr,
                $i64_expr,
            )
        }
    };
}

// ---------------------------------------------------------------------------
// ADD, SUB, MAXIMUM, MINIMUM
// ---------------------------------------------------------------------------

define_binary_op!(
    eval_add,
    f32: |a: f32, b: f32| a + b,
    i8: |a: i8, b: i8| a.wrapping_add(b),
    i16: |a: i16, b: i16| a.wrapping_add(b),
    i32: |a: i32, b: i32| a.wrapping_add(b),
    i64: |a: i64, b: i64| a.wrapping_add(b)
);

define_binary_op!(
    eval_sub,
    f32: |a: f32, b: f32| a - b,
    i8: |a: i8, b: i8| a.wrapping_sub(b),
    i16: |a: i16, b: i16| a.wrapping_sub(b),
    i32: |a: i32, b: i32| a.wrapping_sub(b),
    i64: |a: i64, b: i64| a.wrapping_sub(b)
);

define_binary_op!(
    eval_maximum,
    f32: |a: f32, b: f32| a.max(b),
    i8: |a: i8, b: i8| a.max(b),
    i16: |a: i16, b: i16| a.max(b),
    i32: |a: i32, b: i32| a.max(b),
    i64: |a: i64, b: i64| a.max(b)
);

define_binary_op!(
    eval_minimum,
    f32: |a: f32, b: f32| a.min(b),
    i8: |a: i8, b: i8| a.min(b),
    i16: |a: i16, b: i16| a.min(b),
    i32: |a: i32, b: i32| a.min(b),
    i64: |a: i64, b: i64| a.min(b)
);

// ---------------------------------------------------------------------------
// POW
// ---------------------------------------------------------------------------

pub fn eval_pow(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_elementwise(
        state,
        op,
        |a: f32, b: f32| a.powf(b),
        |a: i8, b: i8| {
            if b < 0 {
                0
            } else {
                a.wrapping_pow(b as u32)
            }
        },
        |a: i16, b: i16| {
            if b < 0 {
                0
            } else {
                a.wrapping_pow(b as u32)
            }
        },
        |a: i32, b: i32| {
            if b < 0 {
                0
            } else {
                a.wrapping_pow(b as u32)
            }
        },
        |a: i64, b: i64| {
            if b < 0 {
                0
            } else {
                a.wrapping_pow(b as u32)
            }
        },
    )
}

// ---------------------------------------------------------------------------
// INTDIV (integer truncation division)
// ---------------------------------------------------------------------------

pub fn eval_intdiv(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_int_only(
        state,
        op,
        |a: i8, b: i8| {
            if b == 0 {
                0
            } else {
                a.wrapping_div(b)
            }
        },
        |a: i16, b: i16| {
            if b == 0 {
                0
            } else {
                a.wrapping_div(b)
            }
        },
        |a: i32, b: i32| {
            if b == 0 {
                0
            } else {
                a.wrapping_div(b)
            }
        },
        |a: i64, b: i64| {
            if b == 0 {
                0
            } else {
                a.wrapping_div(b)
            }
        },
    )
}

// ---------------------------------------------------------------------------
// MUL (with shift for integer types)
// ---------------------------------------------------------------------------

pub fn eval_mul(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // MUL in TOSA has a third input: shift (rank-1 tensor with 1 element for integer types)
    // For floating point, shift is 0 (no shift applied)
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    // Read shift from third input if present
    let shift: i32 = if op.inputs.len() > 2 {
        let shift_tensor = state.get_tensor(&op.inputs[2])?;
        match &shift_tensor.data {
            TensorData::I8(v) => v.first().copied().unwrap_or(0) as i32,
            TensorData::I16(v) => v.first().copied().unwrap_or(0) as i32,
            TensorData::I32(v) => v.first().copied().unwrap_or(0),
            _ => 0,
        }
    } else {
        0
    };

    let data = match (&a.data, &b.data) {
        (TensorData::F32(av), TensorData::F32(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                result.push(av[ai] * bv[bi]);
            }
            TensorData::F32(result)
        }
        (TensorData::F16(av), TensorData::F16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let r = av[ai].to_f32() * bv[bi].to_f32();
                result.push(half::f16::from_f32(r));
            }
            TensorData::F16(result)
        }
        (TensorData::Bf16(av), TensorData::Bf16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let r = av[ai].to_f32() * bv[bi].to_f32();
                result.push(half::bf16::from_f32(r));
            }
            TensorData::Bf16(result)
        }
        (TensorData::I8(av), TensorData::I8(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let product = (av[ai] as i32) * (bv[bi] as i32);
                let shifted = if shift > 0 {
                    let round = 1i32 << (shift - 1);
                    (product + round) >> shift
                } else {
                    product
                };
                result.push(shifted as i8);
            }
            TensorData::I8(result)
        }
        (TensorData::I16(av), TensorData::I16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let product = (av[ai] as i64) * (bv[bi] as i64);
                let shifted = if shift > 0 {
                    let round = 1i64 << (shift - 1);
                    ((product + round) >> shift) as i16
                } else {
                    product as i16
                };
                result.push(shifted);
            }
            TensorData::I16(result)
        }
        (TensorData::I32(av), TensorData::I32(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let product = (av[ai] as i64) * (bv[bi] as i64);
                let shifted = if shift > 0 {
                    let round = 1i64 << (shift - 1);
                    ((product + round) >> shift) as i32
                } else {
                    product as i32
                };
                result.push(shifted);
            }
            TensorData::I32(result)
        }
        (TensorData::I48(av), TensorData::I48(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let product = (av[ai] as i128) * (bv[bi] as i128);
                let shifted = if shift > 0 {
                    let round = 1i128 << (shift - 1);
                    ((product + round) >> shift) as i64
                } else {
                    product as i64
                };
                result.push(shifted);
            }
            TensorData::I48(result)
        }
        (TensorData::I64(av), TensorData::I64(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let product = (av[ai] as i128) * (bv[bi] as i128);
                let shifted = if shift > 0 {
                    let round = 1i128 << (shift - 1);
                    ((product + round) >> shift) as i64
                } else {
                    product as i64
                };
                result.push(shifted);
            }
            TensorData::I64(result)
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "MUL: unsupported dtype combination {:?} and {:?}",
                a.data.dtype(),
                b.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

// ---------------------------------------------------------------------------
// ARITHMETIC_RIGHT_SHIFT (with optional rounding)
// ---------------------------------------------------------------------------

pub fn eval_arithmetic_right_shift(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let round = match &op.attribute {
        OpAttribute::ArithmeticRightShift { round } => *round,
        _ => false,
    };

    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    let data = match (&a.data, &b.data) {
        (TensorData::I8(av), TensorData::I8(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let value = av[ai] as i32;
                let shift = (bv[bi] as i32).clamp(0, 7);
                let shifted = if round && shift > 0 {
                    (value + (1 << (shift - 1))) >> shift
                } else {
                    value >> shift
                };
                result.push(shifted as i8);
            }
            TensorData::I8(result)
        }
        (TensorData::I16(av), TensorData::I16(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let value = av[ai] as i32;
                let shift = (bv[bi] as i32).clamp(0, 15);
                let shifted = if round && shift > 0 {
                    (value + (1 << (shift - 1))) >> shift
                } else {
                    value >> shift
                };
                result.push(shifted as i16);
            }
            TensorData::I16(result)
        }
        (TensorData::I32(av), TensorData::I32(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let value = av[ai] as i64;
                let shift = bv[bi].clamp(0, 31) as i64;
                let shifted = if round && shift > 0 {
                    (value + (1i64 << (shift - 1))) >> shift
                } else {
                    value >> shift
                };
                result.push(shifted as i32);
            }
            TensorData::I32(result)
        }
        (TensorData::I48(av), TensorData::I48(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let value = av[ai] as i128;
                let shift = bv[bi].clamp(0, 47) as u32;
                let shifted = if round && shift > 0 {
                    (value + (1i128 << (shift - 1))) >> shift
                } else {
                    value >> shift
                };
                result.push(shifted as i64);
            }
            TensorData::I48(result)
        }
        (TensorData::I64(av), TensorData::I64(bv)) => {
            let mut result = Vec::with_capacity(out_count);
            for flat in 0..out_count {
                let nd = flat_to_nd(flat, &out_shape);
                let ai = broadcast_index(&nd, &a.shape, &out_shape);
                let bi = broadcast_index(&nd, &b.shape, &out_shape);
                let value = av[ai] as i128;
                let shift = bv[bi].clamp(0, 63) as u32;
                let shifted = if round && shift > 0 {
                    (value + (1i128 << (shift - 1))) >> shift
                } else {
                    value >> shift
                };
                result.push(shifted as i64);
            }
            TensorData::I64(result)
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "ARITHMETIC_RIGHT_SHIFT: unsupported dtype combination {:?} and {:?}",
                a.data.dtype(),
                b.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

// ---------------------------------------------------------------------------
// LOGICAL ops (Bool tensors)
// ---------------------------------------------------------------------------

pub fn eval_logical_and(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_bool(state, op, |a, b| a && b)
}

pub fn eval_logical_or(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_bool(state, op, |a, b| a || b)
}

pub fn eval_logical_xor(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_bool(state, op, |a, b| a ^ b)
}

// ---------------------------------------------------------------------------
// LOGICAL_LEFT_SHIFT / LOGICAL_RIGHT_SHIFT (integer shift ops)
// ---------------------------------------------------------------------------

pub fn eval_logical_left_shift(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_int_only(
        state,
        op,
        |a: i8, b: i8| {
            let shift = (b as u32) & 7;
            ((a as u8).wrapping_shl(shift)) as i8
        },
        |a: i16, b: i16| {
            let shift = (b as u32) & 15;
            ((a as u16).wrapping_shl(shift)) as i16
        },
        |a: i32, b: i32| {
            let shift = (b as u32) & 31;
            ((a as u32).wrapping_shl(shift)) as i32
        },
        |a: i64, b: i64| {
            let shift = (b as u32) & 63;
            ((a as u64).wrapping_shl(shift)) as i64
        },
    )
}

pub fn eval_logical_right_shift(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_int_only(
        state,
        op,
        |a: i8, b: i8| {
            let shift = (b as u32) & 7;
            ((a as u8).wrapping_shr(shift)) as i8
        },
        |a: i16, b: i16| {
            let shift = (b as u32) & 15;
            ((a as u16).wrapping_shr(shift)) as i16
        },
        |a: i32, b: i32| {
            let shift = (b as u32) & 31;
            ((a as u32).wrapping_shr(shift)) as i32
        },
        |a: i64, b: i64| {
            let shift = (b as u32) & 63;
            ((a as u64).wrapping_shr(shift)) as i64
        },
    )
}

// ---------------------------------------------------------------------------
// BITWISE ops (integer tensors)
// ---------------------------------------------------------------------------

pub fn eval_bitwise_and(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_int_only(
        state,
        op,
        |a: i8, b: i8| a & b,
        |a: i16, b: i16| a & b,
        |a: i32, b: i32| a & b,
        |a: i64, b: i64| a & b,
    )
}

pub fn eval_bitwise_or(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_int_only(
        state,
        op,
        |a: i8, b: i8| a | b,
        |a: i16, b: i16| a | b,
        |a: i32, b: i32| a | b,
        |a: i64, b: i64| a | b,
    )
}

pub fn eval_bitwise_xor(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_int_only(
        state,
        op,
        |a: i8, b: i8| a ^ b,
        |a: i16, b: i16| a ^ b,
        |a: i32, b: i32| a ^ b,
        |a: i64, b: i64| a ^ b,
    )
}
