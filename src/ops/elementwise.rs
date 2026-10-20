//! Element-wise operators: unary, binary, comparison, SELECT, activation
//! functions and TABLE.
//!
//! Operands are decoded into one of three lanes -- `bool`, `i64` (every
//! integer type) or `f32` (every float type up to FP32, exactly) -- computed
//! there, and re-encoded into the output type. Float results are rounded to
//! the output type after each operation, which is how the reference model
//! evaluates FP16/BF16 (`fpTrunc`).
//!
//! Binary and ternary operands must have equal ranks; a dimension of size 1
//! broadcasts against any size.
//!
//! Integer overflow, which the specification leaves unpredictable (REQUIRE),
//! wraps. Conditions with no meaningful result at all (division by zero,
//! out-of-range shift amounts) are reported as errors.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, NanMode, OpAttribute, Operator};
use crate::ops::common::{
    check_declared_output, check_nan_mode, expect_operands, float_data, float_values, int_data, int_range, int_values,
    invalid, nan_max, nan_min, round_to, single_value,
};
use crate::ops::math;
use crate::tensor::{Tensor, TensorData};

/// Decoded element values.
enum Lane {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    Float(Vec<f32>),
}

fn lane(op: &str, data: &TensorData) -> Result<Lane> {
    if let TensorData::Bool(v) = data {
        return Ok(Lane::Bool(v.clone()));
    }
    if let Some(v) = int_values(data) {
        return Ok(Lane::Int(v));
    }
    if let Some(v) = float_values(data) {
        return Ok(Lane::Float(v));
    }
    Err(Error::DTypeMismatch(format!("{op}: unsupported dtype {:?}", data.dtype())))
}

fn require_dtype(op: &str, dtype: DType, allowed: &[DType]) -> Result<()> {
    if allowed.contains(&dtype) {
        Ok(())
    } else {
        Err(Error::DTypeMismatch(format!("{op}: unsupported dtype {dtype:?}")))
    }
}

const FLOATS: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const INTS: &[DType] = &[DType::I8, DType::I16, DType::I32, DType::I64];
const I32_I64: &[DType] = &[DType::I32, DType::I64];
const BOOL: &[DType] = &[DType::Bool];

/// Shape resulting from broadcasting `shapes` against each other.
fn broadcast_shape(op: &str, shapes: &[&[usize]]) -> Result<Vec<usize>> {
    let rank = shapes[0].len();
    if shapes.iter().any(|s| s.len() != rank) {
        return Err(Error::ShapeMismatch(format!("{op}: operand ranks differ: {shapes:?}")));
    }
    (0..rank)
        .map(|d| {
            let mut dim = 1;
            for s in shapes {
                if s[d] != 1 {
                    if dim != 1 && dim != s[d] {
                        return Err(Error::Broadcast(format!("{op}: shapes {shapes:?} are not broadcastable")));
                    }
                    dim = s[d];
                }
            }
            Ok(dim)
        })
        .collect()
}

/// `values` (of shape `shape`) broadcast to `out`, in row-major order.
fn expand<T: Copy>(values: &[T], shape: &[usize], out: &[usize]) -> Vec<T> {
    if shape == out {
        return values.to_vec();
    }
    let count: usize = out.iter().product();
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for d in (0..shape.len()).rev() {
        strides[d] = if shape[d] == 1 { 0 } else { stride };
        stride *= shape[d];
    }
    let mut index = vec![0; out.len()];
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(values[index.iter().zip(&strides).map(|(i, s)| i * s).sum::<usize>()]);
        for d in (0..out.len()).rev() {
            index[d] += 1;
            if index[d] < out[d] {
                break;
            }
            index[d] = 0;
        }
    }
    result
}

fn expand_lane(l: Lane, shape: &[usize], out: &[usize]) -> Lane {
    match l {
        Lane::Bool(v) => Lane::Bool(expand(&v, shape, out)),
        Lane::Int(v) => Lane::Int(expand(&v, shape, out)),
        Lane::Float(v) => Lane::Float(expand(&v, shape, out)),
    }
}

/// Check the result against the graph's declaration and store it.
fn finish(state: &mut InterpreterState, op: &Operator, name: &str, shape: Vec<usize>, data: TensorData) -> Result<()> {
    let output = &op.outputs[0];
    if let Some(declared) = check_declared_output(state, name, output, &shape)?
        && declared != data.dtype()
    {
        return Err(Error::DTypeMismatch(format!("{name}: output dtype {declared:?} should be {:?}", data.dtype())));
    }
    state.set_tensor(Tensor::new(output.clone(), shape, data));
    Ok(())
}

/// Encode a lane in `dtype`.
fn encode(dtype: DType, l: Lane) -> Result<TensorData> {
    match l {
        Lane::Bool(v) => Ok(TensorData::Bool(v)),
        Lane::Int(v) => int_data(dtype, v),
        Lane::Float(v) => float_data(dtype, v),
    }
}

fn wrap(dtype: DType, v: i64) -> i64 {
    match dtype {
        DType::I8 => v as i8 as i64,
        DType::I16 => v as i16 as i64,
        DType::I32 => v as i32 as i64,
        _ => v,
    }
}

fn bits(dtype: DType) -> i64 {
    match dtype {
        DType::I8 => 8,
        DType::I16 => 16,
        DType::I32 => 32,
        _ => 64,
    }
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

fn eval_unary(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    allowed: &[DType],
    f: impl Fn(DType, Lane) -> Result<Lane>,
) -> Result<()> {
    expect_operands(op, name, 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let dtype = input.dtype();
    require_dtype(name, dtype, allowed)?;
    let shape = input.shape.clone();
    let result = f(dtype, lane(name, &input.data)?)?;
    finish(state, op, name, shape, encode(dtype, result)?)
}

fn map_float(name: &str, dtype: DType, l: Lane, f: impl Fn(f32) -> f32) -> Result<Lane> {
    match l {
        Lane::Float(v) => Ok(Lane::Float(v.into_iter().map(|x| round_to(dtype, f(x))).collect())),
        _ => Err(Error::DTypeMismatch(format!("{name}: expected a float tensor"))),
    }
}

fn map_int(name: &str, l: Lane, f: impl Fn(i64) -> i64) -> Result<Lane> {
    match l {
        Lane::Int(v) => Ok(Lane::Int(v.into_iter().map(f).collect())),
        _ => Err(Error::DTypeMismatch(format!("{name}: expected an integer tensor"))),
    }
}

fn float_unary(state: &mut InterpreterState, op: &Operator, name: &str, f: fn(f32) -> f32) -> Result<()> {
    eval_unary(state, op, name, FLOATS, |dt, l| map_float(name, dt, l, f))
}

pub fn eval_abs(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_unary(state, op, "ABS", &[DType::F16, DType::Bf16, DType::F32, DType::I32, DType::I64], |dt, l| match l {
        Lane::Int(v) => Ok(Lane::Int(v.into_iter().map(|x| wrap(dt, x.wrapping_abs())).collect())),
        l => map_float("ABS", dt, l, f32::abs),
    })
}

pub fn eval_bitwise_not(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_unary(state, op, "BITWISE_NOT", INTS, |_, l| map_int("BITWISE_NOT", l, |x| !x))
}

pub fn eval_ceil(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "CEIL", f32::ceil)
}

pub fn eval_floor(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "FLOOR", f32::floor)
}

pub fn eval_clz(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_unary(state, op, "CLZ", I32_I64, |dt, l| {
        map_int(
            "CLZ",
            l,
            |x| if dt == DType::I32 { (x as u32).leading_zeros() } else { (x as u64).leading_zeros() } as i64,
        )
    })
}

pub fn eval_exp(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "EXP", math::expf)
}

pub fn eval_log(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "LOG", libm::logf)
}

pub fn eval_sin(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "SIN", libm::sinf)
}

pub fn eval_cos(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "COS", libm::cosf)
}

pub fn eval_reciprocal(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    // `1.0 / a` is evaluated in double precision, then narrowed via f32.
    float_unary(state, op, "RECIPROCAL", |x| (1.0 / x as f64) as f32)
}

pub fn eval_rsqrt(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "RSQRT", |x| (1.0 / x.sqrt() as f64) as f32)
}

pub fn eval_erf(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "ERF", |x| libm::erf(x as f64) as f32)
}

pub fn eval_sigmoid(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    float_unary(state, op, "SIGMOID", |x| 1.0 / (1.0 + math::expf(-x)))
}

pub fn eval_tanh(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_unary(state, op, "TANH", FLOATS, |dt, l| {
        // (1 - e^-2x) / (1 + e^-2x), with numerator and denominator clamped to
        // the finite range of the data type; zeros keep their sign.
        let max = match dt {
            DType::F16 => 65504.0,
            DType::Bf16 => f32::from_bits(0x7f7f_0000),
            _ => f32::MAX,
        };
        map_float("TANH", dt, l, |x| {
            if x == 0.0 {
                return x;
            }
            let e = math::expf(-2.0 * x);
            (1.0 - e).clamp(-max, max) / (1.0 + e).clamp(-max, max)
        })
    })
}

pub fn eval_logical_not(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_unary(state, op, "LOGICAL_NOT", BOOL, |_, l| match l {
        Lane::Bool(v) => Ok(Lane::Bool(v.into_iter().map(|b| !b).collect())),
        _ => unreachable!(),
    })
}

pub fn eval_negate(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "NEGATE";
    expect_operands(op, OP, 3, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let dtype = input.dtype();
    require_dtype(OP, dtype, &[DType::F16, DType::Bf16, DType::F32, DType::I8, DType::I16, DType::I32, DType::I64])?;
    let in_zp = single_value(OP, "input1_zp", state.get_tensor(&op.inputs[1])?)?;
    let out_zp = single_value(OP, "output_zp", state.get_tensor(&op.inputs[2])?)?;
    if in_zp.dtype() != dtype || out_zp.dtype() != dtype {
        return Err(Error::DTypeMismatch(format!("{OP}: zero points must have the input dtype {dtype:?}")));
    }
    let shape = input.shape.clone();
    let result = match lane(OP, &input.data)? {
        Lane::Int(v) => {
            let (izp, ozp) = (int_values(&in_zp).unwrap()[0], int_values(&out_zp).unwrap()[0]);
            if dtype != DType::I8 && (izp != 0 || ozp != 0) {
                return Err(invalid(OP, "zero points must be zero for non-int8 data"));
            }
            let (lo, hi) = int_range(dtype);
            Lane::Int(v.into_iter().map(|x| (ozp.wrapping_sub(x.wrapping_sub(izp))).clamp(lo, hi)).collect())
        }
        l => {
            if float_values(&in_zp).unwrap()[0] != 0.0 || float_values(&out_zp).unwrap()[0] != 0.0 {
                return Err(invalid(OP, "zero points must be zero for floating-point data"));
            }
            map_float(OP, dtype, l, |x| -x)?
        }
    };
    finish(state, op, OP, shape, encode(dtype, result)?)
}

/// CLAMP's min/max attribute values, serialized in the input type.
fn clamp_bounds(op: &str, dtype: DType, min: &[u8], max: &[u8]) -> Result<(TensorData, TensorData)> {
    let decode = |b: &[u8]| TensorData::from_raw_bytes(dtype, b, 1).map_err(|_| invalid(op, "malformed min/max"));
    Ok((decode(min)?, decode(max)?))
}

pub fn eval_clamp(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CLAMP";
    let OpAttribute::Clamp { min_val, max_val, nan_mode } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    let nan_mode = check_nan_mode(OP, *nan_mode)?;
    let dtype = state.get_tensor(op.inputs.first().map_or("", String::as_str))?.dtype();
    let (min, max) = clamp_bounds(OP, dtype, min_val, max_val)?;
    eval_unary(state, op, OP, &[DType::F16, DType::Bf16, DType::F32, DType::I8, DType::I16], |dt, l| match l {
        Lane::Int(v) => {
            let (lo, hi) = (int_values(&min).unwrap()[0], int_values(&max).unwrap()[0]);
            if hi < lo {
                return Err(invalid(OP, "max_val must not be smaller than min_val"));
            }
            Ok(Lane::Int(v.into_iter().map(|x| x.max(lo).min(hi)).collect()))
        }
        l => {
            let (lo, hi) = (float_values(&min).unwrap()[0], float_values(&max).unwrap()[0]);
            if lo.is_nan() || hi.is_nan() || hi < lo {
                return Err(invalid(OP, "min_val/max_val must be ordered and not NaN"));
            }
            map_float(OP, dt, l, |x| nan_min(nan_max(x, lo, nan_mode), hi, nan_mode))
        }
    })
}

// ---------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------

/// Broadcast operands of a binary op: dtype, output shape and expanded lanes.
fn binary_operands(state: &InterpreterState, op: &Operator, name: &str) -> Result<(DType, Vec<usize>, Lane, Lane)> {
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch(format!("{name}: input dtypes {:?} and {:?} differ", a.dtype(), b.dtype())));
    }
    let shape = broadcast_shape(name, &[&a.shape, &b.shape])?;
    let la = expand_lane(lane(name, &a.data)?, &a.shape, &shape);
    let lb = expand_lane(lane(name, &b.data)?, &b.shape, &shape);
    Ok((a.dtype(), shape, la, lb))
}

/// A binary op whose output has the input type.
fn eval_binary(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    allowed: &[DType],
    int_op: Option<&dyn Fn(DType, i64, i64) -> Result<i64>>,
    float_op: Option<&dyn Fn(f32, f32) -> f32>,
    bool_op: Option<&dyn Fn(bool, bool) -> bool>,
) -> Result<()> {
    expect_operands(op, name, 2, 1)?;
    let (dtype, shape, a, b) = binary_operands(state, op, name)?;
    require_dtype(name, dtype, allowed)?;
    let result = match (a, b) {
        (Lane::Int(a), Lane::Int(b)) if int_op.is_some() => {
            let f = int_op.unwrap();
            Lane::Int(a.into_iter().zip(b).map(|(x, y)| f(dtype, x, y)).collect::<Result<_>>()?)
        }
        (Lane::Float(a), Lane::Float(b)) if float_op.is_some() => {
            let f = float_op.unwrap();
            Lane::Float(a.into_iter().zip(b).map(|(x, y)| round_to(dtype, f(x, y))).collect())
        }
        (Lane::Bool(a), Lane::Bool(b)) if bool_op.is_some() => {
            let f = bool_op.unwrap();
            Lane::Bool(a.into_iter().zip(b).map(|(x, y)| f(x, y)).collect())
        }
        _ => return Err(Error::DTypeMismatch(format!("{name}: unsupported dtype {dtype:?}"))),
    };
    finish(state, op, name, shape, encode(dtype, result)?)
}

const ARITH: &[DType] = &[DType::F16, DType::Bf16, DType::F32, DType::I32, DType::I64];

pub fn eval_add(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "ADD", ARITH, Some(&|dt, a, b| Ok(wrap(dt, a.wrapping_add(b)))), Some(&|a, b| a + b), None)
}

pub fn eval_sub(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "SUB", ARITH, Some(&|dt, a, b| Ok(wrap(dt, a.wrapping_sub(b)))), Some(&|a, b| a - b), None)
}

fn nan_mode_attr(op: &Operator, name: &str) -> Result<NanMode> {
    match op.attribute {
        OpAttribute::NanPropagation { nan_mode } => check_nan_mode(name, nan_mode),
        _ => Err(invalid(name, "missing attribute")),
    }
}

pub fn eval_maximum(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let mode = nan_mode_attr(op, "MAXIMUM")?;
    eval_binary(state, op, "MAXIMUM", ARITH, Some(&|_, a, b| Ok(a.max(b))), Some(&|a, b| nan_max(a, b, mode)), None)
}

pub fn eval_minimum(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let mode = nan_mode_attr(op, "MINIMUM")?;
    eval_binary(state, op, "MINIMUM", ARITH, Some(&|_, a, b| Ok(a.min(b))), Some(&|a, b| nan_min(a, b, mode)), None)
}

pub fn eval_pow(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "POW", FLOATS, None, Some(&|x, y| math::expf(y * libm::logf(x))), None)
}

pub fn eval_intdiv(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(
        state,
        op,
        "INTDIV",
        I32_I64,
        Some(&|dt, a, b| {
            if b == 0 {
                return Err(invalid("INTDIV", "division by zero"));
            }
            Ok(wrap(dt, a.wrapping_div(b)))
        }),
        None,
        None,
    )
}

fn check_shift(name: &str, dtype: DType, shift: i64) -> Result<u32> {
    if shift < 0 || shift >= bits(dtype) {
        return Err(invalid(name, format!("shift {shift} out of range for {dtype:?}")));
    }
    Ok(shift as u32)
}

pub fn eval_arithmetic_right_shift(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "ARITHMETIC_RIGHT_SHIFT";
    let OpAttribute::ArithmeticRightShift { round } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    eval_binary(
        state,
        op,
        OP,
        INTS,
        Some(&|dt, a, b| {
            let s = check_shift(OP, dt, b)?;
            let mut r = a >> s;
            if round && s > 0 && (a >> (s - 1)) & 1 != 0 {
                r += 1;
            }
            Ok(r)
        }),
        None,
        None,
    )
}

pub fn eval_logical_left_shift(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "LOGICAL_LEFT_SHIFT";
    eval_binary(state, op, OP, INTS, Some(&|dt, a, b| Ok(wrap(dt, a << check_shift(OP, dt, b)?))), None, None)
}

pub fn eval_logical_right_shift(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "LOGICAL_RIGHT_SHIFT";
    eval_binary(
        state,
        op,
        OP,
        INTS,
        Some(&|dt, a, b| {
            let s = check_shift(OP, dt, b)?;
            let width = bits(dt) as u32;
            let unsigned = (a as u64) & (u64::MAX >> (64 - width));
            Ok(wrap(dt, (unsigned >> s) as i64))
        }),
        None,
        None,
    )
}

pub fn eval_bitwise_and(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "BITWISE_AND", INTS, Some(&|_, a, b| Ok(a & b)), None, None)
}

pub fn eval_bitwise_or(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "BITWISE_OR", INTS, Some(&|_, a, b| Ok(a | b)), None, None)
}

pub fn eval_bitwise_xor(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "BITWISE_XOR", INTS, Some(&|_, a, b| Ok(a ^ b)), None, None)
}

pub fn eval_logical_and(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "LOGICAL_AND", BOOL, None, None, Some(&|a, b| a && b))
}

pub fn eval_logical_or(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "LOGICAL_OR", BOOL, None, None, Some(&|a, b| a || b))
}

pub fn eval_logical_xor(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary(state, op, "LOGICAL_XOR", BOOL, None, None, Some(&|a, b| a ^ b))
}

/// MUL: inputs a, b and an INT8 `shift` [1]. INT8/INT16 inputs produce INT32;
/// INT32 products are rounded and shifted right by `shift` (which must be
/// zero for every other type).
pub fn eval_mul(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "MUL";
    expect_operands(op, OP, 3, 1)?;
    let shift = single_value(OP, "shift", state.get_tensor(&op.inputs[2])?)?;
    if shift.dtype() != DType::I8 {
        return Err(Error::DTypeMismatch(format!("{OP}: shift must be INT8, got {:?}", shift.dtype())));
    }
    let shift = int_values(&shift).unwrap()[0];
    let (dtype, shape, a, b) = binary_operands(state, op, OP)?;
    require_dtype(OP, dtype, &[DType::F16, DType::Bf16, DType::F32, DType::I8, DType::I16, DType::I32, DType::I64])?;
    if dtype != DType::I32 && shift != 0 {
        return Err(invalid(OP, "shift must be zero for non-INT32 inputs"));
    }
    let data = match (a, b) {
        (Lane::Float(a), Lane::Float(b)) => {
            float_data(dtype, a.into_iter().zip(b).map(|(x, y)| round_to(dtype, x * y)).collect())?
        }
        (Lane::Int(a), Lane::Int(b)) => match dtype {
            DType::I64 => TensorData::I64(a.into_iter().zip(b).map(|(x, y)| x.wrapping_mul(y)).collect()),
            DType::I32 => {
                if !(0..=63).contains(&shift) {
                    return Err(invalid(OP, format!("shift {shift} out of range")));
                }
                // In i128: rounding (x * y) with shift 63 overflows i64.
                let product = |x: i64, y: i64| {
                    let p = x as i128 * y as i128;
                    if shift > 0 { (p + (1i128 << (shift - 1))) >> shift } else { p }
                };
                TensorData::I32(a.into_iter().zip(b).map(|(x, y)| product(x, y) as i32).collect())
            }
            _ => TensorData::I32(a.into_iter().zip(b).map(|(x, y)| (x * y) as i32).collect()),
        },
        _ => unreachable!("require_dtype admits only numeric types"),
    };
    finish(state, op, OP, shape, data)
}

// ---------------------------------------------------------------------------
// Comparison and SELECT
// ---------------------------------------------------------------------------

fn eval_compare(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    int_cmp: fn(&i64, &i64) -> bool,
    float_cmp: fn(&f32, &f32) -> bool,
) -> Result<()> {
    expect_operands(op, name, 2, 1)?;
    let (dtype, shape, a, b) = binary_operands(state, op, name)?;
    require_dtype(name, dtype, ARITH)?;
    let out = match (a, b) {
        (Lane::Int(a), Lane::Int(b)) => a.iter().zip(&b).map(|(x, y)| int_cmp(x, y)).collect(),
        (Lane::Float(a), Lane::Float(b)) => a.iter().zip(&b).map(|(x, y)| float_cmp(x, y)).collect(),
        _ => unreachable!("require_dtype admits only numeric types"),
    };
    finish(state, op, name, shape, TensorData::Bool(out))
}

pub fn eval_equal(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_compare(state, op, "EQUAL", i64::eq, f32::eq)
}

pub fn eval_greater(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_compare(state, op, "GREATER", i64::gt, f32::gt)
}

pub fn eval_greater_equal(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_compare(state, op, "GREATER_EQUAL", i64::ge, f32::ge)
}

pub fn eval_select(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "SELECT";
    expect_operands(op, OP, 3, 1)?;
    let cond = state.get_tensor(&op.inputs[0])?;
    let on_true = state.get_tensor(&op.inputs[1])?;
    let on_false = state.get_tensor(&op.inputs[2])?;
    let TensorData::Bool(c) = &cond.data else {
        return Err(Error::DTypeMismatch(format!("{OP}: condition must be BOOL, got {:?}", cond.dtype())));
    };
    let dtype = on_true.dtype();
    if on_false.dtype() != dtype {
        return Err(Error::DTypeMismatch(format!("{OP}: input dtypes {dtype:?} and {:?} differ", on_false.dtype())));
    }
    require_dtype(
        OP,
        dtype,
        &[DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64, DType::F16, DType::Bf16, DType::F32],
    )?;
    let shape = broadcast_shape(OP, &[&cond.shape, &on_true.shape, &on_false.shape])?;
    let c = expand(c, &cond.shape, &shape);
    let t = expand_lane(lane(OP, &on_true.data)?, &on_true.shape, &shape);
    let f = expand_lane(lane(OP, &on_false.data)?, &on_false.shape, &shape);
    fn choose<T>(c: &[bool], t: Vec<T>, f: Vec<T>) -> Vec<T> {
        c.iter().zip(t.into_iter().zip(f)).map(|(&c, (t, f))| if c { t } else { f }).collect()
    }
    let out = match (t, f) {
        (Lane::Bool(t), Lane::Bool(f)) => Lane::Bool(choose(&c, t, f)),
        (Lane::Int(t), Lane::Int(f)) => Lane::Int(choose(&c, t, f)),
        (Lane::Float(t), Lane::Float(f)) => Lane::Float(choose(&c, t, f)),
        _ => unreachable!("both inputs have the same dtype"),
    };
    finish(state, op, OP, shape, encode(dtype, out)?)
}

// ---------------------------------------------------------------------------
// TABLE
// ---------------------------------------------------------------------------

/// TABLE: INT8 input with an INT8 [256] table gives INT8; INT16 input with an
/// INT16 [513] table gives an INT32 interpolated value with 7 fraction bits.
pub fn eval_table(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "TABLE";
    expect_operands(op, OP, 2, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let table = state.get_tensor(&op.inputs[1])?;
    let shape = input.shape.clone();
    let data = match (&input.data, &table.data) {
        (TensorData::I8(x), TensorData::I8(t)) => {
            if t.len() != 256 {
                return Err(invalid(OP, "INT8 table must have 256 entries"));
            }
            TensorData::I8(x.iter().map(|&v| t[(v as i32 + 128) as usize]).collect())
        }
        (TensorData::I16(x), TensorData::I16(t)) => {
            if t.len() != 513 {
                return Err(invalid(OP, "INT16 table must have 513 entries"));
            }
            TensorData::I32(
                x.iter()
                    .map(|&v| {
                        let v = v as i32;
                        let index = ((v >> 7) + 256) as usize;
                        let (base, next) = (t[index] as i32, t[index + 1] as i32);
                        (base << 7) + (next - base) * (v & 0x7f)
                    })
                    .collect(),
            )
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "{OP}: unsupported input/table dtypes {:?}/{:?}",
                input.dtype(),
                table.dtype()
            )));
        }
    };
    finish(state, op, OP, shape, data)
}
