//! Validation and element-conversion helpers shared by operator implementations.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, NanMode, Operator};
use crate::tensor::{Tensor, TensorData};

/// Error for a graph that violates an operator's ERROR_IF conditions.
pub fn invalid(op: &str, msg: impl std::fmt::Display) -> Error {
    Error::GraphError(format!("{op}: {msg}"))
}

pub fn expect_operands(op: &Operator, name: &str, inputs: usize, outputs: usize) -> Result<()> {
    if op.inputs.len() != inputs || op.outputs.len() != outputs {
        return Err(invalid(
            name,
            format!("expected {inputs} inputs and {outputs} outputs, got {} and {}", op.inputs.len(), op.outputs.len()),
        ));
    }
    Ok(())
}

pub fn expect_rank(op: &str, what: &str, t: &Tensor, rank: usize) -> Result<()> {
    if t.shape.len() != rank {
        return Err(Error::ShapeMismatch(format!("{op}: {what} must have rank {rank}, got shape {:?}", t.shape)));
    }
    Ok(())
}

/// The element data of a rank-1, single-element operand such as a zero point.
pub fn single_value(op: &str, what: &str, t: &Tensor) -> Result<TensorData> {
    expect_rank(op, what, t, 1)?;
    if t.data.len() != 1 {
        return Err(Error::ShapeMismatch(format!(
            "{op}: {what} must have exactly one element, got shape {:?}",
            t.shape
        )));
    }
    Ok(t.data.clone())
}

/// Check a computed output against the shape and dtype declared in the graph.
///
/// Returns the declared dtype, or `None` if the graph has no definition for
/// the output tensor.
pub fn check_declared_output(state: &InterpreterState, op: &str, name: &str, shape: &[usize]) -> Result<Option<DType>> {
    let Some(def) = state.find_tensor_def(name) else {
        return Ok(None);
    };
    let static_shape = def.shape.iter().all(|&d| d >= 0);
    if static_shape {
        let declared: Vec<usize> = def.shape.iter().map(|&d| d as usize).collect();
        if declared != shape {
            return Err(Error::ShapeMismatch(format!(
                "{op}: output shape {shape:?} does not match declared shape {declared:?}"
            )));
        }
    }
    Ok(Some(def.dtype))
}

/// Integer element values widened to i64. `None` for non-integer data.
pub fn int_values(data: &TensorData) -> Option<Vec<i64>> {
    Some(match data {
        TensorData::I4(v) | TensorData::I8(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I16(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I48(v) | TensorData::I64(v) => v.clone(),
        _ => return None,
    })
}

/// Floating-point element values widened to f32 (exact for every format
/// narrower than f32). `None` for non-float or f64 data.
pub fn float_values(data: &TensorData) -> Option<Vec<f32>> {
    Some(match data {
        TensorData::Fp8E4M3(v) => v.iter().map(|x| x.to_f32()).collect(),
        TensorData::Fp8E5M2(v) => v.iter().map(|x| x.to_f32()).collect(),
        TensorData::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| x.to_f32()).collect(),
        TensorData::F32(v) => v.clone(),
        _ => return None,
    })
}

/// Pack f32 values into tensor data of a float dtype, rounding to nearest even.
pub fn float_data(dtype: DType, values: Vec<f32>) -> Result<TensorData> {
    use crate::float8::{Fp8E4M3, Fp8E5M2};
    Ok(match dtype {
        DType::F32 => TensorData::F32(values),
        DType::F16 => TensorData::F16(values.into_iter().map(half::f16::from_f32).collect()),
        DType::Bf16 => TensorData::Bf16(values.into_iter().map(half::bf16::from_f32).collect()),
        DType::Fp8E4M3 => TensorData::Fp8E4M3(values.into_iter().map(Fp8E4M3::from_f32).collect()),
        DType::Fp8E5M2 => TensorData::Fp8E5M2(values.into_iter().map(Fp8E5M2::from_f32).collect()),
        other => return Err(Error::DTypeMismatch(format!("{other:?} is not a float dtype"))),
    })
}

/// Round an f32 to the nearest value representable in `dtype`.
pub fn round_to(dtype: DType, x: f32) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_f32(x).to_f32(),
        DType::Bf16 => half::bf16::from_f32(x).to_f32(),
        DType::Fp8E4M3 => crate::float8::Fp8E4M3::from_f32(x).to_f32(),
        DType::Fp8E5M2 => crate::float8::Fp8E5M2::from_f32(x).to_f32(),
        _ => x,
    }
}

/// Pack integer values into tensor data of an integer dtype, truncating
/// (wrapping) values that do not fit.
pub fn int_data(dtype: DType, values: Vec<i64>) -> Result<TensorData> {
    Ok(match dtype {
        DType::I4 => TensorData::I4(values.into_iter().map(|v| ((v as i8) << 4) >> 4).collect()),
        DType::I8 => TensorData::I8(values.into_iter().map(|v| v as i8).collect()),
        DType::I16 => TensorData::I16(values.into_iter().map(|v| v as i16).collect()),
        DType::I32 => TensorData::I32(values.into_iter().map(|v| v as i32).collect()),
        DType::I48 => TensorData::I48(values.into_iter().map(|v| (v << 16) >> 16).collect()),
        DType::I64 | DType::Shape => TensorData::I64(values),
        other => return Err(Error::DTypeMismatch(format!("{other:?} is not an integer dtype"))),
    })
}

/// Inclusive value range of a signed integer dtype.
pub fn int_range(dtype: DType) -> (i64, i64) {
    match dtype {
        DType::I4 => (-8, 7),
        DType::I8 => (i8::MIN as i64, i8::MAX as i64),
        DType::I16 => (i16::MIN as i64, i16::MAX as i64),
        DType::I32 => (i32::MIN as i64, i32::MAX as i64),
        DType::I48 => (-(1 << 47), (1 << 47) - 1),
        _ => (i64::MIN, i64::MAX),
    }
}

/// Lowest value of a float dtype: -infinity, or -448 for FP8E4M3, which has
/// no infinities.
pub fn float_lowest(dtype: DType) -> f32 {
    if dtype == DType::Fp8E4M3 { -448.0 } else { f32::NEG_INFINITY }
}

/// Highest value of a float dtype (see [`float_lowest`]).
pub fn float_highest(dtype: DType) -> f32 {
    -float_lowest(dtype)
}

/// Reject graphs with a missing or unknown NaN propagation mode.
pub fn check_nan_mode(op: &str, mode: NanMode) -> Result<NanMode> {
    if mode == NanMode::Unknown {
        return Err(invalid(op, "nan_mode must be PROPAGATE or IGNORE"));
    }
    Ok(mode)
}

/// The reference model's `applyMax`: ties keep `a`.
pub fn nan_max(a: f32, b: f32, mode: NanMode) -> f32 {
    if a.is_nan() || b.is_nan() {
        return nan_pick(a, b, mode);
    }
    if a >= b { a } else { b }
}

/// The reference model's `applyMin`: ties keep `b`.
pub fn nan_min(a: f32, b: f32, mode: NanMode) -> f32 {
    if a.is_nan() || b.is_nan() {
        return nan_pick(a, b, mode);
    }
    if a < b { a } else { b }
}

fn nan_pick(a: f32, b: f32, mode: NanMode) -> f32 {
    match mode {
        NanMode::Ignore if a.is_nan() => b,
        NanMode::Ignore => a,
        _ => f32::NAN,
    }
}
