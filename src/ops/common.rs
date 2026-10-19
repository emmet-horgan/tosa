//! Validation and element-conversion helpers shared by operator implementations.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, Operator};
use crate::tensor::{Tensor, TensorData};

/// Error for a graph that violates an operator's ERROR_IF conditions.
pub fn invalid(op: &str, msg: impl std::fmt::Display) -> Error {
    Error::GraphError(format!("{op}: {msg}"))
}

pub fn expect_operands(op: &Operator, name: &str, inputs: usize, outputs: usize) -> Result<()> {
    if op.inputs.len() != inputs || op.outputs.len() != outputs {
        return Err(invalid(
            name,
            format!(
                "expected {inputs} inputs and {outputs} outputs, got {} and {}",
                op.inputs.len(),
                op.outputs.len()
            ),
        ));
    }
    Ok(())
}

pub fn expect_rank(op: &str, what: &str, t: &Tensor, rank: usize) -> Result<()> {
    if t.shape.len() != rank {
        return Err(Error::ShapeMismatch(format!(
            "{op}: {what} must have rank {rank}, got shape {:?}",
            t.shape
        )));
    }
    Ok(())
}

/// Check a computed output against the shape and dtype declared in the graph.
///
/// Returns the declared dtype, or `None` if the graph has no definition for
/// the output tensor.
pub fn check_declared_output(
    state: &InterpreterState,
    op: &str,
    name: &str,
    shape: &[usize],
) -> Result<Option<DType>> {
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
        _ => x,
    }
}
