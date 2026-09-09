use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::{Tensor, TensorData};

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Dispatch a unary element-wise operation over all float types.
/// F16 and BF16 are promoted to f32, computed, then truncated back.
macro_rules! dispatch_unary_float {
    ($state:expr, $op:expr, $f32_fn:expr) => {{
        let input = $state.get_tensor(&$op.inputs[0])?;
        let output_name = &$op.outputs[0];
        let shape = input.shape.clone();
        let f = $f32_fn;

        let data = match &input.data {
            TensorData::F32(v) => TensorData::F32(v.iter().map(|&x| f(x)).collect()),
            TensorData::F16(v) => TensorData::F16(
                v.iter()
                    .map(|&x| half::f16::from_f32(f(x.to_f32())))
                    .collect(),
            ),
            TensorData::Bf16(v) => TensorData::Bf16(
                v.iter()
                    .map(|&x| half::bf16::from_f32(f(x.to_f32())))
                    .collect(),
            ),
            TensorData::F64(v) => TensorData::F64(v.iter().map(|&x| f(x as f32) as f64).collect()),
            _ => {
                return Err(Error::DTypeMismatch(format!(
                    "unary float op {:?}: unsupported dtype {:?}",
                    $op.op,
                    input.data.dtype(),
                )));
            }
        };

        $state.set_tensor(Tensor::new(output_name.clone(), shape, data));
        Ok(())
    }};
}

/// Dispatch a unary element-wise operation over all numeric types (int + float).
/// Float types use `$f32_fn`, integer types use their respective closures.
macro_rules! dispatch_unary_all {
    ($state:expr, $op:expr, $f32_fn:expr, $i8_fn:expr, $i16_fn:expr, $i32_fn:expr, $i64_fn:expr) => {{
        let input = $state.get_tensor(&$op.inputs[0])?;
        let output_name = &$op.outputs[0];
        let shape = input.shape.clone();
        let f = $f32_fn;

        let data = match &input.data {
            TensorData::F32(v) => TensorData::F32(v.iter().map(|&x| f(x)).collect()),
            TensorData::F16(v) => TensorData::F16(
                v.iter()
                    .map(|&x| half::f16::from_f32(f(x.to_f32())))
                    .collect(),
            ),
            TensorData::Bf16(v) => TensorData::Bf16(
                v.iter()
                    .map(|&x| half::bf16::from_f32(f(x.to_f32())))
                    .collect(),
            ),
            TensorData::F64(v) => TensorData::F64(v.iter().map(|&x| f(x as f32) as f64).collect()),
            TensorData::I8(v) => {
                let op_fn = $i8_fn;
                TensorData::I8(v.iter().map(|&x| op_fn(x)).collect())
            }
            TensorData::I16(v) => {
                let op_fn = $i16_fn;
                TensorData::I16(v.iter().map(|&x| op_fn(x)).collect())
            }
            TensorData::I32(v) => {
                let op_fn = $i32_fn;
                TensorData::I32(v.iter().map(|&x| op_fn(x)).collect())
            }
            TensorData::I48(v) | TensorData::I64(v) => {
                let op_fn = $i64_fn;
                let result: Vec<i64> = v.iter().map(|&x| op_fn(x)).collect();
                // Preserve the original variant
                match &input.data {
                    TensorData::I48(_) => TensorData::I48(result),
                    _ => TensorData::I64(result),
                }
            }
            _ => {
                return Err(Error::DTypeMismatch(format!(
                    "unary op {:?}: unsupported dtype {:?}",
                    $op.op,
                    input.data.dtype(),
                )));
            }
        };

        $state.set_tensor(Tensor::new(output_name.clone(), shape, data));
        Ok(())
    }};
}

/// Dispatch a unary element-wise operation over integer types only.
macro_rules! dispatch_unary_int {
    ($state:expr, $op:expr, $i8_fn:expr, $i16_fn:expr, $i32_fn:expr, $i64_fn:expr) => {{
        let input = $state.get_tensor(&$op.inputs[0])?;
        let output_name = &$op.outputs[0];
        let shape = input.shape.clone();

        let data = match &input.data {
            TensorData::I8(v) => {
                let op_fn = $i8_fn;
                TensorData::I8(v.iter().map(|&x| op_fn(x)).collect())
            }
            TensorData::I16(v) => {
                let op_fn = $i16_fn;
                TensorData::I16(v.iter().map(|&x| op_fn(x)).collect())
            }
            TensorData::I32(v) => {
                let op_fn = $i32_fn;
                TensorData::I32(v.iter().map(|&x| op_fn(x)).collect())
            }
            TensorData::I48(v) | TensorData::I64(v) => {
                let op_fn = $i64_fn;
                let result: Vec<i64> = v.iter().map(|&x| op_fn(x)).collect();
                match &input.data {
                    TensorData::I48(_) => TensorData::I48(result),
                    _ => TensorData::I64(result),
                }
            }
            _ => {
                return Err(Error::DTypeMismatch(format!(
                    "unary int op {:?}: unsupported dtype {:?}",
                    $op.op,
                    input.data.dtype(),
                )));
            }
        };

        $state.set_tensor(Tensor::new(output_name.clone(), shape, data));
        Ok(())
    }};
}

// ---------------------------------------------------------------------------
// Helper: erf approximation (Abramowitz and Stegun 7.1.26, max error ~1.5e-7)
// ---------------------------------------------------------------------------

fn erf_approx(x: f32) -> f32 {
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;

    let sign = if x >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let a = x.abs();
    let t = 1.0 / (1.0 + P * a);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let y = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * (-a * a).exp();
    sign * y
}

// ---------------------------------------------------------------------------
// Public eval functions
// ---------------------------------------------------------------------------

pub fn eval_abs(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_all!(
        state,
        op,
        f32::abs,
        |x: i8| x.wrapping_abs(),
        |x: i16| x.wrapping_abs(),
        |x: i32| x.wrapping_abs(),
        |x: i64| x.wrapping_abs()
    )
}

pub fn eval_negate(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_all!(
        state,
        op,
        |x: f32| -x,
        |x: i8| x.wrapping_neg(),
        |x: i16| x.wrapping_neg(),
        |x: i32| x.wrapping_neg(),
        |x: i64| x.wrapping_neg()
    )
}

pub fn eval_ceil(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::ceil)
}

pub fn eval_floor(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::floor)
}

pub fn eval_exp(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::exp)
}

pub fn eval_log(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::ln)
}

pub fn eval_sin(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::sin)
}

pub fn eval_cos(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::cos)
}

pub fn eval_reciprocal(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, |x: f32| 1.0 / x)
}

pub fn eval_rsqrt(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, |x: f32| 1.0 / x.sqrt())
}

pub fn eval_erf(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, erf_approx)
}

pub fn eval_sigmoid(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, |x: f32| 1.0 / (1.0 + (-x).exp()))
}

pub fn eval_tanh(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_float!(state, op, f32::tanh)
}

pub fn eval_clz(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_int!(
        state,
        op,
        |x: i8| x.leading_zeros() as i8,
        |x: i16| x.leading_zeros() as i16,
        |x: i32| x.leading_zeros() as i32,
        |x: i64| x.leading_zeros() as i64
    )
}

pub fn eval_bitwise_not(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    dispatch_unary_int!(
        state,
        op,
        |x: i8| !x,
        |x: i16| !x,
        |x: i32| !x,
        |x: i64| !x
    )
}

pub fn eval_logical_not(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];
    let shape = input.shape.clone();

    let data = match &input.data {
        TensorData::Bool(v) => TensorData::Bool(v.iter().map(|&x| !x).collect()),
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "logical_not: expected Bool tensor, got {:?}",
                input.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), shape, data));
    Ok(())
}

// ---------------------------------------------------------------------------
// CLAMP: needs min_val/max_val from OpAttribute::Clamp (raw bytes per dtype)
// ---------------------------------------------------------------------------

/// Read a scalar f32 value from raw little-endian bytes.
fn read_scalar_f32(bytes: &[u8]) -> f32 {
    if bytes.len() >= 4 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        f32::NAN
    }
}

fn read_scalar_f16(bytes: &[u8]) -> half::f16 {
    if bytes.len() >= 2 {
        half::f16::from_le_bytes([bytes[0], bytes[1]])
    } else {
        half::f16::NAN
    }
}

fn read_scalar_bf16(bytes: &[u8]) -> half::bf16 {
    if bytes.len() >= 2 {
        half::bf16::from_le_bytes([bytes[0], bytes[1]])
    } else {
        half::bf16::NAN
    }
}

fn read_scalar_i8(bytes: &[u8]) -> i8 {
    if bytes.is_empty() {
        0
    } else {
        bytes[0] as i8
    }
}

fn read_scalar_i16(bytes: &[u8]) -> i16 {
    if bytes.len() >= 2 {
        i16::from_le_bytes([bytes[0], bytes[1]])
    } else {
        0
    }
}

fn read_scalar_i32(bytes: &[u8]) -> i32 {
    if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        0
    }
}

fn read_scalar_i64(bytes: &[u8]) -> i64 {
    if bytes.len() >= 8 {
        i64::from_le_bytes(bytes[..8].try_into().unwrap())
    } else {
        0
    }
}

pub fn eval_clamp(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (min_bytes, max_bytes) = match &op.attribute {
        OpAttribute::Clamp { min_val, max_val } => (min_val.as_slice(), max_val.as_slice()),
        _ => {
            return Err(Error::DTypeMismatch(
                "clamp: missing Clamp attribute".to_string(),
            ));
        }
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];
    let shape = input.shape.clone();

    let data = match &input.data {
        TensorData::F32(v) => {
            let min = read_scalar_f32(min_bytes);
            let max = read_scalar_f32(max_bytes);
            TensorData::F32(v.iter().map(|&x| x.clamp(min, max)).collect())
        }
        TensorData::F16(v) => {
            let min = read_scalar_f16(min_bytes).to_f32();
            let max = read_scalar_f16(max_bytes).to_f32();
            TensorData::F16(
                v.iter()
                    .map(|&x| half::f16::from_f32(x.to_f32().clamp(min, max)))
                    .collect(),
            )
        }
        TensorData::Bf16(v) => {
            let min = read_scalar_bf16(min_bytes).to_f32();
            let max = read_scalar_bf16(max_bytes).to_f32();
            TensorData::Bf16(
                v.iter()
                    .map(|&x| half::bf16::from_f32(x.to_f32().clamp(min, max)))
                    .collect(),
            )
        }
        TensorData::I8(v) => {
            let min = read_scalar_i8(min_bytes);
            let max = read_scalar_i8(max_bytes);
            TensorData::I8(v.iter().map(|&x| x.clamp(min, max)).collect())
        }
        TensorData::I16(v) => {
            let min = read_scalar_i16(min_bytes);
            let max = read_scalar_i16(max_bytes);
            TensorData::I16(v.iter().map(|&x| x.clamp(min, max)).collect())
        }
        TensorData::I32(v) => {
            let min = read_scalar_i32(min_bytes);
            let max = read_scalar_i32(max_bytes);
            TensorData::I32(v.iter().map(|&x| x.clamp(min, max)).collect())
        }
        TensorData::I48(v) => {
            let min = read_scalar_i64(min_bytes);
            let max = read_scalar_i64(max_bytes);
            TensorData::I48(v.iter().map(|&x| x.clamp(min, max)).collect())
        }
        TensorData::I64(v) => {
            let min = read_scalar_i64(min_bytes);
            let max = read_scalar_i64(max_bytes);
            TensorData::I64(v.iter().map(|&x| x.clamp(min, max)).collect())
        }
        _ => {
            return Err(Error::DTypeMismatch(format!(
                "clamp: unsupported dtype {:?}",
                input.data.dtype(),
            )));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), shape, data));
    Ok(())
}
