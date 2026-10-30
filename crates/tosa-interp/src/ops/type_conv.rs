//! CAST and RESCALE.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator, RoundingMode};
use crate::ops::common::{
    check_declared_output, expect_operands, float_data, float_values, int_data, int_range, int_values, invalid,
    round_to,
};
use crate::ops::quant::apply_scale_32;
use crate::tensor::{Tensor, TensorData};

/// The reference model's CAST type pairs (excluding FP64 and block-scaled types).
fn cast_supported(from: DType, to: DType) -> bool {
    use DType::*;
    let fp8 = |t: DType| matches!(t, Fp8E4M3 | Fp8E5M2);
    match (from, to) {
        (Bool, I8 | I16 | I32 | I64 | F32) => true,
        (I8 | I16 | I32 | I64 | F32, Bool) => true,
        (I8, I16 | I32) | (I16, I8 | I32) | (I32, I8 | I16 | I64) | (I64, I32) => true,
        (I8 | I16 | I32, F16 | Bf16 | F32) => true,
        (F16 | Bf16 | F32, I8 | I16 | I32) => true,
        (F16 | Bf16, F32) | (F32, F16 | Bf16) => true,
        (F16 | Bf16 | F32, t) if fp8(t) => true,
        (f, F16 | Bf16 | F32) if fp8(f) => true,
        _ => false,
    }
}

/// Float to integer: saturate, otherwise round half to even. The bounds are
/// compared in the input's precision, as the reference does (so e.g. FP16
/// saturates INT16 only at 32768).
fn float_to_int(from: DType, to: DType, x: f32) -> i64 {
    let (lo, hi) = int_range(to);
    if x >= round_to(from, hi as f32) {
        return hi;
    }
    if x <= round_to(from, lo as f32) {
        return lo;
    }
    x.round_ties_even() as i64
}

pub fn eval_cast(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CAST";
    if let OpAttribute::Cast { input_unsigned: true } = op.attribute {
        return Err(Error::UnsupportedOp(format!("{OP} with input_unsigned")));
    }
    expect_operands(op, OP, 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let shape = input.shape.clone();
    let from = input.dtype();
    let to = check_declared_output(state, OP, &op.outputs[0], &shape)?
        .ok_or_else(|| invalid(OP, "output type must be declared"))?;
    if !cast_supported(from, to) {
        return Err(Error::DTypeMismatch(format!("{OP}: unsupported cast {from:?} -> {to:?}")));
    }

    let data = match (&input.data, to) {
        (TensorData::Bool(v), DType::F32) => TensorData::F32(v.iter().map(|&b| b as u8 as f32).collect()),
        (TensorData::Bool(v), _) => int_data(to, v.iter().map(|&b| b as i64).collect())?,
        (d, DType::Bool) => TensorData::Bool(match int_values(d) {
            Some(v) => v.iter().map(|&x| x != 0).collect(),
            None => float_values(d).unwrap().iter().map(|&x| x != 0.0).collect(),
        }),
        (d, _) if from.is_int() => {
            let v = int_values(d).unwrap();
            if to.is_int() {
                int_data(to, v)?
            } else {
                // Integers convert via FP32 (so INT32 -> FP16 rounds twice).
                float_data(to, v.into_iter().map(|x| x as f32).collect())?
            }
        }
        (d, _) => {
            let v = float_values(d).unwrap();
            if to.is_int() {
                int_data(to, v.into_iter().map(|x| float_to_int(from, to, x)).collect())?
            } else {
                float_data(to, v)?
            }
        }
    };
    state.set_tensor(Tensor::new(op.outputs[0].clone(), shape, data));
    Ok(())
}

/// The scale32 RESCALE of every element of `values`, per channel (the last
/// dimension, `channels` wide), with TOSA's single or double rounding.
/// Values the reference model would reject are handed to `slow`, which
/// produces the matching error.
#[allow(clippy::too_many_arguments)]
fn rescale_fast<I: Copy + Into<i64>, O: WrapFromI64>(
    values: &[I],
    params: &[(i64, i64, i64, i64)],
    channels: usize,
    in_zp: i64,
    out_zp: i64,
    lo: i64,
    hi: i64,
    double: bool,
    slow: &dyn Fn(i64, i64, i64) -> Result<i64>,
) -> Result<Vec<O>> {
    let mut out = Vec::with_capacity(values.len());
    for row in values.chunks(channels) {
        for (&x, &(m, s, round_pos, bound)) in row.iter().zip(params) {
            let x: i64 = x.into() - in_zp;
            if x < -bound || x >= bound {
                slow(x, m, s)?;
                return Err(Error::GraphError("apply_scale_32: value out of range".into()));
            }
            let round = if double && s > 31 && x < 0 { bound - (1i64 << 30) } else { round_pos };
            let result = ((x * m + round) >> s) + out_zp;
            if result < i32::MIN as i64 || result > i32::MAX as i64 {
                return Err(invalid("RESCALE", format!("result {result} exceeds the int32 range")));
            }
            // Unsigned outputs hold values up to 255 / 65535 as their bit patterns.
            out.push(O::wrap(result.clamp(lo, hi)));
        }
    }
    Ok(out)
}

/// Truncating conversion from `i64`, as `int_data` performs it.
trait WrapFromI64 {
    fn wrap(v: i64) -> Self;
}

impl WrapFromI64 for i8 {
    fn wrap(v: i64) -> Self {
        v as i8
    }
}

impl WrapFromI64 for i16 {
    fn wrap(v: i64) -> Self {
        v as i16
    }
}

impl WrapFromI64 for i32 {
    fn wrap(v: i64) -> Self {
        v as i32
    }
}

/// `round(value * multiplier / 2^shift)` with a 16-bit multiplier.
fn apply_scale_16(value: i64, multiplier: i16, shift: i32) -> Result<i32> {
    if multiplier < 0 || !(2..=62).contains(&shift) {
        return Err(Error::GraphError(format!("apply_scale_16: invalid multiplier {multiplier} or shift {shift}")));
    }
    let result = (value * multiplier as i64 + (1i64 << (shift - 1))) >> shift;
    i32::try_from(result).map_err(|_| Error::GraphError("apply_scale_16: result exceeds int32".into()))
}

fn bit_width(dtype: DType) -> u32 {
    match dtype {
        DType::I8 => 8,
        DType::I16 => 16,
        DType::I32 => 32,
        _ => 48,
    }
}

/// Reinterpret a sign-extended value as unsigned when `unsigned` is set.
fn extend(dtype: DType, v: i64, unsigned: bool) -> i64 {
    if unsigned { v & ((1i64 << bit_width(dtype)) - 1) } else { v }
}

/// RESCALE: inputs value, multiplier [1 or C], shift [1 or C], input_zp [1],
/// output_zp [1]. Computes `clip(scale(value - input_zp) + output_zp)` with
/// the signedness given by the `*_unsigned` attributes.
pub fn eval_rescale(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "RESCALE";
    let OpAttribute::Rescale { scale32, rounding_mode, per_channel, input_unsigned, output_unsigned } = op.attribute
    else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 5, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let multiplier = state.get_tensor(&op.inputs[1])?;
    let shift = state.get_tensor(&op.inputs[2])?;
    let in_zp = state.get_tensor(&op.inputs[3])?;
    let out_zp = state.get_tensor(&op.inputs[4])?;
    let shape = input.shape.clone();
    let (in_t, out_t) =
        (input.dtype(), check_declared_output(state, OP, &op.outputs[0], &shape)?.unwrap_or(DType::I32));

    use DType::*;
    if !matches!(in_t, I8 | I16 | I32 | I48) || !matches!(out_t, I8 | I16 | I32) {
        return Err(Error::DTypeMismatch(format!("{OP}: unsupported types {in_t:?} -> {out_t:?}")));
    }
    if rounding_mode == RoundingMode::Unknown {
        return Err(invalid(OP, "unsupported rounding mode"));
    }
    if scale32 && in_t == I48 {
        return Err(invalid(OP, "scale32 is not allowed for INT48 input"));
    }
    if !scale32 && rounding_mode == RoundingMode::Double {
        return Err(invalid(OP, "double rounding requires scale32"));
    }
    // Unsigned data must be 8 or 16 bits wide, and only one side may be unsigned.
    let has_32_bit_side = matches!(in_t, I32 | I48) || out_t == I32;
    if (input_unsigned && output_unsigned) || ((input_unsigned || output_unsigned) && has_32_bit_side) {
        return Err(invalid(OP, "invalid combination of input_unsigned/output_unsigned and types"));
    }
    let expected_multiplier = if scale32 { I32 } else { I16 };
    if multiplier.dtype() != expected_multiplier || shift.dtype() != I8 {
        return Err(Error::DTypeMismatch(format!(
            "{OP}: multiplier must be {expected_multiplier:?} and shift INT8, got {:?}/{:?}",
            multiplier.dtype(),
            shift.dtype()
        )));
    }
    if in_zp.dtype() != in_t || out_zp.dtype() != out_t || in_zp.data.len() != 1 || out_zp.data.len() != 1 {
        return Err(invalid(OP, "zero points must be single elements of the input/output types"));
    }
    let channels =
        if per_channel { *shape.last().ok_or_else(|| invalid(OP, "per_channel requires rank >= 1"))? } else { 1 };
    if multiplier.data.len() != channels || shift.data.len() != channels {
        return Err(invalid(OP, format!("multiplier and shift must have {channels} elements")));
    }

    let in_zp = extend(in_t, int_values(&in_zp.data).unwrap()[0], input_unsigned);
    let out_zp = extend(out_t, int_values(&out_zp.data).unwrap()[0], output_unsigned);
    let zp_ok = |t: DType, zp: i64, unsigned: bool| t == I8 || zp == 0 || (t == I16 && unsigned && zp == 32768);
    if !zp_ok(in_t, in_zp, input_unsigned) || !zp_ok(out_t, out_zp, output_unsigned) {
        return Err(invalid(OP, "zero point not allowed for this type"));
    }

    let multipliers = int_values(&multiplier.data).unwrap();
    let shifts = int_values(&shift.data).unwrap();
    let (lo, hi) = if output_unsigned { (0, (1i64 << bit_width(out_t)) - 1) } else { int_range(out_t) };
    let values = int_values(&input.data).unwrap();

    // One element, with every check the reference model performs.
    let slow = |x: i64, m: i64, s: i64| -> Result<i64> {
        let (m, s) = (m as i32, s as i32);
        let scaled = match rounding_mode {
            RoundingMode::Inexact => (x as f32 * m as f32 * libm::ldexpf(1.0, -s)).round_ties_even() as i32,
            _ if scale32 => apply_scale_32(x as i32, m, s, rounding_mode == RoundingMode::Double)?,
            _ => apply_scale_16(x, m as i16, s)?,
        };
        let result = scaled as i64 + out_zp;
        if i32::try_from(result).is_err() {
            return Err(invalid(OP, format!("result {result} exceeds the int32 range")));
        }
        Ok(result.clamp(lo, hi))
    };

    let fast = scale32 && rounding_mode != RoundingMode::Inexact && !input_unsigned;
    if fast {
        // Per-channel constants: (multiplier, shift, rounding term, |value| bound).
        let params: Vec<(i64, i64, i64, i64)> = multipliers
            .iter()
            .zip(&shifts)
            .map(|(&m, &s)| {
                let round = 1i64 << (s - 1).clamp(0, 62);
                let extra = if rounding_mode == RoundingMode::Double && s > 31 { 1i64 << 30 } else { 0 };
                (m, s, round + extra, round)
            })
            .collect();
        let valid = params.iter().all(|&(m, s, _, _)| m >= 0 && (2..=62).contains(&s));
        if valid {
            let double = rounding_mode == RoundingMode::Double;
            // Straight from the input element type to the output element type.
            let data = match (&input.data, out_t) {
                (TensorData::I8(v), I8) => {
                    TensorData::I8(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I8(v), I16) => {
                    TensorData::I16(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I8(v), I32) => {
                    TensorData::I32(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I16(v), I8) => {
                    TensorData::I8(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I16(v), I16) => {
                    TensorData::I16(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I16(v), I32) => {
                    TensorData::I32(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I32(v), I8) => {
                    TensorData::I8(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I32(v), I16) => {
                    TensorData::I16(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                (TensorData::I32(v), I32) => {
                    TensorData::I32(rescale_fast(v, &params, channels, in_zp, out_zp, lo, hi, double, &slow)?)
                }
                _ => return Err(Error::DTypeMismatch(format!("{OP}: unsupported types {in_t:?} -> {out_t:?}"))),
            };
            state.set_tensor(Tensor::new(op.outputs[0].clone(), shape, data));
            return Ok(());
        }
    }
    let mut out = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let ch = if per_channel { i % channels } else { 0 };
        let x = extend(in_t, v, input_unsigned) - in_zp;
        out.push(slow(x, multipliers[ch], shifts[ch])?);
    }
    state.set_tensor(Tensor::new(op.outputs[0].clone(), shape, int_data(out_t, out)?));
    Ok(())
}
