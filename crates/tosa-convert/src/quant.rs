//! Quantization arithmetic shared by frontends: turning real-valued scales
//! into TOSA RESCALE multipliers and shifts, and emitting RESCALE operators.
//!
//! The fixed-point conventions follow the TensorFlow TOSA legalization
//! (`computeMultiplierAndShift` in `legalize_utils.cc`), so quantized models
//! lowered here reproduce what TFLite's reference kernels compute.

use tosa_ir::{DType, Op, OpAttribute, RoundingMode, TensorData, TensorRef};

use crate::error::Result;
use crate::lower::LowerCtx;

/// A fixed-point approximation `multiplier / 2^shift` of a real scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleParams {
    pub multiplier: i32,
    pub shift: i32,
}

/// Approximate `scale` as `multiplier / 2^shift` with a 32-bit multiplier
/// in `[2^30, 2^31)` and a shift in `[2, 62]`, as RESCALE with `scale32`
/// requires. Scales too large for a shift of at least 2 are an error; very
/// small scales lose precision as the multiplier is shifted down.
pub fn scale32(scale: f64) -> Result<ScaleParams> {
    multiplier_and_shift(scale, 32)
}

/// As [`scale32`] with a 16-bit multiplier in `[2^14, 2^15)`, for RESCALE
/// without `scale32` (INT48 inputs).
pub fn scale16(scale: f64) -> Result<ScaleParams> {
    multiplier_and_shift(scale, 16)
}

fn multiplier_and_shift(scale: f64, bits: u32) -> Result<ScaleParams> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(
            tosa_ir::Error::GraphError(format!("quantization scale {scale} must be positive and finite")).into()
        );
    }
    // scale = mantissa * 2^exponent with mantissa in [0.5, 1).
    let (mantissa, mut exponent) = frexp(scale);
    let mut multiplier = (mantissa * (1u64 << (bits - 1)) as f64).round() as i64;
    if multiplier == 1i64 << (bits - 1) {
        multiplier /= 2;
        exponent += 1;
    }
    // value * scale = value * multiplier * 2^(exponent - (bits - 1)).
    let mut shift = (bits as i32 - 1) - exponent;
    if shift > 62 {
        multiplier >>= shift - 62;
        shift = 62;
    }
    if shift < 2 {
        return Err(tosa_ir::Error::GraphError(format!("quantization scale {scale} is too large for RESCALE")).into());
    }
    Ok(ScaleParams { multiplier: multiplier as i32, shift })
}

/// `x = m * 2^e` with `m` in `[0.5, 1)` (C's `frexp`).
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        // Subnormal: normalize first.
        let (m, e) = frexp(x * 2f64.powi(64));
        return (m, e - 64);
    }
    let e = exp_field - 1022;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, e)
}

/// Parameters of one RESCALE operator.
#[derive(Debug, Clone, PartialEq)]
pub struct RescaleSpec {
    /// Real scale factor(s): one, or one per output channel.
    pub scales: Vec<f64>,
    pub input_zp: i64,
    pub output_zp: i64,
    pub rounding: RoundingMode,
    /// Treat the (INT8 or INT16) input's bits as unsigned; `input_zp` is then
    /// in the unsigned range.
    pub input_unsigned: bool,
    /// Produce unsigned bits in the (INT8 or INT16) output; `output_zp` is
    /// then in the unsigned range.
    pub output_unsigned: bool,
}

impl RescaleSpec {
    pub fn per_tensor(scale: f64, input_zp: i64, output_zp: i64) -> Self {
        RescaleSpec {
            scales: vec![scale],
            input_zp,
            output_zp,
            rounding: RoundingMode::Double,
            input_unsigned: false,
            output_unsigned: false,
        }
    }

    pub fn per_channel(scales: Vec<f64>, input_zp: i64, output_zp: i64) -> Self {
        RescaleSpec { scales, ..Self::per_tensor(1.0, input_zp, output_zp) }
    }

    pub fn rounding(mut self, rounding: RoundingMode) -> Self {
        self.rounding = rounding;
        self
    }

    pub fn input_unsigned(mut self, unsigned: bool) -> Self {
        self.input_unsigned = unsigned;
        self
    }

    pub fn output_unsigned(mut self, unsigned: bool) -> Self {
        self.output_unsigned = unsigned;
        self
    }
}

/// Single-element integer data of `dtype`. Values are truncated to the
/// type's width, so an unsigned zero point such as 200 becomes the INT8 bit
/// pattern RESCALE's `*_unsigned` flags expect.
pub fn int_scalar(dtype: DType, value: i64) -> Result<TensorData> {
    Ok(match dtype {
        DType::I8 => TensorData::I8(vec![value as i8]),
        DType::I16 => TensorData::I16(vec![value as i16]),
        DType::I32 => TensorData::I32(vec![value as i32]),
        DType::I48 => TensorData::I48(vec![value]),
        DType::I64 => TensorData::I64(vec![value]),
        other => return Err(tosa_ir::Error::DTypeMismatch(format!("{other:?} is not an integer type")).into()),
    })
}

/// Emit `RESCALE(input)` producing `out_dtype`, with multipliers and shifts
/// derived from `spec`. INT48 inputs use 16-bit multipliers, everything else
/// 32-bit. Per-channel when `spec.scales` has more than one entry (the
/// channel is the last dimension, as TOSA requires).
pub fn emit_rescale(
    ctx: &mut LowerCtx<'_>,
    input: &TensorRef,
    out_dtype: DType,
    spec: &RescaleSpec,
    hint: &str,
) -> Result<TensorRef> {
    let scale32 = input.dtype != DType::I48;
    let per_channel = spec.scales.len() > 1;
    let params: Vec<ScaleParams> =
        spec.scales.iter().map(|&s| if scale32 { scale32_fn(s) } else { scale16(s) }).collect::<Result<_>>()?;
    let multiplier = if scale32 {
        TensorData::I32(params.iter().map(|p| p.multiplier).collect())
    } else {
        TensorData::I16(params.iter().map(|p| p.multiplier as i16).collect())
    };
    let shift = TensorData::I8(params.iter().map(|p| p.shift as i8).collect());
    let n = params.len() as i32;
    let multiplier = ctx.block.constant(&format!("{hint}_multiplier"), &[n], multiplier)?;
    let shift = ctx.block.constant(&format!("{hint}_shift"), &[n], shift)?;
    let input_zp = ctx.scalar(int_scalar(input.dtype, spec.input_zp)?, &format!("{hint}_input_zp"))?;
    let output_zp = ctx.scalar(int_scalar(out_dtype, spec.output_zp)?, &format!("{hint}_output_zp"))?;
    let attr = OpAttribute::Rescale {
        scale32,
        rounding_mode: spec.rounding,
        per_channel,
        input_unsigned: spec.input_unsigned,
        output_unsigned: spec.output_unsigned,
    };
    let shape = input.shape.clone();
    Ok(ctx.emit_temp(Op::Rescale, attr, &[input, &multiplier, &shift, &input_zp, &output_zp], out_dtype, &shape, hint))
}

fn scale32_fn(s: f64) -> Result<ScaleParams> {
    scale32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_matches_libc_convention() {
        assert_eq!(frexp(1.0), (0.5, 1));
        assert_eq!(frexp(0.5), (0.5, 0));
        assert_eq!(frexp(3.0), (0.75, 2));
        assert_eq!(frexp(0.1).0 * 2f64.powi(frexp(0.1).1), 0.1);
    }

    #[test]
    fn scale32_known_values() {
        assert_eq!(scale32(1.0).unwrap(), ScaleParams { multiplier: 1 << 30, shift: 30 });
        assert_eq!(scale32(0.5).unwrap(), ScaleParams { multiplier: 1 << 30, shift: 31 });
        assert_eq!(scale32(0.25).unwrap(), ScaleParams { multiplier: 1 << 30, shift: 32 });
        let p = scale32(0.003).unwrap();
        assert!((1i64 << 30..1i64 << 31).contains(&(p.multiplier as i64)));
        let approx = p.multiplier as f64 / 2f64.powi(p.shift);
        assert!((approx - 0.003).abs() < 1e-9, "{approx}");
    }

    #[test]
    fn scale32_rejects_out_of_range() {
        assert!(scale32(0.0).is_err());
        assert!(scale32(-1.0).is_err());
        assert!(scale32(f64::NAN).is_err());
        assert!(scale32(2f64.powi(31)).is_err());
        // Tiny scales clamp the shift and lose multiplier precision.
        let p = scale32(2f64.powi(-40)).unwrap();
        assert_eq!(p.shift, 62);
        assert!(p.multiplier < 1 << 30);
    }

    #[test]
    fn scale16_known_values() {
        assert_eq!(scale16(1.0).unwrap(), ScaleParams { multiplier: 1 << 14, shift: 14 });
        assert_eq!(scale16(0.75).unwrap(), ScaleParams { multiplier: 3 << 13, shift: 15 });
    }
}

// ---------------------------------------------------------------------------
// Exact float-to-integer quantization
// ---------------------------------------------------------------------------

/// How a quotient exactly half way between two integers is rounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ties {
    /// To the even neighbour (onnxruntime's `QuantizeLinear`, `nearbyint`).
    Even,
    /// Away from zero (TFLite's `QUANTIZE`, `std::round`).
    AwayFromZero,
}

/// Emit `round(x / scale)` as INT32, with the rounding of the *exact*
/// quotient, for FP32 `x` and a positive constant `scale`.
///
/// Reference runtimes divide by the scale; TOSA has no float division, and
/// `x * (1 / scale)` differs from the correctly rounded quotient by an ulp,
/// which changes the rounded integer when the quotient is within an ulp of
/// a half-integer boundary. The emitted graph therefore computes
/// `r = x * (1 / scale)`, rounds it to the candidate `k`, and decides
/// exactly whether `x / scale` lies beyond the boundary `b = k ± 0.5` on
/// `r`'s side by evaluating the sign of `x - b * scale` with Dekker's
/// error-free product (Veltkamp splitting), all in FP32 operators. Ties
/// (`x == b * scale` exactly) follow `ties`.
pub fn quantize_round_exact(
    ctx: &mut LowerCtx<'_>,
    x: &TensorRef,
    scale: f32,
    ties: Ties,
    hint: &str,
) -> Result<TensorRef> {
    if !(scale > 0.0 && scale.is_finite()) {
        return ctx.fail(format!("quantization scale {scale} must be positive and finite"));
    }
    let shape = x.shape.clone();
    let rank = shape.len();
    let f32c = |ctx: &mut LowerCtx<'_>, v: f32, name: &str| -> Result<TensorRef> {
        Ok(ctx.block.constant(&format!("{hint}_{name}"), &vec![1; rank], TensorData::F32(vec![v]))?)
    };
    let i32c = |ctx: &mut LowerCtx<'_>, v: i32, name: &str| -> Result<TensorRef> {
        Ok(ctx.block.constant(&format!("{hint}_{name}"), &vec![1; rank], TensorData::I32(vec![v]))?)
    };
    let shift = ctx.scalar(TensorData::I8(vec![0]), &format!("{hint}_shift"))?;
    let mul = |ctx: &mut LowerCtx<'_>, a: &TensorRef, b: &TensorRef, name: &str| -> TensorRef {
        ctx.emit_temp(Op::Mul, OpAttribute::None, &[a, b, &shift], DType::F32, &shape, &format!("{hint}_{name}"))
    };
    let add = |ctx: &mut LowerCtx<'_>, a: &TensorRef, b: &TensorRef, name: &str| -> TensorRef {
        ctx.emit_temp(Op::Add, OpAttribute::None, &[a, b], DType::F32, &shape, &format!("{hint}_{name}"))
    };
    let sub = |ctx: &mut LowerCtx<'_>, a: &TensorRef, b: &TensorRef, name: &str| -> TensorRef {
        ctx.emit_temp(Op::Sub, OpAttribute::None, &[a, b], DType::F32, &shape, &format!("{hint}_{name}"))
    };
    let cast = |ctx: &mut LowerCtx<'_>, a: &TensorRef, to: DType, name: &str| -> TensorRef {
        ctx.emit_temp(
            Op::Cast,
            OpAttribute::Cast { input_unsigned: false },
            &[a],
            to,
            &shape,
            &format!("{hint}_{name}"),
        )
    };
    let cmp = |ctx: &mut LowerCtx<'_>, op: Op, a: &TensorRef, b: &TensorRef, name: &str| -> TensorRef {
        ctx.emit_temp(op, OpAttribute::None, &[a, b], DType::Bool, &shape, &format!("{hint}_{name}"))
    };

    // Candidate: k = round_half_even(x * (1 / scale)).
    let inv = f32c(ctx, 1.0 / scale, "inv_scale")?;
    let r = mul(ctx, x, &inv, "r");
    let k = cast(ctx, &r, DType::I32, "k");
    let kf = cast(ctx, &k, DType::F32, "kf");

    // The boundary on r's side of k: b = k + sigma, sigma = +-0.5.
    let zero = f32c(ctx, 0.0, "zero")?;
    let half = f32c(ctx, 0.5, "half")?;
    let neg_half = f32c(ctx, -0.5, "neg_half")?;
    let above = cmp(ctx, Op::GreaterEqual, &r, &kf, "above");
    let sigma = ctx.emit_temp(
        Op::Select,
        OpAttribute::None,
        &[&above, &half, &neg_half],
        DType::F32,
        &shape,
        &format!("{hint}_sigma"),
    );
    let b = add(ctx, &kf, &sigma, "b");

    // Exact sign of x - b * scale. Veltkamp split of b (factor 2^12 + 1) and
    // of the constant scale; Dekker's product error e = b * scale - p exactly.
    let factor = f32c(ctx, 4097.0, "split_factor")?;
    let c = mul(ctx, &b, &factor, "split_c");
    let c_minus_b = sub(ctx, &c, &b, "split_cb");
    let bh = sub(ctx, &c, &c_minus_b, "bh");
    let bl = sub(ctx, &b, &bh, "bl");
    let s_c = scale * 4097.0;
    let sh_v = s_c - (s_c - scale);
    let sl_v = scale - sh_v;
    let s_t = f32c(ctx, scale, "scale")?;
    let sh = f32c(ctx, sh_v, "scale_hi")?;
    let sl = f32c(ctx, sl_v, "scale_lo")?;
    let p = mul(ctx, &b, &s_t, "p");
    let hh = mul(ctx, &bh, &sh, "hh");
    let e1 = sub(ctx, &hh, &p, "e1");
    let hl = mul(ctx, &bh, &sl, "hl");
    let e2 = add(ctx, &e1, &hl, "e2");
    let lh = mul(ctx, &bl, &sh, "lh");
    let e3 = add(ctx, &e2, &lh, "e3");
    let ll = mul(ctx, &bl, &sl, "ll");
    let e = add(ctx, &e3, &ll, "e");
    let d1 = sub(ctx, x, &p, "d1");
    let d = sub(ctx, &d1, &e, "d");

    // Crossed the boundary (in sigma's direction), or exactly on it.
    let d_sigma = mul(ctx, &d, &sigma, "d_sigma");
    let crossed = cmp(ctx, Op::Greater, &d_sigma, &zero, "crossed");
    let tie = cmp(ctx, Op::Equal, &d, &zero, "tie");
    let tie_moves = match ties {
        Ties::Even => {
            // From k to k + 2 sigma: move when k is odd.
            let one = i32c(ctx, 1, "one_i32")?;
            let low_bit = ctx.emit_temp(
                Op::BitwiseAnd,
                OpAttribute::None,
                &[&k, &one],
                DType::I32,
                &shape,
                &format!("{hint}_low_bit"),
            );
            cmp(ctx, Op::Equal, &low_bit, &one, "k_odd")
        }
        Ties::AwayFromZero => {
            // Away from zero is in sigma's direction when b and sigma share a sign.
            let b_sigma = mul(ctx, &b, &sigma, "b_sigma");
            cmp(ctx, Op::Greater, &b_sigma, &zero, "away")
        }
    };
    let tie_move = ctx.emit_temp(
        Op::LogicalAnd,
        OpAttribute::None,
        &[&tie, &tie_moves],
        DType::Bool,
        &shape,
        &format!("{hint}_tie_move"),
    );
    let moves = ctx.emit_temp(
        Op::LogicalOr,
        OpAttribute::None,
        &[&crossed, &tie_move],
        DType::Bool,
        &shape,
        &format!("{hint}_moves"),
    );
    let step = cast(ctx, &moves, DType::I32, "step");
    let two = f32c(ctx, 2.0, "two")?;
    let sigma2 = mul(ctx, &sigma, &two, "sigma2");
    let direction = cast(ctx, &sigma2, DType::I32, "direction");
    let delta = ctx.emit_temp(
        Op::Mul,
        OpAttribute::None,
        &[&step, &direction, &shift],
        DType::I32,
        &shape,
        &format!("{hint}_delta"),
    );
    Ok(ctx.emit_temp(Op::Add, OpAttribute::None, &[&k, &delta], DType::I32, &shape, hint))
}

/// `saturate(round(x / scale) + zero_point)` as INT32 in the range of
/// `out_dtype` (INT8 or INT16), with the exact rounding of
/// [`quantize_round_exact`].
pub fn quantize_exact(
    ctx: &mut LowerCtx<'_>,
    x: &TensorRef,
    scale: f32,
    zero_point: i32,
    out_dtype: DType,
    ties: Ties,
    hint: &str,
) -> Result<TensorRef> {
    let (lo, hi) = match out_dtype {
        DType::I8 => (-128, 127),
        DType::I16 => (-32768, 32767),
        other => return ctx.fail(format!("cannot quantize to {other:?}")),
    };
    let shape = x.shape.clone();
    let rank = shape.len();
    let rounded = quantize_round_exact(ctx, x, scale, ties, &format!("{hint}_rounded"))?;
    let zp = ctx.block.constant(&format!("{hint}_zp"), &vec![1; rank], TensorData::I32(vec![zero_point]))?;
    let shifted =
        ctx.emit_temp(Op::Add, OpAttribute::None, &[&rounded, &zp], DType::I32, &shape, &format!("{hint}_shifted"));
    let attr = OpAttribute::NanPropagation { nan_mode: tosa_ir::NanMode::Propagate };
    let hi_t = ctx.block.constant(&format!("{hint}_hi"), &vec![1; rank], TensorData::I32(vec![hi]))?;
    let lo_t = ctx.block.constant(&format!("{hint}_lo"), &vec![1; rank], TensorData::I32(vec![lo]))?;
    let m = ctx.emit_temp(Op::Minimum, attr.clone(), &[&shifted, &hi_t], DType::I32, &shape, &format!("{hint}_min"));
    Ok(ctx.emit_temp(Op::Maximum, attr, &[&m, &lo_t], DType::I32, &shape, &format!("{hint}_saturated")))
}
