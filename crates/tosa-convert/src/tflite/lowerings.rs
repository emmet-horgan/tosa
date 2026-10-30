//! Lowerings of TFLite builtin operators to TOSA.
//!
//! Float operators map directly. Quantized (int8, and uint8 re-expressed as
//! int8) operators follow TFLite's reference kernels: convolutions
//! accumulate in INT32 with the input and weight zero points and requantize
//! per channel with RESCALE in double-rounding mode; ADD and SUB rescale
//! both operands to a common INT32 scale with TFLite's 20-bit left shift;
//! MUL multiplies in INT32 and requantizes; pooling and concatenation stay
//! in the quantized domain. SOFTMAX, LOGISTIC and TANH on quantized data are
//! computed in FP32 between a dequantize and a quantize, which TFLite's
//! lookup-table kernels approximate within a few units.
//!
//! uint8 graph inputs arrive as INT8 tensors holding the uint8 bits; the
//! input hook converts them with RESCALE (`input_unsigned`), and the output
//! hook converts back (`output_unsigned`).

use tosa_ir::{DType, NanMode, Op, OpAttribute, ResizeMode, RoundingMode, TensorData, TensorRef};

use crate::error::Result;
use crate::lower::{LowerCtx, LoweringRegistry};
use crate::quant::{self, RescaleSpec, emit_rescale};
use crate::source::{Quantization, SourceDType, SourceNode, TensorId};

pub fn registry() -> LoweringRegistry {
    let mut r = LoweringRegistry::new();
    r.register("CONV_2D", |n, c| conv(n, c, ConvKind::Conv));
    r.register("DEPTHWISE_CONV_2D", |n, c| conv(n, c, ConvKind::Depthwise));
    r.register("FULLY_CONNECTED", fully_connected);
    r.register("AVERAGE_POOL_2D", |n, c| pool(n, c, true));
    r.register("MAX_POOL_2D", |n, c| pool(n, c, false));
    r.register("ADD", |n, c| binary(n, c, Op::Add));
    r.register("SUB", |n, c| binary(n, c, Op::Sub));
    r.register("MUL", |n, c| binary(n, c, Op::Mul));
    r.register("RESHAPE", reshape);
    r.register("SQUEEZE", reshape_to_output);
    r.register("EXPAND_DIMS", reshape_to_output);
    r.register("SOFTMAX", softmax);
    r.register("LOGISTIC", |n, c| float_unary(n, c, Op::Sigmoid));
    r.register("TANH", |n, c| float_unary(n, c, Op::Tanh));
    r.register("RELU", |n, c| activation(n, c, "RELU"));
    r.register("RELU6", |n, c| activation(n, c, "RELU6"));
    r.register("RELU_N1_TO_1", |n, c| activation(n, c, "RELU_N1_TO_1"));
    r.register("QUANTIZE", quantize);
    r.register("DEQUANTIZE", dequantize);
    r.register("PAD", pad);
    r.register("MEAN", mean);
    r.register("CONCATENATION", concatenation);
    register_more(&mut r);
    r.set_input_hook(input_hook);
    r.set_output_hook(output_hook);
    r
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_quantized(ctx: &LowerCtx<'_>, id: TensorId) -> bool {
    let t = ctx.tensor(id);
    matches!(t.dtype, SourceDType::I8 | SourceDType::I16) && t.quant.is_some()
}

fn quant(ctx: &LowerCtx<'_>, id: TensorId) -> Result<Quantization> {
    match &ctx.tensor(id).quant {
        Some(q) => Ok(q.clone()),
        None => ctx.fail(format!("tensor '{}' has no quantization parameters", ctx.tensor(id).name)),
    }
}

/// Per-tensor scale and zero point.
fn scale_zp(ctx: &LowerCtx<'_>, id: TensorId) -> Result<(f64, i64)> {
    let q = quant(ctx, id)?;
    match (q.scale(), q.zero_point()) {
        (Some(s), Some(z)) => Ok((s, z)),
        _ => ctx.fail(format!("tensor '{}' is quantized per channel where per-tensor is needed", ctx.tensor(id).name)),
    }
}

fn static_shape(ctx: &LowerCtx<'_>, id: TensorId) -> Result<Vec<i32>> {
    let t = ctx.tensor(id);
    if t.is_static() { Ok(t.tosa_shape()) } else { ctx.fail(format!("tensor '{}' has a dynamic shape", t.name)) }
}

fn shape_of(ctx: &LowerCtx<'_>, id: TensorId) -> Vec<i32> {
    ctx.tensor(id).tosa_shape()
}

fn ones(rank: usize) -> Vec<i32> {
    vec![1; rank]
}

/// A constant holding `value` broadcast-ready for a rank-`rank` operand.
fn f32_const(ctx: &mut LowerCtx<'_>, value: f32, rank: usize, hint: &str) -> Result<TensorRef> {
    Ok(ctx.block.constant(hint, &ones(rank), TensorData::F32(vec![value]))?)
}

/// The INT8 `shift` operand of MUL.
fn mul_shift(ctx: &mut LowerCtx<'_>) -> Result<TensorRef> {
    ctx.scalar(TensorData::I8(vec![0]), "shift")
}

fn float_mul(ctx: &mut LowerCtx<'_>, a: &TensorRef, b: &TensorRef, shape: &[i32], hint: &str) -> Result<TensorRef> {
    let shift = mul_shift(ctx)?;
    Ok(ctx.emit_temp(Op::Mul, OpAttribute::None, &[a, b, &shift], a.dtype, shape, hint))
}

/// Bytes of a CLAMP bound in `dtype`.
fn bound_bytes(ctx: &LowerCtx<'_>, dtype: DType, value: f64) -> Result<Vec<u8>> {
    Ok(match dtype {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f64(value).to_le_bytes().to_vec(),
        DType::Bf16 => half::bf16::from_f64(value).to_le_bytes().to_vec(),
        DType::I8 => vec![(value as i64).clamp(-128, 127) as i8 as u8],
        DType::I16 => ((value as i64).clamp(-32768, 32767) as i16).to_le_bytes().to_vec(),
        other => return ctx.fail(format!("cannot clamp a {other:?} tensor")),
    })
}

/// `CLAMP(input)` producing source tensor `out`.
fn clamp(ctx: &mut LowerCtx<'_>, input: &TensorRef, out: TensorId, lo: f64, hi: f64) -> Result<TensorRef> {
    let attr = OpAttribute::Clamp {
        min_val: bound_bytes(ctx, input.dtype, lo)?,
        max_val: bound_bytes(ctx, input.dtype, hi)?,
        nan_mode: NanMode::Propagate,
    };
    ctx.emit(Op::Clamp, attr, &[input], out)
}

/// TFLite's quantized representation of real value `x` in tensor `id`.
fn quantize_value(ctx: &LowerCtx<'_>, id: TensorId, x: f64) -> Result<i64> {
    let (scale, zp) = scale_zp(ctx, id)?;
    let (lo, hi) = int_range(ctx.dtype(id)?);
    Ok(((x / scale).round() as i64 + zp).clamp(lo, hi))
}

fn int_range(dtype: DType) -> (i64, i64) {
    match dtype {
        DType::I8 => (-128, 127),
        DType::I16 => (-32768, 32767),
        _ => (i32::MIN as i64, i32::MAX as i64),
    }
}

/// Apply the node's fused activation to `value`, producing source tensor `out`.
fn fused_activation(node: &SourceNode, ctx: &mut LowerCtx<'_>, value: TensorRef, out: TensorId) -> Result<()> {
    let act = node
        .attrs
        .get("fused_activation_function")
        .map_or(Ok("NONE"), |_| node.attr_str("fused_activation_function"))?;
    apply_activation(ctx, act, value, out)
}

fn apply_activation(ctx: &mut LowerCtx<'_>, act: &str, value: TensorRef, out: TensorId) -> Result<()> {
    let (lo, hi) = match act {
        "NONE" => {
            ctx.bind(out, value);
            return Ok(());
        }
        "RELU" => (0.0, f64::INFINITY),
        "RELU6" => (0.0, 6.0),
        "RELU_N1_TO_1" => (-1.0, 1.0),
        "TANH" if !is_quantized(ctx, out) => {
            ctx.emit(Op::Tanh, OpAttribute::None, &[&value], out)?;
            return Ok(());
        }
        other => return ctx.fail(format!("fused activation {other} is not supported")),
    };
    if is_quantized(ctx, out) {
        let (qmin, qmax) = int_range(ctx.dtype(out)?);
        let lo = quantize_value(ctx, out, lo)?.max(qmin);
        let hi = if hi.is_infinite() { qmax } else { quantize_value(ctx, out, hi)? };
        clamp(ctx, &value, out, lo as f64, hi as f64)?;
    } else {
        let hi = if hi.is_infinite() { f32::MAX as f64 } else { hi };
        clamp(ctx, &value, out, lo, hi)?;
    }
    Ok(())
}

/// TFLite SAME/VALID padding as TOSA `[top, bottom, left, right]`, plus the
/// spatial size the input must be sliced to first.
///
/// TFLite floors the output size (`VALID`: `(in - k) / stride + 1`), so an
/// input may have trailing rows or columns no output window reads. TOSA
/// requires the window to tile the padded input exactly, so those are cut
/// off with a SLICE (a no-op on data any window would use).
struct Geometry {
    pad: Vec<i32>,
    /// `[H, W]` to slice the input to, if smaller than the input.
    sliced: Option<[i32; 2]>,
}

fn geometry(
    ctx: &LowerCtx<'_>,
    node: &SourceNode,
    input: &[i32],
    kernel: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
) -> Result<Geometry> {
    let mode = node.attr_str("padding")?;
    let mut pad = Vec::with_capacity(4);
    let mut sliced = [input[1], input[2]];
    let mut any_slice = false;
    for d in 0..2 {
        let (size, k, s, dil) = (input[1 + d] as i64, kernel[d] as i64, stride[d] as i64, dilation[d] as i64);
        let effective = (k - 1) * dil + 1;
        let (out, before, mut after) = match mode {
            "SAME" => {
                let out = (size + s - 1) / s;
                let total = ((out - 1) * s + effective - size).max(0);
                (out, total / 2, total - total / 2)
            }
            "VALID" => {
                if size < effective {
                    return ctx.fail(format!("input size {size} is smaller than the {effective}-wide window"));
                }
                ((size - effective) / s + 1, 0, 0)
            }
            other => return ctx.fail(format!("padding mode {other} is not supported")),
        };
        let needed = (out - 1) * s + effective;
        let excess = size + before + after - needed;
        after -= excess;
        if after < 0 {
            sliced[d] = (size + after) as i32;
            any_slice = true;
            after = 0;
        }
        if before < 0 {
            return ctx.fail("negative padding");
        }
        pad.push(before as i32);
        pad.push(after as i32);
    }
    Ok(Geometry { pad, sliced: any_slice.then_some(sliced) })
}

/// Slice a NHWC tensor to the spatial size `hw` (from the origin).
fn slice_spatial(ctx: &mut LowerCtx<'_>, input: &TensorRef, hw: [i32; 2], hint: &str) -> TensorRef {
    let shape = vec![input.shape[0], hw[0], hw[1], input.shape[3]];
    let start = ctx.shape_const(&[0, 0, 0, 0], &format!("{hint}_start"));
    let size = ctx.shape_const(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>(), &format!("{hint}_size"));
    ctx.emit_temp(Op::Slice, OpAttribute::None, &[input, &start, &size], input.dtype, &shape, hint)
}

/// Reshape a tensor to `shape` (as an intermediate).
fn reshape_temp(ctx: &mut LowerCtx<'_>, input: &TensorRef, shape: &[i32], hint: &str) -> TensorRef {
    let s = ctx.shape_const(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>(), &format!("{hint}_shape"));
    ctx.emit_temp(Op::Reshape, OpAttribute::None, &[input, &s], input.dtype, shape, hint)
}

/// Give two operands equal rank for TOSA broadcasting by prepending 1s to
/// the lower-rank one.
fn align_ranks(ctx: &mut LowerCtx<'_>, a: TensorRef, b: TensorRef) -> (TensorRef, TensorRef) {
    let (ra, rb) = (a.shape.len(), b.shape.len());
    if ra == rb {
        return (a, b);
    }
    let pad = |t: &TensorRef, rank: usize| {
        let mut s = vec![1; rank - t.shape.len()];
        s.extend_from_slice(&t.shape);
        s
    };
    if ra < rb {
        let s = pad(&a, rb);
        let a2 = reshape_temp(ctx, &a, &s, &format!("{}_bcast", a.name));
        (a2, b)
    } else {
        let s = pad(&b, ra);
        let b2 = reshape_temp(ctx, &b, &s, &format!("{}_bcast", b.name));
        (a, b2)
    }
}

/// Dequantize an INT8 tensor to FP32: `scale * (q - zero_point)`.
fn dequantize_temp(ctx: &mut LowerCtx<'_>, input: &TensorRef, id: TensorId, hint: &str) -> Result<TensorRef> {
    let (scale, zp) = scale_zp(ctx, id)?;
    let shape = input.shape.clone();
    let rank = shape.len();
    let f = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[input],
        DType::F32,
        &shape,
        &format!("{hint}_f32"),
    );
    let zp_c = f32_const(ctx, zp as f32, rank, &format!("{hint}_zp"))?;
    let centred =
        ctx.emit_temp(Op::Sub, OpAttribute::None, &[&f, &zp_c], DType::F32, &shape, &format!("{hint}_centred"));
    let s_c = f32_const(ctx, scale as f32, rank, &format!("{hint}_scale"))?;
    float_mul(ctx, &centred, &s_c, &shape, &format!("{hint}_real"))
}

/// Quantize an FP32 tensor into source tensor `out`:
/// `saturate(round(x / scale) + zero_point)` with TFLite's rounding
/// (`std::round`, half away from zero) of the exact quotient
/// (see [`crate::quant::quantize_exact`]).
fn quantize_into(ctx: &mut LowerCtx<'_>, input: &TensorRef, out: TensorId) -> Result<TensorRef> {
    let (scale, zp) = scale_zp(ctx, out)?;
    let dtype = ctx.dtype(out)?;
    let saturated =
        quant::quantize_exact(ctx, input, scale as f32, zp as i32, dtype, quant::Ties::AwayFromZero, "quant")?;
    ctx.emit(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&saturated], out)
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvKind {
    Conv,
    Depthwise,
}

/// Parameters shared by CONV_2D, DEPTHWISE_CONV_2D and FULLY_CONNECTED.
struct ConvOperands {
    input: TensorRef,
    /// TOSA layout: `[OC, KH, KW, IC]` or `[KH, KW, C, M]`.
    weight: TensorRef,
    bias: TensorRef,
    out_channels: i32,
    /// Source tensors supplying the quantization parameters.
    input_id: TensorId,
    weight_id: TensorId,
    /// The node's output and the shape of the convolution result.
    out: TensorId,
    out_shape: Vec<i32>,
}

/// Emit the convolution `op` on `operands` (already in TOSA layout), with
/// the requantization and fused activation the node asks for, producing
/// source tensor `out` of shape `out_shape`.
fn emit_conv(
    node: &SourceNode,
    ctx: &mut LowerCtx<'_>,
    op: Op,
    attr: impl Fn(DType) -> OpAttribute,
    operands: ConvOperands,
    finish: impl FnOnce(&mut LowerCtx<'_>, TensorRef) -> Result<TensorRef>,
) -> Result<()> {
    let ConvOperands { input, weight, bias, out_channels, input_id, weight_id, out, out_shape } = operands;
    let out_shape = &out_shape[..];
    if is_quantized(ctx, input_id) {
        let (in_scale, in_zp) = scale_zp(ctx, input_id)?;
        let wq = quant(ctx, weight_id)?;
        let w_zp = match wq.zero_points.iter().min().zip(wq.zero_points.iter().max()) {
            Some((lo, hi)) if lo == hi => *lo,
            _ => return ctx.fail("weights with differing per-channel zero points are not supported"),
        };
        let (out_scale, out_zp) = scale_zp(ctx, out)?;
        if bias.dtype != DType::I32 {
            return ctx.fail(format!("quantized bias must be INT32, got {:?}", bias.dtype));
        }
        let in_zp_t = ctx.scalar(TensorData::I8(vec![in_zp as i8]), "input_zp")?;
        let w_zp_t = ctx.scalar(TensorData::I8(vec![w_zp as i8]), "weight_zp")?;
        let acc = ctx.emit_temp(
            op,
            attr(DType::I32),
            &[&input, &weight, &bias, &in_zp_t, &w_zp_t],
            DType::I32,
            out_shape,
            "conv_acc",
        );
        // TFLite derives the multiplier of its legacy uint8 kernels from the
        // float32 product of the input and weight scales (converted to
        // double only afterwards), and the int8 per-channel multipliers from
        // a double product. The difference moves the rounding boundary by a
        // few parts in 10^8, which decides about 1 element in 20,000.
        let legacy_uint8 = ctx.tensor(input_id).original_dtype == Some(SourceDType::U8);
        let product = |w: f64| if legacy_uint8 { (in_scale as f32 * w as f32) as f64 } else { in_scale * w };
        let scales: Vec<f64> = if wq.scales.len() == 1 {
            vec![product(wq.scales[0]) / out_scale]
        } else {
            if wq.scales.len() != out_channels as usize {
                return ctx
                    .fail(format!("{} per-channel weight scales for {out_channels} output channels", wq.scales.len()));
            }
            wq.scales.iter().map(|&w| product(w) / out_scale).collect()
        };
        let spec = RescaleSpec::per_channel(scales, 0, out_zp).rounding(RoundingMode::Double);
        let requantized = emit_rescale(ctx, &acc, ctx.dtype(out)?, &spec, "conv_out")?;
        let value = finish(ctx, requantized)?;
        fused_activation(node, ctx, value, out)
    } else {
        let dtype = input.dtype;
        let zp = ctx.zero(dtype, "zp")?;
        let acc_type = if dtype == DType::F16 { DType::F16 } else { DType::F32 };
        let conv = ctx.emit_temp(op, attr(acc_type), &[&input, &weight, &bias, &zp, &zp], dtype, out_shape, "conv");
        let value = finish(ctx, conv)?;
        fused_activation(node, ctx, value, out)
    }
}

fn bias_or_zeros(
    ctx: &mut LowerCtx<'_>,
    node: &SourceNode,
    index: usize,
    dtype: DType,
    channels: i32,
) -> Result<TensorRef> {
    match node.optional_input(index) {
        Some(id) => ctx.value(id),
        None => {
            let bias_dtype = if dtype.is_float() { dtype } else { DType::I32 };
            Ok(ctx.block.constant("bias", &[channels], TensorData::zeros(bias_dtype, channels as usize)?)?)
        }
    }
}

fn conv(node: &SourceNode, ctx: &mut LowerCtx<'_>, kind: ConvKind) -> Result<()> {
    let (input_id, weight_id, out) = (node.input(0)?, node.input(1)?, node.output(0)?);
    let input = ctx.value(input_id)?;
    let in_shape = static_shape(ctx, input_id)?;
    if in_shape.len() != 4 {
        return ctx.fail(format!("input must be 4-D, got {in_shape:?}"));
    }
    let w_shape = static_shape(ctx, weight_id)?;
    if w_shape.len() != 4 {
        return ctx.fail(format!("weights must be 4-D, got {w_shape:?}"));
    }
    let stride = [node.attr_int("stride_h")? as i32, node.attr_int("stride_w")? as i32];
    let dilation = [node.attr_int_or("dilation_h", 1)? as i32, node.attr_int_or("dilation_w", 1)? as i32];
    let kernel = [w_shape[1], w_shape[2]];
    let Geometry { pad, sliced } = geometry(ctx, node, &in_shape, kernel, stride, dilation)?;
    let input = match sliced {
        Some(hw) => slice_spatial(ctx, &input, hw, "conv_input"),
        None => input,
    };
    let out_shape = shape_of(ctx, out);
    let out_channels = *out_shape.last().unwrap_or(&0);

    let weight = ctx.value(weight_id)?;
    let (op, weight) = match kind {
        ConvKind::Conv => (Op::Conv2d, weight),
        ConvKind::Depthwise => {
            // TFLite: [1, KH, KW, C * M]; TOSA: [KH, KW, C, M].
            let channels = in_shape[3];
            let multiplier = node.attr_int_or("depth_multiplier", 0)? as i32;
            let multiplier = if multiplier > 0 {
                multiplier
            } else if channels > 0 {
                w_shape[3] / channels
            } else {
                1
            };
            if channels * multiplier != w_shape[3] {
                return ctx.fail(format!(
                    "depthwise weights {w_shape:?} do not match {channels} input channels x {multiplier}"
                ));
            }
            (
                Op::DepthwiseConv2d,
                reshape_temp(ctx, &weight, &[kernel[0], kernel[1], channels, multiplier], "dw_weight"),
            )
        }
    };
    let bias = bias_or_zeros(ctx, node, 2, input.dtype, out_channels)?;
    let attr = move |acc_type: DType| match kind {
        ConvKind::Conv => OpAttribute::Conv2d {
            pad: pad.clone(),
            stride: stride.to_vec(),
            dilation: dilation.to_vec(),
            local_bound: false,
            acc_type,
        },
        ConvKind::Depthwise => OpAttribute::DepthwiseConv2d {
            pad: pad.clone(),
            stride: stride.to_vec(),
            dilation: dilation.to_vec(),
            local_bound: false,
            acc_type,
        },
    };
    let operands = ConvOperands { input, weight, bias, out_channels, input_id, weight_id, out, out_shape };
    emit_conv(node, ctx, op, attr, operands, |_, v| Ok(v))
}

/// FULLY_CONNECTED as a 1x1 CONV2D over a `[N, 1, 1, K]` view of the input.
fn fully_connected(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (input_id, weight_id, out) = (node.input(0)?, node.input(1)?, node.output(0)?);
    if node.attr_str("weights_format").unwrap_or("DEFAULT") != "DEFAULT" {
        return ctx.fail("only the DEFAULT weights format is supported");
    }
    let input = ctx.value(input_id)?;
    let in_shape = static_shape(ctx, input_id)?;
    let w_shape = static_shape(ctx, weight_id)?;
    if w_shape.len() != 2 {
        return ctx.fail(format!("weights must be 2-D, got {w_shape:?}"));
    }
    let (out_channels, k) = (w_shape[0], w_shape[1]);
    let total: i64 = in_shape.iter().map(|&d| d as i64).product();
    if k == 0 || total % k as i64 != 0 {
        return ctx.fail(format!("input {in_shape:?} is not a multiple of {k} features"));
    }
    let batch = (total / k as i64) as i32;
    let input4 = reshape_temp(ctx, &input, &[batch, 1, 1, k], "fc_input");
    let weight = ctx.value(weight_id)?;
    let weight4 = reshape_temp(ctx, &weight, &[out_channels, 1, 1, k], "fc_weight");
    let bias = bias_or_zeros(ctx, node, 2, input.dtype, out_channels)?;
    let out_shape = shape_of(ctx, out);
    let conv_shape = [batch, 1, 1, out_channels];
    let attr = |acc_type: DType| OpAttribute::Conv2d {
        pad: vec![0; 4],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        local_bound: false,
        acc_type,
    };
    let operands = ConvOperands {
        input: input4,
        weight: weight4,
        bias,
        out_channels,
        input_id,
        weight_id,
        out,
        out_shape: conv_shape.to_vec(),
    };
    let final_shape = out_shape.clone();
    emit_conv(node, ctx, Op::Conv2d, attr, operands, move |ctx, v| Ok(reshape_temp(ctx, &v, &final_shape, "fc_out")))
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------

fn pool(node: &SourceNode, ctx: &mut LowerCtx<'_>, average: bool) -> Result<()> {
    let (input_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(input_id)?;
    let in_shape = static_shape(ctx, input_id)?;
    if in_shape.len() != 4 {
        return ctx.fail(format!("input must be 4-D, got {in_shape:?}"));
    }
    let kernel = [node.attr_int("filter_height")? as i32, node.attr_int("filter_width")? as i32];
    let stride = [node.attr_int("stride_h")? as i32, node.attr_int("stride_w")? as i32];
    let Geometry { pad, sliced } = geometry(ctx, node, &in_shape, kernel, stride, [1, 1])?;
    let input = match sliced {
        Some(hw) => slice_spatial(ctx, &input, hw, "pool_input"),
        None => input,
    };
    let out_shape = shape_of(ctx, out);
    let value = if average {
        let quantized = is_quantized(ctx, input_id);
        let acc_type = if quantized {
            DType::I32
        } else if input.dtype == DType::F16 {
            DType::F16
        } else {
            DType::F32
        };
        let attr = OpAttribute::AvgPool2d { kernel: kernel.to_vec(), stride: stride.to_vec(), pad, acc_type };
        let zp = if quantized {
            let (_, zp) = scale_zp(ctx, input_id)?;
            ctx.scalar(TensorData::I8(vec![zp as i8]), "pool_zp")?
        } else {
            ctx.zero(input.dtype, "pool_zp")?
        };
        ctx.emit_temp(Op::AvgPool2d, attr, &[&input, &zp, &zp], input.dtype, &out_shape, "avg_pool")
    } else {
        let attr = OpAttribute::MaxPool2d {
            kernel: kernel.to_vec(),
            stride: stride.to_vec(),
            pad,
            nan_mode: NanMode::Propagate,
        };
        ctx.emit_temp(Op::MaxPool2d, attr, &[&input], input.dtype, &out_shape, "max_pool")
    };
    fused_activation(node, ctx, value, out)
}

// ---------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------

/// TFLite's left shift for quantized ADD / SUB.
const ADD_LEFT_SHIFT: i32 = 20;

fn binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let (a_id, b_id, out) = (node.input(0)?, node.input(1)?, node.output(0)?);
    let a = ctx.value(a_id)?;
    let b = ctx.value(b_id)?;
    let out_shape = shape_of(ctx, out);
    if !is_quantized(ctx, out) {
        let (a, b) = align_ranks(ctx, a, b);
        let value = if op == Op::Mul {
            float_mul(ctx, &a, &b, &out_shape, "mul")?
        } else {
            ctx.emit_temp(op, OpAttribute::None, &[&a, &b], a.dtype, &out_shape, "binary")
        };
        return fused_activation(node, ctx, value, out);
    }

    let (sa, za) = scale_zp(ctx, a_id)?;
    let (sb, zb) = scale_zp(ctx, b_id)?;
    let (so, zo) = scale_zp(ctx, out)?;
    let out_dtype = ctx.dtype(out)?;
    let value = if op == Op::Mul {
        let a32 = emit_rescale(
            ctx,
            &a,
            DType::I32,
            &RescaleSpec::per_tensor(1.0, za, 0).rounding(RoundingMode::Single),
            "mul_a",
        )?;
        let b32 = emit_rescale(
            ctx,
            &b,
            DType::I32,
            &RescaleSpec::per_tensor(1.0, zb, 0).rounding(RoundingMode::Single),
            "mul_b",
        )?;
        let (a32, b32) = align_ranks(ctx, a32, b32);
        let shift = mul_shift(ctx)?;
        let prod = ctx.emit_temp(Op::Mul, OpAttribute::None, &[&a32, &b32, &shift], DType::I32, &out_shape, "mul_acc");
        // TFLite computes this multiplier in float32 arithmetic.
        let scale = (sa as f32 * sb as f32 / so as f32) as f64;
        emit_rescale(ctx, &prod, out_dtype, &RescaleSpec::per_tensor(scale, 0, zo), "mul_out")?
    } else {
        // TFLite: (x - zp) << 20, then a doubling high multiply by
        // scale / twice_max with a rounding right shift. The left shift is an
        // exact RESCALE; the multiply is a RESCALE in double-rounding mode,
        // which reproduces the two roundings.
        let twice_max = 2.0 * sa.max(sb);
        let shift = (1i64 << ADD_LEFT_SHIFT) as f64;
        let widen = |ctx: &mut LowerCtx<'_>, x: &TensorRef, scale: f64, zp: i64, hint: &str| -> Result<TensorRef> {
            let shifted = emit_rescale(
                ctx,
                x,
                DType::I32,
                &RescaleSpec::per_tensor(shift, zp, 0).rounding(RoundingMode::Single),
                &format!("{hint}_shifted"),
            )?;
            emit_rescale(ctx, &shifted, DType::I32, &RescaleSpec::per_tensor(scale / twice_max, 0, 0), hint)
        };
        let a32 = widen(ctx, &a, sa, za, "add_a")?;
        let b32 = widen(ctx, &b, sb, zb, "add_b")?;
        let (a32, b32) = align_ranks(ctx, a32, b32);
        let sum = ctx.emit_temp(op, OpAttribute::None, &[&a32, &b32], DType::I32, &out_shape, "add_acc");
        emit_rescale(ctx, &sum, out_dtype, &RescaleSpec::per_tensor(twice_max / (shift * so), 0, zo), "add_out")?
    };
    fused_activation(node, ctx, value, out)
}

fn float_unary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    if !is_quantized(ctx, in_id) {
        ctx.emit(op, OpAttribute::None, &[&input], out)?;
        return Ok(());
    }
    let real = dequantize_temp(ctx, &input, in_id, "unary_in")?;
    let shape = real.shape.clone();
    let result = ctx.emit_temp(op, OpAttribute::None, &[&real], DType::F32, &shape, "unary");
    quantize_into(ctx, &result, out)?;
    Ok(())
}

fn activation(node: &SourceNode, ctx: &mut LowerCtx<'_>, act: &str) -> Result<()> {
    let input = ctx.value(node.input(0)?)?;
    apply_activation(ctx, act, input, node.output(0)?)
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

fn reshape(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    let mut shape: Vec<i64> = match node.optional_input(1) {
        Some(id) if ctx.tensor(id).is_constant() => ctx.constant_ints(id)?,
        _ => match node.attrs.get("new_shape") {
            Some(_) => node.attr_ints("new_shape")?.to_vec(),
            None => shape_of(ctx, out).iter().map(|&d| d as i64).collect(),
        },
    };
    let total: i64 = input.shape.iter().map(|&d| d as i64).product();
    if let Some(i) = shape.iter().position(|&d| d == -1) {
        let known: i64 = shape.iter().filter(|&&d| d != -1).product();
        if known == 0 || total % known != 0 {
            return ctx.fail(format!("cannot infer the -1 dimension of {shape:?} from {} elements", total));
        }
        shape[i] = total / known;
    }
    let s = ctx.shape_const(&shape, "reshape_shape");
    ctx.emit(Op::Reshape, OpAttribute::None, &[&input, &s], out)?;
    Ok(())
}

/// Operators whose only effect is the declared output shape (SQUEEZE, EXPAND_DIMS).
fn reshape_to_output(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    let shape = static_shape(ctx, out)?;
    let s = ctx.shape_const(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>(), "reshape_shape");
    ctx.emit(Op::Reshape, OpAttribute::None, &[&input, &s], out)?;
    Ok(())
}

fn pad(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    let paddings = ctx.constant_ints(node.input(1)?)?;
    if paddings.len() != 2 * input.shape.len() {
        return ctx.fail(format!("paddings has {} values for a rank-{} input", paddings.len(), input.shape.len()));
    }
    let padding = ctx.shape_const(&paddings, "paddings");
    let pad_const = if is_quantized(ctx, in_id) {
        let (_, zp) = scale_zp(ctx, in_id)?;
        ctx.scalar(TensorData::I8(vec![zp as i8]), "pad_value")?
    } else {
        ctx.zero(input.dtype, "pad_value")?
    };
    ctx.emit(Op::Pad, OpAttribute::None, &[&input, &padding, &pad_const], out)?;
    Ok(())
}

fn concatenation(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let out = node.output(0)?;
    let rank = shape_of(ctx, out).len() as i64;
    let mut axis = node.attr_int("axis")?;
    if axis < 0 {
        axis += rank;
    }
    let ids: Vec<TensorId> = node.inputs.iter().flatten().copied().collect();
    let mut values = Vec::with_capacity(ids.len());
    let quantized = is_quantized(ctx, out);
    let (so, zo) = if quantized { scale_zp(ctx, out)? } else { (1.0, 0) };
    for id in ids {
        let mut v = ctx.value(id)?;
        if quantized {
            let (si, zi) = scale_zp(ctx, id)?;
            if si != so || zi != zo {
                let dtype = ctx.dtype(out)?;
                v = emit_rescale(ctx, &v, dtype, &RescaleSpec::per_tensor(si / so, zi, zo), "concat_in")?;
            }
        }
        values.push(v);
    }
    let refs: Vec<&TensorRef> = values.iter().collect();
    let cat = ctx.emit_temp(
        Op::Concat,
        OpAttribute::Concat { axis: axis as i32 },
        &refs,
        values[0].dtype,
        &shape_of(ctx, out),
        "concat",
    );
    fused_activation(node, ctx, cat, out)
}

fn mean(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    let rank = input.shape.len() as i64;
    let mut axes: Vec<usize> = ctx
        .constant_ints(node.input(1)?)?
        .iter()
        .map(|&a| if a < 0 { (a + rank) as usize } else { a as usize })
        .collect();
    axes.sort_unstable();
    axes.dedup();
    let count: i64 = axes.iter().map(|&a| input.shape[a] as i64).product();
    if count <= 0 {
        return ctx.fail("cannot take the mean over an empty or dynamic axis");
    }
    let out_shape = shape_of(ctx, out);
    let quantized = is_quantized(ctx, in_id);
    let mut acc = if quantized {
        let (_, zi) = scale_zp(ctx, in_id)?;
        emit_rescale(
            ctx,
            &input,
            DType::I32,
            &RescaleSpec::per_tensor(1.0, zi, 0).rounding(RoundingMode::Single),
            "mean_in",
        )?
    } else {
        input
    };
    for &a in &axes {
        let mut shape = acc.shape.clone();
        shape[a] = 1;
        acc = ctx.emit_temp(
            Op::ReduceSum,
            OpAttribute::Reduce { axis: a as i32, nan_mode: NanMode::Propagate },
            &[&acc],
            acc.dtype,
            &shape,
            "mean_sum",
        );
    }
    let value = if quantized {
        let (si, _) = scale_zp(ctx, in_id)?;
        let (so, zo) = scale_zp(ctx, out)?;
        let dtype = ctx.dtype(out)?;
        emit_rescale(ctx, &acc, dtype, &RescaleSpec::per_tensor(si / (so * count as f64), 0, zo), "mean_out")?
    } else {
        let inv = f32_const(ctx, 1.0 / count as f32, acc.shape.len(), "mean_inv_count")?;
        let shape = acc.shape.clone();
        float_mul(ctx, &acc, &inv, &shape, "mean")?
    };
    let value = if value.shape != out_shape { reshape_temp(ctx, &value, &out_shape, "mean_shape") } else { value };
    ctx.bind(out, value);
    Ok(())
}

// ---------------------------------------------------------------------------
// Softmax and quantize / dequantize
// ---------------------------------------------------------------------------

fn softmax(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    let beta = node.attrs.get("beta").map_or(Ok(1.0), |_| node.attr_float("beta"))?;
    let quantized = is_quantized(ctx, in_id);
    let x = if quantized { dequantize_temp(ctx, &input, in_id, "softmax_in")? } else { input };
    let shape = x.shape.clone();
    let rank = shape.len();
    let axis = (rank - 1) as i32;
    let mut reduced = shape.clone();
    reduced[axis as usize] = 1;
    let reduce = |nan_mode| OpAttribute::Reduce { axis, nan_mode };
    let max = ctx.emit_temp(Op::ReduceMax, reduce(NanMode::Propagate), &[&x], x.dtype, &reduced, "softmax_max");
    let mut shifted = ctx.emit_temp(Op::Sub, OpAttribute::None, &[&x, &max], x.dtype, &shape, "softmax_shifted");
    if beta != 1.0 {
        let b = f32_const(ctx, beta as f32, rank, "softmax_beta")?;
        shifted = float_mul(ctx, &shifted, &b, &shape, "softmax_scaled")?;
    }
    let e = ctx.emit_temp(Op::Exp, OpAttribute::None, &[&shifted], x.dtype, &shape, "softmax_exp");
    let sum = ctx.emit_temp(Op::ReduceSum, reduce(NanMode::Propagate), &[&e], x.dtype, &reduced, "softmax_sum");
    let inv = ctx.emit_temp(Op::Reciprocal, OpAttribute::None, &[&sum], x.dtype, &reduced, "softmax_inv");
    if quantized {
        let result = float_mul(ctx, &e, &inv, &shape, "softmax")?;
        quantize_into(ctx, &result, out)?;
    } else {
        let shift = mul_shift(ctx)?;
        ctx.emit(Op::Mul, OpAttribute::None, &[&e, &inv, &shift], out)?;
    }
    Ok(())
}

fn quantize(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    if is_quantized(ctx, in_id) {
        // Requantization between quantized types.
        let (si, zi) = scale_zp(ctx, in_id)?;
        let (so, zo) = scale_zp(ctx, out)?;
        let dtype = ctx.dtype(out)?;
        let v = emit_rescale(ctx, &input, dtype, &RescaleSpec::per_tensor(si / so, zi, zo), "requantize")?;
        ctx.bind(out, v);
        return Ok(());
    }
    if input.dtype != DType::F32 {
        return ctx.fail(format!("cannot quantize a {:?} tensor", input.dtype));
    }
    quantize_into(ctx, &input, out)?;
    Ok(())
}

fn dequantize(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let input = ctx.value(in_id)?;
    if !is_quantized(ctx, in_id) {
        // FP16 -> FP32 and similar.
        ctx.emit(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&input], out)?;
        return Ok(());
    }
    let (scale, zp) = scale_zp(ctx, in_id)?;
    let shape = input.shape.clone();
    let rank = shape.len();
    let f = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[&input],
        DType::F32,
        &shape,
        "dequant_f32",
    );
    let zp_c = f32_const(ctx, zp as f32, rank, "dequant_zp")?;
    let centred = ctx.emit_temp(Op::Sub, OpAttribute::None, &[&f, &zp_c], DType::F32, &shape, "dequant_centred");
    let s_c = f32_const(ctx, scale as f32, rank, "dequant_scale")?;
    let shift = mul_shift(ctx)?;
    ctx.emit(Op::Mul, OpAttribute::None, &[&centred, &s_c, &shift], out)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// uint8 at the graph boundary
// ---------------------------------------------------------------------------

fn input_hook(id: TensorId, value: TensorRef, ctx: &mut LowerCtx<'_>) -> Result<TensorRef> {
    let t = ctx.tensor(id);
    if t.original_dtype != Some(SourceDType::U8) {
        return Ok(value);
    }
    let (_, zp) = scale_zp(ctx, id)?;
    let spec = RescaleSpec::per_tensor(1.0, zp + 128, zp).rounding(RoundingMode::Single).input_unsigned(true);
    emit_rescale(ctx, &value, DType::I8, &spec, &format!("{}_i8", t.name))
}

fn output_hook(id: TensorId, value: TensorRef, ctx: &mut LowerCtx<'_>) -> Result<TensorRef> {
    let t = ctx.tensor(id);
    if t.original_dtype != Some(SourceDType::U8) {
        return Ok(value);
    }
    let name = t.name.clone();
    let (_, zp) = scale_zp(ctx, id)?;
    // Free the output's name for the unsigned tensor.
    let value = ctx.block.rename(&value, &format!("{name}_i8")).unwrap_or(value);
    let spec = RescaleSpec::per_tensor(1.0, zp, zp + 128).rounding(RoundingMode::Single).output_unsigned(true);
    emit_rescale(ctx, &value, DType::I8, &spec, &name)
}

// ---------------------------------------------------------------------------
// Second batch of operators: activations, resizing, slicing, reductions,
// comparisons and the remaining element-wise functions.
// ---------------------------------------------------------------------------

/// Register the second batch of lowerings.
pub(super) fn register_more(r: &mut LoweringRegistry) {
    r.register("HARD_SWISH", hard_swish);
    r.register("PRELU", prelu);
    r.register("LEAKY_RELU", leaky_relu);
    r.register("ELU", elu);
    r.register("GELU", gelu);
    r.register("MAXIMUM", |n, c| max_min(n, c, Op::Maximum));
    r.register("MINIMUM", |n, c| max_min(n, c, Op::Minimum));
    r.register("EXP", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Exp, x, "exp"))));
    r.register("LOG", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Log, x, "log"))));
    r.register("RSQRT", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Rsqrt, x, "rsqrt"))));
    r.register("SQRT", |n, c| float_fn(n, c, sqrt_temp));
    r.register("SIN", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Sin, x, "sin"))));
    r.register("COS", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Cos, x, "cos"))));
    r.register("ABS", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Abs, x, "abs"))));
    r.register("FLOOR", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Floor, x, "floor"))));
    r.register("CEIL", |n, c| float_fn(n, c, |ctx, x| Ok(unary_temp(ctx, Op::Ceil, x, "ceil"))));
    r.register("NEG", neg);
    r.register("POW", |n, c| float_binary(n, c, Op::Pow));
    r.register("DIV", div);
    r.register("SQUARED_DIFFERENCE", squared_difference);
    r.register("L2_NORMALIZATION", l2_normalization);
    r.register("LOG_SOFTMAX", log_softmax);
    r.register("RESIZE_BILINEAR", |n, c| resize(n, c, ResizeMode::Bilinear));
    r.register("RESIZE_NEAREST_NEIGHBOR", |n, c| resize(n, c, ResizeMode::Nearest));
    r.register("TRANSPOSE_CONV", transpose_conv);
    r.register("TRANSPOSE", transpose);
    r.register("SLICE", slice);
    r.register("STRIDED_SLICE", strided_slice);
    r.register("SPLIT", split);
    r.register("SPLIT_V", split_v);
    r.register("PACK", pack);
    r.register("UNPACK", unpack);
    r.register("TILE", tile);
    r.register("GATHER", gather);
    r.register("DEPTH_TO_SPACE", depth_to_space);
    r.register("SPACE_TO_DEPTH", space_to_depth);
    r.register("REDUCE_MAX", |n, c| reduce(n, c, Op::ReduceMax));
    r.register("REDUCE_MIN", |n, c| reduce(n, c, Op::ReduceMin));
    r.register("SUM", |n, c| reduce(n, c, Op::ReduceSum));
    r.register("REDUCE_PROD", |n, c| reduce(n, c, Op::ReduceProduct));
    r.register("REDUCE_ANY", |n, c| reduce(n, c, Op::ReduceAny));
    r.register("ARG_MAX", |n, c| arg_max_min(n, c, false));
    r.register("ARG_MIN", |n, c| arg_max_min(n, c, true));
    r.register("CAST", cast);
    r.register("EQUAL", |n, c| compare(n, c, Op::Equal, false, false));
    r.register("NOT_EQUAL", |n, c| compare(n, c, Op::Equal, false, true));
    r.register("GREATER", |n, c| compare(n, c, Op::Greater, false, false));
    r.register("GREATER_EQUAL", |n, c| compare(n, c, Op::GreaterEqual, false, false));
    r.register("LESS", |n, c| compare(n, c, Op::Greater, true, false));
    r.register("LESS_EQUAL", |n, c| compare(n, c, Op::GreaterEqual, true, false));
    r.register("LOGICAL_AND", |n, c| bool_binary(n, c, Op::LogicalAnd));
    r.register("LOGICAL_OR", |n, c| bool_binary(n, c, Op::LogicalOr));
    r.register("LOGICAL_NOT", logical_not);
    r.register("SELECT", select);
    r.register("SELECT_V2", select);
    r.register("BATCH_MATMUL", batch_matmul);
}

fn unary_temp(ctx: &mut LowerCtx<'_>, op: Op, x: &TensorRef, hint: &str) -> TensorRef {
    let shape = x.shape.clone();
    ctx.emit_temp(op, OpAttribute::None, &[x], x.dtype, &shape, hint)
}

/// `sqrt(x)` as `x ^ 0.5`.
fn sqrt_temp(ctx: &mut LowerCtx<'_>, x: &TensorRef) -> Result<TensorRef> {
    let half = f32_const(ctx, 0.5, x.shape.len(), "half")?;
    let shape = x.shape.clone();
    Ok(ctx.emit_temp(Op::Pow, OpAttribute::None, &[x, &half], x.dtype, &shape, "sqrt"))
}

/// Element-wise binary operator on float tensors with rank alignment.
fn binary_temp(ctx: &mut LowerCtx<'_>, op: Op, a: &TensorRef, b: &TensorRef, hint: &str) -> Result<TensorRef> {
    let (a, b) = align_ranks(ctx, a.clone(), b.clone());
    let shape: Vec<i32> = a.shape.iter().zip(&b.shape).map(|(&x, &y)| x.max(y)).collect();
    Ok(if op == Op::Mul {
        float_mul(ctx, &a, &b, &shape, hint)?
    } else if matches!(op, Op::Maximum | Op::Minimum) {
        let attr = OpAttribute::NanPropagation { nan_mode: NanMode::Propagate };
        ctx.emit_temp(op, attr, &[&a, &b], a.dtype, &shape, hint)
    } else {
        ctx.emit_temp(op, OpAttribute::None, &[&a, &b], a.dtype, &shape, hint)
    })
}

/// Bind `value` as output `index` under the source tensor's name.
fn bind_named(ctx: &mut LowerCtx<'_>, node: &SourceNode, index: usize, value: TensorRef) -> Result<()> {
    let out = node.output(index)?;
    let name = ctx.tensor(out).name.clone();
    let value = ctx.block.rename(&value, &name).unwrap_or(value);
    ctx.bind(out, value);
    Ok(())
}

/// Run `f` on the float form of the node's inputs: quantized inputs are
/// dequantized first and the float result is quantized into the output
/// (TFLite's fixed-point and lookup-table kernels for these operators are
/// not reproduced step by step).
fn float_path(
    node: &SourceNode,
    ctx: &mut LowerCtx<'_>,
    n_inputs: usize,
    f: impl FnOnce(&mut LowerCtx<'_>, &[TensorRef]) -> Result<TensorRef>,
) -> Result<()> {
    let out = node.output(0)?;
    let mut xs = Vec::with_capacity(n_inputs);
    let mut quantized = false;
    for i in 0..n_inputs {
        let id = node.input(i)?;
        let v = ctx.value(id)?;
        if is_quantized(ctx, id) {
            quantized = true;
            xs.push(dequantize_temp(ctx, &v, id, &format!("in{i}"))?);
        } else {
            xs.push(v);
        }
    }
    let y = f(ctx, &xs)?;
    if quantized || is_quantized(ctx, out) {
        quantize_into(ctx, &y, out)?;
        Ok(())
    } else {
        fused_activation(node, ctx, y, out)
    }
}

fn float_fn(
    node: &SourceNode,
    ctx: &mut LowerCtx<'_>,
    f: impl FnOnce(&mut LowerCtx<'_>, &TensorRef) -> Result<TensorRef>,
) -> Result<()> {
    float_path(node, ctx, 1, |ctx, xs| f(ctx, &xs[0]))
}

fn float_binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    float_path(node, ctx, 2, |ctx, xs| binary_temp(ctx, op, &xs[0], &xs[1], "binary"))
}

/// `x * relu6(x + 3) / 6`.
fn hard_swish(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 1, |ctx, xs| {
        let x = &xs[0];
        let rank = x.shape.len();
        let shape = x.shape.clone();
        let three = f32_const(ctx, 3.0, rank, "three")?;
        let shifted = ctx.emit_temp(Op::Add, OpAttribute::None, &[x, &three], x.dtype, &shape, "hswish_shift");
        let attr = OpAttribute::Clamp {
            min_val: bound_bytes(ctx, x.dtype, 0.0)?,
            max_val: bound_bytes(ctx, x.dtype, 6.0)?,
            nan_mode: NanMode::Propagate,
        };
        let gate = ctx.emit_temp(Op::Clamp, attr, &[&shifted], x.dtype, &shape, "hswish_gate");
        let prod = float_mul(ctx, x, &gate, &shape, "hswish_prod")?;
        let sixth = f32_const(ctx, 1.0 / 6.0, rank, "sixth")?;
        float_mul(ctx, &prod, &sixth, &shape, "hswish")
    })
}

/// `max(x, 0) + alpha * min(x, 0)` with a broadcast `alpha` tensor.
fn prelu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 2, |ctx, xs| {
        let (x, alpha) = (&xs[0], &xs[1]);
        let zero = f32_const(ctx, 0.0, x.shape.len(), "zero")?;
        let pos = binary_temp(ctx, Op::Maximum, x, &zero, "prelu_pos")?;
        let neg = binary_temp(ctx, Op::Minimum, x, &zero, "prelu_neg")?;
        let scaled = binary_temp(ctx, Op::Mul, &neg, alpha, "prelu_scaled")?;
        binary_temp(ctx, Op::Add, &pos, &scaled, "prelu")
    })
}

fn leaky_relu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let alpha = node.attr_float("alpha")? as f32;
    float_path(node, ctx, 1, |ctx, xs| {
        let x = &xs[0];
        let k = f32_const(ctx, alpha, x.shape.len(), "alpha")?;
        let scaled = binary_temp(ctx, Op::Mul, x, &k, "leaky_scaled")?;
        if alpha <= 1.0 {
            binary_temp(ctx, Op::Maximum, x, &scaled, "leaky_relu")
        } else {
            binary_temp(ctx, Op::Minimum, x, &scaled, "leaky_relu")
        }
    })
}

/// `x > 0 ? x : exp(x) - 1`.
fn elu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 1, |ctx, xs| {
        let x = &xs[0];
        let shape = x.shape.clone();
        let zero = f32_const(ctx, 0.0, x.shape.len(), "zero")?;
        let one = f32_const(ctx, 1.0, x.shape.len(), "one")?;
        let e = unary_temp(ctx, Op::Exp, x, "elu_exp");
        let em1 = binary_temp(ctx, Op::Sub, &e, &one, "elu_neg")?;
        let (xz, zz) = align_ranks(ctx, x.clone(), zero);
        let cond = ctx.emit_temp(Op::Greater, OpAttribute::None, &[&xz, &zz], DType::Bool, &shape, "elu_cond");
        Ok(ctx.emit_temp(Op::Select, OpAttribute::None, &[&cond, x, &em1], x.dtype, &shape, "elu"))
    })
}

/// GELU: exact `0.5 x (1 + erf(x / sqrt 2))`, or the tanh approximation.
fn gelu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let approximate = node.attrs.get("approximate").map_or(Ok(false), |_| node.attr_bool("approximate"))?;
    float_path(node, ctx, 1, |ctx, xs| {
        let x = &xs[0];
        let rank = x.shape.len();
        let half = f32_const(ctx, 0.5, rank, "half")?;
        let one = f32_const(ctx, 1.0, rank, "one")?;
        let inner = if approximate {
            // tanh(sqrt(2/pi) * (x + 0.044715 x^3))
            let c = f32_const(ctx, 0.044715, rank, "gelu_c")?;
            let k = f32_const(ctx, (2.0f64 / std::f64::consts::PI).sqrt() as f32, rank, "gelu_k")?;
            let x2 = binary_temp(ctx, Op::Mul, x, x, "gelu_x2")?;
            let x3 = binary_temp(ctx, Op::Mul, &x2, x, "gelu_x3")?;
            let cx3 = binary_temp(ctx, Op::Mul, &x3, &c, "gelu_cx3")?;
            let sum = binary_temp(ctx, Op::Add, x, &cx3, "gelu_sum")?;
            let arg = binary_temp(ctx, Op::Mul, &sum, &k, "gelu_arg")?;
            unary_temp(ctx, Op::Tanh, &arg, "gelu_tanh")
        } else {
            let inv_sqrt2 = f32_const(ctx, std::f32::consts::FRAC_1_SQRT_2, rank, "inv_sqrt2")?;
            let arg = binary_temp(ctx, Op::Mul, x, &inv_sqrt2, "gelu_arg")?;
            unary_temp(ctx, Op::Erf, &arg, "gelu_erf")
        };
        let gate = binary_temp(ctx, Op::Add, &inner, &one, "gelu_gate")?;
        let hx = binary_temp(ctx, Op::Mul, x, &half, "gelu_hx")?;
        binary_temp(ctx, Op::Mul, &hx, &gate, "gelu")
    })
}

/// MAXIMUM / MINIMUM: quantized inputs share the output's parameters in
/// TFLite, so they are compared directly.
fn max_min(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let (a_id, b_id, out) = (node.input(0)?, node.input(1)?, node.output(0)?);
    let a = ctx.value(a_id)?;
    let b = ctx.value(b_id)?;
    if is_quantized(ctx, out) {
        if a.dtype != DType::I8 || b.dtype != DType::I8 {
            return ctx.fail("quantized MAXIMUM/MINIMUM needs INT8 operands");
        }
        // TOSA MAXIMUM/MINIMUM take INT32; widen, compare, narrow (exact).
        let a32 = emit_rescale(
            ctx,
            &a,
            DType::I32,
            &RescaleSpec::per_tensor(1.0, 0, 0).rounding(RoundingMode::Single),
            "max_a",
        )?;
        let b32 = emit_rescale(
            ctx,
            &b,
            DType::I32,
            &RescaleSpec::per_tensor(1.0, 0, 0).rounding(RoundingMode::Single),
            "max_b",
        )?;
        let y = binary_temp(ctx, op, &a32, &b32, "max_min")?;
        let shape = y.shape.clone();
        let y8 = ctx.emit_temp(
            Op::Cast,
            OpAttribute::Cast { input_unsigned: false },
            &[&y],
            DType::I8,
            &shape,
            "max_min_i8",
        );
        return bind_named(ctx, node, 0, y8);
    }
    let y = binary_temp(ctx, op, &a, &b, "max_min")?;
    fused_activation(node, ctx, y, out)
}

fn neg(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let x = ctx.value(in_id)?;
    if is_quantized(ctx, in_id) {
        let (_, zi) = scale_zp(ctx, in_id)?;
        let (_, zo) = scale_zp(ctx, out)?;
        let izp = ctx.scalar(TensorData::I8(vec![zi as i8]), "neg_in_zp")?;
        let ozp = ctx.scalar(TensorData::I8(vec![zo as i8]), "neg_out_zp")?;
        ctx.emit(Op::Negate, OpAttribute::None, &[&x, &izp, &ozp], out)?;
        return Ok(());
    }
    let zp = ctx.zero(x.dtype, "zp")?;
    ctx.emit(Op::Negate, OpAttribute::None, &[&x, &zp, &zp], out)?;
    Ok(())
}

fn div(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 2, |ctx, xs| {
        let inv = unary_temp(ctx, Op::Reciprocal, &xs[1], "div_inv");
        binary_temp(ctx, Op::Mul, &xs[0], &inv, "div")
    })
}

fn squared_difference(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 2, |ctx, xs| {
        let d = binary_temp(ctx, Op::Sub, &xs[0], &xs[1], "sqdiff_d")?;
        binary_temp(ctx, Op::Mul, &d, &d, "sqdiff")
    })
}

/// `x / sqrt(max(sum(x^2), 1e-6))` over the last axis.
fn l2_normalization(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 1, |ctx, xs| {
        let x = &xs[0];
        let rank = x.shape.len();
        let axis = (rank - 1) as i32;
        let mut reduced = x.shape.clone();
        reduced[axis as usize] = 1;
        let sq = binary_temp(ctx, Op::Mul, x, x, "l2_sq")?;
        let sum = ctx.emit_temp(
            Op::ReduceSum,
            OpAttribute::Reduce { axis, nan_mode: NanMode::Propagate },
            &[&sq],
            x.dtype,
            &reduced,
            "l2_sum",
        );
        let eps = f32_const(ctx, 1e-6, rank, "l2_eps")?;
        let floored = binary_temp(ctx, Op::Maximum, &sum, &eps, "l2_floored")?;
        let inv = unary_temp(ctx, Op::Rsqrt, &floored, "l2_inv");
        binary_temp(ctx, Op::Mul, x, &inv, "l2_norm")
    })
}

/// `x - max - log(sum(exp(x - max)))` over the last axis.
fn log_softmax(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    float_path(node, ctx, 1, |ctx, xs| {
        let x = &xs[0];
        let rank = x.shape.len();
        let axis = (rank - 1) as i32;
        let mut reduced = x.shape.clone();
        reduced[axis as usize] = 1;
        let reduce = |nan_mode| OpAttribute::Reduce { axis, nan_mode };
        let max = ctx.emit_temp(Op::ReduceMax, reduce(NanMode::Propagate), &[x], x.dtype, &reduced, "lsm_max");
        let shifted = binary_temp(ctx, Op::Sub, x, &max, "lsm_shifted")?;
        let e = unary_temp(ctx, Op::Exp, &shifted, "lsm_exp");
        let sum = ctx.emit_temp(Op::ReduceSum, reduce(NanMode::Propagate), &[&e], x.dtype, &reduced, "lsm_sum");
        let log = unary_temp(ctx, Op::Log, &sum, "lsm_log");
        binary_temp(ctx, Op::Sub, &shifted, &log, "log_softmax")
    })
}

// --- resizing ---------------------------------------------------------------

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs().max(1) } else { gcd(b, a % b) }
}

/// TOSA RESIZE parameters `(scale_n, scale_d, offset, border)` for one axis
/// mapping `input` samples to `output`, for the given coordinate convention.
fn resize_axis(
    ctx: &LowerCtx<'_>,
    input: i64,
    output: i64,
    align_corners: bool,
    half_pixel: bool,
    nearest: bool,
) -> Result<(i64, i64, i64, i64)> {
    let (mut n, mut d, offset, border) = if align_corners && output > 1 {
        // src = o * (in - 1) / (out - 1)
        (output - 1, input - 1, 0, 0)
    } else if half_pixel {
        // src = (o + 0.5) * in / out - 0.5; TOSA nearest rounds half up,
        // which with this offset is TFLite's floor((o + 0.5) * in / out).
        (2 * output, 2 * input, input - output, output - input)
    } else if nearest {
        // floor(o * in / out) via round-half-up(v - 0.5)
        (2 * output, 2 * input, -output, output - 2 * input)
    } else {
        // src = o * in / out
        (output, input, 0, output - input)
    };
    let g = gcd(n, d);
    let (mut offset, mut border) = (offset, border);
    if g > 1 && offset % g == 0 && border % g == 0 {
        n /= g;
        d /= g;
        offset /= g;
        border /= g;
    }
    if n <= 0 || d <= 0 || n > 1 << 11 {
        return ctx.fail(format!("resize from {input} to {output} exceeds TOSA's scale limits"));
    }
    Ok((n, d, offset, border))
}

fn resize(node: &SourceNode, ctx: &mut LowerCtx<'_>, mode: ResizeMode) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let size = ctx.constant_ints(node.input(1)?)?;
    if size.len() != 2 {
        return ctx.fail("size must have 2 values");
    }
    let align_corners = node.attrs.get("align_corners").map_or(Ok(false), |_| node.attr_bool("align_corners"))?;
    let half_pixel =
        node.attrs.get("half_pixel_centers").map_or(Ok(false), |_| node.attr_bool("half_pixel_centers"))?;
    let x = ctx.value(in_id)?;
    if x.shape.len() != 4 {
        return ctx.fail("resize input must be 4-D");
    }
    let nearest = mode == ResizeMode::Nearest;
    // An axis resized from a single sample repeats that sample (every output
    // position maps to index 0): a TILE, which RESIZE's parameter ranges
    // cannot express exactly. The remaining axis, if any, is a RESIZE.
    let tile_y = x.shape[1] == 1 && size[0] != 1;
    let tile_x = x.shape[2] == 1 && size[1] != 1;
    let (yn, yd, yo, yb) = if tile_y {
        (1, 1, 0, 0)
    } else {
        resize_axis(ctx, x.shape[1] as i64, size[0], align_corners, half_pixel, nearest)?
    };
    let (xn, xd, xo, xb) = if tile_x {
        (1, 1, 0, 0)
    } else {
        resize_axis(ctx, x.shape[2] as i64, size[1], align_corners, half_pixel, nearest)?
    };
    let out_shape = vec![x.shape[0], size[0] as i32, size[1] as i32, x.shape[3]];
    let emit_resize = |ctx: &mut LowerCtx<'_>, input: &TensorRef, dtype: DType| -> TensorRef {
        let mut cur = input.clone();
        if !(tile_y && tile_x) {
            let shape = vec![
                x.shape[0],
                if tile_y { 1 } else { size[0] as i32 },
                if tile_x { 1 } else { size[1] as i32 },
                x.shape[3],
            ];
            let scale = ctx.shape_const(&[yn, yd, xn, xd], "resize_scale");
            let offset = ctx.shape_const(&[yo, xo], "resize_offset");
            let border = ctx.shape_const(&[yb, xb], "resize_border");
            cur = ctx.emit_temp(
                Op::Resize,
                OpAttribute::Resize { mode },
                &[&cur, &scale, &offset, &border],
                dtype,
                &shape,
                "resize",
            );
        }
        if tile_y || tile_x {
            let multiples = [1, if tile_y { size[0] } else { 1 }, if tile_x { size[1] } else { 1 }, 1];
            let m = ctx.shape_const(&multiples, "resize_tile");
            cur = ctx.emit_temp(Op::Tile, OpAttribute::None, &[&cur, &m], dtype, &out_shape, "resize_tiled");
        }
        cur
    };
    if is_quantized(ctx, in_id) && nearest {
        // Pure data movement: exact in the quantized domain.
        let y = emit_resize(ctx, &x, x.dtype);
        return bind_named(ctx, node, 0, y);
    }
    if is_quantized(ctx, in_id) {
        let xf = dequantize_temp(ctx, &x, in_id, "resize_in")?;
        let y = emit_resize(ctx, &xf, DType::F32);
        quantize_into(ctx, &y, out)?;
        return Ok(());
    }
    let y = emit_resize(ctx, &x, x.dtype);
    bind_named(ctx, node, 0, y)
}

/// TRANSPOSE_CONV: inputs output_shape, weights `[OC, KH, KW, IC]`, input,
/// optional bias.
fn transpose_conv(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let out_shape: Vec<i32> = ctx.constant_ints(node.input(0)?)?.iter().map(|&d| d as i32).collect();
    let (weight_id, input_id, out) = (node.input(1)?, node.input(2)?, node.output(0)?);
    let input = ctx.value(input_id)?;
    let weight = ctx.value(weight_id)?;
    let in_shape = static_shape(ctx, input_id)?;
    let w_shape = static_shape(ctx, weight_id)?;
    if in_shape.len() != 4 || w_shape.len() != 4 || out_shape.len() != 4 {
        return ctx.fail("TRANSPOSE_CONV needs 4-D tensors");
    }
    let stride = [node.attr_int("stride_h")? as i32, node.attr_int("stride_w")? as i32];
    let (kh, kw) = (w_shape[1], w_shape[2]);
    // TFLite: output = (in - 1) * stride + k - pad_total, with SAME padding split
    // like a forward convolution; TOSA out_pad is the negative of that padding.
    let mut out_pad = [0i32; 4];
    for d in 0..2 {
        let (size, k, s, o) = (in_shape[1 + d], [kh, kw][d], stride[d], out_shape[1 + d]);
        let total = (size - 1) * s + k - o;
        let (before, after) = match node.attr_str("padding")? {
            "VALID" => (0, total),
            _ => (total / 2, total - total / 2),
        };
        out_pad[2 * d] = -before;
        out_pad[2 * d + 1] = -after;
    }
    let out_channels = out_shape[3];
    let bias = bias_or_zeros(ctx, node, 3, input.dtype, out_channels)?;
    let attr = move |acc_type: DType| OpAttribute::TransposeConv2d {
        out_pad: out_pad.to_vec(),
        stride: stride.to_vec(),
        local_bound: false,
        acc_type,
    };
    let operands = ConvOperands { input, weight, bias, out_channels, input_id, weight_id, out, out_shape };
    emit_conv(node, ctx, Op::TransposeConv2d, attr, operands, |_, v| Ok(v))
}

// --- data movement ---------------------------------------------------------

fn transpose(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let perms: Vec<i32> = ctx.constant_ints(node.input(1)?)?.iter().map(|&p| p as i32).collect();
    let y = crate::layout::transpose(ctx, &x, &perms, "transpose");
    bind_named(ctx, node, 0, y)
}

fn slice_temp(ctx: &mut LowerCtx<'_>, x: &TensorRef, start: &[i64], size: &[i64], hint: &str) -> TensorRef {
    let shape: Vec<i32> = size.iter().map(|&d| d as i32).collect();
    let s = ctx.shape_const(start, &format!("{hint}_start"));
    let z = ctx.shape_const(size, &format!("{hint}_size"));
    ctx.emit_temp(Op::Slice, OpAttribute::None, &[x, &s, &z], x.dtype, &shape, hint)
}

fn slice(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let begin = ctx.constant_ints(node.input(1)?)?;
    let mut size = ctx.constant_ints(node.input(2)?)?;
    if begin.len() != x.shape.len() || size.len() != x.shape.len() {
        return ctx.fail("begin and size must match the input rank");
    }
    for d in 0..size.len() {
        if size[d] < 0 {
            size[d] = x.shape[d] as i64 - begin[d];
        }
    }
    let y = slice_temp(ctx, &x, &begin, &size, "slice");
    bind_named(ctx, node, 0, y)
}

fn strided_slice(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let begin = ctx.constant_ints(node.input(1)?)?;
    let end = ctx.constant_ints(node.input(2)?)?;
    let strides = ctx.constant_ints(node.input(3)?)?;
    let masks = |k: &str| node.attr_int_or(k, 0);
    let (begin_mask, end_mask) = (masks("begin_mask")?, masks("end_mask")?);
    let (ellipsis, new_axis, shrink) = (masks("ellipsis_mask")?, masks("new_axis_mask")?, masks("shrink_axis_mask")?);
    if ellipsis != 0 || new_axis != 0 {
        return ctx.fail("ellipsis_mask and new_axis_mask are not supported");
    }
    if strides.iter().any(|&s| s < 1) {
        return ctx.fail(format!("negative or zero strides {strides:?} are not supported"));
    }
    let rank = x.shape.len();
    let mut start = vec![0i64; rank];
    let mut size = vec![0i64; rank];
    let mut out_shape = Vec::new();
    // Per axis: the stride and the number of elements it selects.
    let mut strided: Vec<(usize, i64, i64)> = Vec::new();
    for d in 0..rank {
        let dim = x.shape[d] as i64;
        let stride = strides.get(d).copied().unwrap_or(1);
        let mut b = begin.get(d).copied().unwrap_or(0);
        let mut e = end.get(d).copied().unwrap_or(dim);
        if b < 0 {
            b += dim;
        }
        if e < 0 {
            e += dim;
        }
        if begin_mask & (1 << d) != 0 {
            b = 0;
        }
        if end_mask & (1 << d) != 0 {
            e = dim;
        }
        let b = b.clamp(0, dim);
        let e = if shrink & (1 << d) != 0 { b + 1 } else { e.clamp(b, dim) };
        let count = (e - b + stride - 1) / stride;
        start[d] = b;
        // The contiguous span covering the selected elements.
        size[d] = if count == 0 { 0 } else { (count - 1) * stride + 1 };
        if stride > 1 && count > 0 {
            strided.push((d, stride, count));
        }
        if shrink & (1 << d) == 0 {
            out_shape.push(count as i32);
        }
    }
    let mut cur = slice_temp(ctx, &x, &start, &size, "strided_slice");
    // A stride s on an axis of n selected elements: pad the span to n * s,
    // view it as [n, s] and keep column 0.
    for (d, stride, count) in strided {
        let span = cur.shape[d] as i64;
        let mut pads = vec![0i64; 2 * rank];
        pads[2 * d + 1] = count * stride - span;
        let mut padded_shape = cur.shape.clone();
        padded_shape[d] = (count * stride) as i32;
        let p = ctx.shape_const(&pads, "stride_pads");
        let zero = ctx.zero(cur.dtype, "stride_pad_value")?;
        let padded =
            ctx.emit_temp(Op::Pad, OpAttribute::None, &[&cur, &p, &zero], cur.dtype, &padded_shape, "stride_padded");
        let mut split_shape: Vec<i32> = padded_shape[..d].to_vec();
        split_shape.push(count as i32);
        split_shape.push(stride as i32);
        split_shape.extend_from_slice(&padded_shape[d + 1..]);
        let split = reshape_temp(ctx, &padded, &split_shape, "stride_split");
        let mut s_start = vec![0i64; split_shape.len()];
        let mut s_size: Vec<i64> = split_shape.iter().map(|&v| v as i64).collect();
        s_start[d + 1] = 0;
        s_size[d + 1] = 1;
        let picked = slice_temp(ctx, &split, &s_start, &s_size, "stride_picked");
        let mut merged = padded_shape.clone();
        merged[d] = count as i32;
        cur = reshape_temp(ctx, &picked, &merged, "stride_merged");
    }
    let y = if shrink != 0 { reshape_temp(ctx, &cur, &out_shape, "strided_slice_shrunk") } else { cur };
    bind_named(ctx, node, 0, y)
}

fn split_along(ctx: &mut LowerCtx<'_>, node: &SourceNode, x: &TensorRef, axis: usize, sizes: &[i64]) -> Result<()> {
    let mut offset = 0i64;
    for (i, &len) in sizes.iter().enumerate() {
        let mut start = vec![0i64; x.shape.len()];
        let mut size: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
        start[axis] = offset;
        size[axis] = len;
        let piece = slice_temp(ctx, x, &start, &size, &format!("split{i}"));
        bind_named(ctx, node, i, piece)?;
        offset += len;
    }
    Ok(())
}

fn split(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let axis_v = ctx.constant_ints(node.input(0)?)?;
    let x = ctx.value(node.input(1)?)?;
    let rank = x.shape.len() as i64;
    let axis = axis_v.first().copied().unwrap_or(0);
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    let n = node.attr_int("num_splits")?;
    let dim = x.shape[axis] as i64;
    if n <= 0 || dim % n != 0 {
        return ctx.fail(format!("cannot split {dim} into {n} equal parts"));
    }
    let sizes = vec![dim / n; n as usize];
    split_along(ctx, node, &x, axis, &sizes)
}

fn split_v(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let mut sizes = ctx.constant_ints(node.input(1)?)?;
    let axis_v = ctx.constant_ints(node.input(2)?)?;
    let rank = x.shape.len() as i64;
    let axis = axis_v.first().copied().unwrap_or(0);
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    let dim = x.shape[axis] as i64;
    if let Some(i) = sizes.iter().position(|&s| s == -1) {
        let known: i64 = sizes.iter().filter(|&&s| s >= 0).sum();
        sizes[i] = dim - known;
    }
    split_along(ctx, node, &x, axis, &sizes)
}

fn pack(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let ids: Vec<TensorId> = node.inputs.iter().flatten().copied().collect();
    let out = node.output(0)?;
    let out_shape = shape_of(ctx, out);
    let rank = out_shape.len() as i64;
    let mut axis = node.attr_int("axis")?;
    if axis < 0 {
        axis += rank;
    }
    let mut parts = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let v = ctx.value(id)?;
        let mut shape = v.shape.clone();
        shape.insert(axis as usize, 1);
        parts.push(reshape_temp(ctx, &v, &shape, &format!("pack{i}")));
    }
    let refs: Vec<&TensorRef> = parts.iter().collect();
    let y =
        ctx.emit_temp(Op::Concat, OpAttribute::Concat { axis: axis as i32 }, &refs, parts[0].dtype, &out_shape, "pack");
    bind_named(ctx, node, 0, y)
}

fn unpack(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len() as i64;
    let mut axis = node.attr_int("axis")?;
    if axis < 0 {
        axis += rank;
    }
    let axis = axis as usize;
    let n = node.attr_int("num")? as usize;
    let mut out_shape = x.shape.clone();
    out_shape.remove(axis);
    for i in 0..n {
        let mut start = vec![0i64; x.shape.len()];
        let mut size: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
        start[axis] = i as i64;
        size[axis] = 1;
        let piece = slice_temp(ctx, &x, &start, &size, &format!("unpack{i}"));
        let piece = reshape_temp(ctx, &piece, &out_shape, &format!("unpack{i}_shaped"));
        bind_named(ctx, node, i, piece)?;
    }
    Ok(())
}

fn tile(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let multiples = ctx.constant_ints(node.input(1)?)?;
    let shape: Vec<i32> = x.shape.iter().zip(&multiples).map(|(&d, &m)| d * m as i32).collect();
    let m = ctx.shape_const(&multiples, "multiples");
    let y = ctx.emit_temp(Op::Tile, OpAttribute::None, &[&x, &m], x.dtype, &shape, "tile");
    bind_named(ctx, node, 0, y)
}

/// GATHER along `axis` with `batch_dims == 0`: TOSA's GATHER works on
/// `[N, K, C]` values with `[N, W]` indices, so the axis is moved to the
/// front, the result reshaped and the index dimensions moved into place.
fn gather(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let params = ctx.value(node.input(0)?)?;
    let indices = ctx.value(node.input(1)?)?;
    if node.attr_int_or("batch_dims", 0)? != 0 {
        return ctx.fail("batch_dims other than 0 is not supported");
    }
    let rank = params.shape.len() as i64;
    let mut axis = node.attr_int_or("axis", 0)?;
    if axis < 0 {
        axis += rank;
    }
    let axis = axis as usize;
    let indices = if indices.dtype == DType::I64 || indices.dtype == DType::I32 {
        indices
    } else {
        return ctx.fail("indices must be INT32 or INT64");
    };
    // Move the gathered axis first: [K, rest...]
    let mut perm: Vec<i32> = vec![axis as i32];
    perm.extend((0..params.shape.len() as i32).filter(|&d| d as usize != axis));
    let moved = if axis == 0 { params.clone() } else { crate::layout::transpose(ctx, &params, &perm, "gather_perm") };
    let k = moved.shape[0];
    let c: i32 = moved.shape[1..].iter().product();
    let values = reshape_temp(ctx, &moved, &[1, k, c], "gather_values");
    let w: i32 = indices.shape.iter().product();
    let idx = reshape_temp(ctx, &indices, &[1, w], "gather_indices");
    let gathered = ctx.emit_temp(Op::Gather, OpAttribute::None, &[&values, &idx], params.dtype, &[1, w, c], "gather");
    // [indices..., rest...] then move the index dims to where the axis was.
    let mut shape: Vec<i32> = indices.shape.clone();
    shape.extend_from_slice(&moved.shape[1..]);
    let reshaped = reshape_temp(ctx, &gathered, &shape, "gather_shaped");
    let ir = indices.shape.len();
    let mut back: Vec<i32> = Vec::with_capacity(shape.len());
    back.extend((ir..ir + axis).map(|d| d as i32));
    back.extend((0..ir).map(|d| d as i32));
    back.extend((ir + axis..shape.len()).map(|d| d as i32));
    let y = if crate::layout::is_identity(&back) {
        reshaped
    } else {
        crate::layout::transpose(ctx, &reshaped, &back, "gather_out")
    };
    bind_named(ctx, node, 0, y)
}

fn depth_to_space(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let b = node.attr_int("block_size")? as i32;
    let [n, h, w, c] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if c % (b * b) != 0 {
        return ctx.fail("channels must be divisible by block_size^2");
    }
    let c2 = c / (b * b);
    let r1 = reshape_temp(ctx, &x, &[n, h, w, b, b, c2], "d2s_split");
    let t = crate::layout::transpose(ctx, &r1, &[0, 1, 3, 2, 4, 5], "d2s_perm");
    let y = reshape_temp(ctx, &t, &[n, h * b, w * b, c2], "d2s");
    bind_named(ctx, node, 0, y)
}

fn space_to_depth(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let b = node.attr_int("block_size")? as i32;
    let [n, h, w, c] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if h % b != 0 || w % b != 0 {
        return ctx.fail("height and width must be divisible by block_size");
    }
    let r1 = reshape_temp(ctx, &x, &[n, h / b, b, w / b, b, c], "s2d_split");
    let t = crate::layout::transpose(ctx, &r1, &[0, 1, 3, 2, 4, 5], "s2d_perm");
    let y = reshape_temp(ctx, &t, &[n, h / b, w / b, b * b * c], "s2d");
    bind_named(ctx, node, 0, y)
}

// --- reductions and comparisons --------------------------------------------

fn reduce(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let x = ctx.value(in_id)?;
    let rank = x.shape.len() as i64;
    let mut axes: Vec<usize> = ctx
        .constant_ints(node.input(1)?)?
        .iter()
        .map(|&a| if a < 0 { (a + rank) as usize } else { a as usize })
        .collect();
    axes.sort_unstable();
    axes.dedup();
    let quantized = is_quantized(ctx, in_id);
    let floating = op == Op::ReduceProduct;
    if quantized && (floating || op == Op::ReduceSum) {
        // SUM of quantized data: TFLite requantizes the sum; do it in float.
        return float_path(node, ctx, 1, |ctx, xs| {
            let mut acc = xs[0].clone();
            for &a in &axes {
                let mut shape = acc.shape.clone();
                shape[a] = 1;
                acc = ctx.emit_temp(
                    op,
                    OpAttribute::Reduce { axis: a as i32, nan_mode: NanMode::Propagate },
                    &[&acc],
                    acc.dtype,
                    &shape,
                    "reduce",
                );
            }
            let out_shape = shape_of(ctx, node.output(0)?);
            Ok(if acc.shape != out_shape { reshape_temp(ctx, &acc, &out_shape, "reduce_shaped") } else { acc })
        });
    }
    let mut acc = x;
    for &a in &axes {
        let mut shape = acc.shape.clone();
        shape[a] = 1;
        acc = ctx.emit_temp(
            op,
            OpAttribute::Reduce { axis: a as i32, nan_mode: NanMode::Propagate },
            &[&acc],
            acc.dtype,
            &shape,
            "reduce",
        );
    }
    let out_shape = shape_of(ctx, out);
    let y = if acc.shape != out_shape { reshape_temp(ctx, &acc, &out_shape, "reduce_shaped") } else { acc };
    bind_named(ctx, node, 0, y)
}

fn arg_max_min(node: &SourceNode, ctx: &mut LowerCtx<'_>, min: bool) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let x = ctx.value(in_id)?;
    let rank = x.shape.len() as i64;
    let axis_v = ctx.constant_ints(node.input(1)?)?;
    let mut axis = axis_v.first().copied().unwrap_or(0);
    if axis < 0 {
        axis += rank;
    }
    let x = if min {
        if !x.dtype.is_float() {
            return ctx.fail("ARG_MIN is supported for float inputs only");
        }
        let zp = ctx.zero(x.dtype, "zp")?;
        let shape = x.shape.clone();
        ctx.emit_temp(Op::Negate, OpAttribute::None, &[&x, &zp, &zp], x.dtype, &shape, "argmin_neg")
    } else {
        x
    };
    let out_dtype = ctx.dtype(out)?;
    let mut shape = x.shape.clone();
    shape.remove(axis as usize);
    let y = ctx.emit_temp(
        Op::ArgMax,
        OpAttribute::ArgMax { axis: axis as i32, nan_mode: NanMode::Propagate },
        &[&x],
        out_dtype,
        &shape,
        "argmax",
    );
    bind_named(ctx, node, 0, y)
}

fn cast(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (in_id, out) = (node.input(0)?, node.output(0)?);
    let x = ctx.value(in_id)?;
    let to = ctx.dtype(out)?;
    if x.dtype == to {
        return bind_named(ctx, node, 0, x);
    }
    ctx.emit(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], out)?;
    Ok(())
}

/// Widen INT8/INT16 operands to INT32 for operators TOSA defines on INT32.
fn widen_for_compare(ctx: &mut LowerCtx<'_>, x: TensorRef) -> TensorRef {
    if matches!(x.dtype, DType::I8 | DType::I16) {
        let shape = x.shape.clone();
        ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], DType::I32, &shape, "widened")
    } else {
        x
    }
}

fn compare(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op, swap: bool, negate: bool) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let a = widen_for_compare(ctx, a);
    let b = widen_for_compare(ctx, b);
    let (a, b) = if swap { (b, a) } else { (a, b) };
    let (a, b) = align_ranks(ctx, a, b);
    let shape: Vec<i32> = a.shape.iter().zip(&b.shape).map(|(&x, &y)| x.max(y)).collect();
    let mut y = ctx.emit_temp(op, OpAttribute::None, &[&a, &b], DType::Bool, &shape, "compare");
    if negate {
        y = ctx.emit_temp(Op::LogicalNot, OpAttribute::None, &[&y], DType::Bool, &shape, "not");
    }
    bind_named(ctx, node, 0, y)
}

fn bool_binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let (a, b) = align_ranks(ctx, a, b);
    let shape: Vec<i32> = a.shape.iter().zip(&b.shape).map(|(&x, &y)| x.max(y)).collect();
    let y = ctx.emit_temp(op, OpAttribute::None, &[&a, &b], DType::Bool, &shape, "logical");
    bind_named(ctx, node, 0, y)
}

fn logical_not(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let y = unary_temp(ctx, Op::LogicalNot, &x, "logical_not");
    bind_named(ctx, node, 0, y)
}

fn select(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let c = ctx.value(node.input(0)?)?;
    let a = ctx.value(node.input(1)?)?;
    let b = ctx.value(node.input(2)?)?;
    let out = node.output(0)?;
    let shape = shape_of(ctx, out);
    let rank = shape.len();
    let pad = |ctx: &mut LowerCtx<'_>, t: TensorRef, hint: &str| -> TensorRef {
        if t.shape.len() == rank {
            return t;
        }
        let mut s = vec![1; rank - t.shape.len()];
        s.extend_from_slice(&t.shape);
        reshape_temp(ctx, &t, &s, hint)
    };
    let c = pad(ctx, c, "select_c");
    let a = pad(ctx, a, "select_a");
    let b = pad(ctx, b, "select_b");
    let y = ctx.emit_temp(Op::Select, OpAttribute::None, &[&c, &a, &b], a.dtype, &shape, "select");
    bind_named(ctx, node, 0, y)
}

/// BATCH_MATMUL on float tensors: operands are brought to rank 3 with a
/// shared batch dimension.
fn batch_matmul(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let adj_x = node.attrs.get("adj_x").map_or(Ok(false), |_| node.attr_bool("adj_x"))?;
    let adj_y = node.attrs.get("adj_y").map_or(Ok(false), |_| node.attr_bool("adj_y"))?;
    float_path(node, ctx, 2, |ctx, xs| {
        let (mut a, mut b) = (xs[0].clone(), xs[1].clone());
        let ra = a.shape.len();
        let rb = b.shape.len();
        if ra < 2 || rb < 2 {
            return ctx.fail("BATCH_MATMUL operands must have rank 2 or more");
        }
        if adj_x {
            let mut p: Vec<i32> = (0..ra as i32).collect();
            p.swap(ra - 1, ra - 2);
            a = crate::layout::transpose(ctx, &a, &p, "bmm_a_t");
        }
        if adj_y {
            let mut p: Vec<i32> = (0..rb as i32).collect();
            p.swap(rb - 1, rb - 2);
            b = crate::layout::transpose(ctx, &b, &p, "bmm_b_t");
        }
        let batch_a: i32 = a.shape[..ra - 2].iter().product();
        let batch_b: i32 = b.shape[..rb - 2].iter().product();
        let (m, k) = (a.shape[ra - 2], a.shape[ra - 1]);
        let n = b.shape[rb - 1];
        let batch = batch_a.max(batch_b);
        let a3 = reshape_temp(ctx, &a, &[batch_a, m, k], "bmm_a");
        let b3 = reshape_temp(ctx, &b, &[batch_b, k, n], "bmm_b");
        let a3 = if batch_a == batch { a3 } else { tile_batch(ctx, &a3, batch) };
        let b3 = if batch_b == batch { b3 } else { tile_batch(ctx, &b3, batch) };
        let zp = ctx.zero(a.dtype, "zp")?;
        let y = ctx.emit_temp(Op::MatMul, OpAttribute::None, &[&a3, &b3, &zp, &zp], a.dtype, &[batch, m, n], "bmm");
        let mut shape: Vec<i32> = if ra >= rb { a.shape[..ra - 2].to_vec() } else { b.shape[..rb - 2].to_vec() };
        shape.extend([m, n]);
        Ok(reshape_temp(ctx, &y, &shape, "bmm_out"))
    })
}

fn tile_batch(ctx: &mut LowerCtx<'_>, x: &TensorRef, batch: i32) -> TensorRef {
    let m = ctx.shape_const(&[batch as i64, 1, 1], "bmm_multiples");
    let shape = vec![batch, x.shape[1], x.shape[2]];
    ctx.emit_temp(Op::Tile, OpAttribute::None, &[x, &m], x.dtype, &shape, "bmm_tiled")
}
