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

use tosa_ir::{DType, NanMode, Op, OpAttribute, RoundingMode, TensorData, TensorRef};

use crate::error::Result;
use crate::lower::{LowerCtx, LoweringRegistry};
use crate::quant::{RescaleSpec, emit_rescale};
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

/// Quantize an FP32 tensor into source tensor `out`: `round(x / scale) + zero_point`, saturated.
fn quantize_into(ctx: &mut LowerCtx<'_>, input: &TensorRef, out: TensorId) -> Result<TensorRef> {
    let (scale, zp) = scale_zp(ctx, out)?;
    let shape = input.shape.clone();
    let rank = shape.len();
    let inv = f32_const(ctx, (1.0 / scale) as f32, rank, "quant_inv_scale")?;
    let scaled = float_mul(ctx, input, &inv, &shape, "quant_scaled")?;
    let zp_c = f32_const(ctx, zp as f32, rank, "quant_zp")?;
    let shifted = ctx.emit_temp(Op::Add, OpAttribute::None, &[&scaled, &zp_c], DType::F32, &shape, "quant_shifted");
    ctx.emit(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&shifted], out)
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
