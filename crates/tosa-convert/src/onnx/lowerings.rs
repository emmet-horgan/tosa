//! Lowerings of ONNX operators (default domain, float models) to TOSA.
//!
//! ONNX is channels-first. Convolutions and pools transpose their input to
//! NHWC, run the TOSA operator, and transpose the result back; the cleanup
//! passes cancel the transposes that meet between layers and sink those
//! blocked by element-wise operators (Relu, Add, BatchNormalization's
//! multiply and add), so a typical CNN keeps one transpose at the input and
//! one before its first layout-sensitive reshape.
//!
//! Shapes are computed from the operands (ONNX models rarely record
//! intermediate shapes). Shape arithmetic on static tensors (Shape, Gather,
//! Unsqueeze, Concat, Slice, Cast on constants) is evaluated at conversion
//! time, so the `Reshape(x, Concat(Gather(Shape(x), 0), [-1]))` idiom that
//! exporters produce for flattening becomes a plain RESHAPE.

use tosa_ir::{DType, NanMode, Op, OpAttribute, TensorData, TensorRef};

use crate::error::Result;
use crate::layout::{self, NCHW_TO_NHWC, NHWC_TO_NCHW};
use crate::lower::{LowerCtx, LoweringRegistry};
use crate::source::{AttrValue, SourceNode, TensorId};

pub fn registry() -> LoweringRegistry {
    let mut r = LoweringRegistry::new();
    r.register("Conv", conv);
    r.register("MaxPool", |n, c| pool(n, c, PoolKind::Max));
    r.register("AveragePool", |n, c| pool(n, c, PoolKind::Average));
    r.register("GlobalAveragePool", |n, c| pool(n, c, PoolKind::GlobalAverage));
    r.register("GlobalMaxPool", |n, c| pool(n, c, PoolKind::GlobalMax));
    r.register("Gemm", gemm);
    r.register("MatMul", matmul);
    r.register("BatchNormalization", batch_norm);
    r.register("Relu", |n, c| clamp_op(n, c, 0.0, f32::MAX as f64));
    r.register("Clip", clip);
    r.register("LeakyRelu", leaky_relu);
    r.register("Sigmoid", |n, c| unary(n, c, Op::Sigmoid));
    r.register("Tanh", |n, c| unary(n, c, Op::Tanh));
    r.register("Exp", |n, c| unary(n, c, Op::Exp));
    r.register("Log", |n, c| unary(n, c, Op::Log));
    r.register("Abs", |n, c| unary(n, c, Op::Abs));
    r.register("Floor", |n, c| unary(n, c, Op::Floor));
    r.register("Ceil", |n, c| unary(n, c, Op::Ceil));
    r.register("Erf", |n, c| unary(n, c, Op::Erf));
    r.register("Reciprocal", |n, c| unary(n, c, Op::Reciprocal));
    r.register("Neg", neg);
    r.register("Sqrt", sqrt);
    r.register("Add", |n, c| binary(n, c, Op::Add));
    r.register("Sub", |n, c| binary(n, c, Op::Sub));
    r.register("Mul", |n, c| binary(n, c, Op::Mul));
    r.register("Div", div);
    r.register("Pow", |n, c| binary(n, c, Op::Pow));
    r.register("Max", |n, c| nary(n, c, Op::Maximum));
    r.register("Min", |n, c| nary(n, c, Op::Minimum));
    r.register("Softmax", softmax);
    r.register("Flatten", flatten);
    r.register("Reshape", reshape);
    r.register("Squeeze", |n, c| squeeze(n, c, false));
    r.register("Unsqueeze", |n, c| squeeze(n, c, true));
    r.register("Transpose", transpose);
    r.register("Concat", concat);
    r.register("Identity", identity);
    r.register("Dropout", identity);
    r.register("Shape", shape);
    r.register("Gather", gather);
    r.register("Slice", slice);
    r.register("Cast", cast);
    r.register("ReduceMean", reduce_mean);
    r.register("Pad", pad);
    register_more(&mut r);
    r
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn opset(ctx: &LowerCtx<'_>) -> i64 {
    match ctx.source.metadata.get("opset") {
        Some(AttrValue::Int(v)) => *v,
        _ => 0,
    }
}

fn static_shape(ctx: &LowerCtx<'_>, t: &TensorRef) -> Result<Vec<i64>> {
    if t.shape.iter().any(|&d| d < 0) {
        return ctx.fail(format!("tensor '{}' has a dynamic shape {:?}", t.name, t.shape));
    }
    Ok(t.shape.iter().map(|&d| d as i64).collect())
}

fn i32s(shape: &[i64]) -> Vec<i32> {
    shape.iter().map(|&d| d as i32).collect()
}

fn ints_attr_or(node: &SourceNode, key: &str, default: Vec<i64>) -> Result<Vec<i64>> {
    if node.attrs.contains_key(key) { Ok(node.attr_ints(key)?.to_vec()) } else { Ok(default) }
}

fn float_attr_or(node: &SourceNode, key: &str, default: f64) -> Result<f64> {
    if node.attrs.contains_key(key) { node.attr_float(key) } else { Ok(default) }
}

/// A constant holding `value`, shaped `[1; rank]` for broadcasting.
fn f32_const(ctx: &mut LowerCtx<'_>, value: f32, rank: usize, hint: &str) -> Result<TensorRef> {
    Ok(ctx.block.constant(hint, &vec![1; rank], TensorData::F32(vec![value]))?)
}

fn mul_shift(ctx: &mut LowerCtx<'_>) -> Result<TensorRef> {
    ctx.scalar(TensorData::I8(vec![0]), "shift")
}

/// Broadcast shape of two operand shapes (NumPy rules, after rank alignment).
fn broadcast(ctx: &LowerCtx<'_>, a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(a.len());
    for (&x, &y) in a.iter().zip(b) {
        out.push(if x == y || y == 1 {
            x
        } else if x == 1 {
            y
        } else {
            return ctx.fail(format!("shapes {a:?} and {b:?} do not broadcast"));
        });
    }
    Ok(out)
}

fn reshape_temp(ctx: &mut LowerCtx<'_>, input: &TensorRef, shape: &[i32], hint: &str) -> TensorRef {
    if input.shape == shape {
        return input.clone();
    }
    let s = ctx.shape_const(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>(), &format!("{hint}_shape"));
    ctx.emit_temp(Op::Reshape, OpAttribute::None, &[input, &s], input.dtype, shape, hint)
}

/// Give two operands equal rank by prepending 1s to the lower-rank one.
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
        (reshape_temp(ctx, &a, &s, &format!("{}_bcast", a.name)), b)
    } else {
        let s = pad(&b, ra);
        (a, reshape_temp(ctx, &b, &s, &format!("{}_bcast", b.name)))
    }
}

fn binary_temp(ctx: &mut LowerCtx<'_>, op: Op, a: &TensorRef, b: &TensorRef, hint: &str) -> Result<TensorRef> {
    let (a, b) = align_ranks(ctx, a.clone(), b.clone());
    let shape = broadcast(ctx, &a.shape, &b.shape)?;
    Ok(if op == Op::Mul {
        let shift = mul_shift(ctx)?;
        ctx.emit_temp(Op::Mul, OpAttribute::None, &[&a, &b, &shift], a.dtype, &shape, hint)
    } else {
        ctx.emit_temp(op, OpAttribute::None, &[&a, &b], a.dtype, &shape, hint)
    })
}

/// Bind `value` as the node's output `index`, under the source tensor's name.
fn bind_output(ctx: &mut LowerCtx<'_>, node: &SourceNode, index: usize, value: TensorRef) -> Result<()> {
    let out = node.output(index)?;
    let name = ctx.tensor(out).name.clone();
    let value = ctx.block.rename(&value, &name).unwrap_or(value);
    ctx.bind(out, value);
    Ok(())
}

fn bound_bytes(ctx: &LowerCtx<'_>, dtype: DType, value: f64) -> Result<Vec<u8>> {
    Ok(match dtype {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f64(value).to_le_bytes().to_vec(),
        DType::Bf16 => half::bf16::from_f64(value).to_le_bytes().to_vec(),
        DType::I8 => vec![(value as i64).clamp(-128, 127) as i8 as u8],
        DType::I16 => ((value as i64).clamp(-32768, 32767) as i16).to_le_bytes().to_vec(),
        DType::I32 => ((value as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32).to_le_bytes().to_vec(),
        other => return ctx.fail(format!("cannot clamp a {other:?} tensor")),
    })
}

fn clamp_temp(ctx: &mut LowerCtx<'_>, input: &TensorRef, lo: f64, hi: f64, hint: &str) -> Result<TensorRef> {
    let attr = OpAttribute::Clamp {
        min_val: bound_bytes(ctx, input.dtype, lo)?,
        max_val: bound_bytes(ctx, input.dtype, hi)?,
        nan_mode: NanMode::Propagate,
    };
    let shape = input.shape.clone();
    Ok(ctx.emit_temp(Op::Clamp, attr, &[input], input.dtype, &shape, hint))
}

/// Spatial geometry of a 2-D convolution or pool on an NHWC input.
struct Geometry2d {
    /// TOSA `[top, bottom, left, right]`.
    pad: Vec<i32>,
    out: [i32; 2],
    /// `[H, W]` to slice the input to first, if the windows do not cover it.
    sliced: Option<[i32; 2]>,
}

/// Resolve ONNX padding (`pads` / `auto_pad`, `ceil_mode`) for `input_hw`.
#[allow(clippy::too_many_arguments)]
fn geometry(
    ctx: &LowerCtx<'_>,
    node: &SourceNode,
    input_hw: [i64; 2],
    kernel: [i64; 2],
    stride: [i64; 2],
    dilation: [i64; 2],
    ceil_mode: bool,
    pads_allow_window: bool,
) -> Result<Geometry2d> {
    let auto_pad = node.attrs.get("auto_pad").map_or(Ok("NOTSET"), |_| node.attr_str("auto_pad"))?;
    let pads = ints_attr_or(node, "pads", vec![0; 4])?;
    if pads.len() != 4 {
        return ctx.fail(format!("pads must have 4 values for 2-D, got {pads:?}"));
    }
    let mut pad = Vec::with_capacity(4);
    let mut out = [0; 2];
    let mut sliced = [input_hw[0] as i32, input_hw[1] as i32];
    let mut any_slice = false;
    for d in 0..2 {
        let (size, k, s, dil) = (input_hw[d], kernel[d], stride[d], dilation[d]);
        let effective = (k - 1) * dil + 1;
        let (before, mut after) = match auto_pad {
            "NOTSET" => (pads[d], pads[2 + d]),
            "VALID" => (0, 0),
            "SAME_UPPER" | "SAME_LOWER" => {
                let o = (size + s - 1) / s;
                let total = ((o - 1) * s + effective - size).max(0);
                let small = total / 2;
                if auto_pad == "SAME_UPPER" { (small, total - small) } else { (total - small, small) }
            }
            other => return ctx.fail(format!("auto_pad {other} is not supported")),
        };
        let span = size + before + after - effective;
        if span < 0 {
            return ctx.fail(format!("window of {effective} does not fit input of {size} with padding"));
        }
        let o = if ceil_mode { (span + s - 1) / s + 1 } else { span / s + 1 };
        let needed = (o - 1) * s + effective;
        let excess = size + before + after - needed;
        if excess > 0 {
            after -= excess;
            if after < 0 {
                sliced[d] = (size + after) as i32;
                any_slice = true;
                after = 0;
            }
        } else if excess < 0 {
            // ceil_mode: the last window hangs over the end; pad it.
            after -= excess;
            if pads_allow_window && after >= k {
                return ctx.fail("ceil_mode padding exceeds the kernel size");
            }
        }
        if before < 0 || after < 0 {
            return ctx.fail("negative padding");
        }
        pad.push(before as i32);
        pad.push(after as i32);
        out[d] = o as i32;
    }
    Ok(Geometry2d { pad, out, sliced: any_slice.then_some(sliced) })
}

fn slice_spatial(ctx: &mut LowerCtx<'_>, input: &TensorRef, hw: [i32; 2], hint: &str) -> TensorRef {
    let shape = vec![input.shape[0], hw[0], hw[1], input.shape[3]];
    let start = ctx.shape_const(&[0, 0, 0, 0], &format!("{hint}_start"));
    let size = ctx.shape_const(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>(), &format!("{hint}_size"));
    ctx.emit_temp(Op::Slice, OpAttribute::None, &[input, &start, &size], input.dtype, &shape, hint)
}

// ---------------------------------------------------------------------------
// Convolution and pooling
// ---------------------------------------------------------------------------

fn conv(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let w = ctx.value(node.input(1)?)?;
    let x_shape = static_shape(ctx, &x)?;
    let w_shape = static_shape(ctx, &w)?;
    if x_shape.len() != 4 || w_shape.len() != 4 {
        return ctx.fail(format!("only 2-D convolutions are supported (input {x_shape:?}, weights {w_shape:?})"));
    }
    if x.dtype != DType::F32 && x.dtype != DType::F16 && x.dtype != DType::Bf16 {
        return ctx.fail(format!("only float convolutions are supported, got {:?}", x.dtype));
    }
    let [n, c, h, wdt] = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];
    let [m, c_per_group, kh, kw] = [w_shape[0], w_shape[1], w_shape[2], w_shape[3]];
    let group = node.attr_int_or("group", 1)?;
    let kernel = ints_attr_or(node, "kernel_shape", vec![kh, kw])?;
    if kernel != [kh, kw] {
        return ctx.fail(format!("kernel_shape {kernel:?} does not match weights {w_shape:?}"));
    }
    let strides = ints_attr_or(node, "strides", vec![1, 1])?;
    let dilations = ints_attr_or(node, "dilations", vec![1, 1])?;
    if strides.len() != 2 || dilations.len() != 2 {
        return ctx.fail("strides and dilations must have 2 values");
    }
    let g =
        geometry(ctx, node, [h, wdt], [kh, kw], [strides[0], strides[1]], [dilations[0], dilations[1]], false, false)?;

    let mut xt = layout::transpose(ctx, &x, &NCHW_TO_NHWC, "conv_nhwc");
    if let Some(hw) = g.sliced {
        xt = slice_spatial(ctx, &xt, hw, "conv_input");
    }
    let zp = ctx.zero(x.dtype, "zp")?;
    let bias = match node.optional_input(2) {
        Some(id) => ctx.value(id)?,
        None => ctx.block.constant("bias", &[m as i32], TensorData::zeros(x.dtype, m as usize)?)?,
    };
    let out_shape = vec![n as i32, g.out[0], g.out[1], m as i32];
    let acc_type = if x.dtype == DType::F16 { DType::F16 } else { DType::F32 };
    let depthwise = group == c && c_per_group == 1 && group > 1;
    let y = if group == 1 {
        let wt = layout::transpose(ctx, &w, &layout::OIHW_TO_OHWI, "conv_weight");
        let attr = OpAttribute::Conv2d {
            pad: g.pad.clone(),
            stride: i32s(&strides),
            dilation: i32s(&dilations),
            local_bound: false,
            acc_type,
        };
        ctx.emit_temp(Op::Conv2d, attr, &[&xt, &wt, &bias, &zp, &zp], x.dtype, &out_shape, "conv")
    } else if depthwise {
        if m % c != 0 {
            return ctx.fail(format!("depthwise weights {w_shape:?} do not divide {c} channels"));
        }
        let mult = (m / c) as i32;
        // [C*M, 1, KH, KW] -> [KH, KW, 1, C*M] -> [KH, KW, C, M]
        let wt = layout::transpose(ctx, &w, &[2, 3, 1, 0], "dw_weight_t");
        let wt = reshape_temp(ctx, &wt, &[kh as i32, kw as i32, c as i32, mult], "dw_weight");
        let attr = OpAttribute::DepthwiseConv2d {
            pad: g.pad.clone(),
            stride: i32s(&strides),
            dilation: i32s(&dilations),
            local_bound: false,
            acc_type,
        };
        ctx.emit_temp(Op::DepthwiseConv2d, attr, &[&xt, &wt, &bias, &zp, &zp], x.dtype, &out_shape, "dwconv")
    } else {
        return ctx.fail(format!("grouped convolution with group={group} is not supported"));
    };
    let y = layout::transpose(ctx, &y, &NHWC_TO_NCHW, "conv_nchw");
    bind_output(ctx, node, 0, y)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolKind {
    Max,
    Average,
    GlobalAverage,
    GlobalMax,
}

fn pool(node: &SourceNode, ctx: &mut LowerCtx<'_>, kind: PoolKind) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let x_shape = static_shape(ctx, &x)?;
    if x_shape.len() != 4 {
        return ctx.fail(format!("only 2-D pooling is supported, got {x_shape:?}"));
    }
    let [n, c, h, w] = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];
    let global = matches!(kind, PoolKind::GlobalAverage | PoolKind::GlobalMax);
    let kernel = if global { vec![h, w] } else { node.attr_ints("kernel_shape")?.to_vec() };
    let strides = if global { vec![1, 1] } else { ints_attr_or(node, "strides", vec![1, 1])? };
    let dilations = ints_attr_or(node, "dilations", vec![1, 1])?;
    if kernel.len() != 2 || strides.len() != 2 || dilations != [1, 1] {
        return ctx.fail("only 2-D pooling without dilation is supported");
    }
    if node.attr_int_or("count_include_pad", 0)? != 0 && ints_attr_or(node, "pads", vec![0; 4])?.iter().any(|&p| p != 0)
    {
        return ctx.fail("count_include_pad=1 with padding is not supported (TOSA excludes padding)");
    }
    let ceil_mode = node.attr_int_or("ceil_mode", 0)? != 0;
    let g = geometry(ctx, node, [h, w], [kernel[0], kernel[1]], [strides[0], strides[1]], [1, 1], ceil_mode, true)?;
    let mut xt = layout::transpose(ctx, &x, &NCHW_TO_NHWC, "pool_nhwc");
    if let Some(hw) = g.sliced {
        xt = slice_spatial(ctx, &xt, hw, "pool_input");
    }
    let out_shape = vec![n as i32, g.out[0], g.out[1], c as i32];
    let y = match kind {
        PoolKind::Max | PoolKind::GlobalMax => {
            let attr = OpAttribute::MaxPool2d {
                kernel: i32s(&kernel),
                stride: i32s(&strides),
                pad: g.pad,
                nan_mode: NanMode::Propagate,
            };
            ctx.emit_temp(Op::MaxPool2d, attr, &[&xt], x.dtype, &out_shape, "max_pool")
        }
        PoolKind::Average | PoolKind::GlobalAverage => {
            let acc_type = if x.dtype == DType::F16 { DType::F16 } else { DType::F32 };
            let attr = OpAttribute::AvgPool2d { kernel: i32s(&kernel), stride: i32s(&strides), pad: g.pad, acc_type };
            let zp = ctx.zero(x.dtype, "pool_zp")?;
            ctx.emit_temp(Op::AvgPool2d, attr, &[&xt, &zp, &zp], x.dtype, &out_shape, "avg_pool")
        }
    };
    let y = layout::transpose(ctx, &y, &NHWC_TO_NCHW, "pool_nchw");
    bind_output(ctx, node, 0, y)
}

// ---------------------------------------------------------------------------
// Matrix products
// ---------------------------------------------------------------------------

/// `[1, M, K] x [1, K, N]` MATMUL of 2-D operands, as `[M, N]`.
fn matmul_2d(ctx: &mut LowerCtx<'_>, a: &TensorRef, b: &TensorRef, hint: &str) -> Result<TensorRef> {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k != k2 {
        return ctx.fail(format!("inner dimensions {k} and {k2} differ"));
    }
    let a3 = reshape_temp(ctx, a, &[1, m, k], &format!("{hint}_a"));
    let b3 = reshape_temp(ctx, b, &[1, k, n], &format!("{hint}_b"));
    let zp = ctx.zero(a.dtype, "zp")?;
    let out_dtype = if a.dtype == DType::F16 { DType::F16 } else { a.dtype };
    let y = ctx.emit_temp(Op::MatMul, OpAttribute::None, &[&a3, &b3, &zp, &zp], out_dtype, &[1, m, n], hint);
    Ok(reshape_temp(ctx, &y, &[m, n], &format!("{hint}_2d")))
}

fn gemm(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let mut a = ctx.value(node.input(0)?)?;
    let mut b = ctx.value(node.input(1)?)?;
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return ctx.fail("Gemm operands must be 2-D");
    }
    if node.attr_int_or("transA", 0)? != 0 {
        a = layout::transpose(ctx, &a, &[1, 0], "gemm_a_t");
    }
    if node.attr_int_or("transB", 0)? != 0 {
        b = layout::transpose(ctx, &b, &[1, 0], "gemm_b_t");
    }
    let mut y = matmul_2d(ctx, &a, &b, "gemm")?;
    let alpha = float_attr_or(node, "alpha", 1.0)?;
    if alpha != 1.0 {
        let k = f32_const(ctx, alpha as f32, 2, "gemm_alpha")?;
        y = binary_temp(ctx, Op::Mul, &y, &k, "gemm_scaled")?;
    }
    if let Some(c_id) = node.optional_input(2) {
        let mut c = ctx.value(c_id)?;
        let beta = float_attr_or(node, "beta", 1.0)?;
        if beta != 1.0 {
            let k = f32_const(ctx, beta as f32, c.shape.len(), "gemm_beta")?;
            c = binary_temp(ctx, Op::Mul, &c, &k, "gemm_bias_scaled")?;
        }
        y = binary_temp(ctx, Op::Add, &y, &c, "gemm_out")?;
    }
    bind_output(ctx, node, 0, y)
}

fn matmul(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let (ra, rb) = (a.shape.len(), b.shape.len());
    let y = if ra == 2 && rb == 2 {
        matmul_2d(ctx, &a, &b, "matmul")?
    } else if ra >= 2 && rb == 2 {
        // Fold the leading dimensions into the rows.
        let k = a.shape[ra - 1];
        let rows: i32 = a.shape[..ra - 1].iter().product();
        let a2 = reshape_temp(ctx, &a, &[rows, k], "matmul_a");
        let y = matmul_2d(ctx, &a2, &b, "matmul")?;
        let mut shape = a.shape[..ra - 1].to_vec();
        shape.push(b.shape[1]);
        reshape_temp(ctx, &y, &shape, "matmul_out")
    } else if ra == 3 && rb == 3 && a.shape[0] == b.shape[0] {
        let zp = ctx.zero(a.dtype, "zp")?;
        let shape = vec![a.shape[0], a.shape[1], b.shape[2]];
        ctx.emit_temp(Op::MatMul, OpAttribute::None, &[&a, &b, &zp, &zp], a.dtype, &shape, "matmul")
    } else {
        return ctx.fail(format!("MatMul of shapes {:?} and {:?} is not supported", a.shape, b.shape));
    };
    bind_output(ctx, node, 0, y)
}

// ---------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------

fn unary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let folded = match op {
        Op::Floor => fold_unary(node, ctx, f64::floor)?,
        Op::Ceil => fold_unary(node, ctx, f64::ceil)?,
        Op::Abs => fold_unary(node, ctx, f64::abs)?,
        _ => false,
    };
    if folded {
        return Ok(());
    }
    let x = ctx.value(node.input(0)?)?;
    let shape = x.shape.clone();
    let y = ctx.emit_temp(op, OpAttribute::None, &[&x], x.dtype, &shape, "unary");
    bind_output(ctx, node, 0, y)
}

fn neg(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let zp = ctx.zero(x.dtype, "zp")?;
    let shape = x.shape.clone();
    let y = ctx.emit_temp(Op::Negate, OpAttribute::None, &[&x, &zp, &zp], x.dtype, &shape, "neg");
    bind_output(ctx, node, 0, y)
}

fn sqrt(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let half = f32_const(ctx, 0.5, x.shape.len(), "half")?;
    let y = binary_temp(ctx, Op::Pow, &x, &half, "sqrt")?;
    bind_output(ctx, node, 0, y)
}

/// Whether every input of `node` is a conversion-time constant that has not
/// been materialized as a TOSA value (shape arithmetic).
fn all_inputs_constant(node: &SourceNode, ctx: &LowerCtx<'_>) -> bool {
    node.inputs.iter().flatten().all(|&id| ctx.is_constant(id) && !ctx.is_available_as_value(id))
}

/// The elements of a numeric constant as `f64`.
fn const_f64(data: &TensorData) -> Option<Vec<f64>> {
    Some(match data {
        TensorData::I8(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::I16(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::F64(v) => v.clone(),
        TensorData::Bool(v) => v.iter().map(|&x| x as i32 as f64).collect(),
        _ => return None,
    })
}

/// `f64` elements back into a constant of `dtype` (integers truncate).
fn const_from_f64(values: &[f64], dtype: DType) -> Option<TensorData> {
    Some(match dtype {
        DType::I8 => TensorData::I8(values.iter().map(|&x| x as i8).collect()),
        DType::I16 => TensorData::I16(values.iter().map(|&x| x as i16).collect()),
        DType::I32 => TensorData::I32(values.iter().map(|&x| x as i32).collect()),
        DType::I64 => TensorData::I64(values.iter().map(|&x| x as i64).collect()),
        DType::F32 => TensorData::F32(values.iter().map(|&x| x as f32).collect()),
        DType::F64 => TensorData::F64(values.to_vec()),
        DType::Bool => TensorData::Bool(values.iter().map(|&x| x != 0.0).collect()),
        _ => return None,
    })
}

/// Fold an element-wise binary operator on constants (scalar/vector
/// broadcast only). Returns `false` when the node is not foldable.
fn fold_binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, f: impl Fn(f64, f64) -> f64) -> Result<bool> {
    if node.inputs.len() < 2 || !all_inputs_constant(node, ctx) {
        return Ok(false);
    }
    let a = ctx.constant(node.input(0)?)?;
    let b = ctx.constant(node.input(1)?)?;
    let dtype = a.dtype();
    let (Some(av), Some(bv)) = (const_f64(a), const_f64(b)) else { return Ok(false) };
    let values: Vec<f64> = if av.len() == bv.len() {
        av.iter().zip(&bv).map(|(&x, &y)| f(x, y)).collect()
    } else if bv.len() == 1 {
        av.iter().map(|&x| f(x, bv[0])).collect()
    } else if av.len() == 1 {
        bv.iter().map(|&y| f(av[0], y)).collect()
    } else {
        return Ok(false);
    };
    let values: Vec<f64> = if dtype.is_float() { values } else { values.iter().map(|v| v.trunc()).collect() };
    let Some(data) = const_from_f64(&values, dtype) else { return Ok(false) };
    ctx.define_constant(node.output(0)?, data);
    Ok(true)
}

/// Fold an element-wise unary operator on a constant.
fn fold_unary(node: &SourceNode, ctx: &mut LowerCtx<'_>, f: impl Fn(f64) -> f64) -> Result<bool> {
    if !all_inputs_constant(node, ctx) {
        return Ok(false);
    }
    let x = ctx.constant(node.input(0)?)?;
    let dtype = x.dtype();
    let Some(xv) = const_f64(x) else { return Ok(false) };
    let values: Vec<f64> = xv.iter().map(|&v| f(v)).collect();
    let Some(data) = const_from_f64(&values, dtype) else { return Ok(false) };
    ctx.define_constant(node.output(0)?, data);
    Ok(true)
}

fn binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let folded = match op {
        Op::Add => fold_binary(node, ctx, |x, y| x + y)?,
        Op::Sub => fold_binary(node, ctx, |x, y| x - y)?,
        Op::Mul => fold_binary(node, ctx, |x, y| x * y)?,
        _ => false,
    };
    if folded {
        return Ok(());
    }
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let y = binary_temp(ctx, op, &a, &b, "binary")?;
    bind_output(ctx, node, 0, y)
}

fn div(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    if fold_binary(node, ctx, |x, y| x / y)? {
        return Ok(());
    }
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    if !a.dtype.is_float() {
        return ctx.fail("integer division is not supported");
    }
    let shape = b.shape.clone();
    let inv = ctx.emit_temp(Op::Reciprocal, OpAttribute::None, &[&b], b.dtype, &shape, "div_inv");
    let y = binary_temp(ctx, Op::Mul, &a, &inv, "div")?;
    bind_output(ctx, node, 0, y)
}

fn nary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let ids: Vec<TensorId> = node.inputs.iter().flatten().copied().collect();
    let mut acc = ctx.value(ids[0])?;
    let attr = if matches!(op, Op::Maximum | Op::Minimum) {
        OpAttribute::NanPropagation { nan_mode: NanMode::Propagate }
    } else {
        OpAttribute::None
    };
    for &id in &ids[1..] {
        let b = ctx.value(id)?;
        let (a, b) = align_ranks(ctx, acc, b);
        let shape = broadcast(ctx, &a.shape, &b.shape)?;
        acc = ctx.emit_temp(op, attr.clone(), &[&a, &b], a.dtype, &shape, "nary");
    }
    bind_output(ctx, node, 0, acc)
}

fn clamp_op(node: &SourceNode, ctx: &mut LowerCtx<'_>, lo: f64, hi: f64) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let y = clamp_temp(ctx, &x, lo, hi, "clamp")?;
    bind_output(ctx, node, 0, y)
}

fn clip(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let scalar = |ctx: &LowerCtx<'_>, id: Option<TensorId>| -> Result<Option<f64>> {
        match id {
            Some(id) if ctx.is_constant(id) => match ctx.constant(id)? {
                TensorData::F32(v) if v.len() == 1 => Ok(Some(v[0] as f64)),
                TensorData::F16(v) if v.len() == 1 => Ok(Some(v[0].to_f64())),
                other => ctx.fail(format!("Clip bound must be a float scalar, got {other:?}")),
            },
            Some(_) => ctx.fail("Clip bounds must be constants"),
            None => Ok(None),
        }
    };
    let (lo, hi) = if opset(ctx) >= 11 {
        (scalar(ctx, node.optional_input(1))?, scalar(ctx, node.optional_input(2))?)
    } else {
        (
            node.attrs.get("min").map(|_| node.attr_float("min")).transpose()?,
            node.attrs.get("max").map(|_| node.attr_float("max")).transpose()?,
        )
    };
    let y = clamp_temp(ctx, &x, lo.unwrap_or(f32::MIN as f64), hi.unwrap_or(f32::MAX as f64), "clip")?;
    bind_output(ctx, node, 0, y)
}

fn leaky_relu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let alpha = float_attr_or(node, "alpha", 0.01)?;
    let k = f32_const(ctx, alpha as f32, x.shape.len(), "leaky_alpha")?;
    let scaled = binary_temp(ctx, Op::Mul, &x, &k, "leaky_scaled")?;
    let shape = x.shape.clone();
    let attr = OpAttribute::NanPropagation { nan_mode: NanMode::Propagate };
    let y = ctx.emit_temp(Op::Maximum, attr, &[&x, &scaled], x.dtype, &shape, "leaky_relu");
    bind_output(ctx, node, 0, y)
}

/// Inference batch normalization as a per-channel multiply and add with
/// constants computed here: `y = x * a + b`, `a = scale / sqrt(var + eps)`,
/// `b = bias - mean * a`.
fn batch_norm(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let f32s = |ctx: &LowerCtx<'_>, id: TensorId| -> Result<Vec<f32>> {
        match ctx.constant(id)? {
            TensorData::F32(v) => Ok(v.clone()),
            other => ctx.fail(format!("BatchNormalization parameters must be float constants, got {other:?}")),
        }
    };
    let scale = f32s(ctx, node.input(1)?)?;
    let bias = f32s(ctx, node.input(2)?)?;
    let mean = f32s(ctx, node.input(3)?)?;
    let var = f32s(ctx, node.input(4)?)?;
    let eps = float_attr_or(node, "epsilon", 1e-5)? as f32;
    let c = scale.len();
    if bias.len() != c || mean.len() != c || var.len() != c {
        return ctx.fail("BatchNormalization parameters have differing lengths");
    }
    let a: Vec<f32> = (0..c).map(|i| scale[i] / (var[i] + eps).sqrt()).collect();
    let b: Vec<f32> = (0..c).map(|i| bias[i] - mean[i] * a[i]).collect();
    let rank = x.shape.len();
    if rank < 2 {
        return ctx.fail("BatchNormalization input must have a channel dimension");
    }
    let mut shape = vec![1; rank];
    shape[1] = c as i32;
    let a_t = ctx.block.constant("bn_scale", &shape, TensorData::F32(a))?;
    let b_t = ctx.block.constant("bn_shift", &shape, TensorData::F32(b))?;
    let scaled = binary_temp(ctx, Op::Mul, &x, &a_t, "bn_scaled")?;
    let y = binary_temp(ctx, Op::Add, &scaled, &b_t, "bn")?;
    bind_output(ctx, node, 0, y)
}

fn softmax(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len() as i64;
    let default_axis = if opset(ctx) >= 13 { -1 } else { 1 };
    let mut axis = node.attr_int_or("axis", default_axis)?;
    if axis < 0 {
        axis += rank;
    }
    if axis < 0 || axis >= rank {
        return ctx.fail(format!("axis {axis} out of range for rank {rank}"));
    }
    // Before opset 13 the input is treated as 2-D [prod(:axis), prod(axis:)]
    // and normalized over the second part; reshape to make that explicit.
    let (x2, restore) = if opset(ctx) < 13 && axis != rank - 1 {
        let outer: i32 = x.shape[..axis as usize].iter().product();
        let inner: i32 = x.shape[axis as usize..].iter().product();
        (reshape_temp(ctx, &x, &[outer, inner], "softmax_2d"), Some(x.shape.clone()))
    } else {
        (x.clone(), None)
    };
    let axis = if restore.is_some() { 1 } else { axis as i32 };
    let shape = x2.shape.clone();
    let mut reduced = shape.clone();
    reduced[axis as usize] = 1;
    let reduce = OpAttribute::Reduce { axis, nan_mode: NanMode::Propagate };
    let max = ctx.emit_temp(Op::ReduceMax, reduce.clone(), &[&x2], x2.dtype, &reduced, "softmax_max");
    let shifted = ctx.emit_temp(Op::Sub, OpAttribute::None, &[&x2, &max], x2.dtype, &shape, "softmax_shifted");
    let e = ctx.emit_temp(Op::Exp, OpAttribute::None, &[&shifted], x2.dtype, &shape, "softmax_exp");
    let sum = ctx.emit_temp(Op::ReduceSum, reduce, &[&e], x2.dtype, &reduced, "softmax_sum");
    let inv = ctx.emit_temp(Op::Reciprocal, OpAttribute::None, &[&sum], x2.dtype, &reduced, "softmax_inv");
    let shift = mul_shift(ctx)?;
    let mut y = ctx.emit_temp(Op::Mul, OpAttribute::None, &[&e, &inv, &shift], x2.dtype, &shape, "softmax");
    if let Some(original) = restore {
        y = reshape_temp(ctx, &y, &original, "softmax_out");
    }
    bind_output(ctx, node, 0, y)
}

fn reduce_mean(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len() as i64;
    let axes: Vec<i64> = if let Some(id) = node.optional_input(1) {
        ctx.constant_ints(id)?
    } else if node.attrs.contains_key("axes") {
        node.attr_ints("axes")?.to_vec()
    } else {
        (0..rank).collect()
    };
    let mut axes: Vec<usize> = axes.iter().map(|&a| if a < 0 { (a + rank) as usize } else { a as usize }).collect();
    axes.sort_unstable();
    axes.dedup();
    let keep = node.attr_int_or("keepdims", 1)? != 0;
    let count: i64 = axes.iter().map(|&a| x.shape[a] as i64).product();
    let mut acc = x.clone();
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
    let inv = f32_const(ctx, 1.0 / count as f32, acc.shape.len(), "mean_inv")?;
    let mut y = binary_temp(ctx, Op::Mul, &acc, &inv, "mean")?;
    if !keep {
        let shape: Vec<i32> = (0..rank as usize).filter(|d| !axes.contains(d)).map(|d| x.shape[d]).collect();
        y = reshape_temp(ctx, &y, &shape, "mean_squeezed");
    }
    bind_output(ctx, node, 0, y)
}

fn pad(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len();
    let pads: Vec<i64> =
        if let Some(id) = node.optional_input(1) { ctx.constant_ints(id)? } else { node.attr_ints("pads")?.to_vec() };
    if pads.len() != 2 * rank {
        return ctx.fail(format!("pads has {} values for a rank-{rank} input", pads.len()));
    }
    let mode = node.attrs.get("mode").map_or(Ok("constant"), |_| node.attr_str("mode"))?;
    // ONNX: [begin_0, ..., begin_n, end_0, ..., end_n]; TOSA: [begin_0, end_0, ...].
    let tosa_pads: Vec<i64> = (0..rank).flat_map(|d| [pads[d], pads[rank + d]]).collect();
    if tosa_pads.iter().any(|&p| p < 0) {
        return ctx.fail("negative pads are not supported");
    }
    if mode == "reflect" || mode == "edge" {
        let y = pad_by_concat(ctx, &x, &tosa_pads, mode == "reflect")?;
        return bind_output(ctx, node, 0, y);
    }
    if mode != "constant" {
        return ctx.fail(format!("pad mode {mode} is not supported"));
    }
    let value = match node.optional_input(2) {
        Some(id) => ctx.value(id)?,
        None => ctx.zero(x.dtype, "pad_value")?,
    };
    let shape: Vec<i32> = (0..rank).map(|d| x.shape[d] + (pads[d] + pads[rank + d]) as i32).collect();
    let p = ctx.shape_const(&tosa_pads, "paddings");
    let y = ctx.emit_temp(Op::Pad, OpAttribute::None, &[&x, &p, &value], x.dtype, &shape, "pad");
    bind_output(ctx, node, 0, y)
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

fn flatten(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len() as i64;
    let mut axis = node.attr_int_or("axis", 1)?;
    if axis < 0 {
        axis += rank;
    }
    let outer: i32 = x.shape[..axis as usize].iter().product();
    let inner: i32 = x.shape[axis as usize..].iter().product();
    let y = reshape_temp(ctx, &x, &[outer, inner], "flatten");
    bind_output(ctx, node, 0, y)
}

/// Resolve a target shape with ONNX's `0` (copy) and `-1` (infer) rules.
fn resolve_shape(ctx: &LowerCtx<'_>, input: &[i32], target: &[i64], allow_zero_copy: bool) -> Result<Vec<i32>> {
    let total: i64 = input.iter().map(|&d| d as i64).product();
    let mut shape: Vec<i64> = target
        .iter()
        .enumerate()
        .map(|(i, &d)| if d == 0 && allow_zero_copy { input.get(i).copied().unwrap_or(0) as i64 } else { d })
        .collect();
    if let Some(i) = shape.iter().position(|&d| d == -1) {
        let known: i64 = shape.iter().filter(|&&d| d != -1).product();
        if known == 0 || total % known != 0 {
            return ctx.fail(format!("cannot infer -1 in shape {target:?} from {total} elements"));
        }
        shape[i] = total / known;
    }
    if shape.iter().product::<i64>() != total {
        return ctx.fail(format!("shape {shape:?} does not hold {total} elements"));
    }
    Ok(i32s(&shape))
}

fn reshape(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let target =
        if let Some(id) = node.optional_input(1) { ctx.constant_ints(id)? } else { node.attr_ints("shape")?.to_vec() };
    let allow_zero_copy = node.attr_int_or("allowzero", 0)? == 0;
    let shape = resolve_shape(ctx, &x.shape, &target, allow_zero_copy)?;
    let y = reshape_temp(ctx, &x, &shape, "reshape");
    bind_output(ctx, node, 0, y)
}

/// Squeeze (remove size-1 axes) or Unsqueeze (insert size-1 axes).
fn squeeze(node: &SourceNode, ctx: &mut LowerCtx<'_>, unsqueeze: bool) -> Result<()> {
    let in_id = node.input(0)?;
    let axes: Option<Vec<i64>> = if let Some(id) = node.optional_input(1) {
        Some(ctx.constant_ints(id)?)
    } else if node.attrs.contains_key("axes") {
        Some(node.attr_ints("axes")?.to_vec())
    } else {
        None
    };
    // Constant inputs stay constants (shape arithmetic).
    if ctx.is_constant(in_id) && !ctx.is_available_as_value(in_id) {
        let data = ctx.constant(in_id)?.clone();
        ctx.define_constant(node.output(0)?, data);
        return Ok(());
    }
    let x = ctx.value(in_id)?;
    let rank = x.shape.len() as i64;
    let shape: Vec<i32> = if unsqueeze {
        let axes = axes.ok_or(()).or_else(|_| ctx.fail("Unsqueeze needs axes"))?;
        let out_rank = rank + axes.len() as i64;
        let axes: Vec<i64> = axes.iter().map(|&a| if a < 0 { a + out_rank } else { a }).collect();
        let mut shape = Vec::with_capacity(out_rank as usize);
        let mut src = x.shape.iter();
        for d in 0..out_rank {
            if axes.contains(&d) {
                shape.push(1);
            } else {
                shape.push(*src.next().ok_or(()).or_else(|_| ctx.fail("Unsqueeze axes out of range"))?);
            }
        }
        shape
    } else {
        let axes: Vec<i64> = axes
            .map(|a| a.iter().map(|&a| if a < 0 { a + rank } else { a }).collect())
            .unwrap_or_else(|| (0..rank).filter(|&d| x.shape[d as usize] == 1).collect());
        x.shape.iter().enumerate().filter(|(d, _)| !axes.contains(&(*d as i64))).map(|(_, &s)| s).collect()
    };
    let y = reshape_temp(ctx, &x, &shape, if unsqueeze { "unsqueeze" } else { "squeeze" });
    bind_output(ctx, node, 0, y)
}

fn transpose(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len();
    let perm: Vec<i32> = if node.attrs.contains_key("perm") {
        node.attr_ints("perm")?.iter().map(|&p| p as i32).collect()
    } else {
        (0..rank as i32).rev().collect()
    };
    let y = layout::transpose(ctx, &x, &perm, "transpose");
    bind_output(ctx, node, 0, y)
}

fn concat(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let ids: Vec<TensorId> = node.inputs.iter().flatten().copied().collect();
    if ids.is_empty() {
        return ctx.fail("Concat without inputs");
    }
    // Shape arithmetic: 1-D constants of one numeric type stay a constant.
    if ids.iter().all(|&id| ctx.is_constant(id) && !ctx.is_available_as_value(id)) {
        let dtype = ctx.constant(ids[0])?.dtype();
        let mut values = Vec::new();
        let mut foldable = true;
        for &id in &ids {
            let data = ctx.constant(id)?;
            if data.dtype() != dtype || ctx.tensor(id).rank() > 1 {
                foldable = false;
                break;
            }
            match const_f64(data) {
                Some(v) => values.extend(v),
                None => {
                    foldable = false;
                    break;
                }
            }
        }
        if let Some(data) = const_from_f64(&values, dtype).filter(|_| foldable) {
            ctx.define_constant(node.output(0)?, data);
            return Ok(());
        }
    }
    let mut inputs = Vec::with_capacity(ids.len());
    for &id in &ids {
        inputs.push(ctx.value(id)?);
    }
    let rank = inputs[0].shape.len() as i64;
    let mut axis = node.attr_int("axis")?;
    if axis < 0 {
        axis += rank;
    }
    let mut shape = inputs[0].shape.clone();
    shape[axis as usize] = inputs.iter().map(|t| t.shape[axis as usize]).sum();
    let refs: Vec<&TensorRef> = inputs.iter().collect();
    let y =
        ctx.emit_temp(Op::Concat, OpAttribute::Concat { axis: axis as i32 }, &refs, inputs[0].dtype, &shape, "concat");
    bind_output(ctx, node, 0, y)
}

/// Reflect or edge padding, one axis at a time: the border region is
/// sliced out (and reversed, for reflect) and concatenated back on.
fn pad_by_concat(ctx: &mut LowerCtx<'_>, x: &TensorRef, pads: &[i64], reflect: bool) -> Result<TensorRef> {
    let mut cur = x.clone();
    for axis in 0..cur.shape.len() {
        let (before, after) = (pads[2 * axis], pads[2 * axis + 1]);
        if before == 0 && after == 0 {
            continue;
        }
        let dim = cur.shape[axis] as i64;
        let limit = if reflect { dim - 1 } else { dim };
        if before > limit || after > limit {
            return ctx.fail(format!("pad of {before}/{after} exceeds the {dim}-long axis {axis}"));
        }
        let mut parts: Vec<TensorRef> = Vec::new();
        let dims: Vec<i64> = cur.shape.iter().map(|&d| d as i64).collect();
        let border = |ctx: &mut LowerCtx<'_>, start: i64, len: i64, hint: &str| -> TensorRef {
            let mut s = vec![0i64; dims.len()];
            let mut z = dims.clone();
            s[axis] = start;
            z[axis] = len;
            let piece = slice_temp(ctx, &cur, &s, &z, hint);
            if reflect {
                let shape = piece.shape.clone();
                ctx.emit_temp(
                    Op::Reverse,
                    OpAttribute::Reverse { axis: axis as i32 },
                    &[&piece],
                    piece.dtype,
                    &shape,
                    &format!("{hint}_rev"),
                )
            } else if len > 1 {
                // Edge: repeat the single border sample.
                let mut single = z.clone();
                single[axis] = 1;
                let one = slice_temp(ctx, &cur, &s, &single, &format!("{hint}_edge"));
                let mut mult = vec![1i64; dims.len()];
                mult[axis] = len;
                let m = ctx.shape_const(&mult, &format!("{hint}_mult"));
                let shape: Vec<i32> = z.iter().map(|&d| d as i32).collect();
                ctx.emit_temp(Op::Tile, OpAttribute::None, &[&one, &m], one.dtype, &shape, &format!("{hint}_tiled"))
            } else {
                piece
            }
        };
        if before > 0 {
            let start = if reflect { 1 } else { 0 };
            parts.push(border(ctx, start, before, "pad_before"));
        }
        parts.push(cur.clone());
        if after > 0 {
            let start = if reflect { dim - 1 - after } else { dim - 1 };
            let len = if reflect { after } else { 1 };
            let piece = border(ctx, start, len, "pad_after");
            parts.push(if !reflect && after > 1 {
                let mut mult = vec![1i64; dims.len()];
                mult[axis] = after;
                let m = ctx.shape_const(&mult, "pad_after_mult");
                let mut shape = piece.shape.clone();
                shape[axis] = after as i32;
                ctx.emit_temp(Op::Tile, OpAttribute::None, &[&piece, &m], piece.dtype, &shape, "pad_after_tiled")
            } else {
                piece
            });
        }
        let refs: Vec<&TensorRef> = parts.iter().collect();
        let mut shape = cur.shape.clone();
        shape[axis] += (before + after) as i32;
        cur = ctx.emit_temp(
            Op::Concat,
            OpAttribute::Concat { axis: axis as i32 },
            &refs,
            cur.dtype,
            &shape,
            "pad_concat",
        );
    }
    Ok(cur)
}

fn identity(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let in_id = node.input(0)?;
    if ctx.is_constant(in_id) && !ctx.is_available_as_value(in_id) {
        let data = ctx.constant(in_id)?.clone();
        ctx.define_constant(node.output(0)?, data);
        return Ok(());
    }
    let x = ctx.value(in_id)?;
    let shape = x.shape.clone();
    let y = ctx.emit_temp(Op::Identity, OpAttribute::None, &[&x], x.dtype, &shape, "identity");
    bind_output(ctx, node, 0, y)
}

/// Shape of a static tensor, as a conversion-time INT64 constant.
fn shape(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let dims = static_shape(ctx, &x)?;
    let rank = dims.len() as i64;
    let clampi = |v: i64| v.clamp(-rank, rank);
    let start = clampi(node.attr_int_or("start", 0)?);
    let end = clampi(node.attr_int_or("end", rank)?);
    let start = if start < 0 { start + rank } else { start };
    let end = if end < 0 { end + rank } else { end };
    let values = dims[start as usize..end.max(start) as usize].to_vec();
    ctx.define_constant(node.output(0)?, TensorData::I64(values));
    Ok(())
}

fn int_constant(ctx: &LowerCtx<'_>, id: TensorId) -> Result<Vec<i64>> {
    match ctx.constant(id)? {
        TensorData::I64(v) => Ok(v.clone()),
        TensorData::I32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        other => ctx.fail(format!("expected an integer constant, got {other:?}")),
    }
}

/// Gather on constant 1-D data with constant indices (shape arithmetic).
fn gather(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let (data_id, idx_id) = (node.input(0)?, node.input(1)?);
    if !(ctx.is_constant(data_id) && ctx.is_constant(idx_id)) {
        return ctx.fail("Gather is supported only on constants (shape arithmetic)");
    }
    let data = int_constant(ctx, data_id)?;
    let indices = int_constant(ctx, idx_id)?;
    if node.attr_int_or("axis", 0)? != 0 {
        return ctx.fail("Gather on constants supports axis 0 only");
    }
    let n = data.len() as i64;
    let mut out = Vec::with_capacity(indices.len());
    for &i in &indices {
        let i = if i < 0 { i + n } else { i };
        let v = data.get(i as usize).ok_or(()).or_else(|_| ctx.fail(format!("Gather index {i} out of range")))?;
        out.push(*v);
    }
    ctx.define_constant(node.output(0)?, TensorData::I64(out));
    Ok(())
}

/// Slice of constant 1-D data (shape arithmetic).
fn slice(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let data_id = node.input(0)?;
    if !ctx.is_constant(data_id) {
        return ctx.fail("Slice is supported only on constants (shape arithmetic)");
    }
    let data = int_constant(ctx, data_id)?;
    let (starts, ends) = if opset(ctx) >= 10 {
        (ctx.constant_ints(node.input(1)?)?, ctx.constant_ints(node.input(2)?)?)
    } else {
        (node.attr_ints("starts")?.to_vec(), node.attr_ints("ends")?.to_vec())
    };
    let n = data.len() as i64;
    let norm = |v: i64| (if v < 0 { v + n } else { v }).clamp(0, n);
    if starts.len() != 1 || ends.len() != 1 {
        return ctx.fail("Slice on constants supports one axis only");
    }
    let (s, e) = (norm(starts[0]), norm(ends[0]));
    ctx.define_constant(node.output(0)?, TensorData::I64(data[s as usize..e.max(s) as usize].to_vec()));
    Ok(())
}

fn cast(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let in_id = node.input(0)?;
    let to = node.attr_int("to")?;
    let dtype = match to {
        super::proto::data_type::FLOAT => DType::F32,
        // TOSA has no INT64 data tensors: int64 values (ArgMax indices,
        // shape arithmetic) are INT32.
        super::proto::data_type::INT64 => DType::I32,
        super::proto::data_type::INT32 => DType::I32,
        super::proto::data_type::BOOL => DType::Bool,
        super::proto::data_type::FLOAT16 => DType::F16,
        super::proto::data_type::INT8 => DType::I8,
        super::proto::data_type::INT16 => DType::I16,
        other => return ctx.fail(format!("Cast to ONNX type {other} is not supported")),
    };
    if ctx.is_constant(in_id) && !ctx.is_available_as_value(in_id) {
        let src = ctx.constant(in_id)?;
        let data = match const_f64(src).and_then(|v| const_from_f64(&v, dtype)) {
            Some(d) => d,
            None if src.dtype() == dtype => src.clone(),
            None => return ctx.fail(format!("constant Cast from {:?} to {dtype:?} is not supported", src.dtype())),
        };
        ctx.define_constant(node.output(0)?, data);
        return Ok(());
    }
    let x = ctx.value(in_id)?;
    let shape = x.shape.clone();
    let y = ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], dtype, &shape, "cast");
    bind_output(ctx, node, 0, y)
}

// ---------------------------------------------------------------------------
// Second batch: splitting, normalization, resizing, transposed convolution,
// remaining element-wise and reduction operators, and the QLinear family.
// ---------------------------------------------------------------------------

/// Register the second batch of lowerings.
pub(super) fn register_more(r: &mut LoweringRegistry) {
    r.register("Split", split);
    r.register("LRN", lrn);
    r.register("InstanceNormalization", instance_norm);
    r.register("Upsample", |n, c| resize(n, c, true));
    r.register("Resize", |n, c| resize(n, c, false));
    r.register("ConvTranspose", conv_transpose);
    r.register("DepthToSpace", depth_to_space);
    r.register("SpaceToDepth", space_to_depth);
    r.register("PRelu", prelu);
    r.register("Elu", elu);
    r.register("HardSigmoid", hard_sigmoid);
    r.register("HardSwish", hard_swish);
    r.register("Softplus", softplus);
    r.register("Sin", |n, c| unary(n, c, Op::Sin));
    r.register("Cos", |n, c| unary(n, c, Op::Cos));
    r.register("Expand", expand);
    r.register("Tile", tile);
    r.register("ReduceMax", |n, c| reduce_op(n, c, Op::ReduceMax));
    r.register("ReduceMin", |n, c| reduce_op(n, c, Op::ReduceMin));
    r.register("ReduceSum", |n, c| reduce_op(n, c, Op::ReduceSum));
    r.register("ReduceProd", |n, c| reduce_op(n, c, Op::ReduceProduct));
    r.register("ArgMax", |n, c| arg_max_min(n, c, false));
    r.register("ArgMin", |n, c| arg_max_min(n, c, true));
    r.register("Equal", |n, c| compare(n, c, Op::Equal, false));
    r.register("Greater", |n, c| compare(n, c, Op::Greater, false));
    r.register("GreaterOrEqual", |n, c| compare(n, c, Op::GreaterEqual, false));
    r.register("Less", |n, c| compare(n, c, Op::Greater, true));
    r.register("LessOrEqual", |n, c| compare(n, c, Op::GreaterEqual, true));
    r.register("And", |n, c| bool_binary(n, c, Op::LogicalAnd));
    r.register("Or", |n, c| bool_binary(n, c, Op::LogicalOr));
    r.register("Xor", |n, c| bool_binary(n, c, Op::LogicalXor));
    r.register("Not", |n, c| unary(n, c, Op::LogicalNot));
    r.register("Where", where_op);
    r.register("Sum", |n, c| nary(n, c, Op::Add));
    r.register("Mean", mean_nary);
    r.register("LogSoftmax", log_softmax);
    r.register("QuantizeLinear", quantize_linear);
    r.register("DequantizeLinear", dequantize_linear);
    r.register("QLinearConv", qlinear_conv);
    r.register("QLinearMatMul", qlinear_matmul);
    r.register("com.microsoft.QLinearAdd", qlinear_add);
    r.register("com.microsoft.QLinearGlobalAveragePool", qlinear_global_average_pool);
}

fn unary_temp(ctx: &mut LowerCtx<'_>, op: Op, x: &TensorRef, hint: &str) -> TensorRef {
    let shape = x.shape.clone();
    ctx.emit_temp(op, OpAttribute::None, &[x], x.dtype, &shape, hint)
}

fn slice_temp(ctx: &mut LowerCtx<'_>, x: &TensorRef, start: &[i64], size: &[i64], hint: &str) -> TensorRef {
    let shape: Vec<i32> = size.iter().map(|&d| d as i32).collect();
    let s = ctx.shape_const(start, &format!("{hint}_start"));
    let z = ctx.shape_const(size, &format!("{hint}_size"));
    ctx.emit_temp(Op::Slice, OpAttribute::None, &[x, &s, &z], x.dtype, &shape, hint)
}

fn reduce_axis(ctx: &mut LowerCtx<'_>, op: Op, x: &TensorRef, axis: usize, hint: &str) -> TensorRef {
    let mut shape = x.shape.clone();
    shape[axis] = 1;
    ctx.emit_temp(
        op,
        OpAttribute::Reduce { axis: axis as i32, nan_mode: NanMode::Propagate },
        &[x],
        x.dtype,
        &shape,
        hint,
    )
}

fn normalize_axis(ctx: &LowerCtx<'_>, axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a >= rank as i64 {
        return ctx.fail(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

fn split(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let axis = normalize_axis(ctx, node.attr_int_or("axis", 0)?, x.shape.len())?;
    let dim = x.shape[axis] as i64;
    let n_out = node.outputs.len();
    let sizes: Vec<i64> = if let Some(id) = node.optional_input(1) {
        ctx.constant_ints(id)?
    } else if node.attrs.contains_key("split") {
        node.attr_ints("split")?.to_vec()
    } else {
        if n_out == 0 || dim % n_out as i64 != 0 {
            return ctx.fail(format!("cannot split {dim} into {n_out} equal parts"));
        }
        vec![dim / n_out as i64; n_out]
    };
    let mut offset = 0;
    for (i, &len) in sizes.iter().enumerate() {
        let mut start = vec![0i64; x.shape.len()];
        let mut size: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
        start[axis] = offset;
        size[axis] = len;
        let piece = slice_temp(ctx, &x, &start, &size, &format!("split{i}"));
        bind_output(ctx, node, i, piece)?;
        offset += len;
    }
    Ok(())
}

/// Local response normalization across channels:
/// `x / (bias + alpha / size * sum_{window} x^2) ^ beta`. The channel window
/// sum is formed from shifted slices of the zero-padded squares.
fn lrn(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let size = node.attr_int("size")?;
    let alpha = float_attr_or(node, "alpha", 1e-4)? as f32;
    let beta = float_attr_or(node, "beta", 0.75)? as f32;
    let bias = float_attr_or(node, "bias", 1.0)? as f32;
    if x.shape.len() < 2 {
        return ctx.fail("LRN needs a channel dimension");
    }
    let rank = x.shape.len();
    let c = x.shape[1] as i64;
    let sq = binary_temp(ctx, Op::Mul, &x, &x, "lrn_sq")?;
    // Pad the channel axis by floor((size-1)/2) before and ceil after.
    let before = (size - 1) / 2;
    let after = size - 1 - before;
    let mut pads = vec![0i64; 2 * rank];
    pads[2] = before;
    pads[3] = after;
    let mut padded_shape = sq.shape.clone();
    padded_shape[1] += (before + after) as i32;
    let p = ctx.shape_const(&pads, "lrn_pads");
    let zero = ctx.zero(x.dtype, "lrn_zero")?;
    let padded = ctx.emit_temp(Op::Pad, OpAttribute::None, &[&sq, &p, &zero], x.dtype, &padded_shape, "lrn_padded");
    let mut sum: Option<TensorRef> = None;
    for k in 0..size {
        let mut start = vec![0i64; rank];
        start[1] = k;
        let dims: Vec<i64> = x.shape.iter().map(|&d| d as i64).collect();
        let mut sz = dims.clone();
        sz[1] = c;
        let window = slice_temp(ctx, &padded, &start, &sz, &format!("lrn_win{k}"));
        sum = Some(match sum {
            None => window,
            Some(acc) => binary_temp(ctx, Op::Add, &acc, &window, "lrn_sum")?,
        });
    }
    let sum = sum.unwrap();
    let scale = f32_const(ctx, alpha / size as f32, rank, "lrn_alpha")?;
    let scaled = binary_temp(ctx, Op::Mul, &sum, &scale, "lrn_scaled")?;
    let bias_t = f32_const(ctx, bias, rank, "lrn_bias")?;
    let base = binary_temp(ctx, Op::Add, &scaled, &bias_t, "lrn_base")?;
    let beta_t = f32_const(ctx, beta, rank, "lrn_beta")?;
    let denom = binary_temp(ctx, Op::Pow, &base, &beta_t, "lrn_denom")?;
    let inv = unary_temp(ctx, Op::Reciprocal, &denom, "lrn_inv");
    let y = binary_temp(ctx, Op::Mul, &x, &inv, "lrn")?;
    bind_output(ctx, node, 0, y)
}

/// Per-sample, per-channel normalization over the spatial dimensions.
fn instance_norm(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let scale = ctx.value(node.input(1)?)?;
    let bias = ctx.value(node.input(2)?)?;
    let eps = float_attr_or(node, "epsilon", 1e-5)? as f32;
    let rank = x.shape.len();
    if rank < 3 {
        return ctx.fail("InstanceNormalization needs spatial dimensions");
    }
    let count: i64 = x.shape[2..].iter().map(|&d| d as i64).product();
    let inv_count = f32_const(ctx, 1.0 / count as f32, rank, "in_inv_count")?;
    let mut sum = x.clone();
    for a in 2..rank {
        sum = reduce_axis(ctx, Op::ReduceSum, &sum, a, "in_sum");
    }
    let mean = binary_temp(ctx, Op::Mul, &sum, &inv_count, "in_mean")?;
    let centred = binary_temp(ctx, Op::Sub, &x, &mean, "in_centred")?;
    let sq = binary_temp(ctx, Op::Mul, &centred, &centred, "in_sq")?;
    let mut var = sq;
    for a in 2..rank {
        var = reduce_axis(ctx, Op::ReduceSum, &var, a, "in_var_sum");
    }
    let var = binary_temp(ctx, Op::Mul, &var, &inv_count, "in_var")?;
    let eps_t = f32_const(ctx, eps, rank, "in_eps")?;
    let var_eps = binary_temp(ctx, Op::Add, &var, &eps_t, "in_var_eps")?;
    let inv_std = unary_temp(ctx, Op::Rsqrt, &var_eps, "in_inv_std");
    let normed = binary_temp(ctx, Op::Mul, &centred, &inv_std, "in_normed")?;
    // scale and bias are per channel: shape [1, C, 1, ...]
    let mut cshape = vec![1; rank];
    cshape[1] = x.shape[1];
    let scale = reshape_temp(ctx, &scale, &cshape, "in_scale");
    let bias = reshape_temp(ctx, &bias, &cshape, "in_bias");
    let scaled = binary_temp(ctx, Op::Mul, &normed, &scale, "in_scaled")?;
    let y = binary_temp(ctx, Op::Add, &scaled, &bias, "instance_norm")?;
    bind_output(ctx, node, 0, y)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs().max(1) } else { gcd(b, a % b) }
}

/// TOSA RESIZE parameters `(scale_n, scale_d, offset, border)` for one axis.
/// `mode`: "asymmetric", "half_pixel" / "pytorch_half_pixel", "align_corners";
/// `nearest_floor` selects floor rounding for nearest neighbour (otherwise
/// round-half-up, i.e. `round_prefer_ceil`; ONNX's default
/// `round_prefer_floor` differs from it only at exact halves).
fn resize_axis(
    ctx: &LowerCtx<'_>,
    input: i64,
    output: i64,
    mode: &str,
    nearest_floor: bool,
) -> Result<(i64, i64, i64, i64)> {
    let (mut n, mut d, mut offset, mut border) = match mode {
        "align_corners" if output > 1 => (output - 1, input - 1, 0, 0),
        "half_pixel" | "pytorch_half_pixel" => (2 * output, 2 * input, input - output, output - input),
        _ if nearest_floor => (2 * output, 2 * input, -output, output - 2 * input),
        _ => (output, input, 0, output - input),
    };
    let g = gcd(n, d);
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

/// Upsample (opset 7-9) and Resize (opset 10+): nearest and linear on the
/// two spatial dimensions of an NCHW tensor.
fn resize(node: &SourceNode, ctx: &mut LowerCtx<'_>, upsample: bool) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    if x.shape.len() != 4 {
        return ctx.fail("only 4-D resizing is supported");
    }
    let mode_attr = node.attrs.get("mode").map_or(Ok("nearest"), |_| node.attr_str("mode"))?;
    let (coord, nearest_floor) = if upsample {
        // Upsample: asymmetric coordinates; nearest uses floor.
        ("asymmetric", true)
    } else {
        let coord = node
            .attrs
            .get("coordinate_transformation_mode")
            .map_or(Ok("half_pixel"), |_| node.attr_str("coordinate_transformation_mode"))?;
        let nm = node.attrs.get("nearest_mode").map_or(Ok("round_prefer_floor"), |_| node.attr_str("nearest_mode"))?;
        (coord, nm == "floor")
    };
    // Output size: from `sizes`, else from `scales`.
    let (oh, ow) = if upsample {
        let scales: Vec<f64> = if let Some(id) = node.optional_input(1) {
            match ctx.constant(id)? {
                TensorData::F32(v) => v.iter().map(|&s| s as f64).collect(),
                other => return ctx.fail(format!("scales must be float, got {other:?}")),
            }
        } else if node.attrs.contains_key("scales") {
            match node.attr("scales")? {
                AttrValue::Floats(v) => v.clone(),
                other => return ctx.fail(format!("scales attribute {other:?}")),
            }
        } else {
            return ctx.fail("Upsample needs scales");
        };
        if scales.len() != 4 {
            return ctx.fail("scales must have 4 values");
        }
        ((x.shape[2] as f64 * scales[2]).floor() as i64, (x.shape[3] as f64 * scales[3]).floor() as i64)
    } else {
        // Resize: inputs X, roi, scales, sizes.
        let sizes_id = node.optional_input(3).filter(|&id| ctx.is_constant(id));
        if let Some(id) = sizes_id {
            let sizes = ctx.constant_ints(id)?;
            if sizes.len() != 4 {
                return ctx.fail("sizes must have 4 values");
            }
            (sizes[2], sizes[3])
        } else {
            let id = node.optional_input(2).ok_or(()).or_else(|_| ctx.fail("Resize needs scales or sizes"))?;
            let scales: Vec<f64> = match ctx.constant(id)? {
                TensorData::F32(v) => v.iter().map(|&s| s as f64).collect(),
                other => return ctx.fail(format!("scales must be float, got {other:?}")),
            };
            if scales.len() != 4 {
                return ctx.fail("scales must have 4 values");
            }
            ((x.shape[2] as f64 * scales[2]).floor() as i64, (x.shape[3] as f64 * scales[3]).floor() as i64)
        }
    };
    let tosa_mode = match mode_attr {
        "nearest" => tosa_ir::ResizeMode::Nearest,
        "linear" | "bilinear" => tosa_ir::ResizeMode::Bilinear,
        other => return ctx.fail(format!("resize mode {other} is not supported")),
    };
    let nearest = tosa_mode == tosa_ir::ResizeMode::Nearest;
    // An axis resized from a single sample is a TILE (see the TFLite frontend).
    let tile_y = x.shape[2] == 1 && oh != 1;
    let tile_x = x.shape[3] == 1 && ow != 1;
    let (yn, yd, yo, yb) =
        if tile_y { (1, 1, 0, 0) } else { resize_axis(ctx, x.shape[2] as i64, oh, coord, nearest && nearest_floor)? };
    let (xn, xd, xo, xb) =
        if tile_x { (1, 1, 0, 0) } else { resize_axis(ctx, x.shape[3] as i64, ow, coord, nearest && nearest_floor)? };
    let xt = layout::transpose(ctx, &x, &NCHW_TO_NHWC, "resize_nhwc");
    let out_shape = vec![x.shape[0], oh as i32, ow as i32, x.shape[1]];
    let mut y = xt;
    if !(tile_y && tile_x) {
        let shape =
            vec![x.shape[0], if tile_y { 1 } else { oh as i32 }, if tile_x { 1 } else { ow as i32 }, x.shape[1]];
        let scale = ctx.shape_const(&[yn, yd, xn, xd], "resize_scale");
        let offset = ctx.shape_const(&[yo, xo], "resize_offset");
        let border = ctx.shape_const(&[yb, xb], "resize_border");
        y = ctx.emit_temp(
            Op::Resize,
            OpAttribute::Resize { mode: tosa_mode },
            &[&y, &scale, &offset, &border],
            x.dtype,
            &shape,
            "resize",
        );
    }
    if tile_y || tile_x {
        let multiples = [1, if tile_y { oh } else { 1 }, if tile_x { ow } else { 1 }, 1];
        let m = ctx.shape_const(&multiples, "resize_tile");
        y = ctx.emit_temp(Op::Tile, OpAttribute::None, &[&y, &m], x.dtype, &out_shape, "resize_tiled");
    }
    let y = layout::transpose(ctx, &y, &NHWC_TO_NCHW, "resize_nchw");
    bind_output(ctx, node, 0, y)
}

/// ConvTranspose (2-D, group 1): weights `[C_in, C_out, kH, kW]` become TOSA's
/// `[OC, KH, KW, IC]`; ONNX pads become negative TOSA out_pad.
fn conv_transpose(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let w = ctx.value(node.input(1)?)?;
    let x_shape = static_shape(ctx, &x)?;
    let w_shape = static_shape(ctx, &w)?;
    if x_shape.len() != 4 || w_shape.len() != 4 {
        return ctx.fail("only 2-D ConvTranspose is supported");
    }
    if node.attr_int_or("group", 1)? != 1 {
        return ctx.fail("grouped ConvTranspose is not supported");
    }
    let dilations = ints_attr_or(node, "dilations", vec![1, 1])?;
    if dilations != [1, 1] {
        return ctx.fail("dilated ConvTranspose is not supported");
    }
    let strides = ints_attr_or(node, "strides", vec![1, 1])?;
    let pads = ints_attr_or(node, "pads", vec![0; 4])?;
    let output_padding = ints_attr_or(node, "output_padding", vec![0, 0])?;
    let [n, _c_in, h, wd] = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];
    let [_, c_out, kh, kw] = [w_shape[0], w_shape[1], w_shape[2], w_shape[3]];
    if node.attrs.contains_key("output_shape") {
        return ctx.fail("explicit output_shape is not supported");
    }
    let out_h = (h - 1) * strides[0] - pads[0] - pads[2] + (kh - 1) + output_padding[0] + 1;
    let out_w = (wd - 1) * strides[1] - pads[1] - pads[3] + (kw - 1) + output_padding[1] + 1;
    let out_pad = vec![
        -pads[0] as i32,
        (-pads[2] + output_padding[0]) as i32,
        -pads[1] as i32,
        (-pads[3] + output_padding[1]) as i32,
    ];
    let xt = layout::transpose(ctx, &x, &NCHW_TO_NHWC, "convt_nhwc");
    let wt = layout::transpose(ctx, &w, &[1, 2, 3, 0], "convt_weight");
    let zp = ctx.zero(x.dtype, "zp")?;
    let bias = match node.optional_input(2) {
        Some(id) => ctx.value(id)?,
        None => ctx.block.constant("bias", &[c_out as i32], TensorData::zeros(x.dtype, c_out as usize)?)?,
    };
    let acc_type = if x.dtype == DType::F16 { DType::F16 } else { DType::F32 };
    let attr = OpAttribute::TransposeConv2d { out_pad, stride: i32s(&strides), local_bound: false, acc_type };
    let out_shape = vec![n as i32, out_h as i32, out_w as i32, c_out as i32];
    let y = ctx.emit_temp(Op::TransposeConv2d, attr, &[&xt, &wt, &bias, &zp, &zp], x.dtype, &out_shape, "convt");
    let y = layout::transpose(ctx, &y, &NHWC_TO_NCHW, "convt_nchw");
    bind_output(ctx, node, 0, y)
}

/// DepthToSpace (DCR default or CRD) on NCHW: reshape, transpose, reshape.
fn depth_to_space(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let b = node.attr_int("blocksize")? as i32;
    let mode = node.attrs.get("mode").map_or(Ok("DCR"), |_| node.attr_str("mode"))?;
    let [n, c, h, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if c % (b * b) != 0 {
        return ctx.fail("channels must be divisible by blocksize^2");
    }
    let c2 = c / (b * b);
    let (split_shape, perm) = if mode == "DCR" {
        (vec![n, b, b, c2, h, w], vec![0, 3, 4, 1, 5, 2])
    } else {
        (vec![n, c2, b, b, h, w], vec![0, 1, 4, 2, 5, 3])
    };
    let r1 = reshape_temp(ctx, &x, &split_shape, "d2s_split");
    let t = layout::transpose(ctx, &r1, &perm, "d2s_perm");
    let y = reshape_temp(ctx, &t, &[n, c2, h * b, w * b], "d2s");
    bind_output(ctx, node, 0, y)
}

fn space_to_depth(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let b = node.attr_int("blocksize")? as i32;
    let [n, c, h, w] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
    if h % b != 0 || w % b != 0 {
        return ctx.fail("height and width must be divisible by blocksize");
    }
    let r1 = reshape_temp(ctx, &x, &[n, c, h / b, b, w / b, b], "s2d_split");
    let t = layout::transpose(ctx, &r1, &[0, 3, 5, 1, 2, 4], "s2d_perm");
    let y = reshape_temp(ctx, &t, &[n, c * b * b, h / b, w / b], "s2d");
    bind_output(ctx, node, 0, y)
}

fn prelu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let slope = ctx.value(node.input(1)?)?;
    let zero = f32_const(ctx, 0.0, x.shape.len(), "zero")?;
    let attr = OpAttribute::NanPropagation { nan_mode: NanMode::Propagate };
    let shape = x.shape.clone();
    let (xz, zz) = align_ranks(ctx, x.clone(), zero);
    let pos = ctx.emit_temp(Op::Maximum, attr.clone(), &[&xz, &zz], x.dtype, &shape, "prelu_pos");
    let neg = ctx.emit_temp(Op::Minimum, attr, &[&xz, &zz], x.dtype, &shape, "prelu_neg");
    let scaled = binary_temp(ctx, Op::Mul, &neg, &slope, "prelu_scaled")?;
    let y = binary_temp(ctx, Op::Add, &pos, &scaled, "prelu")?;
    bind_output(ctx, node, 0, y)
}

fn elu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let alpha = float_attr_or(node, "alpha", 1.0)? as f32;
    let rank = x.shape.len();
    let shape = x.shape.clone();
    let zero = f32_const(ctx, 0.0, rank, "zero")?;
    let one = f32_const(ctx, 1.0, rank, "one")?;
    let a = f32_const(ctx, alpha, rank, "alpha")?;
    let e = unary_temp(ctx, Op::Exp, &x, "elu_exp");
    let em1 = binary_temp(ctx, Op::Sub, &e, &one, "elu_em1")?;
    let neg = binary_temp(ctx, Op::Mul, &em1, &a, "elu_neg")?;
    let (xz, zz) = align_ranks(ctx, x.clone(), zero);
    let cond = ctx.emit_temp(Op::Greater, OpAttribute::None, &[&xz, &zz], DType::Bool, &shape, "elu_cond");
    let y = ctx.emit_temp(Op::Select, OpAttribute::None, &[&cond, &x, &neg], x.dtype, &shape, "elu");
    bind_output(ctx, node, 0, y)
}

/// `max(0, min(1, alpha * x + beta))`.
fn hard_sigmoid(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let alpha = float_attr_or(node, "alpha", 0.2)? as f32;
    let beta = float_attr_or(node, "beta", 0.5)? as f32;
    let rank = x.shape.len();
    let a = f32_const(ctx, alpha, rank, "alpha")?;
    let b = f32_const(ctx, beta, rank, "beta")?;
    let ax = binary_temp(ctx, Op::Mul, &x, &a, "hsig_ax")?;
    let axb = binary_temp(ctx, Op::Add, &ax, &b, "hsig_axb")?;
    let y = clamp_temp(ctx, &axb, 0.0, 1.0, "hard_sigmoid")?;
    bind_output(ctx, node, 0, y)
}

/// `x * max(0, min(1, x / 6 + 0.5))`.
fn hard_swish(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len();
    let a = f32_const(ctx, 1.0 / 6.0, rank, "sixth")?;
    let b = f32_const(ctx, 0.5, rank, "half")?;
    let ax = binary_temp(ctx, Op::Mul, &x, &a, "hswish_ax")?;
    let axb = binary_temp(ctx, Op::Add, &ax, &b, "hswish_axb")?;
    let gate = clamp_temp(ctx, &axb, 0.0, 1.0, "hswish_gate")?;
    let y = binary_temp(ctx, Op::Mul, &x, &gate, "hard_swish")?;
    bind_output(ctx, node, 0, y)
}

/// `log(1 + exp(x))`.
fn softplus(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let one = f32_const(ctx, 1.0, x.shape.len(), "one")?;
    let e = unary_temp(ctx, Op::Exp, &x, "softplus_exp");
    let e1 = binary_temp(ctx, Op::Add, &e, &one, "softplus_e1")?;
    let y = unary_temp(ctx, Op::Log, &e1, "softplus");
    bind_output(ctx, node, 0, y)
}

/// Expand to a broadcast shape: a reshape to equal rank plus TILE.
fn expand(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let target = ctx.constant_ints(node.input(1)?)?;
    let rank = target.len().max(x.shape.len());
    let mut xs = vec![1; rank - x.shape.len()];
    xs.extend_from_slice(&x.shape);
    let mut ts = vec![1i64; rank - target.len()];
    ts.extend_from_slice(&target);
    let shape: Vec<i32> = xs.iter().zip(&ts).map(|(&a, &b)| if b == 1 { a } else { b as i32 }).collect();
    let x2 = reshape_temp(ctx, &x, &xs, "expand_in");
    let multiples: Vec<i64> = xs.iter().zip(&shape).map(|(&a, &b)| (b / a) as i64).collect();
    let y = if multiples.iter().all(|&m| m == 1) {
        x2
    } else {
        let m = ctx.shape_const(&multiples, "expand_multiples");
        ctx.emit_temp(Op::Tile, OpAttribute::None, &[&x2, &m], x.dtype, &shape, "expand")
    };
    bind_output(ctx, node, 0, y)
}

fn tile(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let repeats = ctx.constant_ints(node.input(1)?)?;
    let shape: Vec<i32> = x.shape.iter().zip(&repeats).map(|(&d, &m)| d * m as i32).collect();
    let m = ctx.shape_const(&repeats, "tile_repeats");
    let y = ctx.emit_temp(Op::Tile, OpAttribute::None, &[&x, &m], x.dtype, &shape, "tile");
    bind_output(ctx, node, 0, y)
}

fn reduce_op(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len();
    let axes: Vec<i64> = if let Some(id) = node.optional_input(1) {
        ctx.constant_ints(id)?
    } else if node.attrs.contains_key("axes") {
        node.attr_ints("axes")?.to_vec()
    } else if node.attr_int_or("noop_with_empty_axes", 0)? != 0 {
        vec![]
    } else {
        (0..rank as i64).collect()
    };
    let mut axes: Vec<usize> = axes.iter().map(|&a| normalize_axis(ctx, a, rank)).collect::<Result<_>>()?;
    axes.sort_unstable();
    axes.dedup();
    let keep = node.attr_int_or("keepdims", 1)? != 0;
    let mut acc = x.clone();
    for &a in &axes {
        acc = reduce_axis(ctx, op, &acc, a, "reduce");
    }
    let y = if keep {
        acc
    } else {
        let shape: Vec<i32> = (0..rank).filter(|d| !axes.contains(d)).map(|d| x.shape[d]).collect();
        reshape_temp(ctx, &acc, &shape, "reduce_squeezed")
    };
    bind_output(ctx, node, 0, y)
}

fn arg_max_min(node: &SourceNode, ctx: &mut LowerCtx<'_>, min: bool) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len();
    let axis = normalize_axis(ctx, node.attr_int_or("axis", 0)?, rank)?;
    let keep = node.attr_int_or("keepdims", 1)? != 0;
    if node.attr_int_or("select_last_index", 0)? != 0 {
        return ctx.fail("select_last_index is not supported");
    }
    let x = if min {
        if !x.dtype.is_float() {
            return ctx.fail("ArgMin is supported for float inputs only");
        }
        let zp = ctx.zero(x.dtype, "zp")?;
        let shape = x.shape.clone();
        ctx.emit_temp(Op::Negate, OpAttribute::None, &[&x, &zp, &zp], x.dtype, &shape, "argmin_neg")
    } else {
        x
    };
    let mut shape = x.shape.clone();
    shape.remove(axis);
    let y = ctx.emit_temp(
        Op::ArgMax,
        OpAttribute::ArgMax { axis: axis as i32, nan_mode: NanMode::Propagate },
        &[&x],
        DType::I32,
        &shape,
        "argmax",
    );
    let y = if keep {
        let mut kept = x.shape.clone();
        kept[axis] = 1;
        reshape_temp(ctx, &y, &kept, "argmax_kept")
    } else {
        y
    };
    bind_output(ctx, node, 0, y)
}

fn widen_for_compare(ctx: &mut LowerCtx<'_>, x: TensorRef) -> TensorRef {
    if matches!(x.dtype, DType::I8 | DType::I16) {
        let shape = x.shape.clone();
        ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], DType::I32, &shape, "widened")
    } else {
        x
    }
}

fn compare(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op, swap: bool) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let a = widen_for_compare(ctx, a);
    let b = widen_for_compare(ctx, b);
    let (a, b) = if swap { (b, a) } else { (a, b) };
    let (a, b) = align_ranks(ctx, a, b);
    let shape = broadcast(ctx, &a.shape, &b.shape)?;
    let y = ctx.emit_temp(op, OpAttribute::None, &[&a, &b], DType::Bool, &shape, "compare");
    bind_output(ctx, node, 0, y)
}

fn bool_binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let (a, b) = align_ranks(ctx, a, b);
    let shape = broadcast(ctx, &a.shape, &b.shape)?;
    let y = ctx.emit_temp(op, OpAttribute::None, &[&a, &b], DType::Bool, &shape, "logical");
    bind_output(ctx, node, 0, y)
}

fn where_op(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let c = ctx.value(node.input(0)?)?;
    let a = ctx.value(node.input(1)?)?;
    let b = ctx.value(node.input(2)?)?;
    let rank = c.shape.len().max(a.shape.len()).max(b.shape.len());
    let pad = |ctx: &mut LowerCtx<'_>, t: TensorRef, hint: &str| -> TensorRef {
        if t.shape.len() == rank {
            return t;
        }
        let mut s = vec![1; rank - t.shape.len()];
        s.extend_from_slice(&t.shape);
        reshape_temp(ctx, &t, &s, hint)
    };
    let c = pad(ctx, c, "where_c");
    let a = pad(ctx, a, "where_a");
    let b = pad(ctx, b, "where_b");
    let shape = broadcast(ctx, &broadcast(ctx, &c.shape, &a.shape)?, &b.shape)?;
    let y = ctx.emit_temp(Op::Select, OpAttribute::None, &[&c, &a, &b], a.dtype, &shape, "where");
    bind_output(ctx, node, 0, y)
}

fn mean_nary(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let ids: Vec<TensorId> = node.inputs.iter().flatten().copied().collect();
    let mut acc = ctx.value(ids[0])?;
    for &id in &ids[1..] {
        let b = ctx.value(id)?;
        acc = binary_temp(ctx, Op::Add, &acc, &b, "mean_sum")?;
    }
    let k = f32_const(ctx, 1.0 / ids.len() as f32, acc.shape.len(), "mean_inv")?;
    let y = binary_temp(ctx, Op::Mul, &acc, &k, "mean")?;
    bind_output(ctx, node, 0, y)
}

fn log_softmax(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let rank = x.shape.len();
    let default_axis = if opset(ctx) >= 13 { -1 } else { 1 };
    let axis = normalize_axis(ctx, node.attr_int_or("axis", default_axis)?, rank)?;
    if opset(ctx) < 13 && axis != rank - 1 {
        return ctx.fail("LogSoftmax before opset 13 is supported on the last axis only");
    }
    let max = reduce_axis(ctx, Op::ReduceMax, &x, axis, "lsm_max");
    let shifted = binary_temp(ctx, Op::Sub, &x, &max, "lsm_shifted")?;
    let e = unary_temp(ctx, Op::Exp, &shifted, "lsm_exp");
    let sum = reduce_axis(ctx, Op::ReduceSum, &e, axis, "lsm_sum");
    let log = unary_temp(ctx, Op::Log, &sum, "lsm_log");
    let y = binary_temp(ctx, Op::Sub, &shifted, &log, "log_softmax")?;
    bind_output(ctx, node, 0, y)
}

// --- QLinear operators ------------------------------------------------------
//
// onnxruntime requantizes in float32: the INT32 accumulator is converted to
// float, multiplied by the (per-channel) scale, rounded to nearest even and
// offset by the zero point. TOSA's CAST rounds to nearest even, so this is
// reproduced exactly with CAST, MUL, CAST, ADD, CLAMP. uint8 tensors are
// INT8 with values and zero points shifted by 128, as in the TFLite frontend.

/// Quantization parameters read from constant operands. Zero points of
/// uint8 tensors are already in the shifted INT8 domain (the loader stores
/// uint8 data as `value - 128`), so no distinction is needed here.
struct QParams {
    scales: Vec<f32>,
    zero_points: Vec<i64>,
}

fn qparams(ctx: &LowerCtx<'_>, scale_id: TensorId, zp_id: Option<TensorId>) -> Result<QParams> {
    let scales = match ctx.constant(scale_id)? {
        TensorData::F32(v) => v.clone(),
        other => return ctx.fail(format!("scale must be float, got {other:?}")),
    };
    let zero_points = match zp_id {
        Some(id) => match ctx.constant(id)? {
            TensorData::I8(v) => v.iter().map(|&z| z as i64).collect(),
            TensorData::I32(v) => v.iter().map(|&z| z as i64).collect(),
            other => return ctx.fail(format!("zero point must be int8/uint8/int32, got {other:?}")),
        },
        None => vec![0; scales.len()],
    };
    Ok(QParams { scales, zero_points })
}

/// Dequantize `x` (INT8 holding shifted uint8 or int8) to FP32.
fn dequant_f32(
    ctx: &mut LowerCtx<'_>,
    x: &TensorRef,
    q: &QParams,
    axis: Option<usize>,
    hint: &str,
) -> Result<TensorRef> {
    let rank = x.shape.len();
    let shape = x.shape.clone();
    let f = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[x],
        DType::F32,
        &shape,
        &format!("{hint}_f32"),
    );
    let (zp_t, s_t) = if q.scales.len() == 1 {
        (
            f32_const(ctx, q.zero_points[0] as f32, rank, &format!("{hint}_zp"))?,
            f32_const(ctx, q.scales[0], rank, &format!("{hint}_scale"))?,
        )
    } else {
        let axis = axis.unwrap_or(1);
        let mut cshape = vec![1; rank];
        cshape[axis] = q.scales.len() as i32;
        (
            ctx.block.constant(
                &format!("{hint}_zp"),
                &cshape,
                TensorData::F32(q.zero_points.iter().map(|&z| z as f32).collect()),
            )?,
            ctx.block.constant(&format!("{hint}_scale"), &cshape, TensorData::F32(q.scales.clone()))?,
        )
    };
    let centred = binary_temp(ctx, Op::Sub, &f, &zp_t, &format!("{hint}_centred"))?;
    binary_temp(ctx, Op::Mul, &centred, &s_t, &format!("{hint}_real"))
}

/// Quantize FP32 `x`: `saturate(round_half_even(x / scale) + zp)` as INT8,
/// with the exact quotient rounding onnxruntime's division produces (see
/// [`crate::quant::quantize_exact`]).
fn quant_i8(ctx: &mut LowerCtx<'_>, x: &TensorRef, q: &QParams, hint: &str) -> Result<TensorRef> {
    let shape = x.shape.clone();
    let saturated = crate::quant::quantize_exact(
        ctx,
        x,
        q.scales[0],
        q.zero_points[0] as i32,
        DType::I8,
        crate::quant::Ties::Even,
        hint,
    )?;
    Ok(ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&saturated], DType::I8, &shape, hint))
}

/// INT32 accumulator -> INT8 output: `saturate(round_half_even(acc * scale) + zp)`.
fn requant_i32(
    ctx: &mut LowerCtx<'_>,
    acc: &TensorRef,
    scales: &[f32],
    zp: i64,
    channel_axis: usize,
    hint: &str,
) -> Result<TensorRef> {
    let rank = acc.shape.len();
    let shape = acc.shape.clone();
    let f = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[acc],
        DType::F32,
        &shape,
        &format!("{hint}_f32"),
    );
    let s_t = if scales.len() == 1 {
        f32_const(ctx, scales[0], rank, &format!("{hint}_scale"))?
    } else {
        let mut cshape = vec![1; rank];
        cshape[channel_axis] = scales.len() as i32;
        ctx.block.constant(&format!("{hint}_scale"), &cshape, TensorData::F32(scales.to_vec()))?
    };
    let scaled = binary_temp(ctx, Op::Mul, &f, &s_t, &format!("{hint}_scaled"))?;
    let rounded = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[&scaled],
        DType::I32,
        &shape,
        &format!("{hint}_rounded"),
    );
    let zp_t = ctx.block.constant(&format!("{hint}_zp"), &vec![1; rank], TensorData::I32(vec![zp as i32]))?;
    let shifted = binary_temp(ctx, Op::Add, &rounded, &zp_t, &format!("{hint}_shifted"))?;
    let clamped = saturate_i32(ctx, &shifted, -128, 127, &format!("{hint}_clamped"))?;
    Ok(ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&clamped], DType::I8, &shape, hint))
}

/// Saturate an INT32 tensor to `[lo, hi]` with MINIMUM/MAXIMUM (CLAMP has
/// no INT32 form).
fn saturate_i32(ctx: &mut LowerCtx<'_>, x: &TensorRef, lo: i32, hi: i32, hint: &str) -> Result<TensorRef> {
    let rank = x.shape.len();
    let shape = x.shape.clone();
    let attr = OpAttribute::NanPropagation { nan_mode: NanMode::Propagate };
    let hi_t = ctx.block.constant(&format!("{hint}_hi"), &vec![1; rank], TensorData::I32(vec![hi]))?;
    let lo_t = ctx.block.constant(&format!("{hint}_lo"), &vec![1; rank], TensorData::I32(vec![lo]))?;
    let m = ctx.emit_temp(Op::Minimum, attr.clone(), &[x, &hi_t], DType::I32, &shape, &format!("{hint}_min"));
    Ok(ctx.emit_temp(Op::Maximum, attr, &[&m, &lo_t], DType::I32, &shape, hint))
}

fn quantize_linear(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let q = qparams(ctx, node.input(1)?, node.optional_input(2))?;
    if q.scales.len() != 1 {
        return ctx.fail("per-axis QuantizeLinear is not supported");
    }
    if !x.dtype.is_float() {
        return ctx.fail("QuantizeLinear of non-float input is not supported");
    }
    let y = quant_i8(ctx, &x, &q, "quantize")?;
    bind_output(ctx, node, 0, y)
}

fn dequantize_linear(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let q = qparams(ctx, node.input(1)?, node.optional_input(2))?;
    let axis = normalize_axis(ctx, node.attr_int_or("axis", 1)?, x.shape.len().max(1)).unwrap_or(0);
    if x.dtype == DType::I32 {
        // INT32 (bias) dequantization: scale only.
        let rank = x.shape.len();
        let shape = x.shape.clone();
        let f =
            ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], DType::F32, &shape, "dq_f32");
        let s = if q.scales.len() == 1 {
            f32_const(ctx, q.scales[0], rank, "dq_scale")?
        } else {
            let mut cshape = vec![1; rank];
            cshape[axis.min(rank.saturating_sub(1))] = q.scales.len() as i32;
            ctx.block.constant("dq_scale", &cshape, TensorData::F32(q.scales.clone()))?
        };
        let y = binary_temp(ctx, Op::Mul, &f, &s, "dequantize")?;
        return bind_output(ctx, node, 0, y);
    }
    let y = dequant_f32(ctx, &x, &q, Some(axis), "dequantize")?;
    bind_output(ctx, node, 0, y)
}

/// QLinearConv: inputs x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B].
fn qlinear_conv(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let xq = qparams(ctx, node.input(1)?, Some(node.input(2)?))?;
    let w = ctx.value(node.input(3)?)?;
    let wq = qparams(ctx, node.input(4)?, Some(node.input(5)?))?;
    let yq = qparams(ctx, node.input(6)?, Some(node.input(7)?))?;
    let x_shape = static_shape(ctx, &x)?;
    let w_shape = static_shape(ctx, &w)?;
    if x_shape.len() != 4 || w_shape.len() != 4 {
        return ctx.fail("only 2-D QLinearConv is supported");
    }
    if x.dtype != DType::I8 || w.dtype != DType::I8 {
        return ctx.fail("QLinearConv needs int8/uint8 operands");
    }
    let w_zp = match (wq.zero_points.iter().min(), wq.zero_points.iter().max()) {
        (Some(lo), Some(hi)) if lo == hi => *lo,
        _ => return ctx.fail("weights with differing per-channel zero points are not supported"),
    };
    let [n, c, h, wd] = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]];
    let [m, c_per_group, kh, kw] = [w_shape[0], w_shape[1], w_shape[2], w_shape[3]];
    let group = node.attr_int_or("group", 1)?;
    let strides = ints_attr_or(node, "strides", vec![1, 1])?;
    let dilations = ints_attr_or(node, "dilations", vec![1, 1])?;
    let g =
        geometry(ctx, node, [h, wd], [kh, kw], [strides[0], strides[1]], [dilations[0], dilations[1]], false, false)?;
    let mut xt = layout::transpose(ctx, &x, &NCHW_TO_NHWC, "qconv_nhwc");
    if let Some(hw) = g.sliced {
        xt = slice_spatial(ctx, &xt, hw, "qconv_input");
    }
    let bias = match node.optional_input(8) {
        Some(id) => ctx.value(id)?,
        None => ctx.block.constant("bias", &[m as i32], TensorData::zeros(DType::I32, m as usize)?)?,
    };
    let x_zp_t = ctx.scalar(TensorData::I8(vec![xq.zero_points[0] as i8]), "qconv_x_zp")?;
    let w_zp_t = ctx.scalar(TensorData::I8(vec![w_zp as i8]), "qconv_w_zp")?;
    let acc_shape = vec![n as i32, g.out[0], g.out[1], m as i32];
    let depthwise = group == c && c_per_group == 1 && group > 1;
    let acc = if group == 1 {
        let wt = layout::transpose(ctx, &w, &layout::OIHW_TO_OHWI, "qconv_weight");
        let attr = OpAttribute::Conv2d {
            pad: g.pad.clone(),
            stride: i32s(&strides),
            dilation: i32s(&dilations),
            local_bound: false,
            acc_type: DType::I32,
        };
        ctx.emit_temp(Op::Conv2d, attr, &[&xt, &wt, &bias, &x_zp_t, &w_zp_t], DType::I32, &acc_shape, "qconv_acc")
    } else if depthwise {
        let mult = (m / c) as i32;
        let wt = layout::transpose(ctx, &w, &[2, 3, 1, 0], "qdw_weight_t");
        let wt = reshape_temp(ctx, &wt, &[kh as i32, kw as i32, c as i32, mult], "qdw_weight");
        let attr = OpAttribute::DepthwiseConv2d {
            pad: g.pad.clone(),
            stride: i32s(&strides),
            dilation: i32s(&dilations),
            local_bound: false,
            acc_type: DType::I32,
        };
        ctx.emit_temp(
            Op::DepthwiseConv2d,
            attr,
            &[&xt, &wt, &bias, &x_zp_t, &w_zp_t],
            DType::I32,
            &acc_shape,
            "qdw_acc",
        )
    } else {
        return ctx.fail(format!("grouped QLinearConv with group={group} is not supported"));
    };
    // y = saturate(round(acc * x_scale * w_scale / y_scale) + y_zp)
    let scales: Vec<f32> = wq.scales.iter().map(|&ws| xq.scales[0] * ws / yq.scales[0]).collect();
    let y = requant_i32(ctx, &acc, &scales, yq.zero_points[0], 3, "qconv_out")?;
    let y = layout::transpose(ctx, &y, &NHWC_TO_NCHW, "qconv_nchw");
    bind_output(ctx, node, 0, y)
}

/// QLinearMatMul: 2-D operands, per-tensor parameters.
fn qlinear_matmul(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let aq = qparams(ctx, node.input(1)?, Some(node.input(2)?))?;
    let b = ctx.value(node.input(3)?)?;
    let bq = qparams(ctx, node.input(4)?, Some(node.input(5)?))?;
    let yq = qparams(ctx, node.input(6)?, Some(node.input(7)?))?;
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return ctx.fail("QLinearMatMul supports 2-D operands only");
    }
    if aq.scales.len() != 1 {
        return ctx.fail("per-axis quantization of the first QLinearMatMul operand is not supported");
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let n = b.shape[1];
    if bq.scales.len() != 1 && bq.scales.len() != n as usize {
        return ctx.fail(format!("{} weight scales for {n} output columns", bq.scales.len()));
    }
    let b_zp = match (bq.zero_points.iter().min(), bq.zero_points.iter().max()) {
        (Some(lo), Some(hi)) if lo == hi => *lo,
        _ => return ctx.fail("weights with differing per-column zero points are not supported"),
    };
    let a3 = reshape_temp(ctx, &a, &[1, m, k], "qmm_a");
    let b3 = reshape_temp(ctx, &b, &[1, k, n], "qmm_b");
    let a_zp = ctx.scalar(TensorData::I8(vec![aq.zero_points[0] as i8]), "qmm_a_zp")?;
    let b_zp = ctx.scalar(TensorData::I8(vec![b_zp as i8]), "qmm_b_zp")?;
    let acc =
        ctx.emit_temp(Op::MatMul, OpAttribute::None, &[&a3, &b3, &a_zp, &b_zp], DType::I32, &[1, m, n], "qmm_acc");
    let acc = reshape_temp(ctx, &acc, &[m, n], "qmm_acc2");
    let scales: Vec<f32> = bq.scales.iter().map(|&bs| aq.scales[0] * bs / yq.scales[0]).collect();
    let y = requant_i32(ctx, &acc, &scales, yq.zero_points[0], 1, "qmm_out")?;
    bind_output(ctx, node, 0, y)
}

/// com.microsoft.QLinearAdd: inputs A, A_scale, A_zp, B, B_scale, B_zp, C_scale, C_zp.
/// onnxruntime (MLAS) computes, in float32,
/// `round((A - A_zp) * (A_scale / C_scale) + (B - B_zp) * (B_scale / C_scale) + C_zp)`;
/// adding the zero point before rounding matters when the sum is within an
/// ulp of a half-integer (verified bit for bit against onnxruntime).
fn qlinear_add(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let aq = qparams(ctx, node.input(1)?, Some(node.input(2)?))?;
    let b = ctx.value(node.input(3)?)?;
    let bq = qparams(ctx, node.input(4)?, Some(node.input(5)?))?;
    let cq = qparams(ctx, node.input(6)?, Some(node.input(7)?))?;
    let ratio = |ctx: &mut LowerCtx<'_>, x: &TensorRef, q: &QParams, hint: &str| -> Result<TensorRef> {
        let rank = x.shape.len();
        let shape = x.shape.clone();
        let f = ctx.emit_temp(
            Op::Cast,
            OpAttribute::Cast { input_unsigned: false },
            &[x],
            DType::F32,
            &shape,
            &format!("{hint}_f32"),
        );
        let zp = f32_const(ctx, q.zero_points[0] as f32, rank, &format!("{hint}_zp"))?;
        let centred = binary_temp(ctx, Op::Sub, &f, &zp, &format!("{hint}_centred"))?;
        let r = f32_const(ctx, q.scales[0] / cq.scales[0], rank, &format!("{hint}_ratio"))?;
        binary_temp(ctx, Op::Mul, &centred, &r, &format!("{hint}_scaled"))
    };
    let af = ratio(ctx, &a, &aq, "qadd_a")?;
    let bf = ratio(ctx, &b, &bq, "qadd_b")?;
    let sum = binary_temp(ctx, Op::Add, &af, &bf, "qadd_sum")?;
    let rank = sum.shape.len();
    let shape = sum.shape.clone();
    // The output zero point is added in float *before* rounding, as MLAS does.
    let zp = f32_const(ctx, cq.zero_points[0] as f32, rank, "qadd_c_zp")?;
    let shifted_f = binary_temp(ctx, Op::Add, &sum, &zp, "qadd_shifted")?;
    let rounded = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[&shifted_f],
        DType::I32,
        &shape,
        "qadd_rounded",
    );
    let clamped = saturate_i32(ctx, &rounded, -128, 127, "qadd_clamped")?;
    let y = ctx.emit_temp(
        Op::Cast,
        OpAttribute::Cast { input_unsigned: false },
        &[&clamped],
        DType::I8,
        &shape,
        "qlinear_add",
    );
    bind_output(ctx, node, 0, y)
}

/// com.microsoft.QLinearGlobalAveragePool: inputs X, x_scale, x_zp, y_scale, y_zp;
/// `channels_last` attribute. onnxruntime sums (x - x_zp) in INT32 and
/// requantizes with `x_scale / (y_scale * count)`.
fn qlinear_global_average_pool(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let xq = qparams(ctx, node.input(1)?, Some(node.input(2)?))?;
    let yq = qparams(ctx, node.input(3)?, Some(node.input(4)?))?;
    let channels_last = node.attr_int_or("channels_last", 0)? != 0;
    let rank = x.shape.len();
    let spatial: Vec<usize> = if channels_last { (1..rank - 1).collect() } else { (2..rank).collect() };
    let count: i64 = spatial.iter().map(|&a| x.shape[a] as i64).product();
    let shape = x.shape.clone();
    let x32 =
        ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], DType::I32, &shape, "qgap_i32");
    let zp = ctx.block.constant("qgap_x_zp", &vec![1; rank], TensorData::I32(vec![xq.zero_points[0] as i32]))?;
    let mut acc = binary_temp(ctx, Op::Sub, &x32, &zp, "qgap_centred")?;
    for &a in &spatial {
        acc = reduce_axis(ctx, Op::ReduceSum, &acc, a, "qgap_sum");
    }
    let y = requant_i32(ctx, &acc, &[xq.scales[0] / (yq.scales[0] * count as f32)], yq.zero_points[0], 1, "qgap_out")?;
    bind_output(ctx, node, 0, y)
}
