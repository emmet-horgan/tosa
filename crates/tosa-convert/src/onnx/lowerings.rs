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

fn binary(node: &SourceNode, ctx: &mut LowerCtx<'_>, op: Op) -> Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    let y = binary_temp(ctx, op, &a, &b, "binary")?;
    bind_output(ctx, node, 0, y)
}

fn div(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> Result<()> {
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
    let attr = OpAttribute::NanPropagation { nan_mode: NanMode::Propagate };
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
    if mode != "constant" {
        return ctx.fail(format!("pad mode {mode} is not supported"));
    }
    // ONNX: [begin_0, ..., begin_n, end_0, ..., end_n]; TOSA: [begin_0, end_0, ...].
    let tosa_pads: Vec<i64> = (0..rank).flat_map(|d| [pads[d], pads[rank + d]]).collect();
    if tosa_pads.iter().any(|&p| p < 0) {
        return ctx.fail("negative pads are not supported");
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
    // Shape arithmetic: all constants of integer type stay a constant.
    if ids.iter().all(|&id| ctx.is_constant(id) && !ctx.is_available_as_value(id)) {
        let mut values = Vec::new();
        for &id in &ids {
            match ctx.constant(id)? {
                TensorData::I64(v) => values.extend_from_slice(v),
                TensorData::I32(v) => values.extend(v.iter().map(|&x| x as i64)),
                _ => {
                    values.clear();
                    break;
                }
            }
        }
        if !values.is_empty() {
            ctx.define_constant(node.output(0)?, TensorData::I64(values));
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
        super::proto::data_type::INT64 => DType::I64,
        super::proto::data_type::INT32 => DType::I32,
        super::proto::data_type::BOOL => DType::Bool,
        super::proto::data_type::FLOAT16 => DType::F16,
        super::proto::data_type::INT8 => DType::I8,
        super::proto::data_type::INT16 => DType::I16,
        other => return ctx.fail(format!("Cast to ONNX type {other} is not supported")),
    };
    if ctx.is_constant(in_id) && !ctx.is_available_as_value(in_id) {
        let data = match (ctx.constant(in_id)?, dtype) {
            (TensorData::I64(v), DType::I32) => TensorData::I32(v.iter().map(|&x| x as i32).collect()),
            (TensorData::I32(v), DType::I64) => TensorData::I64(v.iter().map(|&x| x as i64).collect()),
            (TensorData::I64(v), DType::F32) => TensorData::F32(v.iter().map(|&x| x as f32).collect()),
            (TensorData::F32(v), DType::I64) => TensorData::I64(v.iter().map(|&x| x as i64).collect()),
            (d, t) if d.dtype() == t => d.clone(),
            (d, t) => return ctx.fail(format!("constant Cast from {:?} to {t:?} is not supported", d.dtype())),
        };
        ctx.define_constant(node.output(0)?, data);
        return Ok(());
    }
    let x = ctx.value(in_id)?;
    let shape = x.shape.clone();
    let y = ctx.emit_temp(Op::Cast, OpAttribute::Cast { input_unsigned: false }, &[&x], dtype, &shape, "cast");
    bind_output(ctx, node, 0, y)
}
