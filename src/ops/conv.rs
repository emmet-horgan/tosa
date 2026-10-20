//! Convolutions: CONV2D, CONV3D, DEPTHWISE_CONV2D and TRANSPOSE_CONV2D.
//!
//! All four share operands (input, weight, bias, input_zp, weight_zp), type
//! rules and arithmetic, and differ only in their loop structure:
//!
//! | op               | input          | weight            | output                |
//! |------------------|----------------|-------------------|-----------------------|
//! | CONV2D           | [N,IH,IW,IC]   | [OC,KH,KW,IC]     | [N,OH,OW,OC]          |
//! | CONV3D           | [N,ID,IH,IW,IC]| [OC,KD,KH,KW,IC]  | [N,OD,OH,OW,OC]       |
//! | DEPTHWISE_CONV2D | [N,IH,IW,C]    | [KH,KW,C,M]       | [N,OH,OW,C*M]         |
//! | TRANSPOSE_CONV2D | [N,IH,IW,IC]   | [OC,KH,KW,IC]     | [N,OH,OW,OC]          |
//!
//! Bias is [OC] or [1]. Semantics follow the TOSA reference model bit for bit:
//! operands minus their zero points are multiplied and accumulated in
//! `acc_type` (see [`crate::ops::accum`]) in the reference's loop order,
//! positions in the padding contribute nothing, and the bias is added after
//! the accumulator has been converted to the output type:
//! `out = cast<out>(acc) + bias`. INT48 results are clamped to 48 bits.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::ops::accum::{AccBf16, AccF16, AccF32, AccI32, AccI64, Accumulator};
use crate::ops::common::{
    check_declared_output, expect_operands, expect_rank, float_data, float_values, int_values, invalid, round_to,
    single_value,
};
use crate::tensor::{Tensor, TensorData};

const INT48_MIN: i64 = -(1 << 47);
const INT48_MAX: i64 = (1 << 47) - 1;

/// Loop structure of one convolution kind, with validated geometry.
trait ConvLoop {
    fn out_shape(&self) -> Vec<usize>;

    /// Accumulators for every output element, in row-major output order.
    fn accumulate<A: Accumulator>(&self, input: &[A::Operand], weight: &[A::Operand]) -> Vec<A>;
}

fn to_usize(v: &[i32]) -> Vec<usize> {
    v.iter().map(|&x| x as usize).collect()
}

/// Row-major multi-index of flat index `i` in `shape`.
fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = i % shape[d];
        i /= shape[d];
    }
    idx
}

/// One output extent of a forward convolution; the kernel window must tile
/// the padded input exactly.
fn output_extent(
    op: &str,
    input: usize,
    pad_before: i32,
    pad_after: i32,
    kernel: usize,
    dilation: i32,
    stride: i32,
) -> Result<usize> {
    let full = input as i64 - 1 + pad_before as i64 + pad_after as i64 - (kernel as i64 - 1) * dilation as i64;
    if full % stride as i64 != 0 {
        return Err(invalid(op, "parameters must yield exact integer output dimensions"));
    }
    let out = full / stride as i64 + 1;
    if out < 1 {
        return Err(invalid(op, format!("output dimension {out} must be positive")));
    }
    Ok(out as usize)
}

fn check_conv_attributes(op: &str, dims: usize, pad: &[i32], stride: &[i32], dilation: &[i32]) -> Result<()> {
    if pad.len() != 2 * dims || stride.len() != dims || dilation.len() != dims {
        return Err(invalid(op, format!("pad must have {} elements, stride and dilation {dims}", 2 * dims)));
    }
    if pad.iter().any(|&p| p < 0) {
        return Err(invalid(op, format!("pad {pad:?} must be non-negative")));
    }
    if stride.iter().any(|&s| s < 1) {
        return Err(invalid(op, format!("stride {stride:?} must be at least 1")));
    }
    if dilation.iter().any(|&d| d < 1) {
        return Err(invalid(op, format!("dilation {dilation:?} must be at least 1")));
    }
    Ok(())
}

/// CONV2D and CONV3D: `dims` spatial dimensions, weight [OC, K..., IC].
struct ConvNd {
    n: usize,
    in_sp: Vec<usize>,
    ic: usize,
    oc: usize,
    kernel: Vec<usize>,
    out_sp: Vec<usize>,
    pad_before: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
}

impl ConvNd {
    fn new(
        op: &str,
        dims: usize,
        input: &Tensor,
        weight: &Tensor,
        pad: &[i32],
        stride: &[i32],
        dilation: &[i32],
    ) -> Result<Self> {
        expect_rank(op, "input", input, dims + 2)?;
        expect_rank(op, "weight", weight, dims + 2)?;
        check_conv_attributes(op, dims, pad, stride, dilation)?;
        let (ic, kic) = (input.shape[dims + 1], weight.shape[dims + 1]);
        if kic != ic {
            return Err(Error::ShapeMismatch(format!("{op}: weight input channels {kic} != input channels {ic}")));
        }
        let in_sp = input.shape[1..=dims].to_vec();
        let kernel = weight.shape[1..=dims].to_vec();
        let out_sp = (0..dims)
            .map(|d| output_extent(op, in_sp[d], pad[2 * d], pad[2 * d + 1], kernel[d], dilation[d], stride[d]))
            .collect::<Result<_>>()?;
        Ok(ConvNd {
            n: input.shape[0],
            in_sp,
            ic,
            oc: weight.shape[0],
            kernel,
            out_sp,
            pad_before: (0..dims).map(|d| pad[2 * d] as usize).collect(),
            stride: to_usize(stride),
            dilation: to_usize(dilation),
        })
    }
}

impl ConvLoop for ConvNd {
    fn out_shape(&self) -> Vec<usize> {
        [vec![self.n], self.out_sp.clone(), vec![self.oc]].concat()
    }

    /// Reference loop order: batch, output position, oc, then kernel
    /// position (row-major) and ic innermost.
    fn accumulate<A: Accumulator>(&self, input: &[A::Operand], weight: &[A::Operand]) -> Vec<A> {
        let dims = self.in_sp.len();
        let k_count: usize = self.kernel.iter().product();
        let out_count: usize = self.out_sp.iter().product();
        let kernel_pos: Vec<Vec<usize>> = (0..k_count).map(|i| unravel(i, &self.kernel)).collect();
        let mut out = Vec::with_capacity(self.n * out_count * self.oc);
        for b in 0..self.n {
            for o in 0..out_count {
                let opos = unravel(o, &self.out_sp);
                for oc in 0..self.oc {
                    let mut acc = A::zero();
                    'taps: for (ki, kpos) in kernel_pos.iter().enumerate() {
                        let mut flat = b;
                        for d in 0..dims {
                            let x = (opos[d] * self.stride[d] + kpos[d] * self.dilation[d]) as isize
                                - self.pad_before[d] as isize;
                            if x < 0 || x >= self.in_sp[d] as isize {
                                continue 'taps;
                            }
                            flat = flat * self.in_sp[d] + x as usize;
                        }
                        let i_base = flat * self.ic;
                        let w_base = (oc * k_count + ki) * self.ic;
                        for c in 0..self.ic {
                            acc = acc.mac(input[i_base + c], weight[w_base + c]);
                        }
                    }
                    out.push(acc);
                }
            }
        }
        out
    }
}

/// DEPTHWISE_CONV2D: weight [KH, KW, C, M], output channel `c * M + m`.
struct Depthwise {
    n: usize,
    ih: usize,
    iw: usize,
    c: usize,
    m: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    pad_top: usize,
    pad_left: usize,
    stride: [usize; 2],
    dilation: [usize; 2],
}

impl Depthwise {
    fn new(op: &str, input: &Tensor, weight: &Tensor, pad: &[i32], stride: &[i32], dilation: &[i32]) -> Result<Self> {
        expect_rank(op, "input", input, 4)?;
        expect_rank(op, "weight", weight, 4)?;
        check_conv_attributes(op, 2, pad, stride, dilation)?;
        let [n, ih, iw, c] = input.shape[..] else { unreachable!() };
        let [kh, kw, kc, m] = weight.shape[..] else { unreachable!() };
        if kc != c {
            return Err(Error::ShapeMismatch(format!("{op}: weight channels {kc} != input channels {c}")));
        }
        Ok(Depthwise {
            n,
            ih,
            iw,
            c,
            m,
            kh,
            kw,
            oh: output_extent(op, ih, pad[0], pad[1], kh, dilation[0], stride[0])?,
            ow: output_extent(op, iw, pad[2], pad[3], kw, dilation[1], stride[1])?,
            pad_top: pad[0] as usize,
            pad_left: pad[2] as usize,
            stride: [stride[0] as usize, stride[1] as usize],
            dilation: [dilation[0] as usize, dilation[1] as usize],
        })
    }
}

impl ConvLoop for Depthwise {
    fn out_shape(&self) -> Vec<usize> {
        vec![self.n, self.oh, self.ow, self.c * self.m]
    }

    fn accumulate<A: Accumulator>(&self, input: &[A::Operand], weight: &[A::Operand]) -> Vec<A> {
        let mut out = Vec::with_capacity(self.n * self.oh * self.ow * self.c * self.m);
        for b in 0..self.n {
            for oy in 0..self.oh {
                let iy = (oy * self.stride[0]) as isize - self.pad_top as isize;
                for ox in 0..self.ow {
                    let ix = (ox * self.stride[1]) as isize - self.pad_left as isize;
                    for c in 0..self.c {
                        for m in 0..self.m {
                            let mut acc = A::zero();
                            for ky in 0..self.kh {
                                let y = iy + (ky * self.dilation[0]) as isize;
                                if y < 0 || y >= self.ih as isize {
                                    continue;
                                }
                                for kx in 0..self.kw {
                                    let x = ix + (kx * self.dilation[1]) as isize;
                                    if x < 0 || x >= self.iw as isize {
                                        continue;
                                    }
                                    let i = ((b * self.ih + y as usize) * self.iw + x as usize) * self.c + c;
                                    let w = ((ky * self.kw + kx) * self.c + c) * self.m + m;
                                    acc = acc.mac(input[i], weight[w]);
                                }
                            }
                            out.push(acc);
                        }
                    }
                }
            }
        }
        out
    }
}

/// TRANSPOSE_CONV2D: weight [OC, KH, KW, IC]; `out_pad` may be negative.
struct Transpose {
    n: usize,
    ih: usize,
    iw: usize,
    ic: usize,
    oc: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    out_pad_top: isize,
    out_pad_left: isize,
    stride: [usize; 2],
}

impl Transpose {
    fn new(op: &str, input: &Tensor, weight: &Tensor, out_pad: &[i32], stride: &[i32]) -> Result<Self> {
        expect_rank(op, "input", input, 4)?;
        expect_rank(op, "weight", weight, 4)?;
        if out_pad.len() != 4 || stride.len() != 2 {
            return Err(invalid(op, "out_pad must have 4 elements and stride 2"));
        }
        if stride.iter().any(|&s| s < 1) {
            return Err(invalid(op, format!("stride {stride:?} must be at least 1")));
        }
        let [n, ih, iw, ic] = input.shape[..] else { unreachable!() };
        let [oc, kh, kw, kic] = weight.shape[..] else { unreachable!() };
        if kic != ic {
            return Err(Error::ShapeMismatch(format!("{op}: weight input channels {kic} != input channels {ic}")));
        }
        for (i, &p) in out_pad.iter().enumerate() {
            let k = if i < 2 { kh } else { kw };
            if p as i64 <= -(k as i64) {
                return Err(invalid(op, format!("out_pad {out_pad:?} must be greater than -kernel size")));
            }
        }
        let extent =
            |i: usize, s: i32, a: i32, b: i32, k: usize| (i as i64 - 1) * s as i64 + a as i64 + b as i64 + k as i64;
        let oh = extent(ih, stride[0], out_pad[0], out_pad[1], kh);
        let ow = extent(iw, stride[1], out_pad[2], out_pad[3], kw);
        if oh < 1 || ow < 1 {
            return Err(invalid(op, format!("output size {oh}x{ow} must be positive")));
        }
        Ok(Transpose {
            n,
            ih,
            iw,
            ic,
            oc,
            kh,
            kw,
            oh: oh as usize,
            ow: ow as usize,
            out_pad_top: out_pad[0] as isize,
            out_pad_left: out_pad[2] as isize,
            stride: [stride[0] as usize, stride[1] as usize],
        })
    }
}

impl ConvLoop for Transpose {
    fn out_shape(&self) -> Vec<usize> {
        vec![self.n, self.oh, self.ow, self.oc]
    }

    /// Gather form used by the reference: output (oy, ox) reads input
    /// ((oy - out_pad_top - ky) / stride_y, ...) where that division is exact.
    fn accumulate<A: Accumulator>(&self, input: &[A::Operand], weight: &[A::Operand]) -> Vec<A> {
        let [sy, sx] = self.stride.map(|s| s as isize);
        let mut out = Vec::with_capacity(self.n * self.oh * self.ow * self.oc);
        for b in 0..self.n {
            for oy in 0..self.oh {
                let iy = oy as isize - self.out_pad_top;
                for ox in 0..self.ow {
                    let ix = ox as isize - self.out_pad_left;
                    for oc in 0..self.oc {
                        let mut acc = A::zero();
                        for ky in 0..self.kh {
                            let y = iy - ky as isize;
                            if y < 0 || y >= self.ih as isize * sy || y % sy != 0 {
                                continue;
                            }
                            for kx in 0..self.kw {
                                let x = ix - kx as isize;
                                if x < 0 || x >= self.iw as isize * sx || x % sx != 0 {
                                    continue;
                                }
                                let (y, x) = ((y / sy) as usize, (x / sx) as usize);
                                let i_base = ((b * self.ih + y) * self.iw + x) * self.ic;
                                let w_base = ((oc * self.kh + ky) * self.kw + kx) * self.ic;
                                for c in 0..self.ic {
                                    acc = acc.mac(input[i_base + c], weight[w_base + c]);
                                }
                            }
                        }
                        out.push(acc);
                    }
                }
            }
        }
        out
    }
}

/// CONV2D's (input, weight, acc, output) type combinations.
fn conv2d_types(input: DType, weight: DType, acc: DType, out: DType) -> bool {
    use DType::*;
    let fp8 = |t: DType| matches!(t, Fp8E4M3 | Fp8E5M2);
    match (input, weight, acc, out) {
        (I8, I8 | I4, I32, I32) | (I16, I8, I48, I48) => true,
        (i, w, F32, F32) => [i, w].iter().all(|&t| fp8(t) || matches!(t, F16 | Bf16 | F32)),
        (i, w, F16 | F32, F16) => [i, w].iter().all(|&t| fp8(t) || t == F16),
        (i, w, Bf16 | F32, Bf16) => [i, w].iter().all(|&t| fp8(t) || t == Bf16),
        _ => false,
    }
}

/// Type combinations of CONV3D, DEPTHWISE_CONV2D and TRANSPOSE_CONV2D.
fn standard_types(input: DType, weight: DType, acc: DType, out: DType) -> bool {
    use DType::*;
    match (input, weight, acc, out) {
        (I8, I8 | I4, I32, I32) | (I16, I8, I48, I48) => true,
        (F16, F16, F16 | F32, F16) | (Bf16, Bf16, F32, Bf16) | (F32, F32, F32, F32) => true,
        (Fp8E4M3, Fp8E4M3, _, _) | (Fp8E5M2, Fp8E5M2, _, _) => matches!((acc, out), (F16, F16) | (F32, F32)),
        _ => false,
    }
}

fn eval_conv<L: ConvLoop>(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    acc_t: DType,
    types_ok: fn(DType, DType, DType, DType) -> bool,
    geometry: impl FnOnce(&Tensor, &Tensor) -> Result<L>,
) -> Result<()> {
    expect_operands(op, name, 5, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let weight = state.get_tensor(&op.inputs[1])?;
    let bias = state.get_tensor(&op.inputs[2])?;
    let input_zp = single_value(name, "input_zp", state.get_tensor(&op.inputs[3])?)?;
    let weight_zp = single_value(name, "weight_zp", state.get_tensor(&op.inputs[4])?)?;
    let output_name = &op.outputs[0];

    expect_rank(name, "bias", bias, 1)?;
    let g = geometry(input, weight)?;
    let out_shape = g.out_shape();
    let oc = *out_shape.last().unwrap();
    let bc = bias.shape[0];
    if bc != oc && bc != 1 {
        return Err(Error::ShapeMismatch(format!(
            "{name}: bias channels {bc} must be 1 or match output channels {oc}"
        )));
    }

    let (in_t, w_t, out_t) = (input.dtype(), weight.dtype(), bias.dtype());
    if let Some(declared) = check_declared_output(state, name, output_name, &out_shape)?
        && declared != out_t
    {
        return Err(Error::DTypeMismatch(format!(
            "{name}: output dtype {declared:?} does not match bias dtype {out_t:?}"
        )));
    }
    if !types_ok(in_t, w_t, acc_t, out_t) {
        return Err(Error::DTypeMismatch(format!(
            "{name}: unsupported types input={in_t:?} weight={w_t:?} acc={acc_t:?} output={out_t:?}"
        )));
    }
    if input_zp.dtype() != in_t || weight_zp.dtype() != w_t {
        return Err(Error::DTypeMismatch(format!(
            "{name}: zero points must match input/weight dtypes, got {:?}/{:?}",
            input_zp.dtype(),
            weight_zp.dtype()
        )));
    }

    let bias_idx = |i: usize| if bc == 1 { 0 } else { i % oc };
    let output = if acc_t.is_int() {
        // Every operand is an integer tensor here; see the type tables.
        let in_zp = int_values(&input_zp).unwrap()[0];
        let w_zp = int_values(&weight_zp).unwrap()[0];
        if in_zp != 0 && in_t != DType::I8 {
            return Err(invalid(name, "input zero point must be zero for non-int8 input"));
        }
        if w_zp != 0 && w_t != DType::I8 {
            return Err(invalid(name, "weight zero point must be zero for non-int8 weight"));
        }
        let inp: Vec<i64> = int_values(&input.data).unwrap().iter().map(|&v| v - in_zp).collect();
        let wgt: Vec<i64> = int_values(&weight.data).unwrap().iter().map(|&v| v - w_zp).collect();
        let bis = int_values(&bias.data).unwrap();

        if acc_t == DType::I32 {
            let to_i32 = |v: Vec<i64>| -> Vec<i32> { v.into_iter().map(|x| x as i32).collect() };
            let acc = g.accumulate::<AccI32>(&to_i32(inp), &to_i32(wgt));
            let out = acc.iter().enumerate().map(|(i, a)| a.0.wrapping_add(bis[bias_idx(i)] as i32)).collect();
            TensorData::I32(out)
        } else {
            let acc = g.accumulate::<AccI64>(&inp, &wgt);
            let out = acc
                .iter()
                .enumerate()
                .map(|(i, a)| a.0.wrapping_add(bis[bias_idx(i)]).clamp(INT48_MIN, INT48_MAX))
                .collect();
            TensorData::I48(out)
        }
    } else {
        // Float zero points must be zero; the subtraction is still evaluated
        // (in f32) to mirror the reference.
        let in_zp = float_values(&input_zp).unwrap()[0];
        let w_zp = float_values(&weight_zp).unwrap()[0];
        if in_zp != 0.0 || w_zp != 0.0 {
            return Err(invalid(name, "zero points must be zero for floating-point operands"));
        }
        let inp: Vec<f32> = float_values(&input.data).unwrap().iter().map(|&v| v - in_zp).collect();
        let wgt: Vec<f32> = float_values(&weight.data).unwrap().iter().map(|&v| v - w_zp).collect();
        let bis = float_values(&bias.data).unwrap();

        let acc: Vec<f32> = match acc_t {
            DType::F32 => g.accumulate::<AccF32>(&inp, &wgt).iter().map(|a| a.0).collect(),
            DType::F16 => g.accumulate::<AccF16>(&inp, &wgt).iter().map(|a| a.0).collect(),
            DType::Bf16 => g.accumulate::<AccBf16>(&inp, &wgt).iter().map(|a| a.0).collect(),
            _ => unreachable!("the type tables admit only float accumulators here"),
        };
        let out =
            acc.iter().enumerate().map(|(i, &a)| round_to(out_t, round_to(out_t, a) + bis[bias_idx(i)])).collect();
        float_data(out_t, out)?
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, output));
    Ok(())
}

pub fn eval_conv2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CONV2D";
    let OpAttribute::Conv2d { pad, stride, dilation, acc_type } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    eval_conv(state, op, OP, *acc_type, conv2d_types, |i, w| ConvNd::new(OP, 2, i, w, pad, stride, dilation))
}

pub fn eval_conv3d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CONV3D";
    let OpAttribute::Conv3d { pad, stride, dilation, acc_type } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    eval_conv(state, op, OP, *acc_type, standard_types, |i, w| ConvNd::new(OP, 3, i, w, pad, stride, dilation))
}

pub fn eval_depthwise_conv2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "DEPTHWISE_CONV2D";
    let OpAttribute::DepthwiseConv2d { pad, stride, dilation, acc_type } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    eval_conv(state, op, OP, *acc_type, standard_types, |i, w| Depthwise::new(OP, i, w, pad, stride, dilation))
}

pub fn eval_transpose_conv2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "TRANSPOSE_CONV2D";
    let OpAttribute::TransposeConv2d { out_pad, stride, acc_type } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    eval_conv(state, op, OP, *acc_type, standard_types, |i, w| Transpose::new(OP, i, w, out_pad, stride))
}
