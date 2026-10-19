//! AVG_POOL2D, MAX_POOL2D and ARGMAX.
//!
//! Pooling: input [N,IH,IW,C] -> output [N,OH,OW,C] with `kernel` [KY,KX],
//! `stride` [SY,SX] and `pad` [top,bottom,left,right]; padding positions are
//! excluded from both the maximum and the average (and its divisor).

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, NanMode, OpAttribute, Operator};
use crate::ops::common::{
    check_declared_output, check_nan_mode, expect_operands, expect_rank, float_data, float_lowest,
    float_values, int_data, int_range, int_values, invalid, nan_max, round_to, single_value,
};
use crate::ops::quant::{apply_scale_32, reciprocal_scale};
use crate::tensor::Tensor;

struct PoolGeometry {
    n: usize,
    ih: usize,
    iw: usize,
    c: usize,
    oh: usize,
    ow: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    pad_top: usize,
    pad_left: usize,
}

impl PoolGeometry {
    fn new(op: &str, input: &Tensor, kernel: &[i32], stride: &[i32], pad: &[i32]) -> Result<Self> {
        expect_rank(op, "input", input, 4)?;
        if pad.len() != 4 || kernel.len() != 2 || stride.len() != 2 {
            return Err(invalid(op, "pad must have 4 elements, kernel and stride 2"));
        }
        if pad.iter().any(|&p| p < 0) {
            return Err(invalid(op, format!("pad {pad:?} must be non-negative")));
        }
        if kernel.iter().chain(stride).any(|&k| k < 1) {
            return Err(invalid(op, "kernel and stride must be at least 1"));
        }
        if pad[0] >= kernel[0] || pad[1] >= kernel[0] || pad[2] >= kernel[1] || pad[3] >= kernel[1] {
            return Err(invalid(op, format!("pad {pad:?} must be smaller than kernel {kernel:?}")));
        }
        let [n, ih, iw, c] = input.shape[..] else { unreachable!() };
        if ih == 0 || iw == 0 {
            // Every window must contain at least one input element.
            return Err(invalid(op, "input height and width must be positive"));
        }
        let extent = |size: usize, a: i32, b: i32, k: i32, s: i32| -> Result<usize> {
            let full = size as i64 + a as i64 + b as i64 - k as i64;
            if full < 0 || full % s as i64 != 0 {
                return Err(invalid(op, "parameters must yield exact integer output dimensions"));
            }
            Ok((full / s as i64 + 1) as usize)
        };
        Ok(PoolGeometry {
            n,
            ih,
            iw,
            c,
            oh: extent(ih, pad[0], pad[1], kernel[0], stride[0])?,
            ow: extent(iw, pad[2], pad[3], kernel[1], stride[1])?,
            kernel: [kernel[0] as usize, kernel[1] as usize],
            stride: [stride[0] as usize, stride[1] as usize],
            pad_top: pad[0] as usize,
            pad_left: pad[2] as usize,
        })
    }

    fn out_shape(&self) -> Vec<usize> {
        vec![self.n, self.oh, self.ow, self.c]
    }

    /// Call `f(input_indices)` for every output element, in output order, with the
    /// in-bounds input indices of its window in (ky, kx) order.
    fn for_each_window(&self, mut f: impl FnMut(&[usize])) {
        let mut window = Vec::with_capacity(self.kernel[0] * self.kernel[1]);
        for b in 0..self.n {
            for oy in 0..self.oh {
                let iy = (oy * self.stride[0]) as isize - self.pad_top as isize;
                for ox in 0..self.ow {
                    let ix = (ox * self.stride[1]) as isize - self.pad_left as isize;
                    for c in 0..self.c {
                        window.clear();
                        for ky in 0..self.kernel[0] as isize {
                            let y = iy + ky;
                            if y < 0 || y >= self.ih as isize {
                                continue;
                            }
                            for kx in 0..self.kernel[1] as isize {
                                let x = ix + kx;
                                if x < 0 || x >= self.iw as isize {
                                    continue;
                                }
                                window.push(((b * self.ih + y as usize) * self.iw + x as usize) * self.c + c);
                            }
                        }
                        f(&window);
                    }
                }
            }
        }
    }
}

/// Kernel, stride, pad, accumulator type and NaN mode of a pooling operator.
type PoolAttributes<'a> = (&'a [i32], &'a [i32], &'a [i32], DType, NanMode);

fn pool_attributes<'a>(op: &str, operator: &'a Operator) -> Result<PoolAttributes<'a>> {
    match &operator.attribute {
        OpAttribute::Pool2d { kernel, stride, pad, acc_type, nan_mode } => {
            Ok((kernel, stride, pad, *acc_type, *nan_mode))
        }
        _ => Err(invalid(op, "missing attribute")),
    }
}

/// The output must have the input's type.
fn check_output_dtype(op: &str, declared: Option<DType>, input: DType) -> Result<()> {
    match declared {
        Some(d) if d != input => Err(Error::DTypeMismatch(format!(
            "{op}: output dtype {d:?} must match input dtype {input:?}"
        ))),
        _ => Ok(()),
    }
}

/// Values of the SHAPE operand `index` of an adaptive pool.
fn shape_attribute(state: &InterpreterState, op: &str, operator: &Operator, index: usize) -> Result<Vec<i32>> {
    let t = state.get_tensor(&operator.inputs[index])?;
    int_values(&t.data)
        .and_then(|v| v.into_iter().map(|x| i32::try_from(x).ok()).collect())
        .ok_or_else(|| invalid(op, "kernel, stride and pad must be SHAPE values"))
}

/// Kernel, stride and pad of an adaptive pool, from operands `first..first + 3`.
fn adaptive_parameters(state: &InterpreterState, op: &str, operator: &Operator, first: usize) -> Result<[Vec<i32>; 3]> {
    Ok([
        shape_attribute(state, op, operator, first)?,
        shape_attribute(state, op, operator, first + 1)?,
        shape_attribute(state, op, operator, first + 2)?,
    ])
}

pub fn eval_avg_pool2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "AVG_POOL2D";
    let (kernel, stride, pad, acc_t, _) = pool_attributes(OP, op)?;
    expect_operands(op, OP, 3, 1)?;
    avg_pool(state, op, OP, kernel, stride, pad, acc_t)
}

/// AVG_POOL2D_ADAPTIVE: AVG_POOL2D with kernel, stride and pad given as SHAPE
/// operands 3, 4 and 5.
pub fn eval_avg_pool2d_adaptive(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "AVG_POOL2D_ADAPTIVE";
    let (_, _, _, acc_t, _) = pool_attributes(OP, op)?;
    expect_operands(op, OP, 6, 1)?;
    let [kernel, stride, pad] = adaptive_parameters(state, OP, op, 3)?;
    avg_pool(state, op, OP, &kernel, &stride, &pad, acc_t)
}

/// Average pooling of operand 0, with zero points in operands 1 and 2.
fn avg_pool(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
    acc_t: DType,
) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let input_zp = single_value(name, "input_zp", state.get_tensor(&op.inputs[1])?)?;
    let output_zp = single_value(name, "output_zp", state.get_tensor(&op.inputs[2])?)?;
    let g = PoolGeometry::new(name, input, kernel, stride, pad)?;
    let out_shape = g.out_shape();
    let in_t = input.dtype();
    check_output_dtype(name, check_declared_output(state, name, &op.outputs[0], &out_shape)?, in_t)?;

    use DType::*;
    let supported = matches!(
        (in_t, acc_t),
        (I8 | I16, I32) | (F16, F16 | F32) | (Bf16, F32) | (F32, F32) | (Fp8E4M3 | Fp8E5M2, F16)
    );
    if !supported {
        return Err(Error::DTypeMismatch(format!("{name}: unsupported types input={in_t:?} acc={acc_t:?}")));
    }
    if input_zp.dtype() != in_t || output_zp.dtype() != in_t {
        return Err(Error::DTypeMismatch(format!("{name}: zero points must have the input dtype {in_t:?}")));
    }

    let output = if in_t.is_int() {
        let in_zp = int_values(&input_zp).unwrap()[0];
        let out_zp = int_values(&output_zp).unwrap()[0];
        if in_t != I8 && (in_zp != 0 || out_zp != 0) {
            return Err(invalid(name, "zero points must be zero for non-int8 data"));
        }
        let x = int_values(&input.data).unwrap();
        let (lo, hi) = int_range(in_t);
        let mut out = Vec::with_capacity(x.len());
        let mut err = None;
        g.for_each_window(|w| {
            let acc = w.iter().fold(0i32, |acc, &i| acc.wrapping_add((x[i] - in_zp) as i32));
            let (multiplier, shift) = reciprocal_scale(w.len() as i32);
            match apply_scale_32(acc, multiplier, shift, false) {
                Ok(v) => out.push((v as i64 + out_zp).clamp(lo, hi)),
                Err(e) => err = Some(e),
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        int_data(in_t, out)?
    } else {
        let zp = [float_values(&input_zp).unwrap()[0], float_values(&output_zp).unwrap()[0]];
        if zp.iter().any(|&z| z != 0.0) {
            return Err(invalid(name, "zero points must be zero for floating-point data"));
        }
        let x = float_values(&input.data).unwrap();
        let mut out = Vec::with_capacity(x.len());
        g.for_each_window(|w| {
            // Every add and the division are rounded to the accumulator type.
            let acc = w.iter().fold(0f32, |acc, &i| round_to(acc_t, acc + x[i]));
            out.push(round_to(acc_t, acc / w.len() as f32));
        });
        float_data(in_t, out)?
    };
    state.set_tensor(Tensor::new(op.outputs[0].clone(), out_shape, output));
    Ok(())
}

pub fn eval_max_pool2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "MAX_POOL2D";
    let (kernel, stride, pad, _, nan_mode) = pool_attributes(OP, op)?;
    expect_operands(op, OP, 1, 1)?;
    max_pool(state, op, OP, kernel, stride, pad, nan_mode)
}

/// MAX_POOL2D_ADAPTIVE: MAX_POOL2D with kernel, stride and pad given as SHAPE
/// operands 1, 2 and 3.
pub fn eval_max_pool2d_adaptive(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "MAX_POOL2D_ADAPTIVE";
    let (_, _, _, _, nan_mode) = pool_attributes(OP, op)?;
    expect_operands(op, OP, 4, 1)?;
    let [kernel, stride, pad] = adaptive_parameters(state, OP, op, 1)?;
    max_pool(state, op, OP, &kernel, &stride, &pad, nan_mode)
}

/// Max pooling of operand 0.
fn max_pool(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
    nan_mode: NanMode,
) -> Result<()> {
    let nan_mode = check_nan_mode(name, nan_mode)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let g = PoolGeometry::new(name, input, kernel, stride, pad)?;
    let out_shape = g.out_shape();
    let in_t = input.dtype();
    check_output_dtype(name, check_declared_output(state, name, &op.outputs[0], &out_shape)?, in_t)?;

    use DType::*;
    let output = match in_t {
        I8 | I16 => {
            let x = int_values(&input.data).unwrap();
            let mut out = Vec::with_capacity(x.len());
            g.for_each_window(|w| out.push(w.iter().map(|&i| x[i]).fold(int_range(in_t).0, i64::max)));
            int_data(in_t, out)?
        }
        F16 | Bf16 | F32 | Fp8E4M3 | Fp8E5M2 => {
            let x = float_values(&input.data).unwrap();
            let init = if nan_mode == NanMode::Ignore { f32::NAN } else { float_lowest(in_t) };
            let mut out = Vec::with_capacity(x.len());
            g.for_each_window(|w| out.push(w.iter().fold(init, |m, &i| nan_max(m, x[i], nan_mode))));
            float_data(in_t, out)?
        }
        other => return Err(Error::DTypeMismatch(format!("{name}: unsupported dtype {other:?}"))),
    };
    state.set_tensor(Tensor::new(op.outputs[0].clone(), out_shape, output));
    Ok(())
}

pub fn eval_argmax(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "ARGMAX";
    let OpAttribute::ArgMax { axis, nan_mode } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 1, 1)?;
    let nan_mode = check_nan_mode(OP, nan_mode)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let rank = input.shape.len();
    if rank == 0 || axis < 0 || axis as usize >= rank {
        return Err(invalid(OP, format!("axis {axis} out of range for rank {rank}")));
    }
    let axis = axis as usize;
    let mut out_shape = input.shape.clone();
    out_shape.remove(axis);
    let out_t = check_declared_output(state, OP, &op.outputs[0], &out_shape)?.unwrap_or(DType::I32);
    if !matches!(out_t, DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch(format!("{OP}: output dtype {out_t:?} must be INT32 or INT64")));
    }

    let (outer, len) = (input.shape[..axis].iter().product::<usize>(), input.shape[axis]);
    let inner: usize = input.shape[axis + 1..].iter().product();
    let index = |o: usize, a: usize, i: usize| (o * len + a) * inner + i;

    // Follows the reference: `result = max(value, best)` replaces `best` when
    // it differs from it, except that a NaN never replaces a NaN.
    let mut out = Vec::with_capacity(outer * inner);
    let in_t = input.dtype();
    if let Some(x) = int_values(&input.data) {
        if !matches!(in_t, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
            return Err(Error::DTypeMismatch(format!("{OP}: unsupported dtype {in_t:?}")));
        }
        for o in 0..outer {
            for i in 0..inner {
                let (mut best, mut best_idx) = (int_range(in_t).0, 0);
                for a in 0..len {
                    if x[index(o, a, i)] > best {
                        best = x[index(o, a, i)];
                        best_idx = a as i64;
                    }
                }
                out.push(best_idx);
            }
        }
    } else if let Some(x) = float_values(&input.data) {
        let init = if nan_mode == NanMode::Ignore { f32::NAN } else { float_lowest(in_t) };
        for o in 0..outer {
            for i in 0..inner {
                let (mut best, mut best_idx) = (init, 0);
                for a in 0..len {
                    let result = nan_max(x[index(o, a, i)], best, nan_mode);
                    if result != best && !(result.is_nan() && best.is_nan()) {
                        best = result;
                        best_idx = a as i64;
                    }
                }
                out.push(best_idx);
            }
        }
    } else {
        return Err(Error::DTypeMismatch(format!("{OP}: unsupported dtype {in_t:?}")));
    }
    state.set_tensor(Tensor::new(op.outputs[0].clone(), out_shape, int_data(out_t, out)?));
    Ok(())
}
